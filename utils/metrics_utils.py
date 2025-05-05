import torch
from torchvision.ops import box_iou
from torchvision.ops import masks_to_boxes
import numpy as np


def calculate_metrics(masks, preds,phase="train"):
    try:
        """
        Calculate binary classification metrics for a batch of 2D binary image masks and predictions.

        Parameters:
        - masks: Ground truth binary masks (torch tensor of shape [batch_size, height, width])
        - preds: Predicted binary masks (torch tensor of shape [batch_size, height, width])

        Returns:
        - metrics_dict: Dictionary containing various metrics.
        """
        # Ensure the inputs are binary (0 or 1)
        masks = masks.int()
        preds = (preds >= 0.5).int()  # Threshold predictions at 0.5 for binary classification

        # Flatten the tensors to compare pixel-wise across the entire batch
        masks = masks.view(-1)  # Flatten to [batch_size * height * width]
        preds = preds.view(-1)  # Flatten to [batch_size * height * width]

        # Calculate true positives, false positives, true negatives, false negatives
        tp = (masks * preds).sum().float()  # True Positives
        tn = ((1 - masks) * (1 - preds)).sum().float()  # True Negatives
        fp = ((1 - masks) * preds).sum().float()  # False Positives
        fn = (masks * (1 - preds)).sum().float()  # False Negatives

        # Calculate various metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp + 1e-8)  # Add small epsilon to avoid division by zero
        recall = tp / (tp + fn + 1e-8)
        specificity = tn / (tn + fp + 1e-8)
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)
        iou = tp / (tp + fp + fn + 1e-8)  # Intersection over Union (IoU)
        dice_coeff = (2 * tp) / (2 * tp + fp + fn + 1e-8)

        # Create the metrics dictionary
        metrics_dict = {
            phase+"_PixMetrics/accuracy": float(accuracy.item()),
            phase+"_PixMetrics/precision": float(precision.item()),
            phase+"_PixMetrics/recall": float(recall.item()),
            phase+"_PixMetrics/specificity": float(specificity.item()),
            phase+"_PixMetrics/f1_score": float(f1_score.item()),
            phase+"_PixMetrics/iou": float(iou.item()),
            phase+"_PixMetrics/dice_coeff": float(dice_coeff.item()),
            }

        return metrics_dict,True
    except:
        metrics_dict_ph = {
            phase+"_PixMetrics/accuracy": np.nan,
            phase+"_PixMetrics/precision": np.nan,
            phase+"_PixMetrics/recall": np.nan,
            phase+"_PixMetrics/specificity": np.nan,
            phase+"_PixMetrics/f1_score": np.nan,
            phase+"_PixMetrics/iou": np.nan,
            phase+"_PixMetrics/dice_coeff": np.nan,
            }
        return metrics_dict_ph,True


def calculate_object_metrics(mask, pred, iou_threshold=0.5,phase="train"):
    """
    Calculate object-based metrics for a batch of predicted and ground truth bounding boxes.
    
    Parameters:
    - gt_boxes: Ground truth bounding boxes (list of tensors [num_boxes, 4] for each image)
    - pred_boxes: Predicted bounding boxes (list of tensors [num_boxes, 4] for each image)
    - iou_threshold: IoU threshold to consider a detection as a true positive (default: 0.5)
    - phase: Phase of the model (train or val), to be appended to metrics
    
    Returns:
    - metrics_dict: Dictionary containing precision, recall, and F1-score.
    """
    # Get Valid examples from list of tensors. Valid: Has at least 1 GT and 1 pred
    extract_ls = []
    for v,(m,p) in enumerate(zip(mask,pred)):
        if m.max()>=0.5 and p.max()>=0.5:
            extract_ls.append(v)
    mask = mask[extract_ls]
    pred = pred[extract_ls]

    # If No valid batches, return None and False
    if len(extract_ls)==0:
            metrics_dict_ph = {
                    phase+"_ObjMetrics/precision": np.nan,
                    phase+"_ObjMetrics/recall": np.nan,
                    phase+"_ObjMetrics/f1_score": np.nan,
                    phase+"_ObjMetrics/true_positives": np.nan,
                    phase+"_ObjMetrics/false_positives": np.nan,
                    phase+"_ObjMetrics/false_negatives": np.nan,
                    }
            return metrics_dict_ph,True

        
    gt_boxes,pred_boxes = [],[]
    for m,p in zip(mask,pred):
        # if all 0s, workflow breaks. set 1 corner to 1 in oprder to continue
        """
        if p.max()<=0.5 or :
            p = p.clone()
            p[0, 0] = 1
        if m.max()<=0.5:
            m = m.clone()
            m[0, 0] = 1
        """
        gt_boxes.append(masks_to_boxes(m.squeeze(1)))
        pred_boxes.append(masks_to_boxes(p.squeeze(1)))
    
    # stack boxes of masks and preds
    gt_boxes, pred_boxes = torch.stack(gt_boxes),torch.stack(pred_boxes)

    tp, fp, fn = 0, 0, 0
    for i in range(len(gt_boxes)):
        # Get the IoU between all predicted boxes and ground truth boxes
        if len(pred_boxes[i]) > 0 and len(gt_boxes[i]) > 0:
            ious = box_iou(pred_boxes[i], gt_boxes[i])
            
            # Determine true positives (IoU > threshold)
            gt_matched = torch.zeros(len(gt_boxes[i]), dtype=torch.bool)
            pred_matched = torch.zeros(len(pred_boxes[i]), dtype=torch.bool)
            
            for pred_idx in range(len(pred_boxes[i])):
                iou_max, gt_idx = ious[pred_idx].max(0)
                
                if iou_max >= iou_threshold and not gt_matched[gt_idx]:
                    # True positive if IoU is above the threshold and ground truth has not been matched
                    tp += 1
                    gt_matched[gt_idx] = True
                    pred_matched[pred_idx] = True
                else:
                    # False positive if IoU is below threshold or no match
                    fp += 1

            # False negatives: Ground truth objects not detected
            fn += len(gt_boxes[i]) - gt_matched.sum().item()
        
        # Handle case where no predictions were made
        elif len(pred_boxes[i]) == 0:
            fn += len(gt_boxes[i])
        elif len(gt_boxes[i]) == 0:
            fp += len(pred_boxes[i])

    # Calculate precision, recall, and F1-score
    precision = tp / (tp + fp + 1e-8)  # Add epsilon to avoid division by zero
    recall = tp / (tp + fn + 1e-8)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    metrics_dict = {
        phase+"_ObjMetrics/precision": float(precision),
        phase+"_ObjMetrics/recall": float(recall),
        phase+"_ObjMetrics/f1_score": float(f1_score),
        phase+"_ObjMetrics/true_positives": float(tp),
        phase+"_ObjMetrics/false_positives": float(fp),
        phase+"_ObjMetrics/false_negatives": float(fn),
        }


    return metrics_dict,True



if __name__=="__main__":
    from omegaconf import OmegaConf
    from data.dataset_masks import pl_datamodule
    config = OmegaConf.load("configs/config_hr.yaml")



    pl_dm = pl_datamodule(config)
    images,masks = next(iter(pl_dm.train_dataloader()))
    preds = torch.zeros_like(masks)
    #preds = masks.clone()
    metrics_a = calculate_object_metrics(masks, preds,phase="train")
    print(metrics_a)
    metrics_b = calculate_metrics(masks, preds,phase="train")
    print(metrics_b)

    metrics_a = calculate_object_metrics(torch.zeros(2,3,512,512), torch.zeros(2,1,512,512),phase="train")
