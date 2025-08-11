import torch
import numpy as np

from scipy.ndimage import label


def calculate_metrics(masks, preds, threshhold, phase="train"):
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
        preds = (preds > threshhold).int()  # Threshold predictions at 0.5 for binary classification

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

def calculate_segmentation_metrics(mask, pred):
    # Calculate true positives, false positives, true negatives, false negatives
    tp = (mask * pred).sum()  # True Positives
    tn = ((1 - mask) * (1 - pred)).sum()  # True Negatives
    fp = ((1 - mask) * pred).sum()  # False Positives
    fn = (mask * (1 - pred)).sum()  # False Negatives

    # Calculate various metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp + 1e-8)  # Add small epsilon to avoid division by zero
    recall = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)  # Intersection over Union (IoU)
    dice_coeff = (2 * tp) / (2 * tp + fp + fn + 1e-8)

    return accuracy, precision, recall, specificity, f1_score, iou, dice_coeff

def calculate_object_metrics(mask, pred,
                             object_detection_treshhold=0.5,):
    """
        average_prediction-score (overall_prediction_avg):
             Calculates the overall average prediction score for all objects. calculated as the total sum of pixels detected by the model in the object masks of the gt.
             should reflect how much of the buildings AREA was correctly identified. Per image pixel based method with object mask (extractd from gt)
        found_objects_percentage (overall_found_fraction):
            Calculates the percentage of ground truth objects that are considered 'found' based on the predicted mask.
            An object is considered 'found' if the average predicted score within its region is above the confidence threshold.
            More of an object count metric, counts how many of the buildings can be considered hit by the model

        NONE OF thes consider overestimatioN!!!!!!

    """
    size_ranges = {'0-9': (0, 9),
                   '10-19': (10, 19),
                   '20-34': (20, 34),
                   '35-49': (35, 49),
                   '50-74': (50, 74),
                   '75+': (75, np.inf)}

    # object_detection_threshold != normal threshold. how much are per object has to be found to count as detected?
    total_sum = 0
    total_objects = 0
    found_objects = 0
    results_per_size = {k: {'px_sum': 0, 'object_count': 0, 'found_count': 0} for k in size_ranges.keys()}

    # get objects from gt mask
    labeled_mask, num_objects = label(mask)

    # Iterate over each object in the current mask
    for object_id in range(1, num_objects + 1):
        object_mask = (labeled_mask == object_id)

        object_size = object_mask.sum()
        avg_value = pred[object_mask].mean()

        # Accumulate the sum and count of objects
        total_sum += avg_value
        total_objects += 1

        # have we detected enough of the building area?
        if avg_value >= object_detection_treshhold:
            found_objects += 1

        # assign object a sixe range
        for size_range, (min_size, max_size) in size_ranges.items():
            if min_size <= object_size <= max_size:
                # gt reference of availabele objects
                results_per_size[size_range]['object_count'] += 1

                # pixel sum in these objects we have detected
                results_per_size[size_range]['px_sum'] += avg_value

                # have we detected enough of the building area?
                if avg_value >= object_detection_treshhold:
                    results_per_size[size_range]['found_count'] += 1
                break

    # Compute the overall average prediction score across all objects
    overall_prediction_avg = total_sum / total_objects if total_objects > 0 else 0
    overall_found_fraction = (found_objects / num_objects) if num_objects > 0 else 0

    # compute metrics per size bin
    # overall_prediction_avg_per_size = {}
    # overall_found_fraction_per_size = {}
    overall_per_size = {}
    for size_range, data in results_per_size.items():
        if data['object_count'] > 0:
            overall_per_size[f'prediction_avg_{size_range}'] = data['px_sum'] / data['object_count']
            overall_per_size[f'found_fraction_{size_range}'] = data['found_count'] / data['object_count']
            # overall_prediction_avg_per_size[size_range] = data['px_sum'] / data['object_count']
            # overall_found_fraction_per_size[size_range] = data['found_count'] / data['object_count']
        else:
            # overall_prediction_avg_per_size[size_range] = 0
            # overall_found_fraction_per_size[size_range] = 0
            overall_per_size[f'prediction_avg_{size_range}'] = 0
            overall_per_size[f'found_fraction_{size_range}'] = 0

    return overall_prediction_avg, overall_found_fraction, overall_per_size

def calculate_test_metrics(image_ids, masks, preds, threshhold, phase="test"):
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
    preds = (preds > threshhold).int()  # Threshold predictions at 0.5 for binary classification

    segmentation_metrics = {
                "image_id": [],
                "accuracy": [],
                "precision": [],
                "recall": [],
                "specificity": [],
                "f1_score": [],
                "iou": [],
                "dice_coeff": [],
            }

    object_metrics = {
                "image_id": [],
                "object_prediction_average": [],
                "overall_found_fraction": [],
            }

    object_metrics_per_size = {
        "image_id": [],
            }

    # iterate over batches (slower but wtf) to get individual mask and pred
    # calculate segmentation metrics AND object detextion metrics
    for image_id, mask, pred in zip(image_ids, masks, preds):
        # map to 256x256 np arrays
        mask, pred = mask.squeeze(0).cpu().numpy(), pred.squeeze(0).cpu().numpy()

        accuracy, precision, recall, specificity, f1_score, iou, dice_coeff = calculate_segmentation_metrics(mask, pred)
        overall_prediction_avg, overall_found_fraction, overall_per_size = calculate_object_metrics(mask, pred)

        # utils
        segmentation_metrics["image_id"].append(image_id)
        object_metrics["image_id"].append(image_id)
        object_metrics_per_size["image_id"].append(image_id)

        # segmentation metrics
        segmentation_metrics["accuracy"].append(round(float(accuracy), 4))
        segmentation_metrics["precision"].append(round(float(precision), 4))
        segmentation_metrics["recall"].append(round(float(recall), 4))
        segmentation_metrics["specificity"].append(round(float(specificity), 4))
        segmentation_metrics["f1_score"].append(round(float(f1_score), 4))
        segmentation_metrics["iou"].append(round(float(iou), 4))
        segmentation_metrics["dice_coeff"].append(round(float(dice_coeff), 4))

        # object based metrics
        object_metrics["object_prediction_average"].append(round(float(overall_prediction_avg), 4))
        object_metrics["overall_found_fraction"].append(round(float(overall_found_fraction), 4))

        # object based metrics per size
        for k, v in overall_per_size.items():
            if k not in object_metrics_per_size:
                object_metrics_per_size[k] = [round(float(v), 4)]
            else:
                object_metrics_per_size[k].append(round(float(v), 4))

    return (segmentation_metrics, object_metrics, object_metrics_per_size), True

if __name__=="__main__":
    from omegaconf import OmegaConf
    from data.dataset_austria_v2 import pl_datamodule
    import pandas as pd
    config = OmegaConf.load("/configs/samuel_remodel_configs/full_config_diffusion.yaml")

    pl_dm = pl_datamodule(config)
    #images, masks = next(iter(pl_dm.train_dataloader()))
    ids, images, masks = next(iter(pl_dm.test_dataloader()))
    preds = torch.zeros_like(masks)
    #metrics_b = calculate_metrics(masks, preds, threshhold=0.75, phase="train")

    (segmentation_metrics, object_metrics, object_metrics_per_size), _ = calculate_test_metrics(ids, masks, preds, threshhold=0.75, phase='test')

    segmentation_metrics, object_metrics, object_metrics_per_size = [segmentation_metrics], [object_metrics], [object_metrics_per_size]

    agg_metric = {k: [] for k in segmentation_metrics[0].keys()}
    for batch in segmentation_metrics:
        for metric_key, metric_values in batch.items():
            agg_metric[metric_key].extend(metric_values)
    df = pd.DataFrame.from_dict(agg_metric)
    print(df)

    agg_metric = {k: [] for k in object_metrics[0].keys()}
    for batch in object_metrics:
        for metric_key, metric_values in batch.items():
            agg_metric[metric_key].extend(metric_values)
    df = pd.DataFrame.from_dict(agg_metric)
    print(df)

    agg_metric = {k: [] for k in object_metrics_per_size[0].keys()}
    for batch in object_metrics_per_size:
        for metric_key, metric_values in batch.items():
            agg_metric[metric_key].extend(metric_values)
    df = pd.DataFrame.from_dict(agg_metric)
    print(df)