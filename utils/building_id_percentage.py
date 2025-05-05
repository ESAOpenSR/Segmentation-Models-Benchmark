import numpy as np
from scipy.ndimage import label, find_objects
import torch


def calculate_object_identification(mask_pred, mask_true, size_categories=None):
    """
     Evaluates the detection performance of segmented objects (e.g., building blocks) by comparing predicted masks
     to ground truth masks. Objects are grouped into size-based categories, and identification statistics are computed
     for each group.

     Parameters:
         mask_pred (numpy.ndarray or torch.Tensor): Binary prediction mask with segmented objects.
         mask_true (numpy.ndarray or torch.Tensor): Binary ground truth mask with actual objects.
         size_categories (dict, optional): Dictionary mapping category names to size ranges (in pixels).
             If None, defaults to predefined categories:
                 {
                     "xxs": (0, 4),
                     "xs":  (4, 8),
                     "s":   (8, 12),
                     "m":   (12, 20),
                     "l":   (20, 50),
                     "xl":  (50, 999999)
                 }

     Returns:
         dict: A dictionary containing:
             - "absolute_values": {
                   "correctly_identified": dict with counts per category,
                   "missed": dict with counts per category,
                   "total_objects_in_target": total ground truth objects per category
               }
             - "percentages": {
                   "correctly_identified_percentage": percentage of correctly identified objects per category,
                   "missed_percentage": percentage of missed objects per category
               }

     Notes:
         - An object is considered "correctly identified" if any part of it overlaps with a predicted object.
         - Both predicted and ground truth masks are internally labeled to identify individual objects.
     """
    if size_categories is None:
        size_categories = {
            "xxs": (0, 4),
            "xs": (4, 8),
            "s": (8, 12),
            "m": (12, 20),
            "l": (20, 50),
            "xl": (50, 999999)
            }
    
    # Ensure the inputs are numpy arrays
    mask_pred = mask_pred.cpu().numpy() if isinstance(mask_pred, torch.Tensor) else mask_pred
    mask_true = mask_true.cpu().numpy() if isinstance(mask_true, torch.Tensor) else mask_true
    
    # Label the objects in the true and predicted masks
    true_labeled, num_true_objects = label(mask_true)
    pred_labeled, num_pred_objects = label(mask_pred)
    
    # Get bounding boxes for each object in the true mask
    true_slices = find_objects(true_labeled)
    pred_slices = find_objects(pred_labeled)

    # Initialize counters for each size category
    correctly_identified = {category: 0 for category in size_categories}
    missed = {category: 0 for category in size_categories}
    total_objects_in_target = {category: 0 for category in size_categories}

    for i in range(1, num_true_objects + 1):
        # Get the mask for the current true object
        true_object = (true_labeled == i)
        
        # Calculate the size of the object
        object_size = np.sum(true_object)
        
        # Determine which size category the object belongs to
        for category, (min_size, max_size) in size_categories.items():
            if min_size <= object_size < max_size:
                # Count the object in total target objects
                total_objects_in_target[category] += 1
                
                # Check for overlap with any predicted object
                overlap = np.logical_and(mask_pred, true_object)
                if np.sum(overlap) > 0:
                    correctly_identified[category] += 1
                else:
                    missed[category] += 1
                break  # Exit loop once the correct category is found

    # Calculate percentages
    correctly_identified_percentage = {}
    missed_percentage = {}

    for category in size_categories:
        if total_objects_in_target[category] > 0:
            correctly_identified_percentage[category] = (correctly_identified[category] / total_objects_in_target[category]) * 100
            missed_percentage[category] = (missed[category] / total_objects_in_target[category]) * 100
        else:
            correctly_identified_percentage[category] = 0
            missed_percentage[category] = 0

    # Return both absolute values and percentages
    return {
        "absolute_values": {
            "correctly_identified": correctly_identified,
            "missed": missed,
            "total_objects_in_target": total_objects_in_target
        },
        "percentages": {
            "correctly_identified_percentage": correctly_identified_percentage,
            "missed_percentage": missed_percentage
        }
    }


def calculate_batched_averages(preds, targets, size_categories=None):
    """
    Computes average object identification statistics over a batch of predicted and ground truth masks.
    Each object is classified by size, and detection accuracy is averaged across the batch.

    Parameters:
        preds (numpy.ndarray or torch.Tensor): Batch of predicted binary masks with shape (B, H, W),
            where B is the batch size.
        targets (numpy.ndarray or torch.Tensor): Batch of ground truth binary masks with shape (B, H, W).
        size_categories (dict, optional): Dictionary mapping category names to size ranges (in pixels).
            If None, defaults to:
                {
                    "xxs": (0, 4),
                    "xs":  (4, 8),
                    "s":   (8, 12),
                    "m":   (12, 20),
                    "l":   (20, 50),
                    "xl":  (50, 999999)
                }

    Returns:
        dict: A dictionary containing:
            - "average_absolute_values": {
                  "avg_correctly_identified": dict of average counts per category,
                  "avg_missed": dict of average missed counts per category,
                  "avg_total_objects_in_target": dict of average total objects per category
              }
            - "average_percentages": {
                  "avg_correctly_identified_percentage": dict of average correct detection rates per category (%),
                  "avg_missed_percentage": dict of average miss rates per category (%)
              }

    Notes:
        - Uses `calculate_object_identification` internally on each batch element.
        - Averaging is done over the number of batch elements, not per-object.
        - Percentages are based on total ground truth objects across the batch.
    """

    #assert type(size_categories) in [dict,None], "Size categories must be a dictionary or None"
    
    if size_categories is None:
        size_categories = {
        "xxs": (0, 4),   
        "xs": (4, 8),   
        "s": (8, 12),
        "m": (12, 20),
        "l": (20, 50),  
        "xl": (50, 999999)
    }
        
    # Initialize cumulative sums for the batch
    cumulative_correctly_identified = {category: 0 for category in size_categories}
    cumulative_missed = {category: 0 for category in size_categories}
    cumulative_total_objects_in_target = {category: 0 for category in size_categories}

    # Track the number of batch entries
    batch_size = preds.shape[0]

    # Iterate through each batch element
    for i in range(batch_size):
        mask_pred = preds[i]  # Predicted mask for the ith batch element
        mask_true = targets[i]  # True mask for the ith batch element

        # Call the original function for each batch element
        result = calculate_object_identification(mask_pred, mask_true, size_categories)

        # Accumulate the values from each batch
        for category in size_categories:
            cumulative_correctly_identified[category] += result['absolute_values']['correctly_identified'][category]
            cumulative_missed[category] += result['absolute_values']['missed'][category]
            cumulative_total_objects_in_target[category] += result['absolute_values']['total_objects_in_target'][category]

    # Calculate the average values for absolute counts
    avg_correctly_identified = {category: cumulative_correctly_identified[category] / batch_size for category in size_categories}
    avg_missed = {category: cumulative_missed[category] / batch_size for category in size_categories}
    avg_total_objects_in_target = {category: cumulative_total_objects_in_target[category] / batch_size for category in size_categories}

    # Calculate the percentage averages (only if total objects in target are greater than 0)
    avg_correctly_identified_percentage = {}
    avg_missed_percentage = {}

    for category in size_categories:
        if cumulative_total_objects_in_target[category] > 0:
            avg_correctly_identified_percentage[category] = (cumulative_correctly_identified[category] / cumulative_total_objects_in_target[category]) * 100
            avg_missed_percentage[category] = (cumulative_missed[category] / cumulative_total_objects_in_target[category]) * 100
        else:
            avg_correctly_identified_percentage[category] = 0
            avg_missed_percentage[category] = 0

    # Return the averaged results as a single dictionary
    return {
        "average_absolute_values": {
            "avg_correctly_identified": avg_correctly_identified,
            "avg_missed": avg_missed,
            "avg_total_objects_in_target": avg_total_objects_in_target
        },
        "average_percentages": {
            "avg_correctly_identified_percentage": avg_correctly_identified_percentage,
            "avg_missed_percentage": avg_missed_percentage
        }
    }


if __name__ == "__main__":
    # Define size categories (e.g., small, medium, large)
    size_categories = {
            "xxs": (0, 4),   
            "xs": (4, 8),   
            "s": (8, 12),
            "m": (12, 20),
            "l": (20, 50),  
            "xl": (50, 999)
        }
    
    for i in size_categories.keys():
        print("Size Category",i.upper(),":\t"+str(size_categories[i][0])+" to "+str(int(size_categories[i][1])),"pixels.\t\tMax Sqm:",int(size_categories[i][1]*6.25))

    # Example usage
    mask_pred = torch.tensor([[0, 0, 0, 1, 1], [0, 1, 1, 0, 0], [0, 0, 1, 1, 0], [1, 0, 0, 0, 0], [0, 0, 0, 0, 1]])
    mask_true = torch.tensor([[0, 0, 0, 1, 1], [0, 0, 1, 1, 0], [0, 0, 1, 0, 0], [1, 1, 0, 0, 0], [0, 0, 0, 0, 1]])

    result = calculate_object_identification(mask_pred, mask_true, size_categories)
    print(result)

    m = (torch.rand(512,512)>0.5)*1.
    p = (torch.rand(512,512)>0.5)*1.
    calculate_object_identification(m, p, size_categories)
    calculate_batched_averages(m,p, size_categories=size_categories)





