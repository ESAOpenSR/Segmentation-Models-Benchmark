import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import torch
torch.set_float32_matmul_precision("medium")

# General Imports
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

# Local Imports
from data.dataset_masks import pl_datamodule
from model_files.model_wrapper import model_pl

 
# 0. Prepare Validation --------------------------------------------------------
from opensr_usecases import Validator
val_obj = Validator(output_folder="data_folder", device="cpu", force_recalc= True, debugging=False)
global_threshold = 0.75

# Set Config and CKPT base paths
model_type = "unet" # ["fcn_v1", "unet", "dl","farseg"]
iteration = "v1"
cfg_base_path = "configs/MODEL/config_TYPE.yaml"
ckpt_base_path = "logs/MODEL_v1/TYPE_MODEL.ckpt"
cfg_base_path = cfg_base_path.replace("MODEL", model_type)
ckpt_base_path = ckpt_base_path.replace("MODEL", model_type)

# 1. LR ---------------------------------------------------------------------
# 1.1 Load Model and weights
config = OmegaConf.load(cfg_base_path.replace("TYPE", "lr"))
model_lr = model_pl(config)
ckpt = torch.load(ckpt_base_path.replace("TYPE", "lr"), map_location='cuda:0')
model_lr.load_state_dict(ckpt['state_dict'])
# 1.2 Load Data
data_module = pl_datamodule(config)
dataloader_lr = data_module.test_dataloader()
# 1.3 Validate
val_obj.run_predictions(dataloader_lr, model_lr, pred_type="LR", load_pkl=False)
val_obj.calculate_segmentation_metrics(pred_type="LR", threshold=global_threshold)
val_obj.calculate_object_detection_metrics(pred_type="LR", threshold=global_threshold)
val_obj.calculate_object_detection_metrics_by_size(pred_type="LR", threshold=global_threshold)
val_obj.calculate_percent_objects_found_by_size(pred_type="LR", threshold=global_threshold)


# 2. HR ---------------------------------------------------------------------
# 2.1 Load Model and weights
config = OmegaConf.load(cfg_base_path.replace("TYPE", "hr"))
model_hr = model_pl(config)
ckpt = torch.load(ckpt_base_path.replace("TYPE", "hr"), map_location='cuda:0')
model_hr.load_state_dict(ckpt['state_dict'])
# 2.2 Load Data
data_module = pl_datamodule(config)
dataloader_hr = data_module.test_dataloader()
# 2.3 Validate
val_obj.run_predictions(dataloader_hr, model_hr, pred_type="HR", load_pkl=False)
val_obj.calculate_segmentation_metrics(pred_type="HR", threshold=global_threshold)
val_obj.calculate_object_detection_metrics(pred_type="HR", threshold=global_threshold)
val_obj.calculate_object_detection_metrics_by_size(pred_type="HR", threshold=global_threshold)
val_obj.calculate_percent_objects_found_by_size(pred_type="HR", threshold=global_threshold)


# 3. SR ---------------------------------------------------------------------
# 3.1 Load Model and weights
config = OmegaConf.load(cfg_base_path.replace("TYPE", "sr"))
model_sr = model_pl(config)
ckpt = torch.load(ckpt_base_path.replace("TYPE", "sr"), map_location='cuda:0')
model_sr.load_state_dict(ckpt['state_dict'])
# 3.2 Load Data
data_module = pl_datamodule(config)
dataloader_sr = data_module.test_dataloader()
# 3.3 Validate
val_obj.run_predictions(dataloader_sr, model_sr, pred_type="SR", load_pkl=False)
val_obj.calculate_segmentation_metrics(pred_type="SR", threshold=global_threshold)
val_obj.calculate_object_detection_metrics(pred_type="SR", threshold=global_threshold)
val_obj.calculate_object_detection_metrics_by_size(pred_type="SR", threshold=global_threshold)
val_obj.calculate_percent_objects_found_by_size(pred_type="SR", threshold=global_threshold)



# 4. Results ----------------------------------------------------------------
val_obj.save_results_examples(num_examples=5)

# 4.2 Check Segmentation Metrics
val_obj.print_segmentation_metrics(save_csv=True)
val_obj.print_segmentation_improvements(save_csv=True)

# 4.3 Check Object Detection Metrics
val_obj.print_object_detection_metrics(save_csv=True)
val_obj.print_object_detection_improvements(save_csv=True)

# 4.4 Check Object Detection Metrics by Size
val_obj.print_object_detection_metrics_by_size(save_csv=True)
val_obj.print_object_detection_improvements_by_size(save_csv=True)

# 4.5 Check Object Detection Percent of Objects found - by Size
val_obj.print_percent_objects_found_by_size(save_csv=True)
val_obj.print_percent_objects_found_improvements_by_size(save_csv=True)

# 4.4 Check Threshold Curves
val_obj.plot_threshold_curves(metric="all")