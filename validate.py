import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

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
val_obj = Validator(device="cuda", debugging=True)

# Set Config and CKPT base paths
model_type = "fcn"
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
val_obj.calculate_masks_metrics(dataloader=dataloader_lr, model=model_lr, pred_type="LR")


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
val_obj.calculate_masks_metrics(dataloader=dataloader_hr, model=model_hr, pred_type="HR")



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
val_obj.calculate_masks_metrics(dataloader=dataloader_sr, model=model_sr, pred_type="SR")





# 4. Get Metrics -------------------------------------------------------------
# Retrieve and print the raw metrics
metrics = val_obj.return_raw_metrics()
val_obj.print_sr_improvement(save_to_txt=True)

# calculate mAP curves
val_obj.get_mAP_curve(dataloader_lr, model_lr, pred_type="LR", amount_batches=25)
val_obj.get_mAP_curve(dataloader_hr, model_hr, pred_type="HR", amount_batches=25)
val_obj.get_mAP_curve(dataloader_sr, model_sr, pred_type="SR", amount_batches=25)

# plot mAP curve
mAP_plot = val_obj.plot_mAP_curve()
mAP_plot.save("results/mAP_plot.png")

# save examples
val_obj.save_pred_images(output_path="results/example_images")

