import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
torch.set_float32_matmul_precision("medium")

# General Imports
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from pathlib import Path
import shutil
import rasterio
import numpy as np

# Local Imports
from data.dataset_austria import pl_datamodule
from model_files.model_wrapper import model_pl


# 0. Prepare Validation --------------------------------------------------------
from opensr_usecases import Validator
val_obj = Validator(device="cuda", debugging=False)

# Set Config and CKPT base paths
model_type = "unet_pp"
output_name = f'080525_bilinear_ortho_SEN2SR_{model_type}'
cfg_base_path = "logs/Samuel_building_segmentation/RUN/train_config.yaml"
ckpt_base_path = "logs/Samuel_building_segmentation/RUN/"


# cfg_base_path = "configs/MODEL/config_TYPE.yaml"
# ckpt_base_path = "logs/MODEL_v1/TYPE_MODEL.ckpt"
# cfg_base_path = cfg_base_path.replace("MODEL", model_type)
# ckpt_base_path = ckpt_base_path.replace("MODEL", model_type)

# 1. BILINEAR ---------------------------------------------------------------------
# 1.2 Load Data

# 1.1 Load Model and weights

model_bilinear_name = '2025-05-06_17-08-07'
config = OmegaConf.load(cfg_base_path.replace("RUN", model_bilinear_name))

# load data
data_module = pl_datamodule(config)
dataloader_bilinear = data_module.test_dataloader()

model_bilinear = model_pl(config)
ortho_weights = list(Path(ckpt_base_path.replace("RUN", model_bilinear_name)).glob('epoch=*.ckpt'))[0]
ckpt = torch.load(ortho_weights, map_location='cuda:0')
model_bilinear.load_state_dict(ckpt['state_dict'])
# # 2.3 Validate
val_obj.calculate_masks_metrics(dataloader=dataloader_bilinear, model=model_bilinear, pred_type="HR")


#
# # 2. ORTHOPHOTO ---------------------------------------------------------------------
# # 2.1 Load Model and weights
# ortho_run = '2025-05-06_17-06-13'
# config = OmegaConf.load(cfg_base_path.replace("RUN", ortho_run))
# model_hr = model_pl(config)
# ortho_weights = list(Path(ckpt_base_path.replace("RUN", ortho_run)).glob('epoch=*.ckpt'))[0]
# ckpt = torch.load(ortho_weights, map_location='cuda:0')
# model_hr.load_state_dict(ckpt['state_dict'])
# # 2.2 Load Data
# data_module = pl_datamodule(config)
# dataloader_hr = data_module.test_dataloader()
# # 2.3 Validate
# val_obj.calculate_masks_metrics(dataloader=dataloader_hr, model=model_hr, pred_type="HR")
#
#
#
# # 3. SR ---------------------------------------------------------------------
# # 3.1 Load Model and weights
# # diffusion
# sr_run = '2025-05-07_15-57-13'
# # sen2sr
# sr_run = '2025-05-07_21-59-34'
# config = OmegaConf.load(cfg_base_path.replace("RUN", sr_run))
# model_sr = model_pl(config)
# ortho_weights = list(Path(ckpt_base_path.replace("RUN", sr_run)).glob('epoch=*.ckpt'))[0]
# ckpt = torch.load(ortho_weights, map_location='cuda:0')
#
# model_sr.load_state_dict(ckpt['state_dict'])
# # 3.2 Load Data
# data_module = pl_datamodule(config)
# dataloader_sr = data_module.test_dataloader()
# # 3.3 Validate
# val_obj.calculate_masks_metrics(dataloader=dataloader_sr, model=model_sr, pred_type="SR")
#
# # 4. Get Metrics -------------------------------------------------------------
# # Retrieve and print the raw metrics
# metrics = val_obj.return_raw_metrics()
# val_obj.print_sr_improvement(save_to_txt=True)
#
# # calculate mAP curves
# val_obj.get_mAP_curve(dataloader_lr, model_lr, pred_type="LR", amount_batches=25)
# val_obj.get_mAP_curve(dataloader_hr, model_hr, pred_type="HR", amount_batches=25)
# val_obj.get_mAP_curve(dataloader_sr, model_sr, pred_type="SR", amount_batches=25)
#
# # create folder
# outpath = Path(f'results/{output_name}')
# outpath.mkdir(exist_ok=True)
# shutil.move('results/tabular_results.txt', str(outpath / 'tabular_results.txt'))
#
# with open(outpath/'config.txt', 'w') as file:
#     file.write(f'bilinear: {bilinear_run}, ortho: {ortho_run}, sr: {sr_run}')
#
# # plot mAP curve
# mAP_plot = val_obj.plot_mAP_curve()
# mAP_plot.save(outpath / "mAP_plot.png")
#
# outpath_images = Path(f'results/{output_name}')
# outpath_images.mkdir(exist_ok=True)
# # save examples
#val_obj.save_pred_images(output_path=outpath_images / "example_images")
#
