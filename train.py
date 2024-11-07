import torch
torch.set_float32_matmul_precision("medium")

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from einops import rearrange
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import wandb
from pytorch_lightning import Trainer
from datetime import datetime
import os,sys

# Load the model --------------------------------------------------------------
from model_files import model_pl
config = OmegaConf.load("configs/config_hr.yaml")
model = model_pl(config) # model selection is handled by the model_pl function


# Continue Training PL --------------------------------------------------------
continue_training = config.training.pl_settings.continue_training
# logic to set continued training variable for Trainer
if continue_training in [False,None]:
    continue_training = None
    print("Not loading Lightning-Style CKPT.")
else:
    if not os.path.exists(continue_training):
        print("Model path does not exist. Training from scratch")
        continue_training = None
    else:
        print("Continuing training from:",continue_training)
        
if config.training.pl_settings.load_weights_only not in [False,None]:
    ckpt_path = config.training.pl_settings.load_weights_only
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt['state_dict'])
    print("Loaded weights only from:",ckpt_path)



# Load the data ---------------------------------------------------------------
if config.data.dataset_type=="fake":
    from data.fake_dataset import pl_datamodule
elif config.data.dataset_type=="RS":
    from data.dataset_masks import pl_datamodule
else:
    print("Invalid Dataset Type: ",config.data.dataset_type)
    sys.exit(1)
data_module = pl_datamodule(config)


# Define the Loggers
project_name = config.training.wandb_project_name

# Logging - TF
from pytorch_lightning import loggers as pl_loggers
tb_logger = pl_loggers.TensorBoardLogger(save_dir="logs/")

# Logging - WandB
from pytorch_lightning.loggers import WandbLogger
wandb_logger = WandbLogger(project=project_name)# ,mode="disabled")

# Saving Callbacks
from pytorch_lightning.callbacks import ModelCheckpoint
dir_save_checkpoints = os.path.join(tb_logger.save_dir,project_name,datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
print("Experiment Path:",dir_save_checkpoints)
checkpoint_callback = ModelCheckpoint(dirpath=dir_save_checkpoints,
                                        monitor=config.training.pl_settings.checkpoint_saving_metric,
                                        mode='min',
                                        save_last=True,
                                        save_top_k=1)
config.training.log_dir = dir_save_checkpoints # save into config to log into wandb later

# Learning Rate Monitor
from pytorch_lightning.callbacks import LearningRateMonitor
lr_monitor = LearningRateMonitor(logging_interval='epoch')

# Early Stopping
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
early_stop_callback = EarlyStopping(monitor=config.training.pl_settings.early_stop_metric,
                                    min_delta=0.00, patience=2500, verbose=True,
                                    mode="min",check_finite=True) # patience in epochs

trainer = Trainer(
    accelerator=config.training.pl_settings.accelerator,
    devices=config.training.pl_settings.devices,
    strategy=config.training.pl_settings.strategy,
    check_val_every_n_epoch=config.training.pl_settings.check_val_every_n_epoch,
    log_every_n_steps=config.training.pl_settings.log_every_n_steps,
    #val_check_interval=config.training.pl_settings.val_check_interval,
    max_epochs=config.training.pl_settings.max_epochs,
    limit_val_batches = config.training.pl_settings.limit_val_batches,
    resume_from_checkpoint=continue_training,
    logger=[ wandb_logger,],
    callbacks=[ checkpoint_callback,
                early_stop_callback,
                lr_monitor]  )



# Train the model ---------------------------------------------------------------
trainer.fit(model, datamodule=data_module)
wandb.finish()

