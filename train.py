import glob
import os, sys, wandb
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from einops import rearrange
from omegaconf import OmegaConf
from datetime import datetime

# torch
import torch
torch.set_float32_matmul_precision("medium")
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

# internal
from model_files import model_pl

# Load the config
default_config_path = "configs/samuel_remodel_configs/filtered_config_s2_nn.yaml"
model_name = 1#'bicubic' #bilinear
train = False
log = True
tracasa = True
compare_tracasa = True
tracasa_submodel = 'tracasa'
# diffusion_updated_simon, evoland, sr4rs, swin2_mose, tracasa, bilinear, SEN2SRLite_RGBN, SEN2SR_RGBN, deepsent

paths = {
    "sr4rs": "logs/Samuel_building_segmentation/2025-08-16_22-48-42/",
    "swin2mose": "logs/Samuel_building_segmentation/2025-08-16_22-49-46/",
    "sen2sr_rgbn": "logs/Samuel_building_segmentation/2025-08-16_17-11-20/",
    "bilinear": "logs/Samuel_building_segmentation/2025-08-14_11-16-38/",
    "deepsent": "logs/Samuel_building_segmentation/2025-08-15_18-25-42/",
    "ldrs2": "logs/Samuel_building_segmentation/2025-08-15_18-33-55/",
    "evoland": "logs/Samuel_building_segmentation/2025-08-16_17-20-26/",
    "sen2sr_lite": "logs/Samuel_building_segmentation/2025-08-16_17-22-41/",
    "ortho": "logs/Samuel_building_segmentation/2025-08-11_12-05-45/",
    "nn": "logs/Samuel_building_segmentation/2025-08-20_13-16-27/",
    "bicubic": "logs/Samuel_building_segmentation/2025-08-20_13-13-19/",
}

paths = {
    "bilinear": "logs/Samuel_building_segmentation/2025-08-14_11-16-38/",
}

# paths = {'01': 'logs/Samuel_building_segmentation/2025-08-11_12-01-07/',
#     '02': 'logs/Samuel_building_segmentation/2025-08-11_12-05-45/',
#     '03': 'logs/Samuel_building_segmentation/2025-08-11_21-42-20/',
#     '04': 'logs/Samuel_building_segmentation/2025-08-11_16-40-21/',
#     '042': 'logs/Samuel_building_segmentation/2025-08-12_16-51-32/',
#     '07': 'logs/Samuel_building_segmentation/2025-08-13_11-52-56/',
#     '041': 'logs/Samuel_building_segmentation/2025-08-12_16-06-23/',
#     '05': 'logs/Samuel_building_segmentation/2025-08-12_10-18-39/',
#     '06': 'logs/Samuel_building_segmentation/2025-08-12_10-16-06/',
#     '08': 'logs/Samuel_building_segmentation/2025-08-13_12-10-43/',
#     '09': 'logs/Samuel_building_segmentation/2025-08-13_17-38-11/',
#     '10': 'logs/Samuel_building_segmentation/2025-08-13_17-39-17/',
#     '11': 'logs/Samuel_building_segmentation/2025-08-14_11-15-26/',
#     '12': 'logs/Samuel_building_segmentation/2025-08-14_11-16-38/',
#     '13': 'logs/Samuel_building_segmentation/2025-08-14_14-22-40/',}


for model_name in paths.keys():
    if not train:
        config_path = Path(paths[model_name]) / 'train_config.yaml'
    else:
        # Check if a config path is provided as a command-line argument
        config_path = sys.argv[1] if len(sys.argv) > 1 else default_config_path  # get argument

    config = OmegaConf.load(config_path)
    config.training.pl_settings.devices = [1]
    print("Loaded Config from:", config_path)
    model = model_pl(config, name=model_name)  # model selection is handled by the model_pl function

    # Continue Training PL
    continue_training = config.training.pl_settings.continue_training
    # logic to set continued training variable for Trainer
    if continue_training in [False, None]:
        continue_training = None
        print("Not loading Lightning-Style CKPT.")
    else:
        if not os.path.exists(continue_training):
            print("Model path does not exist. Training from scratch")
            continue_training = None
        else:
            print("Continuing training from:", continue_training)

    # if config.training.pl_settings.load_weights_only not in [False, None]:
    #     ckpt_path = config.training.pl_settings.load_weights_only
    #     ckpt = torch.load(ckpt_path)
    #     model.load_state_dict(ckpt["state_dict"])
    #     print("Loaded weights only from:", ckpt_path)

    # Load the data ---------------------------------------------------------------
    if config.data.dataset_type == "fake":
        from data.fake_dataset import pl_datamodule
    elif config.data.dataset_type == "RS":
        from data.dataset_austria_v2 import pl_datamodule
    else:
        print("Invalid Dataset Type: ", config.data.dataset_type)
        sys.exit(1)
    # data_module = pl_datamodule(config)
    # data_module.train_dataset.validate(idx=10, verbose=True)

    if not train:
        print('doing testing')

        if compare_tracasa:
            config.data.data_path = '/data/USERS/shollend/tracasa_test/'
            config.data.dataset = 'tables_comparison'

        if tracasa:
            to_path = Path('/data/USERS/shollend/inferred_buildings') / 'tracasa' / tracasa_submodel / 'predicted'
            rates_path = Path('/data/USERS/shollend/inferred_buildings') / 'tracasa'/ tracasa_submodel / 'colored'

            config.data.data_path = '/data/USERS/shollend/tracasa_test/'
            config.data.input_path = f'/data/USERS/shollend/tracasa_test/{tracasa_submodel}/images'
            #config.data.target_path = '/data/USERS/shollend/tracasa_test/hr_mask/split_images'
            config.data.dataset = 'tables_comparison'
            config.data.data_type = 'sr_4band'
            config.data.use_256_subsample = True
            data_module = pl_datamodule(config)
        else:
            to_path = Path('/data/USERS/shollend/inferred_buildings') / model_name / 'predicted'
            rates_path = Path('/data/USERS/shollend/inferred_buildings') / model_name / 'colored'
            data_module = pl_datamodule(config)

        to_path.mkdir(exist_ok=True, parents=True)
        rates_path.mkdir(exist_ok=True)

        if compare_tracasa:
            ckpt_path = glob.glob(paths['bilinear'] + 'epoch=*-step=*.ckpt')[0]
        else:
            ckpt_path = glob.glob(paths[model_name] + 'epoch=*-step=*.ckpt')[0]
        print("Loaded weights only from:", ckpt_path)
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["state_dict"])

        if log:
            # log test metrics AND test inference
            wandb_logger = WandbLogger(entity="opensr", project="Samuel_building_segmentation",
                                       name=f'8__{model_name}')

            # Log the config.yaml as a W&B artifact under the current run
            artifact = wandb.Artifact("config-file", type="config")
            artifact.add_file(default_config_path)
            wandb_logger.experiment.log_artifact(artifact)  # use the current run from the logger

            dir_save_checkpoints = os.path.join("logs/", config.training.wandb_project_name,
                                                datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
            config.training.log_dir = dir_save_checkpoints

            trainer = Trainer(
                accelerator=config.training.pl_settings.accelerator,
                devices=config.training.pl_settings.devices,
                strategy=config.training.pl_settings.strategy,
                check_val_every_n_epoch=config.training.pl_settings.check_val_every_n_epoch,
                log_every_n_steps=config.training.pl_settings.log_every_n_steps,
                # val_check_interval=config.training.pl_settings.val_check_interval,
                max_epochs=config.training.pl_settings.max_epochs,
                limit_val_batches=config.training.pl_settings.limit_val_batches,
                logger=[
                    wandb_logger,
                ],

            )
        else:
            # test inference
            trainer = Trainer(
                accelerator=config.training.pl_settings.accelerator,
                devices=config.training.pl_settings.devices,
                strategy=config.training.pl_settings.strategy,
                check_val_every_n_epoch=config.training.pl_settings.check_val_every_n_epoch,
                log_every_n_steps=config.training.pl_settings.log_every_n_steps,
                # val_check_interval=config.training.pl_settings.val_check_interval,
                max_epochs=config.training.pl_settings.max_epochs,
                limit_val_batches=config.training.pl_settings.limit_val_batches,

            )

        trainer.test(model, datamodule=data_module)
        wandb.finish()
        pass

    if train:
        # Define Callbacks and Loggers ------------------------------------------------
        # Logging - WandB
        # own
        #wandb_logger = WandbLogger(entity="zerhigh-tu-wien", project="sr_validation",)
        # isp
        wandb_logger = WandbLogger(entity="opensr", project="Samuel_building_segmentation",)

        # Log the config.yaml as a W&B artifact under the current run
        artifact = wandb.Artifact("config-file", type="config")
        artifact.add_file(default_config_path)
        wandb_logger.experiment.log_artifact(artifact)  # use the current run from the logger

        # Saving Callbacks
        dir_save_checkpoints = os.path.join("logs/", config.training.wandb_project_name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        print("Experiment Path:", dir_save_checkpoints)

        # save config file to dict
        config_dict = OmegaConf.to_container(config)
        # if path doesnt exist, create
        if not os.path.exists(dir_save_checkpoints):
            os.makedirs(dir_save_checkpoints)
        with open(os.path.join(dir_save_checkpoints, "train_config.yaml"), "w") as f:
            f.write(str(config_dict))

        checkpoint_callback = ModelCheckpoint(
            dirpath=dir_save_checkpoints,
            monitor=config.training.pl_settings.checkpoint_saving_metric,
            mode="min",
            save_last=True,
            save_top_k=1,
        )
        config.training.log_dir = dir_save_checkpoints

        # callbacks
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        early_stop_callback = EarlyStopping(
            monitor=config.training.pl_settings.early_stop_metric,
            min_delta=0.00,
            patience=config.training.early_stopping_patience,
            verbose=True,
            mode="min",
            check_finite=True,
        )  # patience in epochs

        # Configure PL Trainer ---------------------------------------------------------
        trainer = Trainer(
            accelerator=config.training.pl_settings.accelerator,
            devices=config.training.pl_settings.devices,
            strategy=config.training.pl_settings.strategy,
            check_val_every_n_epoch=config.training.pl_settings.check_val_every_n_epoch,
            log_every_n_steps=config.training.pl_settings.log_every_n_steps,
            # val_check_interval=config.training.pl_settings.val_check_interval,
            max_epochs=config.training.pl_settings.max_epochs,
            limit_val_batches=config.training.pl_settings.limit_val_batches,
            #resume_from_checkpoint=continue_training,
            logger=[
                wandb_logger,
            ],
            callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        )

        # Train the model ---------------------------------------------------------------
        trainer.fit(model, datamodule=data_module)
        trainer.test(model, datamodule=data_module)
        wandb.finish()
