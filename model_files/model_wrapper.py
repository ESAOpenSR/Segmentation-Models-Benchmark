import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
# not used
#from torchmetrics import Dice
import wandb
from omegaconf import OmegaConf, DictConfig
from utils.metrics_utils import calculate_metrics, calculate_object_metrics
from utils.logging_utils import log_images
from model_files.model_losses import dice_loss
from utils.building_id_percentage import (
    calculate_object_identification,
    calculate_batched_averages,
)


class model_pl(pl.LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()

        # get config
        self.config = config
        self.amp = config.training.amp

        # extract model settings
        self.n_classes = config.model.n_classes
        self.conf_threshold = config.model.conf_threshold
        self.model = self.get_model(config)  # get model from function
        self.get_loss_fn()  # get loss function

    def get_loss_fn(self):
        loss_command = self.config.training.loss
        print("Creating loss of type:", loss_command)
        if loss_command == "BCEWithLogitsLoss":
            self.criterion = (
                nn.CrossEntropyLoss() if self.n_classes > 1 else nn.BCEWithLogitsLoss()
            )
        elif loss_command == "BoundaryAwareLoss":
            from utils.losses import BoundaryAwareLoss

            self.criterion = BoundaryAwareLoss(dilation_ratio=0.02, alpha=1.0, beta=1.0)
        elif loss_command == "QRLoss":
            from torchgeo.losses import QRLoss as TorchgeoQRLoss

            self.criterion = TorchgeoQRLoss()
        elif loss_command == "RQLoss":
            from torchgeo.losses import RQLoss

            self.criterion = RQLoss()
        elif loss_command == "FocalLoss":
            from utils.losses import FocalTverskyLoss

            self.criterion = FocalTverskyLoss().forward
        else:
            raise ValueError("Invalid Loss Function")

    def get_model(self, config):
        print("Creating Model of Type", config.model.model_type)
        # Load model parameters from config
        if config.model.model_type == "unet":
            from model_files.unet_model import UNet

            model = UNet(
                n_channels=config.model.n_channels, n_classes=config.model.n_classes
            )
        elif config.model.model_type == "unet_pp":
            import segmentation_models_pytorch as smp
            model = smp.Unet(
                encoder_name=config.model.encoder,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights=None,  # use `imagenet` pre-trained weights for encoder initialization
                in_channels=config.model.n_channels,  # model input channels (4 for RGB-NIR, 3 for RGB, etc.)
                classes=1,
            )  # model output channels (number of classes in your dataset)
        elif config.model.model_type == "DeepLabV3Plus":
            import segmentation_models_pytorch as smp

            model = smp.DeepLabV3Plus(
                encoder_name=config.model.encoder,
                encoder_depth=5,
                encoder_weights=None,
                encoder_output_stride=8,  # changed from 16 to 8 for sharper borders
                decoder_channels=256,
                decoder_atrous_rates=(12, 24, 36),
                in_channels=4,
                classes=1,
                activation=None,
                upsampling=4,
                aux_params=None,
            )

        elif "torchgeo" in config.model.model_type:
            from model_files.torchgeo_models import create_torchgeo_models

            model = create_torchgeo_models(config)
        else:
            raise ValueError("Invalid Model Type")
        return model

    def forward(self, x):
        return self.model(x)

    @torch.no_grad()
    def predict(self, x):
        if self.config.training.loss_req_sig:
            return torch.sigmoid(self.forward(x))
        else:
            return self.forward(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        # Use AMP (Automatic Mixed Precision) during forward pass
        with torch.cuda.amp.autocast(enabled=self.amp):
            y_hat = self.forward(x)

            # Assuming binary segmentation (1 channel output)
            if self.config.training.loss_req_sig:
                loss = self.criterion(
                    torch.sigmoid(y_hat), y.float()
                )  # Main loss (e.g., BCEWithLogitsLoss) #.squeeze(1)
            else:
                loss = self.criterion(y_hat, y.float())
        self.log("train_loss", loss)

        # apply sigmoid, then thresh
        if self.config.training.loss_req_sig:
            y_hat_thresh = (torch.sigmoid(y_hat.clone()) > self.conf_threshold) * 1
        else:
            y_hat_thresh = (y_hat.clone() > self.conf_threshold) * 1

        # Metrics
        if self.is_trainer_attached():
            # get training dict
            if batch_idx % 50 == 0:
                loss_px_dict, status_px = calculate_metrics(
                    y, y_hat_thresh, phase="train"
                )
                loss_obj_dict, status_obj = calculate_object_metrics(
                    y, y_hat_thresh, phase="train"
                )
                if status_px:
                    self.log_dict(
                        loss_px_dict,
                        prog_bar=False,
                        logger=True,
                        on_step=True,
                        on_epoch=False,
                        sync_dist=True,
                    )
                if status_obj:
                    self.log_dict(
                        loss_obj_dict,
                        prog_bar=False,
                        logger=True,
                        on_step=True,
                        on_epoch=False,
                        sync_dist=True,
                    )
            if batch_idx % 50 == 0:
                # get building id metrics
                y_hat_clone = y_hat.clone().detach()
                building_id_dict = self.get_building_id_metrics(
                    y_hat_clone, y, phase="train"
                )
                self.log_dict(
                    building_id_dict,
                    prog_bar=False,
                    logger=True,
                    on_step=True,
                    on_epoch=False,
                    sync_dist=True,
                )
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        x, y = batch  # get Data
        y_hat = self.predict(x)  # Forward pass
        val_loss = self.criterion(
            y_hat, y.float()
        )  # Main loss (e.g., BCEWithLogitsLoss)
        # val_loss += dice_loss(y_hat, y.float(), multiclass=False)  # Dice loss to  main loss
        self.log("val_loss", val_loss)  # Log

        y_hat_thresh = (
            y_hat.clone() > self.conf_threshold
        ) * 1  # Thresholding, sigmoid in predict

        # Metrics
        if self.is_trainer_attached():
            loss_px_dict, status_px = calculate_metrics(y, y_hat_thresh, phase="val")
            loss_obj_dict, status_obj = calculate_object_metrics(
                y, y_hat_thresh, phase="val"
            )
            if status_px:  # log only if valid metrics are returned
                self.log_dict(
                    loss_px_dict,
                    prog_bar=False,
                    logger=True,
                    # on_step=True,
                    # on_epoch=False,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )
            if status_obj:  # log only if valid metrics are returned
                self.log_dict(
                    loss_obj_dict,
                    prog_bar=False,
                    logger=True,
                    # on_step=True,
                    # on_epoch=False,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )
            if batch_idx < 5:  # log only first 5 val batches
                val_image = log_images(x, y, y_hat, title="Training")
                self.logger.experiment.log(
                    {"images/Validation": [wandb.Image(val_image)]}
                )

                # get building id metrics
                # y_hat_thres = (y_hat>self.conf_threshold)*1
                y_hat_clone = y_hat.clone().detach()
                building_id_dict = self.get_building_id_metrics(
                    y_hat_clone, y, phase="val"
                )

                self.log_dict(
                    building_id_dict,
                    prog_bar=False,
                    logger=True,
                    # on_step=True,
                    # on_epoch=False,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )
        return val_loss

    def get_building_id_metrics(self, mask_pred, mask_true, phase="train"):
        if mask_pred.dim() == 4:
            mask_pred = mask_pred.squeeze(1)
        if mask_true.dim() == 4:
            mask_true = mask_true.squeeze(1)
        res_dict = calculate_batched_averages(mask_pred, mask_true)
        res_dict = res_dict["average_percentages"]
        # rename keys by appending "test"
        p_n = phase + "_BuildID"

        # fix samuel: dict of dicts not loggable
        #res_dict = {f"{p_n}/{k}": v for k, v in res_dict.items()}
        res_dicts = {f"{p_n}/{k}": v for k, v in res_dict.items()}
        flattened_res_dict = {
            f"{outer_key}/{inner_key}": value
            for outer_key, inner_dict in res_dicts.items()
            for inner_key, value in inner_dict.items()
        }
        return flattened_res_dict

    def is_trainer_attached(self):
        try:
            return self.trainer is not None
        except RuntimeError:
            return False

    def configure_optimizers(self):
        if self.config.training.optim == "RMSprop":
            optimizer = optim.RMSprop(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
                momentum=self.config.training.momentum,
                foreach=True,
            )
        elif self.config.training.optim == "adam":
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
            )
        else:
            raise ValueError("Invalid Optimizer")
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=self.config.training.reduce_lr_patience,
            factor=self.config.training.reduce_lr_factor,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": self.config.training.reduce_lr_metric,  # Maximize the criterion score
                "interval": "epoch",
                "frequency": 1,
            },
        }


# Testing ---------------------------------------------------------------
if __name__ == "__main__":
    config = OmegaConf.load("configs/config_hr.yaml")
    model = model_pl(config)

    from data.dataset_masks import pl_datamodule

    data_module = pl_datamodule(config)

    batch = next(iter(data_module.train_dataloader()))
    model.training_step(batch, 50)
    model.validation_step(batch, 2)

    a, b = torch.rand(1, 256, 256).cuda(), torch.rand(1, 1, 256, 256).cuda()
    model.criterion.forward(a, b)
