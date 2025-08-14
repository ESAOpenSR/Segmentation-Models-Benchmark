import os.path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl

import wandb
from omegaconf import OmegaConf, DictConfig
from utils.metrics_utils import calculate_metrics, calculate_test_metrics
from utils.logging_utils import log_images

# losses
from utils.losses import BoundaryAwareLoss, FocalTverskyLoss, LossWrapper, BCEAndFocalTverskyLoss, BCELoss
from torchgeo.losses import QRLoss as TorchgeoQRLoss
from model_files.model_losses import dice_loss
from torchgeo.losses import RQLoss


class model_pl(pl.LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()

        # get config
        self.config = config
        self.amp = config.training.amp

        # extract model settings
        self.n_classes = config.model.n_classes
        self.conf_threshold = config.model.conf_threshold
        self.model = self.get_model(config)
        self.criterion = self.get_loss_fn()
        self.test_segmentation_metrics = []
        self.test_object_metrics = []
        self.test_object_metrics_per_size = []

    def get_loss_fn(self):
        loss_command = self.config.training.loss.fn
        print("Creating loss of type:", loss_command)
        if loss_command == "BCEWithLogitsLoss":
            loss_fn = (nn.CrossEntropyLoss() if self.n_classes > 1 else nn.BCEWithLogitsLoss())
            return LossWrapper(loss_fn, expects_probs=True)

        elif loss_command == "BoundaryAwareLoss":
            loss_fn = BoundaryAwareLoss(dilation_ratio=0.02, alpha=1.0, beta=1.0)
            return LossWrapper(loss_fn, expects_probs=True)

        elif loss_command == "QRLoss":
            loss_fn = TorchgeoQRLoss()
            return LossWrapper(loss_fn, expects_probs=True)

        elif loss_command == "RQLoss":
            loss_fn = RQLoss()
            return LossWrapper(loss_fn, expects_probs=True)

        elif loss_command == "FocalLoss":
            loss_fn = FocalTverskyLoss(alpha=self.config.training.loss.a,
                                       beta=self.config.training.loss.b,
                                       gamma=self.config.training.loss.g)
            return LossWrapper(loss_fn, expects_probs=True)

        elif loss_command == "BCE":
            loss_fn = BCELoss()
            return LossWrapper(loss_fn, expects_probs=True)

        elif loss_command == "BCE+FTL":
            loss_fn = BCEAndFocalTverskyLoss(
                alpha=self.config.training.loss.a,
                beta=self.config.training.loss.b,
                gamma=self.config.training.loss.g,
                bce_weight=0.5,
                ftl_weight=0.5
            )
            return LossWrapper(loss_fn, expects_probs=True)
        else:
            raise ValueError("Invalid Loss Function")

    def get_model(self, config):
        print("Creating Model of Type", config.model.model_type)
        # Load model parameters from config

        if config.model.model_type == "unet":
            from model_files.unet_model import UNet
            print("WARNING; this is the old weird unet!")

            model = UNet(
                n_channels=config.model.n_channels, n_classes=config.model.n_classes
            )
        elif config.model.model_type == "new_unet":
            import segmentation_models_pytorch as smp
            model = smp.Unet(
                encoder_name=config.model.encoder,
                encoder_weights=None,
                encoder_depth=config.model.encoder_depth,
                decoder_channels=config.model.decoder_channels,
                decoder_use_norm=config.model.decoder_use_norm,
                decoder_attention_type=config.model.decoder_attention_type,
                decoder_interpolation=config.model.decoder_interpolation,
                in_channels=config.model.n_channels,
                classes=config.model.n_classes,
                activation=None
            )
        elif config.model.model_type == "unet_pp":
            import segmentation_models_pytorch as smp
            model = smp.UnetPlusPlus(
                encoder_name=config.model.encoder,
                encoder_weights=None,
                encoder_depth=config.model.encoder_depth,
                decoder_channels=config.model.decoder_channels,
                decoder_use_norm=config.model.decoder_use_norm,
                decoder_attention_type=config.model.decoder_attention_type,
                decoder_interpolation=config.model.decoder_interpolation,
                in_channels=config.model.n_channels,
                classes=config.model.n_classes,
                activation=None
            )
        elif config.model.model_type == "segformer":
            import segmentation_models_pytorch as smp
            model = smp.Segformer(
                encoder_name=config.model.encoder,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights=None,  # use `imagenet` pre-trained weights for encoder initialization
                in_channels=config.model.n_channels,  # model input channels (4 for RGB-NIR, 3 for RGB, etc.)
                classes=1,
            )  # model output channels (number of classes in your dataset)
        elif config.model.model_type == "manet":
            import segmentation_models_pytorch as smp
            model = smp.MAnet(
                encoder_name=config.model.encoder,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights=None,  # use `imagenet` pre-trained weights for encoder initialization
                in_channels=config.model.n_channels,  # model input channels (4 for RGB-NIR, 3 for RGB, etc.)
                classes=1,
            )  # model output channels (number of classes in your dataset)
        elif config.model.model_type == "pan":
            import segmentation_models_pytorch as smp
            model = smp.PAN(
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
        elif config.model.model_type == "hrnet":
            from model_files.HRNet.lib.models import hrnet, seg_hrnet, seg_hrnet_ocr
            print('loading hr net')
            config = OmegaConf.load('./model_files/HRNet/configs/seg_hrnet_ocr.yaml')
            model = seg_hrnet_ocr.get_seg_model(cfg=config)

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
        return self.forward(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        # Use AMP (Automatic Mixed Precision) during forward pass
        with torch.cuda.amp.autocast(enabled=False):
            y_hat = self.forward(x)
            loss = self.criterion(y_hat, y.float())

        self.log("train_loss", loss)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        x, y = batch  # get Data
        y_hat = self.predict(x)  # Forward pass
        val_loss = self.criterion(y_hat, y.float())

        self.log("val_loss", val_loss)  # Log

        # check if aux, main from hr net
        if isinstance(y_hat, list):
            y_hat_pred = y_hat[1]
        else:
            y_hat_pred = y_hat

        # Thresholding, sigmoid in predict
        y_hat_thresh = (torch.sigmoid(y_hat_pred.detach()) > self.conf_threshold) * 1

        # Metrics
        if self.is_trainer_attached():
            loss_px_dict, status_px = calculate_metrics(y, y_hat_thresh, self.conf_threshold, phase="val")
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

            if batch_idx < 5:  # log only first 5 val batches
                val_image = log_images(x, y, y_hat_thresh, title="Training")
                self.logger.experiment.log(
                    {"images/Validation": [wandb.Image(val_image)]}
                )
        return val_loss

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        print('test step')
        # calculate several metrics at once! log to returns and then create tables in on_test_epoch_end()
        image_ids, x, y = batch  # get Data
        y_hat = self.predict(x)  # Forward pass
        val_loss = self.criterion(y_hat, y.float())

        #self.log("test_loss", val_loss)  # Log

        # check if aux, main from hr net
        if isinstance(y_hat, list):
            y_hat_pred = y_hat[1]
        else:
            y_hat_pred = y_hat

        # Thresholding, sigmoid in predict
        y_hat_thresh = (torch.sigmoid(y_hat_pred.detach()) > self.conf_threshold) * 1
        (segmentation_metrics, object_metrics, object_metrics_per_size), _ = calculate_test_metrics(image_ids, y, y_hat_thresh, self.conf_threshold, phase="test")
        self.test_segmentation_metrics.append(segmentation_metrics)
        self.test_object_metrics.append(object_metrics)
        self.test_object_metrics_per_size.append(object_metrics_per_size)

        save_to_disk = False
        if save_to_disk:
            for image_id, pred in zip(image_ids, y_hat_thresh):
                np_pred = pred.cpu().numpy().squeeze()

                # load georeference
                # crop georeference to 256x256 tile
                # save img


        # Metrics
        if self.is_trainer_attached():
            if batch_idx < 5:  # log only first 5 val batches
                val_image = log_images(x, y, y_hat_thresh, title="Testing")
                self.logger.experiment.log(
                    {"images/Testing": [wandb.Image(val_image)]}
                )
        return segmentation_metrics, object_metrics, object_metrics_per_size

    def on_test_epoch_end(self):
        self.aggregate_metrics(metric_dict=self.test_segmentation_metrics,
                               metric_name='segmentation_metrics')
        self.aggregate_metrics(metric_dict=self.test_object_metrics,
                               metric_name='object_metrics')
        self.aggregate_metrics(metric_dict=self.test_object_metrics_per_size,
                               metric_name='object_metrics_per_size')
        return

    def aggregate_metrics(self, metric_dict, metric_name):
        agg_metric = {k: [] for k in metric_dict[0].keys()}

        for batch in metric_dict:
            for metric_key, metric_values in batch.items():
                agg_metric[metric_key].extend(metric_values)

        df = pd.DataFrame.from_dict(agg_metric)

        full_table = wandb.Table(dataframe=df)
        wandb.log({f"full test {metric_name}": full_table})

        mean_df = pd.DataFrame([df.drop(columns=['image_id']).mean()])
        mean_table = wandb.Table(dataframe=mean_df)
        wandb.log({f"mean test {metric_name}": mean_table})

        if not os.path.exists(self.config.training.log_dir):
            os.mkdir(self.config.training.log_dir)

        df.to_csv(os.path.join(self.config.training.log_dir, f'all_{metric_name}.csv'), index=False)
        mean_df.to_csv(os.path.join(self.config.training.log_dir, f'mean_{metric_name}.csv'), index=False)

        return

    def is_trainer_attached(self):
        try:
            return self.trainer is not None
        except RuntimeError:
            return False

    def configure_optimizers(self):
        print('optim')
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
