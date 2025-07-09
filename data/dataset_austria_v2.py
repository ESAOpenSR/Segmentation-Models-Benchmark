import numpy as np
import pandas as pd
import rasterio
import random
import torch
import pytorch_lightning as pl
import os
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pathlib
from pathlib import Path


class pl_datamodule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.batch_size = self.config.data.batch_size
        self.no_workers = self.config.data.no_workers
        self.image_type = self.config.data.data_type
        self.use_256_subsample = self.config.data.use_256_subsample
        bands = self.config.data.bands
        resample_512 = getattr(config.data, "resample_512", False)
        lr_interpolation = getattr(config.data, "lr_interpolation", False)

        self.data_path = self.config.data.data_path
        self.input_path = self.config.data.input_path
        self.target_path = self.config.data.target_path

        train = Path(self.data_path) / 'train.csv'
        test = Path(self.data_path) / 'test.csv'
        val = Path(self.data_path) / 'val.csv'

        # instanciate datasets for phases
        print("Creating dataset for Type:", self.image_type.upper())
        self.train_dataset = TIFDataset(
            data_table=train,
            input_path=self.input_path,
            target_path=self.target_path,
            phase="train",
            image_type=self.image_type,
            bands=bands,
            use_subsample=self.use_256_subsample,
            resample_512=resample_512,
            lr_interpolation=lr_interpolation,
        )
        self.test_dataset = TIFDataset(
            data_table=test,
            input_path=self.input_path,
            target_path=self.target_path,
            phase="test",
            image_type=self.image_type,
            bands=bands,
            use_subsample=self.use_256_subsample,
            resample_512=resample_512,
            lr_interpolation=lr_interpolation,
        )
        self.val_dataset = TIFDataset(
            data_table=val,
            input_path=self.input_path,
            target_path=self.target_path,
            phase="val",
            image_type=self.image_type,
            bands=bands,
            use_subsample=self.use_256_subsample,
            resample_512=resample_512,
            lr_interpolation=lr_interpolation,
        )

    def train_dataloader(self):
        # Return the DataLoader for training
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.no_workers,
            prefetch_factor=4,
            persistent_workers=True,
        )

    def val_dataloader(self):
        # Optionally, create a validation DataLoader
        # Here we are using the same dataset for simplicity
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.no_workers,
                          prefetch_factor=4,
                          persistent_workers=True,)

    def test_dataloader(self):
        # Optionally, create a test DataLoader
        # Here we are using the same dataset for simplicity
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.no_workers,
                          prefetch_factor=4,
                          persistent_workers=True,)


class PercentileScaleClip:
    def __init__(self, pmin=2, pmax=98):
        self.pmin = pmin
        self.pmax = pmax

    def __call__(self, img: np.ndarray) -> torch.Tensor:
        """
        Args:
            img (np.ndarray): C x H x W
        Returns:
            torch.Tensor: C x H x W scaled to [0, 1]
        """
        assert img.ndim == 3, "Expected C x H x W"
        img = img.astype(np.float32)

        for c in range(img.shape[0]):
            band = img[c]
            vmin = np.percentile(band, self.pmin)
            vmax = np.percentile(band, self.pmax)
            if vmax - vmin > 1e-6:
                img[c] = (band - vmin) / (vmax - vmin)
            else:
                img[c] = 0.0  # handle uniform bands

        img = np.clip(img, 0.0, 1.0)
        return torch.from_numpy(img)



# read in panda file
class TIFDataset(Dataset):
    def __init__(
        self,
        data_table: str | Path = "",
        input_path='',
        target_path='',
        transform=A.Compose([A.Normalize(mean=0, std=1), ToTensorV2()]), # removed A.Normalize(mean=0, std=1),
        phase="test",
        image_type="lr",
        mask_class=41,
        band_indices=None,
        bands=4,
        use_subsample=True,
        resample_512=True,
        lr_interpolation=False,
    ):
        # own, defintely needed
        assert Path(data_table).exists()
        self.data = pd.read_csv(data_table)
        self.input_path = input_path
        self.target_path = target_path

        ### attenntion different normlaiyation now
        #self.transform = transform
        self.transform = PercentileScaleClip(pmin=2, pmax=98)


        self.image_type = image_type  # Either LR od HR
        self.mask_class = mask_class
        self.phase = phase
        self.use_subsample = use_subsample
        self.resample_512 = resample_512
        self.lr_interpolation = lr_interpolation

        # maybe, dont want to do 3band stuff
        self.bands = bands

        # assertion and validation
        assert self.image_type in ["hr", "sr", "sr_4band"]
        assert bands in [3, 4]
        # assert input_path in self.data.columns
        # assert target_path in self.data.columns
        #self.validate_data()
        # generated
        # allows adding of individual sr-indexing of channel bands
        if band_indices is None:
            if self.image_type == 'hr':
                # assume hr is orthophoto
                self.band_indices = [1, 2, 3, 4]
                self.id_prefix = 'HR_ortho'
            elif self.image_type == 'sr':
                # Select bands: Red (B4), Green (B3), Blue (B2), and NIR (B8)
                self.band_indices = [4, 3, 2, 8]
                self.id_prefix = 'S2'
            elif self.image_type == 'sr_4band':
                # Select bands: Red (B4), Green (B3), Blue (B2), and NIR (B8)
                self.band_indices = [1, 2, 3, 4]
                self.id_prefix = 'S2'

    def validate_data(self):
        for i, col in self.data.iterrows():
            img, mask = Path(col[self.input_path]), Path(col[self.target_path])
            assert img.exists()
            assert mask.exists()
            # other validation
        pass

    def resample_mask_torch(self, arr: np.ndarray, scale_factor: float | int) -> np.ndarray:
        """
        Args:
            arr: (channel x width x height) array will be resampled to (channel x scale_factor*width x scale_factor*height)
            scale_factor: float, 0.25 for HR -> LR
                                 4 for HR -> LR

        Returns: ret_arr

        """
        # nearest neigbor interpolation for masks allows strict adherence to lr mask
        if arr.ndim == 2:
            arr = np.expand_dims(arr, axis=0)
        ret_arr = torch.nn.functional.interpolate(
            torch.from_numpy(arr).unsqueeze(0).float(),
            scale_factor=scale_factor,
            mode="nearest",
            # antialias=True
        ).squeeze().numpy()
        return ret_arr.squeeze()

    def resample_torch(self, arr: np.ndarray, scale_factor: float | int, mode: str='bilinear') -> np.ndarray:
        """
        Args:
            arr: (channel x width x height) array will be resampled to (channel x scale_factor*width x scale_factor*height)
            scale_factor: float, 0.25 for HR -> LR
                                 4 for HR -> LR

        Returns: ret_arr

        """
        anti_alias = False if mode=='nearest' else True
        ret_arr = torch.nn.functional.interpolate(
            torch.from_numpy(arr).unsqueeze(0),
            scale_factor=scale_factor,
            mode=mode,
            antialias=anti_alias
        ).squeeze().numpy()
        return ret_arr

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # tiel indexing

        image_id = f"{self.id_prefix}_{self.data.loc[idx, 'id']:05d}.tif"
        mask_id = f"HR_mask_{self.data.loc[idx, 'id']:05d}.tif"

        img_profile = None
        # Load image
        with rasterio.open(Path(self.input_path) / image_id) as src:
            img = src.read(self.band_indices).astype(np.float32)
            img_profile = src.profile
            if img_profile['dtype'] == 'float32':
                img = img
            elif img_profile['dtype'] == 'uint16':
                img = img / 65535.0

        if self.lr_interpolation:
            img = self.resample_torch(img, scale_factor=4, mode='nearest')

        # Load mask
        with rasterio.open(Path(self.target_path) / mask_id) as src:
            mask = src.read(1).astype(np.uint8)  # Read first band only

        # Convert mask to binary if needed (Assumes 0/1 classes)
        mask = (mask == self.mask_class).astype(np.float32)

        # use a 256 subsample with most building pixels
        if self.use_subsample:
            tile_coords = [(0, 0), (0, 256), (256, 0), (256, 256)]

            # Dictionary to store all tiles with their perc_count
            tiles_dict = {}

            for top, left in tile_coords:
                img_tile = img[:, top:top + 256, left:left + 256]
                mask_tile = mask[top:top + 256, left:left + 256]
                perc_count = np.count_nonzero(mask_tile)

                # Store tiles in dictionary with their perc_count as value
                tiles_dict[(top, left)] = (img_tile, mask_tile, perc_count)

            # Select the tile with the highest perc_count
            (top, left), (img, mask, _) = max(tiles_dict.items(), key=lambda x: x[1][2])

        if self.resample_512:
            # resample from bilinear: (4, 256, 256) -> (4, 512, 512)
            #               nearest: (256, 256) -> (512, 512)
            img = self.resample_torch(img, scale_factor=2)
            mask = self.resample_mask_torch(mask, scale_factor=2)

        # apply transform manually here - only scaling to 0-1
        img_trafo = torch.from_numpy(img)
        mask_trafo = torch.from_numpy(mask).unsqueeze(0)

        # if self.transform:
        #     transformed = self.transform(image=img.transpose(1, 2, 0), mask=mask)
        #     img_trafo = transformed["image"]
        #     mask_trafo = transformed["mask"]
        # if self.transform:
        #     img_trafo = self.transform(img)  # returns torch.Tensor C x H x W scaled [0, 1]
        #     mask_trafo = torch.from_numpy(mask).float()

            # print(img_trafo.shape, mask_trafo.shape)

        return img_trafo, mask_trafo  # Add channel dimension to mask

