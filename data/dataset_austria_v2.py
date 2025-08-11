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
        interpolate_lr = getattr(config.data, "interpolate_lr", False)
        lr_interpolation_type = getattr(config.data, "lr_interpolation_type", None)

        self.data_path = self.config.data.data_path
        self.input_path = self.config.data.input_path
        self.target_path = self.config.data.target_path

        train = Path(self.data_path) / 'train.csv'
        test = Path(self.data_path) / 'test.csv'
        val = Path(self.data_path) / 'val.csv'

        # instanciate datasets for phases
        print("Creating dataset AUSTRIA_V2 for Type:", self.image_type.upper())
        self.train_dataset = TIFDataset(
            data_table=train,
            input_path=self.input_path,
            target_path=self.target_path,
            phase="train",
            image_type=self.image_type,
            bands=bands,
            use_subsample=self.use_256_subsample,
            resample_512=resample_512,
            interpolate_lr=interpolate_lr,
            lr_interpolation_type=lr_interpolation_type,
            return_index=False,
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
            interpolate_lr=interpolate_lr,
            lr_interpolation_type=lr_interpolation_type,
            return_index=True,
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
            interpolate_lr=interpolate_lr,
            lr_interpolation_type=lr_interpolation_type,
            return_index=False,
        )

    def train_dataloader(self):
        # Return the DataLoader for training
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.no_workers,
                          prefetch_factor=4,
                          persistent_workers=True,)

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

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Args:
            img (np.ndarray): C x H x W
        Returns:
            torch.Tensor: C x H x W scaled to [0, 1]
        """
        assert img.ndim == 3, "Expected C x H x W"
        out = np.empty_like(img, dtype=np.float32)

        for c in range(img.shape[0]):
            band = img[c]
            vmin = np.percentile(band, self.pmin)
            vmax = np.percentile(band, self.pmax)
            if vmax - vmin > 1e-6:
                out[c] = (band - vmin) / (vmax - vmin)
            else:
                out[c] = 0.0  # handle uniform bands

        return np.clip(out, 0.0, 1.0, out=out)


# read in panda file
class TIFDataset(Dataset):
    def __init__(
        self,
        data_table: str | Path = "",
        input_path='',
        target_path='',
        phase="test",
        image_type="lr",
        mask_class=41,
        band_indices=None,
        bands=4,
        use_subsample=True,
        resample_512=True,
        interpolate_lr=False,
        lr_interpolation_type=None,
        return_index=False,
    ):
        # own, defintely needed
        assert Path(data_table).exists()
        self.data = pd.read_csv(data_table)
        self.input_path = input_path
        self.target_path = target_path
        self.transform = PercentileScaleClip(pmin=2, pmax=98)
        self.image_type = image_type  # Either LR od HR
        self.mask_class = mask_class
        self.phase = phase
        self.use_subsample = use_subsample
        self.resample_512 = resample_512
        self.interpolate_lr = interpolate_lr
        self.lr_interpolation_type = lr_interpolation_type
        self.return_index = return_index

        # maybe, dont want to do 3band stuff
        self.bands = bands

        # assertion and validation
        assert self.image_type in ["hr", "sr", "sr_4band"]
        assert bands in [3, 4]
        if self.interpolate_lr:
            assert self.lr_interpolation_type in ['bilinear', 'nearest', 'bicubic']
        if self.return_index and self.phase != 'test':
            raise TypeError('Selected image index returning for non-test phase!')

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

        # validate for a singel image selected randomly
        if not self.return_index:
            self.validate(idx=np.random.randint(low=0, high=len(self.data)), verbose=False)

    def validate(self, idx=0, verbose=False):
        if verbose:
            print(f'Validating dataset on image idx=={idx}')

        img, mask = self.__getitem__(idx)
        img, mask = img.numpy(), mask.numpy()
        cimg_min, cimg_max = np.min(img), np.max(img)

        if verbose:
            print('    Image statistics:')
            print(f'       shape: {img.shape}, {mask.shape}')
            print(f'       value range: {cimg_min}, {cimg_max}')

        if cimg_min < 0 or cimg_max > 1.0:
            print(cimg_min, cimg_max)
            raise TypeError(f'   Validator: {cimg_min} is smaller than 0 or {cimg_max} is greater than 1 after conversion.')

        return img, mask

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
        ).squeeze().numpy()
        return ret_arr.squeeze()

    def resample_torch(self, arr: np.ndarray, scale_factor: float | int, mode: str ='bilinear') -> np.ndarray:
        """
        Args:
            arr: (channel x width x height) array will be resampled to (channel x scale_factor*width x scale_factor*height)
            scale_factor: float, 0.25 for HR -> LR
                                 4 for HR -> LR

        Returns: ret_arr

        """
        ret_arr = torch.nn.functional.interpolate(
            torch.from_numpy(arr).unsqueeze(0),
            scale_factor=scale_factor,
            mode=mode,
            antialias=True
        ).squeeze().numpy()
        return ret_arr

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # tile indexing
        image_id = f"{self.id_prefix}_{self.data.loc[idx, 'id']:05d}.tif"
        mask_id = f"HR_mask_{self.data.loc[idx, 'id']:05d}.tif"

        # Load image
        with rasterio.open(Path(self.input_path) / image_id) as src:
            img = src.read(self.band_indices).astype(np.float32)
            img_profile = src.profile

            # validate data is in the range as well
            if img_profile['dtype'] == 'float32':
                img = img
            elif img_profile['dtype'] == 'uint16':
                img = img / 10000.0 #65535.0  # 10_000 #10000
            elif img_profile['dtype'] == 'uint8':
                img = img / 256

        if self.interpolate_lr:
            img = self.resample_torch(img, scale_factor=2, mode=self.lr_interpolation_type)

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
        #img_trafo = torch.from_numpy(self.transform(img))
        img_trafo = torch.from_numpy(img)
        mask_trafo = torch.from_numpy(mask).float().unsqueeze(0)

        if self.return_index:
            return f"{self.data.loc[idx, 'id']:05d}", img_trafo, mask_trafo
        else:
            return img_trafo, mask_trafo

