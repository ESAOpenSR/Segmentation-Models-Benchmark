import numpy as np
import pandas as pd
import rasterio
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
        bands = self.config.data.bands

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
        )
        self.test_dataset = TIFDataset(
            data_table=test,
            input_path=self.input_path,
            target_path=self.target_path,
            phase="test",
            image_type=self.image_type,
            bands=bands,
        )
        self.val_dataset = TIFDataset(
            data_table=val,
            input_path=self.input_path,
            target_path=self.target_path,
            phase="val",
            image_type=self.image_type,
            bands=bands,
        )

    def train_dataloader(self):
        # Return the DataLoader for training
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.no_workers,
            prefetch_factor=4,
        )

    def val_dataloader(self):
        # Optionally, create a validation DataLoader
        # Here we are using the same dataset for simplicity
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        # Optionally, create a test DataLoader
        # Here we are using the same dataset for simplicity
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)


# read in panda file
class TIFDataset(Dataset):
    # def __init__(self, df, input_path, target_path, mask_class=41,
    #              transform=A.Compose([A.Normalize(mean=0, std=1), ToTensorV2()])):
    #     self.image_paths = df[input_path].values  # Get image paths from DataFrame
    #     self.mask_paths = df[target_path].values  # Get mask paths from DataFrame
    #     self.transform = transform
    #     self.mask_class = mask_class
    def __init__(
        self,
        data_table: str | Path = "",
        input_path='',
        target_path='',
        transform=A.Compose([A.Normalize(mean=0, std=1), ToTensorV2()]),
        phase="test",
        image_type="lr",
        mask_class=41,
        band_indices=None,
        bands=4
    ):
        # own, defintely needed
        assert Path(data_table).exists()
        self.data = pd.read_csv(data_table)
        self.input_path = input_path
        self.target_path = target_path
        self.transform = transform
        self.image_type = image_type  # Either LR od HR
        self.mask_class = mask_class
        self.phase = phase

        # maybe, dont want to do 3band stuff
        self.bands = bands

        # assertion and validation
        assert self.image_type in ["hr", "sr"]
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
                self.band_indices = [4, 3, 2, 8]  # Rasterio uses 1-based indexing
                self.id_prefix = 'S2'

    def validate_data(self):
        for i, col in self.data.iterrows():
            img, mask = Path(col[self.input_path]), Path(col[self.target_path])
            assert img.exists()
            assert mask.exists()
            # other validation
        pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_id = f"{self.id_prefix}_{self.data.loc[idx, 'id']:05d}.tif"
        mask_id = f"HR_mask_{self.data.loc[idx, 'id']:05d}.tif"

        # Load image
        with rasterio.open(Path(self.input_path) / image_id) as src:
            img = src.read(self.band_indices).astype(np.float32)

        # Load mask
        with rasterio.open(Path(self.target_path) / mask_id) as src:
            mask = src.read(1).astype(np.uint8)  # Read first band only

        # Convert mask to binary if needed (Assumes 0/1 classes)
        mask = (mask == self.mask_class).astype(np.float32)

        # Apply augmentations
        if self.transform:
            transformed = self.transform(image=img.transpose(1, 2, 0), mask=mask)
            img, mask = transformed["image"], transformed["mask"]
        else:
            raise 'No transform selected: apply at least a normalization'

        return img, mask.unsqueeze(0)  # Add channel dimension to mask

