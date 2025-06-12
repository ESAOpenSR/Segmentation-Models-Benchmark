import torch
from torch.utils.data import Dataset, DataLoader
import rasterio as rio
import numpy as np
import tacoreader
import os
import pathlib
from glob import glob
from tqdm import tqdm
import rasterio
import random
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import random
import pandas as pd
import pytorch_lightning as pl


class AustriaData(Dataset):
    def __init__(self,
            phase="train",
            image_type="lr",
            interpolation_size=512,
            bands=4,
            crop=True,
            int_type="bicubic"):
        
        self.interpolation_size = interpolation_size
        self.int_type = int_type
        self.bands = bands
        self.crop = crop
        self.image_type = image_type
        
        # select data
        if image_type == "lr":
            im_dir="/data2/simon/austria_buildings/lr_s2"
        elif image_type == "hr":
            im_dir="/data2/simon/austria_buildings/hr_orthofoto"
        elif image_type == "sr":
            im_dir="/data2/simon/austria_buildings/sr_ims_Austria" 
        
        mask_dir="/data2/simon/austria_buildings/hr_mask"
        
        im_paths = glob(os.path.join(im_dir, "*.tif"))
        mask_paths = glob(os.path.join(mask_dir, "*.tif"))
        
        def extract_id(path):
            return int(os.path.splitext(os.path.basename(path))[0].split("_")[-1])
        
        im_dict = {extract_id(p): p for p in im_paths}
        mask_dict = {extract_id(p): p for p in mask_paths}
        common_ids = sorted(set(im_dict) & set(mask_dict))
        self.data = [{"dataset_id": i, "img": im_dict[i], "mask": mask_dict[i]} for i in common_ids]
    
    
        # read val and test csvs
        self.df_val = pd.read_csv("/data2/simon/austria_buildings/splits/val.csv")
        self.df_test = pd.read_csv("/data2/simon/austria_buildings/splits/test.csv")
        ids_val = self.df_val["Unnamed: 0"].tolist()
        ids_test = self.df_test["Unnamed: 0"].tolist()
    
        # split by phase
        if phase == "train":
            self.data = [entry for entry in self.data if entry["dataset_id"] not in ids_val and entry["dataset_id"] not in ids_test]
        elif phase == "test":
            self.data = [entry for entry in self.data if entry["dataset_id"] in ids_test]
        elif phase == "val":
            self.data = [entry for entry in self.data if entry["dataset_id"] in ids_val]
        elif phase == "all":
            pass
        else:
            raise ValueError("Phase must be one of ['train', 'val', 'test', 'all']")
        
        print("Instanciated dataset for phase ", phase," with ", len(self.data), " entries.")
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]

        # Read Img
        with rasterio.open(entry["img"]) as src:
            img = src.read().astype("float32")
            # if whole S2 image, extract RGB-NIR
            if self.image_type == "lr":
                img = img[[3,2,1,7],:,:]
        
        # Reas Mask
        with rasterio.open(entry["mask"]) as src:
            mask = src.read().astype("float32")
            mask = (mask == 41.)

        img = torch.from_numpy(img)/10_000

        # replace noData values with 0
        img[img == -9999] = 0
        # replace NaN values with 0
        img[torch.isnan(img)] = 0
        
        # treat specific image types
        if self.image_type == "lr":
            # interpoalte image upo to size
            img = F.interpolate(
                img.unsqueeze(0),
                size=(self.interpolation_size, self.interpolation_size),
                mode=self.int_type,
            )
            img = img.squeeze(0)
        
        img = torch.Tensor(img)
        mask = torch.Tensor(mask)
            
        return (img, mask)
        

class pl_datamodule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.batch_size = self.config.data.batch_size
        self.no_workers = self.config.data.no_workers
        self.image_type = self.config.data.data_type
        interpolation_size = self.config.data.interpolation_size
        bands = self.config.data.bands
        crop = self.config.data.crop
        int_type = self.config.data.interpolation_type

        # instanciate datasets for phases
        print("Creating dataset for Type:", self.image_type.upper())
        self.train_dataset = AustriaData(
            phase="train",
            image_type=self.image_type,
            interpolation_size=interpolation_size,
            bands=bands,
            crop=crop,
            int_type=int_type,
        )
        self.test_dataset = AustriaData(
            phase="test",
            image_type=self.image_type,
            interpolation_size=interpolation_size,
            bands=bands,
            crop=crop,
            int_type=int_type,
        )
        self.val_dataset = AustriaData(
            phase="val",
            image_type=self.image_type,
            interpolation_size=interpolation_size,
            bands=bands,
            crop=crop,
            int_type=int_type,
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
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        # Optionally, create a test DataLoader
        # Here we are using the same dataset for simplicity
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

        
if __name__ == "__main__":
    ds = AustriaData(image_type="sr",phase="val")
    batch = ds.__getitem__(0)
    mask = batch[1]
    img = batch[0]
    
    # test datamodule
    from omegaconf import OmegaConf
    conf = OmegaConf.load("configs/unet/config_hr.yaml")
    dm = pl_datamodule(conf)
    
    # plot mask
    import matplotlib.pyplot as plt
    plt.imshow(batch[1].squeeze().numpy(), cmap="gray")
    plt.savefig("mask.png")