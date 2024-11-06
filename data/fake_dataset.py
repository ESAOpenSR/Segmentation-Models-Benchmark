import torch
import torchvision

from tqdm import tqdm
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch
from einops import rearrange
import random
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
import geopandas as gpd
from shapely.geometry import Polygon
import cv2
import pytorch_lightning as pl

class synth_ds(Dataset):
    def __init__(self,phase="train",interpolation_size=512):
        # define folders
        if phase=="train":
            self.images = [1] * 10000
        elif phase=="val":
            self.images = [1] * 100
        elif phase=="test":
            self.images = [1] * 100
        self.interpolation_size = interpolation_size

    def __len__(self):
        return len(self.images)
    
    def create_square(self,tensor):
        # Define the colors for red, green, and blue
        colors = {
            'red': (1, 0, 0),
            'green': (0, 1, 0),
            'blue': (0, 0, 1)
        }

        # Select a random color
        color_name = random.choice(list(colors.keys()))
        color = colors[color_name]

        # Randomly choose the size and position of the box
        x1, y1 = random.randint(0, 412), random.randint(0, 412)  # Upper-left corner
        width, height = random.randint(20, 100), random.randint(20, 100)  # Width and height

        # Ensure the box stays within the image boundaries
        x2, y2 = min(x1 + width, 512), min(y1 + height, 512)  # Lower-right corner

        # Draw the box on the tensor
        tensor[0, y1:y2, x1:x2] = color[0]  # Red channel
        tensor[1, y1:y2, x1:x2] = color[1]  # Green channel
        tensor[2, y1:y2, x1:x2] = color[2]  # Blue channel
        
        # Create the binary mask (1 inside the square, 0 outside)
        mask = torch.zeros((tensor.shape[1], tensor.shape[2]), dtype=torch.float32)
        mask[y1:y2, x1:x2] = 1.
        #print("Bbox:",x1,y1,x2,y2)
        #print("Tensor max value:", tensor.shape,tensor.max().item())
        #print("Mask max value:", mask.shape,mask.max().item())

        return tensor, mask

            
    def create_batch_data(self,idx=0):

        # load random image from folder
        folder = "/data3/cv_images/test2015"
        files = os.listdir(folder)
        # keep only images
        files = [f for f in files if f.endswith(".png") or f.endswith(".jpg") or f.endswith(".jpeg")]
        file = random.choice(files)
        image = Image.open(os.path.join(folder,file))

        image = image.resize((self.interpolation_size,self.interpolation_size))
        # if image is grayscale, convert to RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Apply the transformation
        image = np.array(image)
        image = image/255
        image = image.transpose(2,0,1)
        
    
        image,mask = self.create_square(image)
        image = torch.Tensor(image).float()
        mask = torch.Tensor(mask).float()
            
            
        mean_band = torch.mean(image, dim=0)
        image = torch.cat((image, mean_band.unsqueeze(0)), dim=0)
        mask = mask.unsqueeze(0)

        return image,mask
    
    def __getitem__(self, idx):
        try:
            image, target = self.create_batch_data()
            image = torch.Tensor(image)
        except KeyboardInterrupt:
            print("Process interrupted by user.")
            raise  # Re-raise the exception to actually stop the loop/operation
        except Exception as e:
            print(f"Error with image {idx}: {e}")
            # Safely generate a new index and recursively call __getitem__
            random_no = random.randint(0, len(self.images) - 1)
            return self.__getitem__(random_no)
        return image, target



class pl_datamodule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.batch_size = self.config.data.batch_size
        self.no_workers = self.config.data.no_workers
        self.image_type = self.config.data.data_type
        interpolation_size = self.config.data.interpolation_size
        bands = self.config.data.bands

        # instanciate datasets for phases
        print("Creating dataset for Type:",self.image_type.upper())
        self.train_dataset = synth_ds(phase="train", interpolation_size=interpolation_size)
        self.test_dataset = synth_ds(phase="test",interpolation_size=interpolation_size)
        self.val_dataset = synth_ds(phase="val",interpolation_size=interpolation_size)

    
    def train_dataloader(self):
        # Return the DataLoader for training
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        # Optionally, create a validation DataLoader
        # Here we are using the same dataset for simplicity
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        # Optionally, create a test DataLoader
        # Here we are using the same dataset for simplicity
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)







if __name__ == "__main__":
    ds = synth_ds(phase="train",interpolation_size=512)
    im,t = ds.__getitem__(100)
