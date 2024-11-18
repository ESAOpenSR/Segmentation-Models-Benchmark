import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
import os
from torchvision import transforms
import mlstac
from io import BytesIO
import numpy as np
import h5py
import pandas as pd
import geopandas as gpd
import pytorch_lightning as pl
import rasterio


class SegmentationDataset(Dataset):
    def __init__(self,root_path_jsons="/data1/simon/GitHub2/building_segmentation/data",
                 phase='test',
                 image_type="lr",
                 interpolation_size=512,
                 bands=4,
                 crop=False):
        self.image_type = image_type # Either LR od HR
        self.interpolation_size = interpolation_size
        assert self.image_type in ["lr","hr","sr"]
        assert bands in [3,4]
        self.bands = bands
        self.crop = crop

        # build path to geojson file and open
        file_path = os.path.join(root_path_jsons,"gdf_geoms_"+phase+"_buildings.geojson")
        self.data = gpd.read_file(file_path)

        # filter for data that has images and masks, and for buildings > 0
        len_1 = len(self.data)
        self.data = self.data[self.data['mask_file'] != 'null']
        len_2 = len(self.data)        
        self.data['amount_buildings'] = self.data['amount_buildings'].astype(int)
        self.data = self.data[self.data['amount_buildings'] > 0]
        len_3 = len(self.data) 
        status_string = "Original Dataset Lenght for phase "+phase+": "+str(len_1)
        status_string+= " - Filtered for valid masks: "+str(len_2)
        status_string+= " - Filtered for buildings > 0: "+str(len_3)

        if self.image_type=="sr":
            self.validate_sr_data()
            len_4 = len(self.data)
            status_string+= " - Filtered for SR images:"+str(len_4)
            
        # print status
        print(status_string)



    def validate_sr_data(self):
        image_folder_path = "/data3/use_cases/buildings_us/sr_images/"
        
        sr_validity = []
        for index, row in self.data.iterrows():
            file_name = row['datapoint_id']+ "_SR.tif"
            im_path = os.path.join(image_folder_path,file_name)
            if os.path.exists(im_path):
                sr_validity.append(True)
            else:
                sr_validity.append(False)
        self.data['sr_validity'] = sr_validity
        self.data = self.data[self.data['sr_validity']==True]


    def __len__(self):
        return len(self.data)
    
    def get_data(self,datapoint):
        data_bytes = mlstac.get_data(dataset=datapoint,
            backend="bytes",
            save_metadata_datapoint=True,
            quiet=True)

        with BytesIO(data_bytes[0][0]) as f:
            with h5py.File(f, "r") as g:
                #metadata = eval(g.attrs["metadata"])
                if self.image_type=="lr":
                    im = np.moveaxis(g["input"][0:4], 0, -1)
                elif self.image_type=="hr":
                    im = np.moveaxis(g["target"][0:4], 0, -1)
                else:
                    raise ValueError("Image type must be either 'lr' or 'hr'")
                
        im = im.astype(np.float32)
        return(im)

    def get_sat_image(self,datapoint):
        im = self.get_data(datapoint)
        im = im.transpose(2,0,1)
        if self.image_type=="lr":
            im = im[:,:128,:128]
        elif self.image_type=="hr" or self.image_type=="sr":
            im = im[:,:512,:512]
        im = torch.tensor(im).float()
        im = im/10000.
        return(im,True)

    def get_mask(self,datapoint):
        mask_path = datapoint['mask_file']
        if os.path.exists(mask_path):
            mask = Image.open(mask_path)
            mask = np.array(mask)
            mask = torch.tensor(mask).float()
            return mask,True
        else:
            return None,False
            
    def get_sr(self, datapoint):
        image_folder_path = "/data3/use_cases/buildings_us/sr_images/"
        image_path = datapoint['datapoint_id'] + "_SR.tif"
        image_path = os.path.join(image_folder_path, image_path)
        
        if os.path.exists(image_path):
            # Open the .tif file with rasterio
            with rasterio.open(image_path) as src:
                im = src.read()  # Reads all bands of the image
            
            # Convert to torch tensor and normalize
            im = torch.tensor(im).float()
            #im = im / 10000.0  # Normalization
            
            return im, True
        else:
            print("Path doesn't exist:", image_path)
            return None, False

    def find_max_one_section(self,mask,image):
        # Ensure mask is 512x512
        if mask.shape != (1,512, 512):
            raise ValueError("Mask must be of size 512x512")
        #if image.shape!= (4,512,512):
        #    raise ValueError("Image must be of size 4x512x512")
        
        # Split mask into four sections
        sections = {
            "top_left": [mask[:,:256, :256],image[:,:256, :256]],
            "top_right": [mask[:,:256, 256:],image[:,:256, 256:]],
            "bottom_left": [mask[:,256:, :256],image[:,256:, :256]],
            "bottom_right": [mask[:,256:, 256:],image[:,256:, 256:]]
        }
        
        # Count 1 pixels in each section
        section_counts = {k: torch.sum(v[0] == 1).item() for k, v in sections.items()}
        
        # Find section with max 1 pixels
        max_section = max(section_counts, key=section_counts.get)
        
        # extract mask and image of max section
        max_section_mask = sections[max_section][0]
        max_section_image = sections[max_section][1]
        
        return max_section_mask, max_section_image


    def __getitem__(self, idx):
        datapoint = self.data.iloc[idx]
        # get image
        if self.image_type in ["lr","hr"]:
            im,statuts_img = self.get_sat_image(datapoint)
        elif self.image_type=="sr":
            im,statuts_img = self.get_sr(datapoint)

        # get mask
        mask,statuts_mask = self.get_mask(datapoint)

        # recursive call is anything went wrong in the data retrieval
        if statuts_img == False or statuts_mask == False:
            random_int = np.random.randint(0,len(self.data))
            return self.__getitem__(random_int)
        
        if self.bands!=4:
            im = im[:self.bands,:,:]

        # adjust image size to interpolation size
        if im.shape[-1] != self.interpolation_size:
            int_mode = self.config.data.interpolation_type
            im = torch.nn.functional.interpolate(im.unsqueeze(0),size=(self.interpolation_size,self.interpolation_size),mode=int_mode)
            im = im.squeeze(0)
            
        # adjust mask size to image size
        mask = mask.unsqueeze(0).unsqueeze(0)
        mask = torch.nn.functional.interpolate(mask,size=(im.shape[1],im.shape[2]),mode="nearest")
        mask = mask.squeeze(0) # squeeze 1x less to receive 1xHxW
        
        # get crop if wanted
        if self.crop:
            mask, im = self.find_max_one_section(mask,im)

        return(im,mask)



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

        # instanciate datasets for phases
        print("Creating dataset for Type:",self.image_type.upper())
        self.train_dataset = SegmentationDataset(phase="train",
                                                 image_type= self.image_type,
                                                 interpolation_size=interpolation_size,
                                                 bands=bands,
                                                 crop=crop)
        self.test_dataset = SegmentationDataset(phase="test",
                                                image_type= self.image_type,
                                                interpolation_size=interpolation_size,
                                                bands=bands,
                                                crop=crop)
        self.val_dataset = SegmentationDataset(phase="val",
                                               image_type= self.image_type,
                                               interpolation_size=interpolation_size,
                                               bands=bands,
                                               crop=crop)

    
    def train_dataloader(self):
        # Return the DataLoader for training
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.no_workers,
                          prefetch_factor=4)
    
    def val_dataloader(self):
        # Optionally, create a validation DataLoader
        # Here we are using the same dataset for simplicity
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        # Optionally, create a test DataLoader
        # Here we are using the same dataset for simplicity
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)





if __name__ == "__main__":
    # dataset
    ds = SegmentationDataset(phase="train",image_type="sr")
    im,mask = ds.__getitem__(10)

    # datamodule
    from omegaconf import OmegaConf
    config = OmegaConf.load("configs/config_sr.yaml")
    pl_dm = pl_datamodule(config)
    im,mask = next(iter(pl_dm.train_dataloader()))
    
    
    # iterate over dataloaders to prove validity
    from tqdm import tqdm
    for i in tqdm(pl_dm.train_dataloader(),desc="Train"):
        assert i[0].shape == (32,4,256,256), "Wrong shape: "+str(i[0].shape)
        assert i[1].shape == (32,1,256,256),"Wrong shape: "+str(i[1].shape)
        pass

    for i in tqdm(pl_dm.val_dataloader(),desc="Val"):
        pass
        
    for i in tqdm(pl_dm.test_dataloader(),desc="Test"):
        pass
        
