import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
torch.set_float32_matmul_precision("medium")

# General Imports
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from pathlib import Path
import shutil
import rasterio
import numpy as np
from tqdm import tqdm

# Local Imports
from data.dataset_austria import pl_datamodule
from model_files.model_wrapper import model_pl
from opensr_usecases import Validator


def process_mask(mask, low_threshold=0.2, high_threshold=0.75):
    # Set values below low_threshold to 0
    mask[mask < high_threshold] = 0
    # Set values above high_threshold to 1
    mask[mask >= high_threshold] = 1
    return mask


def infer_images(dataloader, model, outpath, device="cpu"):

    for idx in tqdm(range(len(dataloader))):
        #for idx in tqdm(range(1000)):
        image_id = f"{dataloader.id_prefix}_{dataloader.data.loc[idx, 'id']:05d}.tif"
        mask_id = f"HR_mask_{dataloader.data.loc[idx, 'id']:05d}.tif"

        if Path(Path(outpath / image_id)).exists():
            continue

        img_profile = None
        # Load image
        with rasterio.open(Path(dataloader.input_path) / image_id) as src:
            img = src.read(dataloader.band_indices).astype(np.float32)
            img_profile = src.profile

            # if dataloader.use_subsample:
            if False:
                tile_coords = [(0, 0), (0, 256), (256, 0), (256, 256)]
                inferred_img = np.zeros(shape=(1, 512, 512))

                for top, left in tile_coords:
                    img_tile = img[:, top:top + 256, left:left + 256]
                    if dataloader.transform:
                        transformed = dataloader.transform(image=img_tile.transpose(1, 2, 0))
                        img_trafo = transformed["image"]

                        with torch.no_grad():
                            inferred_img[:, top:top + 256, left:left + 256] = model(img_trafo.unsqueeze(0).to(device)).squeeze(0).detach().cpu()
            else:
                if dataloader.transform:
                    transformed = dataloader.transform(image=img.transpose(1, 2, 0))
                    img_trafo = transformed["image"]

                with torch.no_grad():
                    inferred_img = model(img_trafo.unsqueeze(0).to(device)).squeeze(0).detach().cpu().numpy()
                    inferred_img = process_mask(mask=inferred_img, low_threshold=0.75, high_threshold=1)

            img_profile.update({'count': 1, 'compress': 'zstd'})
            with rasterio.open(Path(outpath / image_id), 'w', **img_profile) as dst:
                dst.write(inferred_img)

        if not Path(f'/data/USERS/shollend/inferred_buildings/gt/{mask_id}').exists():
            with rasterio.open(Path(dataloader.target_path) / mask_id) as src:
                mask = src.read(1).astype(np.uint8)  # Read first band only
                mask = np.expand_dims(mask == dataloader.mask_class, axis=0)

                with rasterio.open(Path(f'/data/USERS/shollend/inferred_buildings/gt/{mask_id}'), 'w', **src.profile) as dst:
                    dst.write(mask)


# Set Config and CKPT base paths
model_type = "unet_pp"
output_name = f'080525_bilinear_ortho_SEN2SR_{model_type}'
cfg_base_path = "logs/Samuel_building_segmentation/RUN/train_config.yaml"
ckpt_base_path = "logs/Samuel_building_segmentation/RUN/"

inf_type = 'diffusion'
model_run = '2025-05-07_21-59-34'

inf_type = 'bilinear'
model_run = '2025-05-06_17-08-07'

inf_type = 'orthofoto'
model_run = '2025-05-06_17-06-13'

data = [('diffusion', '2025-05-07_21-59-34'), ('bilinear', '2025-05-06_17-08-07'), ('orthofoto', '2025-05-06_17-06-13')]

for inf_type, model_run in data:
    config = OmegaConf.load(cfg_base_path.replace("RUN", model_run))

    # load data
    data_module = pl_datamodule(config)
    dataloader = data_module.test_dataset

    model = model_pl(config)
    ortho_weights = list(Path(ckpt_base_path.replace("RUN", model_run)).glob('epoch=*.ckpt'))[0]
    ckpt = torch.load(ortho_weights, map_location='cuda:0')
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    outpath = '/data/USERS/shollend/inferred_buildings'
    foutpath = Path(outpath) / inf_type / model_run
    foutpath.mkdir(exist_ok=True, parents=True)
    shutil.copy(cfg_base_path.replace("RUN", model_run), foutpath / 'train_config.yaml')

    img_outpath = foutpath / 'imgs'
    img_outpath.mkdir(exist_ok=True)

    infer_images(dataloader=dataloader, model=model, outpath=img_outpath)
