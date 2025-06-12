import torch
import numpy as np
import random
from omegaconf import OmegaConf
import opensr_model
import pytorch_lightning    
import cv2
import matplotlib.pyplot as plt
from data.dataset_masks import pl_datamodule
from model_files.model_wrapper import model_pl
import opensr_model
from tqdm import tqdm
import os,sys

os.environ['CUDA_VISIBLE_DEVICES'] = '3'


def calc_unc(model_sr,image,iterations=25):
    # get unc for image
    assert len(image.shape) == 4, "Image should be in shape [1,4,H,W]"
    assert image.shape[0]==1, "Image should be in shape [1,4,H,W]"
    
    rand_seed_list = random.sample(range(1, 9999), iterations)
    variations = []
    for seed in rand_seed_list:
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        #pytorch_lightning.utilities.seed.seed_everything(seed=seed,workers=True)
        sr = model_sr(image)
        sr = sr.squeeze(0)
        variations.append(sr)

    variations = torch.stack(variations)
    srs_mean = variations.mean(dim=0)
    srs_stdev = variations.std(dim=0)
    lower_bound = srs_mean-srs_stdev
    upper_bound = srs_mean+srs_stdev
    interval_size = srs_stdev*2
    interval_size = interval_size.mean(dim=0).unsqueeze(0).unsqueeze(0)
    return interval_size
    

def get_mask(model,image,masks):
    #assert image.shape[0]==1, "Image should be in shape [1,4,H,W]"
    #assert len(image.shape) == 4, "Image should be in shape [1,4,H,W]"
    #image = image.squeeze(0)
    #masks = masks.squeeze(0)
    with torch.no_grad():
        pred_masks = model(image).sigmoid()
    return(pred_masks)

def calculate_score_by_object(raster,mask):
    # Convert mask to binary (0 or 255)
    mask_binary = mask>0.75
    mask_binary = mask_binary*255
    raster = raster.squeeze().detach().cpu().numpy()
    mask_binary = mask_binary.squeeze().detach().cpu().numpy().astype(np.uint8)
    num_labels, labels_im = cv2.connectedComponents(mask_binary)
    # Calculate the average values for each object
    averages = []
    for label in range(1, num_labels):  # Start from 1 to ignore the background
        component_mask = (labels_im == label)
        component_values = raster[component_mask]
        average_value = np.mean(component_values)
        averages.append(average_value)
    return averages


# 1. Get Model ---------------------------------------------------------------------
# Set Config and CKPT base paths - Delineeation Model
model_type = "unet" # ["fcn_v1", "unet", "dl","farseg"]
device = "cuda" if torch.cuda.is_available() else "cpu"
iteration = "v1"
cfg_base_path = "configs/MODEL/config_TYPE.yaml"
ckpt_base_path = "logs/MODEL_v1/TYPE_MODEL.ckpt"
cfg_base_path = cfg_base_path.replace("MODEL", model_type)
ckpt_base_path = ckpt_base_path.replace("MODEL", model_type)
# get Model (delineation)
config = OmegaConf.load(cfg_base_path.replace("TYPE", "sr"))
model = model_pl(config).to(device)
ckpt = torch.load(ckpt_base_path.replace("TYPE", "sr"), map_location='cuda:0')
model.load_state_dict(ckpt['state_dict'])
data_module = pl_datamodule(config)
dataloader = data_module.test_dataloader()
# get Model (SR)
model_sr = opensr_model.SRLatentDiffusion(bands="10m",device=device) # 10m
model_sr.load_pretrained("opensr_10m_v4_v6.ckpt") # 10m


unc_per_object_ls = []
pred_per_object_ls = []
for batch in tqdm(dataloader):
    try:
        image,mask = batch[0],batch[1]
        image = image[0].unsqueeze(0).to(device)
        mask = mask[0].unsqueeze(0).to(device)
        mask = torch.nn.functional.interpolate(mask, size=(512,512), mode='nearest') 


        unc = calc_unc(model_sr=model_sr,image=image,iterations=50)
        sr = model_sr(image)
        pred_masks = get_mask(model=model,image=sr,masks=mask)
        mask_up = torch.nn.functional.interpolate(mask, size=(512,512), mode='nearest') 
    
        unc_per_object = calculate_score_by_object(unc,mask_up)
        pred_per_object = calculate_score_by_object(pred_masks,mask)
                        
        for i in unc_per_object:
            unc_per_object_ls.append(i)
        for i in pred_per_object:
            pred_per_object_ls.append(i)
            
        # save to catch errors 
        torch.save(torch.Tensor(unc_per_object_ls), "results/tensors_unc_by_building/unc_per_object_ls.pt")
        torch.save(torch.Tensor(pred_per_object_ls), "results/tensors_unc_by_building/pred_per_object_ls.pt")
            
    except ValueError as e:
        print("Error:",e)
        

# ------------------------------------------------------------------------------------------------------------------
# plot result
import torch
import matplotlib.pyplot as plt
pred_per_object_ls = torch.load("results/tensors_unc_by_building/pred_per_object_ls.pt")
unc_per_object_ls = torch.load("results/tensors_unc_by_building/unc_per_object_ls.pt")
plt.figure(figsize=(10,10))
plt.scatter(x=pred_per_object_ls,y=unc_per_object_ls)
# add labels
plt.title("Uncertainty vs Prediction Score per Building")
plt.ylabel("Uncertainty")
plt.xlabel("Prediction Logit Score")
plt.savefig("results/unc_vs_pred.png")
plt.close()
    
# plot Result w/ Regression and Heatmap
# ------------------------------------------------------------------------------------------------------------------
import torch
import matplotlib.pyplot as plt
import numpy as np

# Load data
pred_per_object_ls = torch.load("results/tensors_unc_by_building/pred_per_object_ls.pt")
unc_per_object_ls = torch.load("results/tensors_unc_by_building/unc_per_object_ls.pt")

# Fit a linear regression line
coefficients = np.polyfit(pred_per_object_ls, unc_per_object_ls, 1)
polynomial = np.poly1d(coefficients)
x_lin_reg = np.linspace(pred_per_object_ls.min(), pred_per_object_ls.max(), 100)
y_lin_reg = polynomial(x_lin_reg)

# Plotting
plt.figure(figsize=(10, 10))
plt.scatter(x=pred_per_object_ls, y=unc_per_object_ls)
plt.plot(x_lin_reg, y_lin_reg, color='red')  # Regression line in red for visibility
plt.title("Uncertainty vs Prediction Score per Building")
plt.ylabel("Uncertainty")
plt.xlabel("Prediction Logit Score")
plt.savefig("results/unc_vs_pred_linear.png")
plt.close()

# ------------------------------------------------------------------------------------------------------------------
import torch
import matplotlib.pyplot as plt
import numpy as np

# Load data
pred_per_object_ls = torch.load("results/tensors_unc_by_building/pred_per_object_ls.pt")
unc_per_object_ls = torch.load("results/tensors_unc_by_building/unc_per_object_ls.pt")

max = unc_per_object_ls.max()
unc_per_object_ls = unc_per_object_ls/max
unc_per_object_ls = 1-unc_per_object_ls

max = pred_per_object_ls.max()
pred_per_object_ls = pred_per_object_ls/max


# Fit a linear regression line
coefficients = np.polyfit(pred_per_object_ls, unc_per_object_ls, 1)
polynomial = np.poly1d(coefficients)
x_lin_reg = np.linspace(pred_per_object_ls.min(), pred_per_object_ls.max(), 100)
y_lin_reg = polynomial(x_lin_reg)

# Plotting using hexbin for heatmap effect
plt.figure(figsize=(15, 10))
hb = plt.hexbin(pred_per_object_ls, unc_per_object_ls, gridsize=50, cmap='inferno', bins='log')
cb = plt.colorbar(hb)
cb.set_label('log10(N)')

# Add regression line over the heatmap, thickness
plt.plot(x_lin_reg, y_lin_reg, color='red', linewidth=3, linestyle=':')  # Cyan color for visibility on dark backgrounds

plt.title("Heatmap of Uncertainty vs Prediction Score per Building")
plt.ylabel("Uncertainty")
plt.xlabel("Prediction Logit Score")
plt.savefig("results/heatmap_unc_vs_pred_lin_heat.png")
plt.close()



# Remove Outliers
# ------------------------------------------------------------------------------------------------------------------

import torch
import numpy as np

# Load data
pred_per_object_ls = torch.load("results/tensors_unc_by_building/pred_per_object_ls.pt")
unc_per_object_ls = torch.load("results/tensors_unc_by_building/unc_per_object_ls.pt")

# Convert tensors to numpy arrays if not already
pred_per_object_ls = pred_per_object_ls.numpy()
unc_per_object_ls = unc_per_object_ls.numpy()

# Define a function to remove outliers using the IQR
def remove_outliers(data):
    q25, q75 = np.percentile(data, [25, 75])
    iqr = q75 - q25
    lower_bound = q25 - 1.5 * iqr
    upper_bound = q75 + 1.5 * iqr
    return data[(data >= lower_bound) & (data <= upper_bound)]

# Filter outliers from both datasets
filtered_pred = remove_outliers(pred_per_object_ls)
filtered_unc = remove_outliers(unc_per_object_ls)

# Ensure that we have the same number of elements in both filtered arrays
# This aligns the pairs correctly after outlier removal
min_length = min(len(filtered_pred), len(filtered_unc))
filtered_pred = filtered_pred[:min_length]
filtered_unc = filtered_unc[:min_length]

# Fit a linear regression line to the filtered data
coefficients = np.polyfit(filtered_pred, filtered_unc, 1)
polynomial = np.poly1d(coefficients)
x_lin_reg = np.linspace(filtered_pred.min(), filtered_pred.max(), 100)
y_lin_reg = polynomial(x_lin_reg)

# Plotting using hexbin for the heatmap
plt.figure(figsize=(10, 10))
hb = plt.hexbin(filtered_pred, filtered_unc, gridsize=50, cmap='inferno', bins='log')
cb = plt.colorbar(hb)
cb.set_label('log10(N)')

# Add regression line over the heatmap
plt.plot(x_lin_reg, y_lin_reg, color='cyan')  # Cyan color for visibility on dark backgrounds

plt.title("Filtered Heatmap of Uncertainty vs Prediction Score per Building")
plt.ylabel("Uncertainty")
plt.xlabel("Prediction Logit Score")
plt.savefig("results/filtered_heatmap_unc_vs_pred.png")
plt.close()