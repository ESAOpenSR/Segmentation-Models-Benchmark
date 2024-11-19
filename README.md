# Building Segmentation on LR-HR-SR Satellite Imagery
![Example Image](resources/hr_example.png)
This repository contains code for training and validating segmentation models to perform building delineation on different types of satellite imagery: Low-Resolution (LR), High-Resolution (HR), and Super-Resolution (SR). The goal is to compare the performance of segmentation models across these varying resolutions.

## Overview
The project leverages PyTorch Lightning for model training and Weights & Biases (W&B) for experiment tracking. It includes scripts to train segmentation models and validate them by calculating relevant metrics.

## Project Structure
- train.py: Script to train the segmentation models using configurations specified in YAML files.
- validate.py: Script to validate the trained models and calculate segmentation metrics.
- configs/: Directory containing YAML configuration files for different training setups.
- model_files/: Contains model definitions and utilities.

## Models
The following segmentation models are implemented and can be selected through the configuration files in the configs/ directory:  
| Model Name         | Number of Bands | Pretrained Status                       |
|---------------------|-----------------|-----------------------------------------|
| UNet               | 4               | Scratch                                      |
| UNet++             | 4               | Scratch                                      |
| DeepLabV3          | 4               | Scratch                                      |
| DeepLabV3+         | 4               | Scratch                                      |
| TorchGeo ResNet18  | 3               | Backbone pretrained on S2 RGB                  |
| TorchGeo FarSeg    | 3               | Backbone pretrained on S2 RGB                  |  

These models are customizable via YAML configurations and are compatible with LR, HR, and SR imagery workflows. Important settings when changing models:
- Set number of bands in both model and data section
- Set appropriate loss, define wether sigmoid needs to be applied



## Usage
To train a segmentation model:
1. **Update Configuration**: Modify the configuration files in the configs/ directory to set your training parameters. Things to consider:
- Model Selection: Currently implemented are DeepLabV3, UNet and UNet++
- Training parameters: Optimizers, Schedulers, LRs etc
- Set the LR-SR-HR paramter
- if using dataloaders from this project, make sure to change the data information like path and interpolation setttings

2. **Run Training**: Run train.py to start training, adjust which config to use. Either pass config file as argument from CL, or hardcode.  
```bash
python train.py configs/config_hr.yaml
```

3. **Validate**: Run validate.py
- Give models and loaded weights + dataloaders to opensr-usecases package to get validation metrics, hardcoded in script.
- output in package format, as both text and mAP curve image.
- optionally, send output to txt file 
```bash
python validate.py >> logs/output.txt
```

### Example Output
The validation is based on an external package. It outputs a numerical caluclation of the improvement of SR basic imagery over LR, as well as mAP curves for all data types  
![example_output](resources/example_output.png)  
![example_output](resources/mAP_plot.png)  
