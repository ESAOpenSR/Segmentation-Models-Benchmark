# plot 3 iamges next to each other
import matplotlib.pyplot as plt
import torch
from PIL import Image
import numpy as np
import io

def minmax_percentile(im,percentile=3):
    """
    Min-Max Normalization with Percentile Clipping
    """
    im_min = np.percentile(im,percentile)
    im_max = np.percentile(im,100-percentile)
    im = (im-im_min)/(im_max-im_min)
    im = np.clip(im,0,1)
    return im

def visualize_red_and_gray(image,v=0.5):
    """
    Visualizes areas over 0.5 in red and the rest in shades of gray.

    Parameters:
    - image: A 2D numpy array with values ranging from 0 to 1.
    """
    # Create an empty RGB image with the same height and width as the input image
    rgb_image = np.zeros((*image.shape, 3))

    # Areas over 0.5: paint them in red (set the red channel to 1)
    red_mask = image > v
    rgb_image[red_mask, 0] = 1  # Red channel for areas > 0.5

    # Areas 0 to 0.5: paint them in grayscale
    gray_mask = ~red_mask
    rgb_image[gray_mask] = np.stack([image[gray_mask]] * 3, axis=-1)  # Set R, G, and B to the same value
    #rgb_image = rgb_image.clip(0, 1)  # Clip values to the range [0, 1]
    return rgb_image


def log_images(images, masks, preds, title="Training"):
    """
    Plots up to 5 images stacked vertically. For each batch sample, 
    it plots three images next to each other: an RGB image, a ground truth mask, 
    and a predicted mask.

    Parameters:
    - images: Batch of RGB images (tensor with shape [B, C, H, W])
    - masks: Batch of ground truth masks (tensor with shape [B, 1, H, W] or [B, H, W])
    - preds: Batch of predicted masks (tensor with shape [B, 1, H, W] or [B, H, W])
    - title: Title for the plot (default is 'Training')
    """
    # set CMAP
    cmap = "gray"
    # Ensure we're only working with the first 5 images in the batch
    batch_size = min(images.shape[0], 5)
    # batch tensors iof theyre not batched
    if images.ndim == 3:
        images = images.unsqueeze(0)
    if masks.ndim == 2:
        masks = masks.unsqueeze(0)
    if preds.ndim == 2:
        preds = preds.unsqueeze(0)
    if masks.ndim == 3:
        masks = masks.unsqueeze(1)
    if preds.ndim == 3:
        preds = preds.unsqueeze(1)
    
    # Convert the tensors to numpy arrays and prepare for plotting
    images_np = images.cpu().numpy()
    masks_np = masks.cpu().numpy()
    preds_np = preds.cpu().numpy()
    
    # Create a figure with subplots for each image, stacked vertically
    fig, axes = plt.subplots(batch_size, 3, figsize=(12, 4 * batch_size))
    fig.suptitle(title, fontsize=16)
    
    if batch_size == 1:
        axes = [axes]  # Ensure axes is always 2D for uniform indexing
    
    for i in range(batch_size):
        # Get current image, mask, and prediction
        image = images_np[i,:3,:,:].transpose(1, 2, 0)  # Change [C, H, W] -> [H, W, C]
        image = minmax_percentile(image,percentile=4)
        mask = masks_np[i][0] if masks_np[i].ndim == 3 else masks_np[i]  # Handle shape [B, 1, H, W] or [B, H, W]
        pred = preds_np[i][0] if preds_np[i].ndim == 3 else preds_np[i]  # Handle shape [B, 1, H, W] or [B, H, W]

        # Plot RGB image
        axes[i][0].imshow(image, interpolation='none')
        axes[i][0].set_title("RGB Image")
        #axes[i][0].axis('off')

        # Plot mask image
        axes[i][1].imshow(mask, cmap=cmap, interpolation='none')
        axes[i][1].set_title("Ground Truth Mask")
        axes[i][1].axis('off')

        # Plot predicted mask image
        draw_red = False
        if draw_red:
            red_v = 0.75
            pred = visualize_red_and_gray(pred,v=red_v)
            axes[i][2].imshow(pred, cmap=cmap, interpolation='none')
            axes[i][2].set_title("Predicted Mask\n>"+str(red_v)+" conf in red")
        else:
            axes[i][2].imshow(pred, cmap=cmap, interpolation='none')
            axes[i][2].set_title("Predicted Mask")
        axes[i][2].axis('off')
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    #plt.savefig("sample_images.png")
    
    # Convert the plot to a PIL image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=200)
    buf.seek(0)
    pil_image = Image.open(buf)
    plt.close(fig)  # Close the figure to free up memory

    return pil_image


if __name__ == "__main__":
    # datamodule
    from omegaconf import OmegaConf
    from data.dataset_masks import pl_datamodule
    config = OmegaConf.load("configs/config_hr.yaml")
    pl_dm = pl_datamodule(config)
    images,masks = next(iter(pl_dm.train_dataloader()))
    preds = torch.rand_like(masks)

    # Plot the images
    pil_image = log_images(images, masks, preds, title="Validation")
    pil_image.save("sample_images_2.png")


