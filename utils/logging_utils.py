# plot 3 iamges next to each other
import matplotlib.pyplot as plt
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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
            pred = plot_mask_with_threshold(pred,
                                            low_threshold=0.1,
                                            high_threshold=0.75)
            axes[i][2].imshow(pred, interpolation='none')
            axes[i][2].set_title(f"Predicted Mask\n({str(0.75)} in red)")
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


def plot_mask_with_threshold(mask, low_threshold=0.2, high_threshold=0.75):
    # Convert the mask to a NumPy array if needed
    mask = mask.numpy() if isinstance(mask, torch.Tensor) else np.array(mask)

    # Set values below low_threshold to 0
    mask[mask < low_threshold] = 0
    # Set values above high_threshold to 1
    mask[mask > high_threshold] = 1

    # Stretch values between low_threshold and high_threshold
    mask_in_range = (mask >= low_threshold) & (mask <= high_threshold)
    mask[mask_in_range] = (mask[mask_in_range] - low_threshold) / (high_threshold - low_threshold) * 0.99

    # Initialize RGB image with zeros (black)
    rgb_image = np.zeros((*mask.shape, 3), dtype=np.uint8)
    
    # Create grayscale for intermediate values (between 0 and 1)
    grayscale_values = (mask * 255).astype(np.uint8)
    rgb_image[mask > 0] = np.stack([grayscale_values[mask > 0]] * 3, axis=-1)
    
    # Set bright red for values that are exactly 1
    rgb_image[mask == 1] = [255, 0, 0]
    
    return rgb_image


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
    
    # Test RGB-Thresholding
    import torch
    res = plot_mask_with_threshold(torch.rand(256,256))
    # Plot the mask
    plt.imshow(res,interpolation='none')
    plt.colorbar()
    plt.axis('off')
    plt.savefig("sample_mask.png")
    plt.close()


