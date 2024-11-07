import torch
import torch.nn.functional as F
import cv2
import numpy as np



class BoundaryAwareLoss(torch.nn.Module):
    def __init__(self, dilation_ratio=0.02, alpha=1.0, beta=1.0):
        """
        Initialize Boundary-Aware Loss.
        
        Parameters:
        - dilation_ratio (float): Ratio for boundary thickness.
        - alpha (float): Weight for Dice loss.
        - beta (float): Weight for boundary BCE loss.
        """
        
        super().__init__()
        self.dilation_ratio = dilation_ratio
        self.alpha = alpha
        self.beta = beta

    def dice_loss(self, pred, target, smooth=1.0):
        intersection = (pred * target).sum()
        dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
        return 1 - dice

    def forward(self, pred, target):
        # Dice loss on the entire mask
        dice = self.dice_loss(pred, target)
        
        # Boundary mask
        boundary_mask = self.get_boundary_mask(target, dilation_ratio=self.dilation_ratio)
        
        # BCE loss only on boundary pixels
        bce_boundary = F.binary_cross_entropy(pred * boundary_mask, target * boundary_mask, reduction='mean')
        
        # Combined loss
        loss = self.alpha * dice + self.beta * bce_boundary
        return loss
    
    
    def get_boundary_mask(self,mask, dilation_ratio=0.02):
        """
        Generate a boundary mask by dilating and eroding the binary mask.
        
        Parameters:
        - mask (torch.Tensor): The input binary mask of shape (H, W).
        - dilation_ratio (float): Ratio to determine the boundary thickness (default: 2% of the image diagonal).
        
        Returns:
        - torch.Tensor: A boundary mask highlighting edges.
        """
        # Convert to numpy for morphological operations
        mask_np = mask.cpu().numpy().astype(np.uint8)
        
        # Calculate the kernel size for boundary extraction based on image size
        kernel_size = int(dilation_ratio * np.sqrt(mask_np.shape[0] ** 2 + mask_np.shape[1] ** 2))
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        # Dilate and erode
        dilated = cv2.dilate(mask_np, kernel, iterations=1)
        eroded = cv2.erode(mask_np, kernel, iterations=1)
        
        # Boundary is the difference between dilation and erosion
        boundary = dilated - eroded
        return torch.from_numpy(boundary).float().to(mask.device)