import torch
import torch.nn.functional as F
import cv2
import numpy as np
import torch.nn as nn


class LossWrapper(nn.Module):
    def __init__(self, base_loss: nn.Module, expects_probs: bool):
        super().__init__()
        self.base_loss = base_loss
        self.expects_probs = expects_probs

    def forward(self, logits, targets):
        # remodel hrnet output to main loss instead of aux+main
        if isinstance(logits, list):
            aux_logits, main_logits = logits  # main output instead of aux loss
            if self.expects_probs:
                aux_preds, main_preds = torch.sigmoid(aux_logits), torch.sigmoid(main_logits)
            else:
                aux_preds, main_preds = aux_logits, main_logits

            return self.base_loss(main_preds, targets) + 0.4 * self.base_loss(aux_preds, targets)

        else:
            if self.expects_probs:
                preds = torch.sigmoid(logits)
            else:
                preds = logits
            return self.base_loss(preds, targets)


class BCEAndFocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75, smooth=1e-6, bce_weight=0.5, ftl_weight=0.5):
        """
        Combines BCE Loss and Focal Tversky Loss.

        Args:
            alpha, beta, gamma, smooth: Parameters for FocalTverskyLoss.
            bce_weight: Weight for BCE loss contribution.
            ftl_weight: Weight for Focal Tversky loss contribution.
        """
        super().__init__()
        self.bce = nn.BCELoss()
        self.ftl = FocalTverskyLoss(alpha, beta, gamma, smooth)
        self.bce_weight = bce_weight
        self.ftl_weight = ftl_weight

    def forward(self, y_pred, y_true):
        bce_loss = self.bce(y_pred, y_true)
        ftl_loss = self.ftl(y_pred, y_true)
        return self.bce_weight * bce_loss + self.ftl_weight * ftl_loss


class BCELoss(nn.Module):
    def __init__(self):
        """
        Standard Binary Cross Entropy Loss (expects probabilities, not logits).
        """
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self, y_pred, y_true):
        return self.bce(y_pred, y_true)



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
        dice = (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
        return 1 - dice

    def forward(self, pred, target):
        # Dice loss on the entire mask
        pred, target = pred.squeeze(), target.squeeze()
        dice = self.dice_loss(pred, target)

        # Boundary mask
        boundary_mask = self.get_boundary_mask(
            target, dilation_ratio=self.dilation_ratio
        )

        # BCE loss only on boundary pixels
        bce_boundary = F.binary_cross_entropy(
            torch.sigmoid(pred) * boundary_mask,
            target * boundary_mask,
            reduction="mean",
        )

        # Combined loss
        loss = self.alpha * dice + self.beta * bce_boundary
        return loss

    def get_boundary_mask(self, mask, dilation_ratio=0.02):
        """
        Generate a boundary mask by dilating and eroding the binary mask.

        Parameters:
        - mask (torch.Tensor): The input binary mask of shape (B, H, W) or (B, 1, H, W).
        - dilation_ratio (float): Ratio to determine the boundary thickness (default: 2% of the image diagonal).

        Returns:
        - torch.Tensor: A boundary mask highlighting edges for each image in the batch.
        """
        boundary_masks = []
        batch_size = mask.shape[0]
        height, width = mask.shape[-2], mask.shape[-1]

        # Calculate the kernel size for boundary extraction based on image size
        kernel_size = int(dilation_ratio * np.sqrt(height**2 + width**2))
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        for i in range(batch_size):
            # Convert each mask to numpy
            mask_np = mask[i].squeeze().cpu().numpy().astype(np.uint8)

            # Dilate and erode -> switch to; scipy.ndimage import binary_dilation
            dilated = cv2.dilate(mask_np, kernel, iterations=1)
            eroded = cv2.erode(mask_np, kernel, iterations=1)

            # Boundary is the difference between dilation and erosion
            boundary = dilated - eroded
            boundary_masks.append(torch.from_numpy(boundary).float())

        # Stack and reshape boundary masks back to match input batch shape
        boundary_masks = torch.stack(boundary_masks).unsqueeze(
            1
        )  # Add channel dimension
        return boundary_masks.to(mask.device)


import torch
import torch.nn as nn
from torchgeo.losses import QRLoss as TorchgeoQRLoss


class QRLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super(QRLoss, self).__init__()
        # Instantiate the QRLoss from torchgeo
        self.criterion = TorchgeoQRLoss(*args, **kwargs)

    def forward(self, probs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward method to compute the QR loss.

        Args:
            probs (torch.Tensor): Predicted probabilities, shape (B, C, H, W).
            target (torch.Tensor): Target probabilities, shape (B, C, H, W).

        Returns:
            torch.Tensor: Computed QR loss.
        """
        # Compute the QR loss using the instantiated criterion
        loss = self.criterion(probs, target)
        return loss


# Example usage
if __name__ == "__main__":
    # Dummy data
    B, C, H, W = 2, 3, 4, 4
    probs = torch.rand(B, C, H, W)
    target = torch.rand(B, C, H, W)

    # Normalize to make them probabilities
    probs = probs / probs.sum(dim=1, keepdim=True)
    target = torch.ones_like(probs).float()

    # Initialize the model and compute the loss
    loss_model = QRLoss()
    loss = loss_model(probs, target)
    print(f"QR Loss: {loss.item()}")


import torch
import torch.nn as nn


class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75, smooth=1e-6):
        """
        Initialize the Focal Tversky Loss.

        Args:
        - alpha: Weight for false positives (default: 0.7).
        - beta: Weight for false negatives (default: 0.3).
        - gamma: Focusing parameter (default: 0.75).
        - smooth: Smoothing factor to avoid division by zero (default: 1e-6).
        """
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        print(
            f"Focal Tversky Loss with alpha={alpha}, beta={beta}, gamma={gamma}, smooth={smooth}"
        )

    def forward(self, y_pred, y_true):
        """
        Compute the Focal Tversky Loss.

        Args:
        - y_pred: Predicted logits or probabilities (BxCxHxW or BxHxW).
        - y_true: Ground truth binary mask (same shape as y_pred).

        Returns:
        - Focal Tversky Loss value.
        """
        # Ensure predictions are probabilities
        # y_pred = torch.sigmoid(y_pred)

        # Flatten tensors to compute the Tversky index
        y_pred_flat = y_pred.view(-1)
        y_true_flat = y_true.view(-1)

        # Compute true positives, false positives, and false negatives
        true_positive = torch.sum(y_pred_flat * y_true_flat)
        false_positive = torch.sum(y_pred_flat * (1 - y_true_flat))
        false_negative = torch.sum((1 - y_pred_flat) * y_true_flat)

        # Compute the Tversky index
        tversky_index = (true_positive + self.smooth) / (
            true_positive
            + self.alpha * false_positive
            + self.beta * false_negative
            + self.smooth
        )

        # Compute the Focal Tversky Loss
        focal_tversky_loss = (1 - tversky_index) ** self.gamma
        return focal_tversky_loss


# Example usage
if __name__ == "__main__":
    # Dummy data
    B, C, H, W = 2, 3, 4, 4
    probs = torch.rand(B, C, H, W)
    target = torch.rand(B, C, H, W)

    # Normalize to make them probabilities
    probs = probs / probs.sum(dim=1, keepdim=True)
    target = torch.ones_like(probs).float()

    # Initialize the model and compute the loss
    loss_model = FocalTverskyLoss()
    loss = loss_model(probs, target)
    print(f"focal_tversky_loss: {loss.item()}")
