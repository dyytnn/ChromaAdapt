import os

import cv2
import numpy as np
import torch
from torch import Tensor


def denormalize_to_uint8(tensor, mean, std):
    """
    Fully denormalize a tensor to [0, 255] uint8 for visualization or saving.
    
    Args:
        tensor: Tensor of shape (C, H, W) or (B, C, H, W), normalized.
        mean: List of means per channel.
        std: List of stds per channel.
    
    Returns:
        Numpy array (uint8), shape (H, W, C) or (B, H, W, C)
    """
    is_batched = tensor.dim() == 4
    if not is_batched:
        tensor = tensor.unsqueeze(0)

    mean = torch.tensor(mean, device=tensor.device).reshape(1, -1, 1, 1)
    std = torch.tensor(std, device=tensor.device).reshape(1, -1, 1, 1)

    tensor = tensor * std + mean  # back to [0, 1]
    tensor = tensor * 255.0       # back to [0, 255]
    tensor = tensor.clamp(0, 255).byte()  # ensure uint8 range

    np_imgs = tensor.permute(0, 2, 3, 1).cpu().numpy()  # (B, H, W, C)

    return np_imgs[0] if not is_batched else np_imgs

def save_data(image: Tensor, segmentation_labels: list[Tensor], mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225], save_path: str = "debug"):
    img_denorm = denormalize_to_uint8(image, mean, std)
    for idx, im in enumerate(img_denorm):
        labels = segmentation_labels[1][idx].squeeze(0).cpu().detach().numpy() * 125
        labels = cv2.cvtColor(labels.astype(np.uint8), cv2.COLOR_GRAY2BGR)

        stacked = np.hstack([im, labels])
        cv2.imwrite(os.path.join(save_path, f"{idx}.png"), stacked)
        
def save_data_with_preds(
    image: Tensor,
    segmentation_labels: list[Tensor],
    segmentation_preds: Tensor,
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
    save_path: str = "debug"
):
    os.makedirs(save_path, exist_ok=True)
    img_denorm = denormalize_to_uint8(image, mean, std)  # [B, H, W, 3]
    pred_mask_all = torch.argmax(segmentation_preds, dim=1).cpu().detach().numpy()  # [B, H, W]

    for idx, im in enumerate(img_denorm):
        gt_mask = segmentation_labels[1][idx].squeeze(0).cpu().detach().numpy() * 125
        gt_mask = cv2.cvtColor(gt_mask.astype(np.uint8), cv2.COLOR_GRAY2BGR)

        pred_mask = pred_mask_all[idx] * 125
        pred_mask = cv2.cvtColor(pred_mask.astype(np.uint8), cv2.COLOR_GRAY2BGR)

        # Ensure all shapes are the same
        if im.shape != gt_mask.shape:
            gt_mask = cv2.resize(gt_mask, (im.shape[1], im.shape[0]))
        if im.shape != pred_mask.shape:
            pred_mask = cv2.resize(pred_mask, (im.shape[1], im.shape[0]))

        stacked = np.hstack([im, gt_mask, pred_mask])
        cv2.imwrite(os.path.join(save_path, f"{idx}.png"), stacked)
