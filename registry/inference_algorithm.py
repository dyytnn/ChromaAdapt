import os

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from abs_interface import AbsAlgorithmProcessHandler
from configs.config import configurable
from data import real_dataset as build_dataset_class

# from data import dataset as build_dataset_class
from data.transformation import build_transforms
from registry import evaluate_segmentation_handler_register


def refine_by_skeleton(mask: np.ndarray) -> np.ndarray:
    """
    Refine segmentation using skeleton-based approach.

    For each chromosome skeleton, enforce a single dominant label along the entire skeleton.
    This is based on the principle that one skeleton = one chromosome = one class.

    Args:
        mask: Segmentation mask with integer labels (H, W)

    Returns:
        Refined mask where each skeleton has consistent labeling
    """
    from skimage.morphology import dilation, skeletonize, square

    refined_mask = mask.copy()

    # Create a binary mask of all foreground pixels (non-background)
    foreground = (mask > 0).astype(np.uint8)

    # Skeletonize the foreground to get centerlines
    skeleton = skeletonize(foreground)

    # Find connected components in the skeleton
    num_skeletons, skeleton_labels = cv2.connectedComponents(skeleton.astype(np.uint8), connectivity=8)

    # For each skeleton component
    for skel_id in range(1, num_skeletons):
        # Get the skeleton pixels for this component
        skeleton_mask = (skeleton_labels == skel_id)

        # Get the labels along this skeleton
        labels_on_skeleton = mask[skeleton_mask]
        labels_on_skeleton = labels_on_skeleton[labels_on_skeleton > 0]  # Exclude background

        if len(labels_on_skeleton) == 0:
            continue

        # Find the dominant label along this skeleton
        dominant_label = np.bincount(labels_on_skeleton).argmax()

        # Now find the entire region connected to this skeleton
        # Dilate the skeleton to get the full chromosome region
        skeleton_dilated = skeleton_mask.astype(np.uint8)

        # Iteratively dilate until we capture the full connected region
        for _ in range(20):
            # Dilate
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            dilated = cv2.dilate(skeleton_dilated, kernel, iterations=1)

            # Only keep pixels that are foreground
            dilated = dilated & foreground

            # If no more growth, stop
            if np.array_equal(dilated, skeleton_dilated):
                break

            skeleton_dilated = dilated

        # Assign the dominant label to this entire region
        refined_mask[skeleton_dilated > 0] = dominant_label

    return refined_mask


def refine_segmentation_mask(mask: np.ndarray, min_size: int = 50, use_morphology: bool = True, median_kernel: int = 5, fill_enclaves: bool = True, use_skeleton: bool = True) -> np.ndarray:
    """
    Refine segmentation mask by removing noise using multiple methods.

    This function applies several noise removal techniques:
    0. Skeleton-based refinement - enforce one label per skeleton (chromosome centerline)
    1. Morphological opening to remove small isolated pixels
    2. Enclave filling - small regions surrounded by another class get absorbed
    3. Connected component analysis to remove small regions
    4. Median filtering for final smoothing

    Args:
        mask: Segmentation mask with integer labels (H, W)
        min_size: Minimum size of connected components to keep. Smaller components
                  will be relabeled to the dominant neighboring label.
        use_morphology: Apply morphological opening operations
        median_kernel: Kernel size for median filter (set to 0 to disable)
        fill_enclaves: Fill small regions that are surrounded by another class (like Iceland example)
        use_skeleton: Use skeleton-based refinement (recommended for chromosome segmentation)

    Returns:
        Refined mask with noise removed
    """
    refined_mask = mask.copy()

    # Step 0: Skeleton-based refinement (best for chromosomes)
    if use_skeleton:
        refined_mask = refine_by_skeleton(refined_mask)

    unique_labels = np.unique(refined_mask)

    # Step 1: Apply morphological opening for each class
    if use_morphology:
        temp_mask = np.zeros_like(refined_mask)
        for label in unique_labels:
            if label == 0:  # Skip background
                continue

            # Create binary mask for current label
            binary_mask = (refined_mask == label).astype(np.uint8)

            # Apply morphological opening (erosion then dilation)
            # This removes small noise while preserving larger structures
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)

            # Add to temp mask
            temp_mask[opened > 0] = label

        refined_mask = temp_mask

    # Step 2: Fill enclaves - small regions surrounded by another class
    if fill_enclaves:
        # Process from smallest to largest labels to handle nested enclaves
        for label in unique_labels:
            if label == 0:  # Skip background
                continue

            # Create binary mask for current label
            binary_mask = (refined_mask == label).astype(np.uint8)

            # Find connected components
            num_labels_comp, labels_im, stats, _ = cv2.connectedComponentsWithStats(
                binary_mask, connectivity=8
            )

            # Process each connected component
            for comp_label in range(1, num_labels_comp):  # Skip background (0)
                component_size = stats[comp_label, cv2.CC_STAT_AREA]

                # Only process small components
                if component_size < min_size:
                    # Get the component mask
                    component_mask = (labels_im == comp_label)

                    # Create a bounding box around the component with padding
                    x, y, w, h = stats[comp_label, cv2.CC_STAT_LEFT:cv2.CC_STAT_LEFT+4]
                    pad = 10
                    x1, y1 = max(0, x - pad), max(0, y - pad)
                    x2, y2 = min(refined_mask.shape[1], x + w + pad), min(refined_mask.shape[0], y + h + pad)

                    # Extract the region around this component
                    region_mask = refined_mask[y1:y2, x1:x2].copy()
                    component_region = component_mask[y1:y2, x1:x2]

                    # Check what's surrounding this component
                    # Create a dilated version to get the surrounding pixels
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                    dilated = cv2.dilate(component_region.astype(np.uint8), kernel, iterations=3)

                    # Get the ring around the component (dilated - original)
                    surrounding_ring = dilated & ~component_region

                    # Get labels in the surrounding ring
                    surrounding_labels = region_mask[surrounding_ring]
                    surrounding_labels = surrounding_labels[(surrounding_labels != label) & (surrounding_labels != 0)]

                    if len(surrounding_labels) > 0:
                        # Find the dominant surrounding label
                        label_counts = np.bincount(surrounding_labels)
                        dominant_surrounding = label_counts.argmax()

                        # Calculate what percentage of the surroundings is the dominant label
                        dominance_ratio = label_counts[dominant_surrounding] / len(surrounding_labels)

                        # If one label dominates the surroundings (>70%), this is an enclave
                        if dominance_ratio > 0.7:
                            # Fill the enclave with the surrounding label
                            refined_mask[component_mask] = dominant_surrounding

    # Step 3: Connected component analysis for remaining small regions
    for label in unique_labels:
        if label == 0:  # Skip background
            continue

        # Create binary mask for current label
        binary_mask = (refined_mask == label).astype(np.uint8)

        # Find connected components
        num_labels_comp, labels_im, stats, _ = cv2.connectedComponentsWithStats(
            binary_mask, connectivity=8
        )

        # Process each connected component
        for comp_label in range(1, num_labels_comp):  # Skip background (0)
            component_size = stats[comp_label, cv2.CC_STAT_AREA]

            # If component is too small, it's likely noise
            if component_size < min_size:
                # Get the component mask
                component_mask = (labels_im == comp_label)

                # Find neighbors by dilating the component
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                dilated = cv2.dilate(component_mask.astype(np.uint8), kernel, iterations=2)
                neighbors_mask = dilated & ~component_mask

                # Get neighbor labels (excluding current label and background)
                neighbor_labels = refined_mask[neighbors_mask]
                neighbor_labels = neighbor_labels[(neighbor_labels != label) & (neighbor_labels != 0)]

                if len(neighbor_labels) > 0:
                    # Find the most common neighboring label
                    dominant_label = np.bincount(neighbor_labels).argmax()
                    # Relabel this noisy component
                    refined_mask[component_mask] = dominant_label

    # Step 4: Apply median filter for final smoothing
    if median_kernel > 0:
        refined_mask = cv2.medianBlur(refined_mask, median_kernel)

    return refined_mask


def calculate_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Calculate Intersection over Union (IoU) between two binary masks.

    Args:
        mask1: Binary mask (H, W)
        mask2: Binary mask (H, W)

    Returns:
        IoU value between 0 and 1
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    if union == 0:
        return 0.0

    return intersection / union


def match_masks_by_iou(masks_original: list, masks_transformed: list) -> dict:
    """
    Match masks from original and transformed images using IoU.

    Implements step 4 from the paper:
    For each mask m_i in original, find the mask m'_j in transformed
    that has the highest IoU.

    Args:
        masks_original: List of binary masks from original image
        masks_transformed: List of binary masks from transformed image

    Returns:
        Dictionary mapping original mask index to transformed mask index
        phi(i) = j means mask_i matches with mask'_j
    """
    matching = {}

    for i, mask_i in enumerate(masks_original):
        best_iou = 0.0
        best_j = -1

        for j, mask_j in enumerate(masks_transformed):
            iou = calculate_iou(mask_i, mask_j)
            if iou > best_iou:
                best_iou = iou
                best_j = j

        if best_j != -1:
            matching[i] = best_j

    return matching


def combine_matched_masks(masks_original: list, masks_transformed: list, matching: dict) -> list:
    """
    Combine matched masks using element-wise minimum (intersection).

    Implements step 5 from the paper:
    m_i = m_i ∧ m'_φ(i)

    This creates more conservative predictions by only keeping pixels
    that are consistently predicted in both views.

    Args:
        masks_original: List of binary masks from original image
        masks_transformed: List of binary masks from transformed image
        matching: Dictionary mapping original index to transformed index

    Returns:
        List of refined masks
    """
    refined_masks = []

    for i, mask_i in enumerate(masks_original):
        if i in matching:
            j = matching[i]
            mask_j = masks_transformed[j]

            # Element-wise minimum (intersection)
            refined_mask = np.logical_and(mask_i, mask_j).astype(np.uint8)
            refined_masks.append(refined_mask)
        else:
            # No match found, use original mask
            refined_masks.append(mask_i)

    return refined_masks


def keep_largest_components(mask: np.ndarray, n: int = 2) -> np.ndarray:
    """
    Keep only the N largest connected components in a binary or multi-class mask.

    For multi-class masks, this processes each class independently.

    Args:
        mask: Binary or multi-class mask (H, W)
        n: Number of largest components to keep per class

    Returns:
        Filtered mask with only N largest components per class
    """
    filtered_mask = np.zeros_like(mask)
    unique_labels = np.unique(mask)

    for label in unique_labels:
        if label == 0:  # Skip background
            continue

        # Create binary mask for this class
        binary_mask = (mask == label).astype(np.uint8)

        # Find connected components
        num_labels, labels_im, stats, _ = cv2.connectedComponentsWithStats(
            binary_mask, connectivity=8
        )

        if num_labels <= 1:  # Only background
            continue

        # Get component sizes (skip background at index 0)
        component_sizes = stats[1:, cv2.CC_STAT_AREA]

        # Get indices of N largest components (add 1 to account for skipping background)
        largest_indices = np.argsort(component_sizes)[::-1][:n] + 1

        # Keep only the N largest components
        for comp_idx in largest_indices:
            component_mask = (labels_im == comp_idx)
            filtered_mask[component_mask] = label

    return filtered_mask


def apply_flip_or_tta(
    model: nn.Module,
    image: torch.Tensor,
    device: str = "cuda",
    n_largest: int = 2,
    inference_mode: str = "both"
) -> tuple:
    """
    Apply simplified TTA using logical OR of original and horizontally flipped predictions.

    This approach:
    1. Generates predictions for original image
    2. Generates predictions for horizontally flipped image
    3. Flips back the flipped predictions
    4. Combines using logical_or (union)
    5. Keeps only N largest connected components to filter noise

    Args:
        model: Segmentation model
        image: Input image tensor (B, C, H, W)
        device: Device to run inference on
        n_largest: Number of largest components to keep (default 2)
        inference_mode: What to infer - "both", "chromosome", or "overlap"

    Returns:
        Tuple of (chromosome_mask, overlap_mask) after flip-or TTA
        Returns None for masks not requested by inference_mode
    """
    model.eval()

    # Step 1: Generate predictions for original image
    with torch.no_grad():
        _, outputs_orig = model(image.to(device))

        if inference_mode in ["both", "chromosome"]:
            chromosome_pred_orig = torch.argmax(outputs_orig[0], dim=1).squeeze(0)  # (H, W)
        else:
            chromosome_pred_orig = None

        if inference_mode in ["both", "overlap"]:
            overlap_pred_orig = (torch.sigmoid(outputs_orig[1]) > 0.85).long().squeeze()  # (H, W)
        else:
            overlap_pred_orig = None

    # Step 2: Generate predictions for horizontally flipped image
    image_flipped = torch.flip(image, dims=[3])  # Horizontal flip

    with torch.no_grad():
        _, outputs_flipped = model(image_flipped.to(device))

        if inference_mode in ["both", "chromosome"]:
            chromosome_pred_flipped = torch.argmax(outputs_flipped[0], dim=1).squeeze(0)  # (H, W)
            # Step 3: Flip back the prediction
            chromosome_pred_flipped = torch.flip(chromosome_pred_flipped.unsqueeze(0), dims=[2]).squeeze(0)
        else:
            chromosome_pred_flipped = None

        if inference_mode in ["both", "overlap"]:
            overlap_pred_flipped = (torch.sigmoid(outputs_flipped[1]) > 0.85).long().squeeze()  # (H, W)
            # Step 3: Flip back the prediction
            overlap_pred_flipped = torch.flip(overlap_pred_flipped.unsqueeze(0), dims=[2]).squeeze(0)
        else:
            overlap_pred_flipped = None

    # Step 4: Combine using logical_or
    chromosome_final = None
    overlap_final = None

    if chromosome_pred_orig is not None:
        # Convert to numpy for processing
        chrom_orig_np = chromosome_pred_orig.cpu().numpy()
        chrom_flip_np = chromosome_pred_flipped.cpu().numpy()

        # For multi-class segmentation, we need to handle each class separately
        # Logical OR for each class: keep pixel if either prediction has that class
        chromosome_combined = np.zeros_like(chrom_orig_np)

        for class_id in [1, 2]:  # Assuming classes 1 and 2 (0 is background)
            mask_orig_class = (chrom_orig_np == class_id)
            mask_flip_class = (chrom_flip_np == class_id)

            # Logical OR
            combined_class = np.logical_or(mask_orig_class, mask_flip_class).astype(np.uint8)

            # Assign class label where combined mask is True
            chromosome_combined[combined_class > 0] = class_id

        # Step 5: Keep only N largest components per class
        chromosome_filtered = keep_largest_components(chromosome_combined, n=n_largest)
        chromosome_final = torch.from_numpy(chromosome_filtered)

    if overlap_pred_orig is not None:
        # Convert to numpy
        overlap_orig_np = overlap_pred_orig.cpu().numpy()
        overlap_flip_np = overlap_pred_flipped.cpu().numpy()

        # Logical OR for binary overlap prediction
        overlap_combined = np.logical_or(overlap_orig_np, overlap_flip_np).astype(np.uint8)

        # Step 5: Keep only N largest components
        overlap_filtered = keep_largest_components(overlap_combined, n=n_largest)
        overlap_final = torch.from_numpy(overlap_filtered)

    return chromosome_final, overlap_final


def apply_tta_refinement(
    model: nn.Module,
    image: torch.Tensor,
    device: str = "cuda",
    transformations: list = None,
    inference_mode: str = "both"
) -> tuple:
    """
    Apply Test-Time Augmentation (TTA) refinement to segmentation predictions.

    Implements the TTA algorithm from the paper:
    1. Generate masks from original image: M = SM(x)
    2. Transform image: x' = T(x)
    3. Generate masks from transformed image and inverse transform: M' = T^-1(SM(x'))
    4. Match masks using IoU
    5. Combine masks using element-wise minimum

    Args:
        model: Segmentation model
        image: Input image tensor (B, C, H, W)
        device: Device to run inference on
        transformations: List of augmentation functions to apply
                        Each function should return (transform_fn, inverse_transform_fn)
                        Default: [horizontal_flip]
        inference_mode: What to infer - "both", "chromosome", or "overlap"

    Returns:
        Tuple of (chromosome_mask, overlap_mask) after TTA refinement
        Returns None for masks not requested by inference_mode
    """
    model.eval()

    if transformations is None:
        # Default: horizontal flip
        transformations = [
            (
                lambda x: torch.flip(x, dims=[3]),  # Horizontal flip
                lambda x: torch.flip(x, dims=[3])   # Inverse is the same
            )
        ]

    # Step 1: Generate masks from original image
    with torch.no_grad():
        _, outputs = model(image.to(device))

        # Process based on inference mode
        if inference_mode in ["both", "chromosome"]:
            chromosome_pred_orig = torch.argmax(outputs[0], dim=1).squeeze(0)  # (H, W)
        else:
            chromosome_pred_orig = None

        if inference_mode in ["both", "overlap"]:
            overlap_pred_orig = (torch.sigmoid(outputs[1]) > 0.85).long().squeeze()  # (H, W)
        else:
            overlap_pred_orig = None

    # Convert to numpy and extract individual masks
    from skimage.measure import label, regionprops

    chromosome_masks_orig = []
    overlap_masks_orig = []
    chromosome_mask_orig = None
    overlap_mask_orig = None
    h, w = None, None

    # For chromosome mask
    if chromosome_pred_orig is not None:
        chromosome_mask_orig = chromosome_pred_orig.cpu().numpy()
        h, w = chromosome_mask_orig.shape
        chromosome_labeled = label(chromosome_mask_orig > 0)
        for region in regionprops(chromosome_labeled):
            mask = (chromosome_labeled == region.label).astype(np.uint8)
            chromosome_masks_orig.append(mask)

    # For overlap mask
    if overlap_pred_orig is not None:
        overlap_mask_orig = overlap_pred_orig.cpu().numpy()
        if h is None:
            h, w = overlap_mask_orig.shape
        overlap_labeled = label(overlap_mask_orig > 0)
        for region in regionprops(overlap_labeled):
            mask = (overlap_labeled == region.label).astype(np.uint8)
            overlap_masks_orig.append(mask)

    # Apply each transformation
    for transform_fn, inverse_transform_fn in transformations:
        # Step 2: Transform the image
        image_transformed = transform_fn(image)

        # Step 3: Generate masks from transformed image
        with torch.no_grad():
            _, outputs_transformed = model(image_transformed.to(device))

            if inference_mode in ["both", "chromosome"]:
                chromosome_pred_trans = torch.argmax(outputs_transformed[0], dim=1).squeeze(0)
                # Inverse transform the mask
                chromosome_pred_trans = inverse_transform_fn(chromosome_pred_trans.unsqueeze(0)).squeeze(0)
                # Convert to numpy
                chromosome_mask_trans = chromosome_pred_trans.cpu().numpy()
            else:
                chromosome_mask_trans = None

            if inference_mode in ["both", "overlap"]:
                overlap_pred_trans = (torch.sigmoid(outputs_transformed[1]) > 0.85).long().squeeze()
                # Inverse transform the mask
                overlap_pred_trans = inverse_transform_fn(overlap_pred_trans.unsqueeze(0)).squeeze(0)
                # Convert to numpy
                overlap_mask_trans = overlap_pred_trans.cpu().numpy()
            else:
                overlap_mask_trans = None

        # Extract connected components and match
        if chromosome_mask_trans is not None:
            chromosome_labeled_trans = label(chromosome_mask_trans > 0)
            chromosome_masks_trans = []
            for region in regionprops(chromosome_labeled_trans):
                mask = (chromosome_labeled_trans == region.label).astype(np.uint8)
                chromosome_masks_trans.append(mask)

            # Step 4 & 5: Match and combine masks
            if len(chromosome_masks_orig) > 0 and len(chromosome_masks_trans) > 0:
                chromosome_matching = match_masks_by_iou(chromosome_masks_orig, chromosome_masks_trans)
                chromosome_masks_orig = combine_matched_masks(
                    chromosome_masks_orig, chromosome_masks_trans, chromosome_matching
                )

        if overlap_mask_trans is not None:
            overlap_labeled_trans = label(overlap_mask_trans > 0)
            overlap_masks_trans = []
            for region in regionprops(overlap_labeled_trans):
                mask = (overlap_labeled_trans == region.label).astype(np.uint8)
                overlap_masks_trans.append(mask)

            # Step 4 & 5: Match and combine masks
            if len(overlap_masks_orig) > 0 and len(overlap_masks_trans) > 0:
                overlap_matching = match_masks_by_iou(overlap_masks_orig, overlap_masks_trans)
                overlap_masks_orig = combine_matched_masks(
                    overlap_masks_orig, overlap_masks_trans, overlap_matching
                )

    # Reconstruct final masks from refined individual masks
    chromosome_final = None
    overlap_final = None

    if inference_mode in ["both", "chromosome"] and h is not None:
        chromosome_final = np.zeros((h, w), dtype=np.uint8)
        for i, mask in enumerate(chromosome_masks_orig):
            chromosome_final[mask > 0] = i + 1  # Assign unique label
        chromosome_final = torch.from_numpy(chromosome_final)

    if inference_mode in ["both", "overlap"] and h is not None:
        overlap_final = np.zeros((h, w), dtype=np.uint8)
        for mask in overlap_masks_orig:
            overlap_final[mask > 0] = 1
        overlap_final = torch.from_numpy(overlap_final)

    return chromosome_final, overlap_final


def save_outputs(chromosome_pred: Tensor, overlap_pred: Tensor, output_folder: str, image_name: str, image_path: str = None, refine: bool = True, min_size: int = 50):
    """
    Save chromosome and/or overlap predictions as colored images, optionally with original image.

    Args:
        chromosome_pred: Chromosome prediction tensor, or None if not predicted
        overlap_pred: Overlap prediction tensor, or None if not predicted
        output_folder: Output directory
        image_name: Output filename
        image_path: Path to original image for side-by-side comparison (optional)
        refine: Whether to apply refinement
        min_size: Minimum component size for refinement
    """
    # Handle None values
    if chromosome_pred is None and overlap_pred is None:
        return  # Nothing to save

    outputs_to_save = []

    # Read and add original image if path is provided
    if image_path is not None:
        original_img = cv2.imread(image_path)
        if original_img is not None:
            # Convert grayscale to BGR if needed
            if len(original_img.shape) == 2:
                original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
            outputs_to_save.append(original_img)

    # Process chromosome predictions
    if chromosome_pred is not None:
        chromosome_pred_numpy = chromosome_pred.cpu().detach().numpy().astype(np.uint8)

        # Apply refinement to remove noise
        if refine:
            chromosome_pred_numpy = refine_segmentation_mask(chromosome_pred_numpy, min_size=min_size, use_skeleton=False)

        # Create color palette for chromosome predictions
        # 0: Background (Black), 1: Class 1 (Blue), 2: Class 2 (Green)
        chromosome_color_palette = np.array([
            [0, 0, 0],       # 0 - Background: Black
            [255, 0, 0],     # 1 - Class 1: Blue (BGR format)
            [0, 255, 0],     # 2 - Class 2: Green
        ], dtype=np.uint8)

        # Apply color palette using lookup table
        chromosome_colored = chromosome_color_palette[chromosome_pred_numpy]
        outputs_to_save.append(chromosome_colored)

    # Process overlap predictions
    if overlap_pred is not None:
        overlap_pred_numpy = overlap_pred.cpu().detach().numpy().astype(np.uint8)

        # Apply refinement
        # if refine:
        #     overlap_pred_numpy = refine_segmentation_mask(overlap_pred_numpy, min_size=min_size)

        # Create color palette for overlap predictions
        # 0: No overlap (Black), 1: Overlap (Red)
        overlap_color_palette = np.array([
            [0, 0, 0],       # 0 - No overlap: Black
            [0, 0, 255],     # 1 - Overlap: Red (BGR format)
        ], dtype=np.uint8)

        # Apply color palette using lookup table
        overlap_colored = overlap_color_palette[overlap_pred_numpy]
        outputs_to_save.append(overlap_colored)

    # Stack horizontally if multiple outputs, otherwise save single output
    if len(outputs_to_save) > 1:
        # Ensure all images have same number of channels (convert to BGR if needed)
        processed_outputs = []
        for img in outputs_to_save:
            if len(img.shape) == 2:  # Grayscale image
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 1:  # Single channel but 3D
                img = cv2.cvtColor(img.squeeze(), cv2.COLOR_GRAY2BGR)
            processed_outputs.append(img)

        # Ensure all images have the same height by resizing if needed
        max_height = max(img.shape[0] for img in processed_outputs)
        resized_outputs = []
        for img in processed_outputs:
            if img.shape[0] != max_height:
                # Resize to match max height while preserving aspect ratio
                scale = max_height / img.shape[0]
                new_width = int(img.shape[1] * scale)
                img = cv2.resize(img, (new_width, max_height), interpolation=cv2.INTER_NEAREST)
            resized_outputs.append(img)

        stacked = np.hstack(resized_outputs)
    else:
        stacked = outputs_to_save[0]

    cv2.imwrite(os.path.join(output_folder, image_name), stacked)

@evaluate_segmentation_handler_register.register("Inference")
class Inference(AbsAlgorithmProcessHandler):
    def __init__(self, config: dict):
        self.config = config

    def _pre_process(self, data):
      # Validate inputs
      model = data.get("model", None)
      output_folder = data.get("save_path", None)
      
      if model is None:
          raise ValueError("Model not found in data dict")

      model_path = self.config.get("model_path", None)
      if model_path is None:
          raise ValueError("model_path not specified in config")

      device = self.config.get("device", "cpu")

      # Load checkpoint with error handling
      try:
          ckpt = torch.load(model_path, map_location=device)
      except Exception as e:
          raise RuntimeError(f"Failed to load checkpoint from {model_path}: {e}")

      # Remove DataParallel/DistributedDataParallel prefix
      non_parallel_state_dict = {k.replace("module.", ""): v for k, v in ckpt.items()}

      # Handle domain adaptation models
      if self.config.get("from_domain_adaptation"):
          domain_adaptation_state_dict = {}
          for k, v in non_parallel_state_dict.items():
              if k.startswith("base_model."):
                  domain_adaptation_state_dict[k.replace("base_model.", "")] = v

          # Check if we found any base_model keys
          if not domain_adaptation_state_dict:
              raise ValueError(
                  "from_domain_adaptation=True but no 'base_model.' keys found in checkpoint. "
                  f"Available keys: {list(non_parallel_state_dict.keys())[:5]}..."
              )
          non_parallel_state_dict = domain_adaptation_state_dict

      # Load state dict and validate
      status = model.load_state_dict(non_parallel_state_dict, strict=True)

      # Check for issues
      if status.missing_keys:
          print(f"Warning: Missing keys in checkpoint: {status.missing_keys}")
      if status.unexpected_keys:
          print(f"Warning: Unexpected keys in checkpoint: {status.unexpected_keys}")

      del ckpt
      return model.to(device), output_folder
    
    def _main_process(self, data):
        model, output_folder = data
        model.eval()
        
        dataset_provider = self.config.get("Dataset")["provider"]
        dataset_config = self.config.get("Dataset")[dataset_provider]
        
        
        dataset_class = getattr(build_dataset_class, dataset_config["dataset"])
        transformation = build_transforms(self.config.get("transformation"))
        batch_size = self.config.get("batch_size")
        dataset = dataset_class(
            image_dir = dataset_config.get("data_dir", None), 
            transformation = transformation, 
            equalization_method_config = self.config.get("equalization_method"))
        
        dataset.inference()
        
        print(f"[INFO] number of instances in dataset: {len(dataset)}")
        
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=self.config.get("num_workers", 10), pin_memory=True)
        
        # Check if TTA is enabled
        use_tta = self.config.get("use_tta", False)
        tta_method = self.config.get("tta_method", "iou_matching")  # "iou_matching" or "flip_or"
        tta_transforms = self.config.get("tta_transforms", None)
        n_largest_components = self.config.get("n_largest_components", 2)  # For flip_or method
        inference_mode = self.config.get("inference_mode", "both")  # "both", "chromosome", or "overlap"

        print(f"[INFO] Inference mode: {inference_mode}")
        if use_tta:
            print(f"[INFO] Test-Time Augmentation (TTA) is ENABLED")
            print(f"[INFO] TTA method: {tta_method}")
            if tta_method == "flip_or":
                print(f"[INFO] Keeping {n_largest_components} largest components per class")
            if tta_transforms and tta_method == "iou_matching":
                print(f"[INFO] TTA transforms: {tta_transforms}")
            elif tta_method == "iou_matching":
                print(f"[INFO] TTA transforms: horizontal_flip (default)")

        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Evaluation Segmentation model")
        with torch.no_grad():
            for _, batches in pbar:
                images, image_paths = batches
                images = images.to(self.config.get("device"))

                if use_tta:
                    # Apply TTA refinement
                    batch_chromosome_preds = []
                    batch_overlap_preds = []

                    for img in images:
                        if tta_method == "flip_or":
                            # Use simpler flip-or TTA method
                            chromosome_pred, overlap_pred = apply_flip_or_tta(
                                model=model,
                                image=img.unsqueeze(0),
                                device=self.config.get("device"),
                                n_largest=n_largest_components,
                                inference_mode=inference_mode
                            )
                        else:
                            # Use IoU matching TTA method (default)
                            chromosome_pred, overlap_pred = apply_tta_refinement(
                                model=model,
                                image=img.unsqueeze(0),
                                device=self.config.get("device"),
                                transformations=tta_transforms,
                                inference_mode=inference_mode
                            )
                        batch_chromosome_preds.append(chromosome_pred)
                        batch_overlap_preds.append(overlap_pred)

                    # Stack only non-None predictions
                    if inference_mode in ["both", "chromosome"]:
                        chromosome_preds = torch.stack(batch_chromosome_preds)
                    else:
                        chromosome_preds = [None] * len(batch_chromosome_preds)

                    if inference_mode in ["both", "overlap"]:
                        overlap_preds = torch.stack(batch_overlap_preds)
                    else:
                        overlap_preds = [None] * len(batch_overlap_preds)
                else:
                    # Standard inference
                    _, outputs = model(images)

                    if inference_mode in ["both", "chromosome"]:
                        chromosome_preds = torch.argmax(outputs[0], dim=1)
                    else:
                        chromosome_preds = [None] * images.size(0)

                    if inference_mode in ["both", "overlap"]:
                        overlap_preds = (torch.sigmoid(outputs[1]) > float(self.config.get("overlap_conf"))).long().squeeze(1)
                    else:
                        overlap_preds = [None] * images.size(0)

                for chromosome_pred, overlap_pred, image_path in zip(chromosome_preds, overlap_preds, image_paths):
                    save_outputs(
                        chromosome_pred = chromosome_pred,
                        overlap_pred = overlap_pred,
                        output_folder = output_folder,
                        image_name = image_path.split("/")[-1],
                        image_path = image_path  # Pass original image path for side-by-side visualization
                    )
        
        return None
    
    def _post_process(self, data):
        return data
    
    def handle(self, data):
        data = self.process(data)
        return super().handle(data)