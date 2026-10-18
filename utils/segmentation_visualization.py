"""
Segmentation Visualization Utilities for Multi-class Chromosome Segmentation
Supports visualization of 47 classes (0=background, 1-46=chromosomes)
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch
from matplotlib.patches import Patch
from pathlib import Path
from typing import Union, Optional, Tuple, List


def generate_chromosome_colormap(num_classes: int = 47) -> np.ndarray:
    """
    Generate a distinctive color palette for chromosome segmentation.

    Args:
        num_classes: Number of classes (default 47: 1 background + 46 chromosomes)

    Returns:
        colors: RGB color array of shape (num_classes, 3) with values in [0, 1]
    """
    colors = np.zeros((num_classes, 3))

    # Background: Black
    colors[0] = [0, 0, 0]

    # Generate distinct colors for chromosomes 1-46
    # Use HSV color space for better perceptual distinction
    for i in range(1, num_classes):
        # Distribute hues evenly across the color wheel
        hue = (i - 1) / (num_classes - 1)

        # Alternate saturation and value for better distinction
        saturation = 0.7 + 0.3 * ((i % 2))  # 0.7 or 1.0
        value = 0.7 + 0.3 * ((i % 3) / 2)   # 0.7, 0.85, or 1.0

        # Convert HSV to RGB
        hsv = np.array([hue, saturation, value])
        rgb = mcolors.hsv_to_rgb(hsv)
        colors[i] = rgb

    return colors


def generate_tab_colormap(num_classes: int = 47) -> np.ndarray:
    """
    Alternative colormap using matplotlib's color palettes for better distinction.
    Combines multiple palettes for maximum variety.

    Args:
        num_classes: Number of classes (default 47)

    Returns:
        colors: RGB color array of shape (num_classes, 3)
    """
    colors = np.zeros((num_classes, 3))

    # Background: Black
    colors[0] = [0, 0, 0]

    # Combine multiple matplotlib colormaps for variety
    tab20 = plt.cm.tab20(np.linspace(0, 1, 20))[:, :3]
    tab20b = plt.cm.tab20b(np.linspace(0, 1, 20))[:, :3]
    tab20c = plt.cm.tab20c(np.linspace(0, 1, 20))[:, :3]

    # Interleave colors from different palettes for maximum distinction
    all_colors = []
    for i in range(20):
        all_colors.append(tab20[i])
        if i < 20:
            all_colors.append(tab20b[i])
        if i < 20:
            all_colors.append(tab20c[i])

    # Assign colors to classes (cycling if needed)
    for i in range(1, num_classes):
        colors[i] = all_colors[(i - 1) % len(all_colors)]

    return colors


def apply_colormap_to_segmentation(
    segmentation: Union[torch.Tensor, np.ndarray],
    colormap: np.ndarray
) -> np.ndarray:
    """
    Apply color palette to segmentation map.

    Args:
        segmentation: Segmentation map of shape (H, W) with class indices
        colormap: Color palette of shape (num_classes, 3)

    Returns:
        colored: RGB image of shape (H, W, 3)
    """
    # Convert to numpy if tensor
    if isinstance(segmentation, torch.Tensor):
        segmentation = segmentation.cpu().numpy()

    # Ensure integer type
    segmentation = segmentation.astype(np.int32)

    # Create RGB image
    h, w = segmentation.shape
    colored = np.zeros((h, w, 3), dtype=np.float32)

    # Map each class to its color
    for class_id in np.unique(segmentation):
        if 0 <= class_id < len(colormap):
            mask = segmentation == class_id
            colored[mask] = colormap[class_id]

    return colored


def visualize_segmentation_batch(
    images: Union[torch.Tensor, np.ndarray],
    predictions: Union[torch.Tensor, np.ndarray],
    labels: Optional[Union[torch.Tensor, np.ndarray]] = None,
    num_samples: int = 8,
    colormap: Optional[np.ndarray] = None,
    figsize: Tuple[int, int] = (20, 10),
    save_path: Optional[str] = None,
    show_legend: bool = True,
    title: str = "Segmentation Results"
) -> plt.Figure:
    """
    Visualize a batch of segmentation results.

    Args:
        images: Input images (B, C, H, W) or (B, H, W, C)
        predictions: Predicted segmentation (B, H, W) with class indices
        labels: Ground truth segmentation (B, H, W) [optional]
        num_samples: Number of samples to display
        colormap: Color palette (num_classes, 3) [optional, will generate if None]
        figsize: Figure size
        save_path: Path to save visualization [optional]
        show_legend: Whether to show color legend
        title: Plot title

    Returns:
        fig: Matplotlib figure object
    """
    # Convert to numpy
    if isinstance(images, torch.Tensor):
        images = images.cpu().numpy()
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if labels is not None and isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    # Normalize images to [0, 1]
    if images.max() > 1.0:
        images = images / 255.0

    # Handle channel dimension: (B, C, H, W) -> (B, H, W, C)
    if images.ndim == 4 and images.shape[1] in [1, 3]:
        images = np.transpose(images, (0, 2, 3, 1))

    # Convert grayscale to RGB
    if images.shape[-1] == 1:
        images = np.repeat(images, 3, axis=-1)

    # Generate colormap if not provided
    if colormap is None:
        num_classes = int(predictions.max()) + 1
        colormap = generate_tab_colormap(num_classes)

    # Determine layout
    batch_size = min(num_samples, len(images))
    has_labels = labels is not None
    num_cols = 3 if has_labels else 2

    fig, axes = plt.subplots(batch_size, num_cols, figsize=figsize)
    if batch_size == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(title, fontsize=16, fontweight='bold')

    for i in range(batch_size):
        # Original image
        axes[i, 0].imshow(images[i])
        axes[i, 0].set_title(f"Sample {i+1}: Input")
        axes[i, 0].axis('off')

        # Prediction
        pred_colored = apply_colormap_to_segmentation(predictions[i], colormap)
        axes[i, 1].imshow(pred_colored)
        unique_preds = np.unique(predictions[i])
        axes[i, 1].set_title(f"Prediction ({len(unique_preds)} classes)")
        axes[i, 1].axis('off')

        # Ground truth (if available)
        if has_labels:
            label_colored = apply_colormap_to_segmentation(labels[i], colormap)
            axes[i, 2].imshow(label_colored)
            unique_labels = np.unique(labels[i])
            axes[i, 2].set_title(f"Ground Truth ({len(unique_labels)} classes)")
            axes[i, 2].axis('off')

    # Add legend showing class colors
    if show_legend:
        unique_classes = np.unique(predictions)
        legend_elements = []

        # Show up to 20 classes in legend to avoid clutter
        display_classes = unique_classes[:20] if len(unique_classes) > 20 else unique_classes

        for class_id in display_classes:
            if class_id == 0:
                label = "Background"
            else:
                label = f"Chr {class_id}"

            color = colormap[class_id] if class_id < len(colormap) else [0.5, 0.5, 0.5]
            legend_elements.append(Patch(facecolor=color, label=label))

        # Add legend outside the plot
        if len(unique_classes) > 20:
            legend_elements.append(Patch(facecolor='white', label=f'... +{len(unique_classes)-20} more'))

        fig.legend(
            handles=legend_elements,
            loc='center left',
            bbox_to_anchor=(1.0, 0.5),
            ncol=1,
            fontsize=10
        )

    plt.tight_layout(rect=[0, 0, 0.95, 0.96] if show_legend else [0, 0, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")

    return fig


def visualize_single_segmentation(
    image: Union[torch.Tensor, np.ndarray],
    prediction: Union[torch.Tensor, np.ndarray],
    label: Optional[Union[torch.Tensor, np.ndarray]] = None,
    colormap: Optional[np.ndarray] = None,
    figsize: Tuple[int, int] = (15, 5),
    save_path: Optional[str] = None,
    title: str = "Segmentation Visualization"
) -> plt.Figure:
    """
    Visualize a single segmentation result with detailed class information.

    Args:
        image: Input image (C, H, W) or (H, W, C)
        prediction: Predicted segmentation (H, W)
        label: Ground truth segmentation (H, W) [optional]
        colormap: Color palette
        figsize: Figure size
        save_path: Path to save
        title: Plot title

    Returns:
        fig: Matplotlib figure
    """
    # Convert to numpy
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.cpu().numpy()
    if label is not None and isinstance(label, torch.Tensor):
        label = label.cpu().numpy()

    # Normalize image
    if image.max() > 1.0:
        image = image / 255.0

    # Handle channel dimension: (C, H, W) -> (H, W, C)
    if image.ndim == 3 and image.shape[0] in [1, 3]:
        image = np.transpose(image, (1, 2, 0))

    # Convert grayscale to RGB
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)

    # Generate colormap if not provided
    if colormap is None:
        num_classes = int(prediction.max()) + 1
        colormap = generate_tab_colormap(num_classes)

    # Create subplots
    has_label = label is not None
    num_cols = 4 if has_label else 3

    fig, axes = plt.subplots(1, num_cols, figsize=figsize)
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Input Image")
    axes[0].axis('off')

    # Prediction colored
    pred_colored = apply_colormap_to_segmentation(prediction, colormap)
    axes[1].imshow(pred_colored)
    unique_preds = np.unique(prediction)
    axes[1].set_title(f"Prediction\n({len(unique_preds)} classes detected)")
    axes[1].axis('off')

    # Overlay
    overlay = 0.6 * image + 0.4 * pred_colored
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis('off')

    # Ground truth (if available)
    if has_label:
        label_colored = apply_colormap_to_segmentation(label, colormap)
        axes[3].imshow(label_colored)
        unique_labels = np.unique(label)
        axes[3].set_title(f"Ground Truth\n({len(unique_labels)} classes)")
        axes[3].axis('off')

    # Add detailed class information
    info_text = f"Detected classes: {sorted(unique_preds.tolist())}"
    fig.text(0.5, 0.02, info_text, ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout(rect=[0, 0.05, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")

    return fig


def create_class_distribution_plot(
    predictions: Union[torch.Tensor, np.ndarray],
    labels: Optional[Union[torch.Tensor, np.ndarray]] = None,
    num_classes: int = 47,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a histogram showing the distribution of predicted classes.

    Args:
        predictions: Predicted segmentation maps (B, H, W)
        labels: Ground truth segmentation maps (B, H, W) [optional]
        num_classes: Total number of classes
        figsize: Figure size
        save_path: Path to save plot

    Returns:
        fig: Matplotlib figure
    """
    # Convert to numpy
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if labels is not None and isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    # Flatten predictions
    pred_flat = predictions.flatten()

    # Count pixels per class
    pred_counts = np.bincount(pred_flat, minlength=num_classes)
    pred_percentages = 100 * pred_counts / pred_flat.size

    # Create figure
    fig, axes = plt.subplots(1, 2 if labels is not None else 1, figsize=figsize)
    if labels is None:
        axes = [axes]

    # Plot predictions
    axes[0].bar(range(num_classes), pred_percentages, color='steelblue', alpha=0.7)
    axes[0].set_xlabel('Class ID')
    axes[0].set_ylabel('Percentage (%)')
    axes[0].set_title('Predicted Class Distribution')
    axes[0].grid(axis='y', alpha=0.3)

    # Mark present classes
    present_classes = np.where(pred_counts > 0)[0]
    axes[0].set_xticks(present_classes)
    axes[0].set_xticklabels(present_classes, rotation=45, ha='right')

    # Plot ground truth if available
    if labels is not None:
        label_flat = labels.flatten()
        label_counts = np.bincount(label_flat, minlength=num_classes)
        label_percentages = 100 * label_counts / label_flat.size

        axes[1].bar(range(num_classes), label_percentages, color='coral', alpha=0.7)
        axes[1].set_xlabel('Class ID')
        axes[1].set_ylabel('Percentage (%)')
        axes[1].set_title('Ground Truth Class Distribution')
        axes[1].grid(axis='y', alpha=0.3)

        present_labels = np.where(label_counts > 0)[0]
        axes[1].set_xticks(present_labels)
        axes[1].set_xticklabels(present_labels, rotation=45, ha='right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Distribution plot saved to: {save_path}")

    return fig


def save_colormap_legend(
    colormap: np.ndarray,
    save_path: str,
    num_classes: Optional[int] = None,
    figsize: Tuple[int, int] = (10, 15)
) -> None:
    """
    Save a standalone legend showing all class colors.

    Args:
        colormap: Color palette (num_classes, 3)
        save_path: Path to save legend image
        num_classes: Number of classes to display (default: all)
        figsize: Figure size
    """
    if num_classes is None:
        num_classes = len(colormap)

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')

    # Create legend patches
    legend_elements = []
    for i in range(num_classes):
        if i == 0:
            label = f"Class {i}: Background"
        elif i <= 46:
            label = f"Class {i}: Chromosome {i}"
        else:
            label = f"Class {i}"

        color = colormap[i] if i < len(colormap) else [0.5, 0.5, 0.5]
        legend_elements.append(Patch(facecolor=color, label=label))

    # Display legend
    ax.legend(
        handles=legend_elements,
        loc='center',
        ncol=3,
        fontsize=10,
        title="Segmentation Color Map",
        title_fontsize=12
    )

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Colormap legend saved to: {save_path}")
    plt.close()


# Example usage and testing
if __name__ == "__main__":
    print("Segmentation Visualization Utilities")
    print("=" * 60)

    # Generate colormap
    colormap = generate_tab_colormap(47)
    print(f"Generated colormap with {len(colormap)} colors")

    # Create dummy data for testing
    dummy_image = torch.rand(4, 3, 512, 512)  # Batch of 4 images
    dummy_preds = torch.randint(0, 46, (4, 512, 512))  # Random predictions

    print(f"\nDummy data shapes:")
    print(f"  Images: {dummy_image.shape}")
    print(f"  Predictions: {dummy_preds.shape}")
    print(f"  Unique classes: {torch.unique(dummy_preds).tolist()}")

    # Test visualization
    fig = visualize_segmentation_batch(
        images=dummy_image,
        predictions=dummy_preds,
        num_samples=2,
        colormap=colormap,
        title="Test Segmentation Visualization"
    )

    print("\n✓ Visualization utilities ready to use!")