"""
DANN Combined Dataset
Balances synthetic (source) and real (target) data for domain adaptation.
Handles imbalanced dataset sizes by oversampling the smaller dataset.
"""

import numpy as np
from torch.utils.data import Dataset


class DANNCombinedDataset(Dataset):
    """
    Combines source (synthetic) and target (real) datasets for DANN training.

    Automatically handles imbalanced dataset sizes by:
    - Using the larger dataset size as the epoch length
    - Cycling through the smaller dataset with random sampling

    This ensures all data from both domains is used efficiently.
    """

    def __init__(
        self,
        source_dataset: Dataset,
        target_dataset: Dataset,
        balance_strategy: str = "oversample_minority"
    ):
        """
        Args:
            source_dataset: Synthetic/source domain dataset
            target_dataset: Real/target domain dataset
            balance_strategy: How to handle imbalance
                - "oversample_minority": Repeat smaller dataset (default)
                - "match_majority": Make both equal to larger size
                - "undersample_majority": Use smaller dataset size
        """
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
        self.balance_strategy = balance_strategy

        self.source_size = len(source_dataset)
        self.target_size = len(target_dataset)

        # Determine epoch length based on strategy
        if balance_strategy == "oversample_minority":
            # Use larger dataset size - oversample the smaller one
            self.epoch_length = max(self.source_size, self.target_size)
        elif balance_strategy == "match_majority":
            # Make both equal to larger size
            self.epoch_length = max(self.source_size, self.target_size)
        elif balance_strategy == "undersample_majority":
            # Use smaller dataset size
            self.epoch_length = min(self.source_size, self.target_size)
        else:
            raise ValueError(f"Unknown balance_strategy: {balance_strategy}")

        print(f"[INFO] DANN Combined Dataset:")
        print(f"  Source (synthetic) size: {self.source_size}")
        print(f"  Target (real) size: {self.target_size}")
        print(f"  Strategy: {balance_strategy}")
        print(f"  Epoch length: {self.epoch_length}")
        print(f"  Imbalance ratio: {max(self.source_size, self.target_size) / min(self.source_size, self.target_size):.2f}x")

    def __len__(self):
        return self.epoch_length

    def __getitem__(self, index):
        """
        Get a paired sample from source and target datasets.
        Handles index wrapping for the smaller dataset.
        """
        # Source sample
        if self.source_size <= self.target_size:
            # Source is smaller - cycle through with random sampling
            source_idx = np.random.randint(0, self.source_size)
        else:
            # Source is larger - use direct indexing
            source_idx = index % self.source_size

        # Target sample
        if self.target_size <= self.source_size:
            # Target is smaller - cycle through with random sampling
            target_idx = np.random.randint(0, self.target_size)
        else:
            # Target is larger - use direct indexing
            target_idx = index % self.target_size

        # Get samples
        source_data = self.source_dataset[source_idx]
        target_data = self.target_dataset[target_idx]

        # Return in format expected by DANN
        return {
            'source': source_data,
            'target': target_data
        }


def dann_collate_fn(batch):
    """
    Collate function for DANN training with multi-label segmentation.
    Handles chromosome overlap-touching segmentation with two label images.

    Args:
        batch: List of dicts with 'source' and 'target' keys
            - source: (image, (mask1, mask2), domain_label) - synthetic with 2 masks
            - target: (image, domain_label) - real without labels

    Returns:
        Tuple of (source_batch, target_batch) where:
        - source_batch: (images [B,C,H,W], labels [tuple of tensors], domain_labels [B])
        - target_batch: (images [B,C,H,W], domain_labels [B])
    """
    import torch

    source_images = []
    source_labels_list = []  # Will contain tuples of (mask1, mask2)
    source_domain_labels = []
    target_images = []
    target_domain_labels = []

    for sample in batch:
        # Source data: (image, (chromosome_mask, overlap_mask), domain_label)
        source_data = sample['source']
        if len(source_data) == 3:
            src_img, src_labels, src_domain_label = source_data
        else:
            raise ValueError(f"Expected source data to have 3 elements, got {len(source_data)}")

        source_images.append(src_img)
        source_labels_list.append(src_labels)  # This is a tuple of (mask1, mask2)
        source_domain_labels.append(src_domain_label)

        # Target data: (image, domain_label)
        target_data = sample['target']
        if len(target_data) == 2:
            tgt_img, tgt_domain_label = target_data
        else:
            raise ValueError(f"Expected target data to have 2 elements, got {len(target_data)}")

        target_images.append(tgt_img)
        target_domain_labels.append(tgt_domain_label)

    # Stack images
    source_images = torch.stack(source_images)
    target_images = torch.stack(target_images)

    # Stack domain labels
    source_domain_labels = torch.stack(source_domain_labels)
    target_domain_labels = torch.stack(target_domain_labels)

    # Handle multi-label stacking
    # source_labels_list is a list of tuples: [(mask1, mask2), (mask1, mask2), ...]
    # We need to stack each mask type separately
    chromosome_masks = []
    overlap_masks = []
    for mask_tuple in source_labels_list:
        if isinstance(mask_tuple, (tuple, list)) and len(mask_tuple) == 2:
            chromosome_masks.append(mask_tuple[0])
            overlap_masks.append(mask_tuple[1])
        else:
            raise ValueError(f"Expected label tuple to have 2 elements, got {len(mask_tuple)}")

    # Stack masks
    chromosome_masks = torch.stack(chromosome_masks)
    overlap_masks = torch.stack(overlap_masks)

    # Return as tuple: labels as tuple of stacked masks
    source_labels = (chromosome_masks, overlap_masks)

    # Return as tuple for DANN training loop
    source_batch = (source_images, source_labels, source_domain_labels)
    target_batch = (target_images, target_domain_labels)

    return source_batch, target_batch