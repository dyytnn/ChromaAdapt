"""
Instance-Aware Loss Functions for Chromosome Segmentation
Helps model distinguish WHICH chromosome is WHICH (not just chromosome vs background)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from registry import segmentation_loss_register


class InstanceContrastiveLoss(nn.Module):
    """
    Contrastive loss for instance discrimination.
    Forces model to learn discriminative features for each chromosome instance.

    Key idea:
    - Features of pixels from SAME chromosome should be similar (pull together)
    - Features of pixels from DIFFERENT chromosomes should be different (push apart)
    """
    def __init__(self, temperature: float = 0.07, min_instance_pixels: int = 50):
        super().__init__()
        self.temperature = temperature
        self.min_instance_pixels = min_instance_pixels

    def forward(self, features: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, C, H, W) - feature maps from encoder
            targets: (B, H, W) - ground truth instance IDs (0=background, 1-46=chromosomes)

        Returns:
            Contrastive loss scalar
        """
        B, C, H, W = features.shape

        # Normalize features to unit sphere
        features = F.normalize(features, p=2, dim=1)

        # Flatten for easier processing
        features_flat = features.view(B, C, -1).permute(0, 2, 1)  # (B, H*W, C)
        targets_flat = targets.view(B, -1)  # (B, H*W)

        total_loss = 0.0
        num_valid_batches = 0

        for b in range(B):
            feat = features_flat[b]  # (H*W, C)
            target = targets_flat[b]  # (H*W,)

            # Get unique chromosome instances (excluding background=0)
            unique_instances = torch.unique(target)
            unique_instances = unique_instances[unique_instances > 0]

            if len(unique_instances) < 2:
                continue  # Need at least 2 instances for contrastive learning

            batch_loss = 0.0
            num_valid_instances = 0

            for inst_id in unique_instances:
                # Positive samples: pixels belonging to this instance
                pos_mask = (target == inst_id)
                pos_features = feat[pos_mask]  # (N_pos, C)

                if len(pos_features) < self.min_instance_pixels:
                    continue  # Skip very small instances

                # Negative samples: pixels from OTHER chromosome instances (not background)
                neg_mask = (target > 0) & (target != inst_id)
                neg_features = feat[neg_mask]  # (N_neg, C)

                if len(neg_features) == 0:
                    continue

                # Compute instance prototype (center of features)
                prototype = pos_features.mean(dim=0, keepdim=True)  # (1, C)

                # Sample negatives to avoid memory issues (max 1000 samples)
                if len(neg_features) > 1000:
                    indices = torch.randperm(len(neg_features), device=neg_features.device)[:1000]
                    neg_features = neg_features[indices]

                # Compute similarities
                pos_sim = torch.matmul(pos_features, prototype.T).squeeze()  # (N_pos,)
                neg_sim = torch.matmul(neg_features, prototype.T).squeeze()  # (N_neg,)

                # Apply temperature
                pos_sim = pos_sim / self.temperature
                neg_sim = neg_sim / self.temperature

                # InfoNCE loss: log(exp(pos) / (exp(pos) + sum(exp(neg))))
                # Equivalent to: -log(exp(pos) / Z) where Z is partition function
                pos_exp = torch.exp(pos_sim)
                neg_exp_sum = torch.exp(neg_sim).sum()

                # Loss for this instance
                instance_loss = -torch.log(pos_exp / (pos_exp + neg_exp_sum + 1e-8)).mean()

                batch_loss += instance_loss
                num_valid_instances += 1

            if num_valid_instances > 0:
                total_loss += batch_loss / num_valid_instances
                num_valid_batches += 1

        return total_loss / max(num_valid_batches, 1)


class CenterPredictionLoss(nn.Module):
    """
    Auxiliary task: Predict center point of each chromosome instance.
    Generates Gaussian heatmaps around instance centers.

    Benefits:
    - Helps model learn instance-specific spatial features
    - Provides additional supervision signal
    - Useful for post-processing (separate touching chromosomes)
    """
    def __init__(self, sigma: float = 3.0, background_weight: float = 0.1):
        super().__init__()
        self.sigma = sigma
        self.background_weight = background_weight

    def generate_center_heatmap(self, targets: torch.Tensor) -> torch.Tensor:
        """
        Generate Gaussian heatmap around center of each chromosome instance.

        Args:
            targets: (B, H, W) - instance IDs

        Returns:
            heatmaps: (B, 47, H, W) - Gaussian heatmap for each class
        """
        B, H, W = targets.shape
        device = targets.device
        heatmaps = torch.zeros(B, 47, H, W, device=device)

        # Precompute coordinate grids
        y_grid, x_grid = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij'
        )

        for b in range(B):
            target = targets[b]
            unique_instances = torch.unique(target)
            unique_instances = unique_instances[unique_instances > 0]

            for inst_id in unique_instances:
                mask = (target == inst_id)

                # Find center of mass
                y_coords, x_coords = torch.where(mask)
                if len(y_coords) == 0:
                    continue

                cy = y_coords.float().mean()
                cx = x_coords.float().mean()

                # Generate Gaussian heatmap
                gaussian = torch.exp(-((y_grid - cy)**2 + (x_grid - cx)**2) / (2 * self.sigma**2))
                heatmaps[b, inst_id] = gaussian

        return heatmaps

    def forward(self, center_predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            center_predictions: (B, 47, H, W) - predicted center heatmaps
            targets: (B, H, W) - instance IDs

        Returns:
            MSE loss between predicted and target heatmaps
        """
        target_heatmaps = self.generate_center_heatmap(targets)

        # Create weight map: higher weight for foreground
        weight_map = torch.ones_like(target_heatmaps)
        weight_map[:, 0] = self.background_weight  # Lower weight for background class

        # Weighted MSE loss
        loss = ((center_predictions - target_heatmaps) ** 2 * weight_map).mean()

        return loss


class InstanceEmbeddingLoss(nn.Module):
    """
    Learn embedding space where same instance has similar embeddings.
    Based on "Semantic Instance Segmentation with a Discriminative Loss Function" (De Brabandere et al., 2017)

    Loss components:
    1. Variance loss: Pull embeddings of same instance to cluster center
    2. Distance loss: Push different instance cluster centers apart
    3. Regularization: Prevent embeddings from becoming too large
    """
    def __init__(
        self,
        delta_v: float = 0.5,  # Variance margin
        delta_d: float = 1.5,  # Distance margin
        alpha: float = 1.0,    # Variance weight
        beta: float = 1.0,     # Distance weight
        gamma: float = 0.001,  # Regularization weight
        min_instance_pixels: int = 50
    ):
        super().__init__()
        self.delta_v = delta_v
        self.delta_d = delta_d
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.min_instance_pixels = min_instance_pixels

    def forward(self, embeddings: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            embeddings: (B, D, H, W) - learned embeddings (D = embedding dimension)
            targets: (B, H, W) - instance IDs

        Returns:
            Total loss and loss components dictionary
        """
        B, D, H, W = embeddings.shape

        total_var_loss = 0.0
        total_dist_loss = 0.0
        total_reg_loss = 0.0
        num_valid_batches = 0

        for b in range(B):
            emb = embeddings[b].view(D, -1).T  # (H*W, D)
            target = targets[b].view(-1)  # (H*W,)

            unique_instances = torch.unique(target)
            unique_instances = unique_instances[unique_instances > 0]

            if len(unique_instances) == 0:
                continue

            # Filter out very small instances
            valid_instances = []
            valid_centers = []

            for inst_id in unique_instances:
                mask = (target == inst_id)
                if mask.sum() < self.min_instance_pixels:
                    continue

                inst_emb = emb[mask]
                center = inst_emb.mean(dim=0)

                valid_instances.append(inst_id)
                valid_centers.append(center)

            if len(valid_instances) == 0:
                continue

            centers = torch.stack(valid_centers)  # (N_inst, D)

            # 1. Variance loss: Pull embeddings to their cluster center
            var_loss = 0.0
            for i, inst_id in enumerate(valid_instances):
                mask = (target == inst_id)
                inst_emb = emb[mask]

                # Distance to center
                dist = torch.norm(inst_emb - centers[i], dim=1)

                # Hinge loss: max(0, dist - delta_v)^2
                var_loss += torch.clamp(dist - self.delta_v, min=0).pow(2).mean()

            var_loss /= len(valid_instances)

            # 2. Distance loss: Push different cluster centers apart
            dist_loss = 0.0
            num_pairs = 0

            for i in range(len(centers)):
                for j in range(i + 1, len(centers)):
                    dist = torch.norm(centers[i] - centers[j])

                    # Hinge loss: max(0, 2*delta_d - dist)^2
                    dist_loss += torch.clamp(2 * self.delta_d - dist, min=0).pow(2)
                    num_pairs += 1

            if num_pairs > 0:
                dist_loss /= num_pairs

            # 3. Regularization: Prevent embeddings from growing too large
            reg_loss = torch.norm(centers, dim=1).mean()

            total_var_loss += var_loss
            total_dist_loss += dist_loss
            total_reg_loss += reg_loss
            num_valid_batches += 1

        if num_valid_batches == 0:
            return torch.tensor(0.0, device=embeddings.device), {}

        # Average over batch
        var_loss_avg = total_var_loss / num_valid_batches
        dist_loss_avg = total_dist_loss / num_valid_batches
        reg_loss_avg = total_reg_loss / num_valid_batches

        # Combine losses
        total_loss = (
            self.alpha * var_loss_avg +
            self.beta * dist_loss_avg +
            self.gamma * reg_loss_avg
        )

        loss_dict = {
            'loss_embedding_variance': var_loss_avg,
            'loss_embedding_distance': dist_loss_avg,
            'loss_embedding_regularization': reg_loss_avg
        }

        return total_loss, loss_dict


@segmentation_loss_register.register("InstanceAwareSegmentationLoss")
class InstanceAwareSegmentationLoss(nn.Module):
    """
    Combined loss for instance-aware chromosome segmentation.

    Components:
    1. Segmentation loss (CrossEntropy): Pixel-wise classification
    2. Contrastive loss: Learn discriminative features per instance
    3. Center prediction loss: Auxiliary task for instance localization

    This helps the model learn to distinguish WHICH chromosome is WHICH.
    """
    def __init__(
        self,
        seg_weight: float = 0.5,
        contrastive_weight: float = 0.3,
        center_weight: float = 0.2,
        temperature: float = 0.07,
        center_sigma: float = 3.0,
        ignore_index: int = -100
    ):
        super().__init__()
        self.seg_weight = seg_weight
        self.contrastive_weight = contrastive_weight
        self.center_weight = center_weight

        # Loss functions
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.contrastive_loss = InstanceContrastiveLoss(temperature=temperature)
        self.center_loss = CenterPredictionLoss(sigma=center_sigma)

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        features: Optional[torch.Tensor] = None,
        center_preds: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            predictions: (B, 47, H, W) - segmentation logits
            targets: (B, H, W) - instance IDs (0-46)
            features: (B, C, H, W) - encoder features for contrastive loss
            center_preds: (B, 47, H, W) - center heatmap predictions

        Returns:
            Dictionary with total_loss and component losses
        """
        # 1. Segmentation loss (always computed)
        seg_loss = self.ce_loss(predictions, targets.long())

        total_loss = self.seg_weight * seg_loss
        loss_dict = {
            'total_loss': total_loss,
            'loss_segmentation': seg_loss
        }

        # 2. Contrastive loss (if features provided)
        if features is not None and self.contrastive_weight > 0:
            contrastive_loss = self.contrastive_loss(features, targets)
            total_loss = total_loss + self.contrastive_weight * contrastive_loss
            loss_dict['loss_contrastive'] = contrastive_loss

        # 3. Center prediction loss (if center predictions provided)
        if center_preds is not None and self.center_weight > 0:
            center_loss = self.center_loss(center_preds, targets)
            total_loss = total_loss + self.center_weight * center_loss
            loss_dict['loss_center'] = center_loss

        loss_dict['total_loss'] = total_loss

        return loss_dict


@segmentation_loss_register.register("InstanceEmbeddingSegmentationLoss")
class InstanceEmbeddingSegmentationLoss(nn.Module):
    """
    Segmentation loss + Instance embedding loss.
    Alternative to contrastive approach.
    """
    def __init__(
        self,
        seg_weight: float = 0.6,
        embedding_weight: float = 0.4,
        delta_v: float = 0.5,
        delta_d: float = 1.5,
        ignore_index: int = -100
    ):
        super().__init__()
        self.seg_weight = seg_weight
        self.embedding_weight = embedding_weight

        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.embedding_loss = InstanceEmbeddingLoss(delta_v=delta_v, delta_d=delta_d)

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        embeddings: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            predictions: (B, 47, H, W) - segmentation logits
            targets: (B, H, W) - instance IDs
            embeddings: (B, D, H, W) - learned embeddings

        Returns:
            Dictionary with losses
        """
        seg_loss = self.ce_loss(predictions, targets.long())

        total_loss = self.seg_weight * seg_loss
        loss_dict = {
            'total_loss': total_loss,
            'loss_segmentation': seg_loss
        }

        if embeddings is not None and self.embedding_weight > 0:
            emb_loss, emb_components = self.embedding_loss(embeddings, targets)
            total_loss = total_loss + self.embedding_weight * emb_loss
            loss_dict['loss_embedding'] = emb_loss
            loss_dict.update(emb_components)

        loss_dict['total_loss'] = total_loss

        return loss_dict


# Export all loss functions
__all__ = [
    'InstanceContrastiveLoss',
    'CenterPredictionLoss',
    'InstanceEmbeddingLoss',
    'InstanceAwareSegmentationLoss',
    'InstanceEmbeddingSegmentationLoss'
]