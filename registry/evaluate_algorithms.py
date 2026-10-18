import albumentations as A
import numpy as np
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from tqdm import tqdm

from abs_interface import AbsAlgorithmProcessHandler
from configs.config import configurable
from data import dataset as build_dataset_class
from data.transformation import build_transforms
from registry import evaluate_segmentation_handler_register
from utils.segmentation_visualization import (
    generate_tab_colormap,
    visualize_segmentation_batch
)


class SegmentationMetrics:
    def __init__(self, num_classes, ignore_background=True):
        """
        Initialize the metrics computation for multi-class segmentation.

        Args:
        num_classes (int): Number of classes in the segmentation task.
        ignore_background (bool): Whether to ignore background class (class 0) in metrics. Default: True.
        """
        self.num_classes = num_classes
        self.ignore_background = ignore_background
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def _compute_confusion_matrix(self, pred, target):
        """
        Compute confusion matrix for one batch.
        
        Args:
        pred (Tensor): Predicted segmentation map (batch_size, H, W).
        target (Tensor): Ground truth segmentation map (batch_size, H, W).
        """
        mask = (target >= 0) & (target < self.num_classes)  
        label = self.num_classes * target[mask] + pred[mask]
        count = np.bincount(label, minlength=self.num_classes**2)
        return count.reshape(self.num_classes, self.num_classes)
    
    def update(self, preds, targets):
        """
        Update confusion matrix with new batch predictions.
        
        Args:
        preds (Tensor): Model predictions of shape (batch_size, H, W).
        targets (Tensor): Ground truth of shape (batch_size, H, W).
        """
        preds = preds.cpu().numpy()
        targets = targets.cpu().numpy()
        self.confusion_matrix += self._compute_confusion_matrix(preds, targets)

    def get_pixel_accuracy(self):
        """Compute Pixel Accuracy."""
        if self.ignore_background:
            # Exclude background class (class 0) from pixel accuracy calculation
            confusion_no_bg = self.confusion_matrix[1:, 1:]
            acc = np.diag(confusion_no_bg).sum() / confusion_no_bg.sum()
        else:
            # Include all classes
            acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return acc

    def get_classwise_iou(self):
        """Compute IoU for each class."""
        intersection = np.diag(self.confusion_matrix)
        union = (
            self.confusion_matrix.sum(axis=1) + 
            self.confusion_matrix.sum(axis=0) - 
            intersection
        )
        iou = intersection / (union + 1e-6)
        return iou  # Returns an array with IoU per class

    def get_mean_iou(self):
        """Compute mean IoU across all classes."""
        iou = self.get_classwise_iou()
        if self.ignore_background:
            # Exclude background class (class 0) from mean calculation
            return np.mean(iou[1:])
        else:
            # Include all classes
            return np.mean(iou)

    def get_classwise_f1(self):
        """Compute F1-Score (Dice Coefficient) for each class."""
        intersection = np.diag(self.confusion_matrix)
        union = (
            self.confusion_matrix.sum(axis=1) + 
            self.confusion_matrix.sum(axis=0)
        )
        f1_score = (2 * intersection) / (union + 1e-6)
        return f1_score  # Returns an array with F1-Score per class

    def get_mean_f1(self):
        """Compute mean F1-Score (Dice Score) across all classes."""
        f1 = self.get_classwise_f1()
        if self.ignore_background:
            # Exclude background class (class 0) from mean calculation
            return np.mean(f1[1:])
        else:
            # Include all classes
            return np.mean(f1)

    def compute_all_metrics(self):
        """
        Compute all metrics including Pixel Accuracy, Mean IoU, and Mean F1-Score.
        """
        metrics = {
            "Pixel Accuracy": self.get_pixel_accuracy(),
            "Mean IoU": self.get_mean_iou(),
            "Mean F1-Score": self.get_mean_f1(),
        }
        return metrics


class BinarySegmentationMetrics:
    def __init__(self):
        """Initialize the binary segmentation metrics computation."""
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.tp = 0  # True Positives
        self.fp = 0  # False Positives
        self.fn = 0  # False Negatives
        self.tn = 0  # True Negatives

    def update(self, preds, targets):
        """
        Update confusion values with new batch predictions.

        Args:
        preds (Tensor): Predicted segmentation map (batch_size, H, W), values should be {0, 1}.
        targets (Tensor): Ground truth segmentation map (batch_size, H, W), values should be {0, 1}.
        """
        preds = preds.cpu().numpy().flatten()
        targets = targets.cpu().numpy().flatten()

        # Skip if ground truth is empty (all background)
        # to avoid 0/0 in IoU and Dice when prediction is also correct (all background)
        if np.sum(targets == 1) == 0:
            return

        self.tp += np.sum((preds == 1) & (targets == 1))
        self.fp += np.sum((preds == 1) & (targets == 0))
        self.fn += np.sum((preds == 0) & (targets == 1))
        self.tn += np.sum((preds == 0) & (targets == 0))

    def get_pixel_accuracy(self):
        """Compute Pixel Accuracy: (TP + TN) / (Total Pixels)"""
        total = self.tp + self.fp + self.fn + self.tn
        return (self.tp + self.tn) / total if total != 0 else 0.0

    def get_iou(self):
        """Compute Intersection over Union (IoU): TP / (TP + FP + FN)"""
        denominator = self.tp + self.fp + self.fn
        return self.tp / denominator if denominator != 0 else 0.0

    def get_dice_score(self):
        """Compute Dice Score (F1-Score): 2TP / (2TP + FP + FN)"""
        denominator = (2 * self.tp) + self.fp + self.fn
        return (2 * self.tp) / denominator if denominator != 0 else 0.0

    def compute_all_metrics(self):
        """Compute all metrics and return a dictionary."""
        return {
            "Pixel Accuracy": self.get_pixel_accuracy(),
            "IoU": self.get_iou(),
            "Dice Score (F1)": self.get_dice_score(),
        }


@evaluate_segmentation_handler_register.register("EvaluateSegmentation")
class EvaluateSegmentation(AbsAlgorithmProcessHandler):
    def __init__(self, config: dict, **kwargs) -> None:
        # Accept and ignore extra parameters like is_evaluation
        self.config = config
        ignore_bg = self.config.get("ignore_background", True)
        self.chromosome_evaluator = SegmentationMetrics(num_classes=3, ignore_background=ignore_bg)
        self.overlap_evaluator = BinarySegmentationMetrics()
        
    def _pre_process(self, data):
      # Validate inputs
      model = data.get("model", None)
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
      return model.to(device)
    
    def _main_process(self, data):
        model = data
        model.eval()
        dataset_class = getattr(build_dataset_class, self.config.get("dataset"))
        transformation = build_transforms(self.config.get("transformation"))
        batch_size = self.config.get("batch_size")
        dataset = dataset_class(image_dir=self.config.get("data_dir", None), transformation=transformation, equalization_method_config = self.config.get("equalization_method"))
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=self.config.get("num_workers", 10), pin_memory=True)

        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Evaluation Segmentation model")
        with torch.no_grad():
            for _, batches in pbar:
                images, labels = batches
                images = images.to(self.config.get("device"))
                labels = [lbl.to(self.config.get("device")) for lbl in labels]

                if self.config.get("isPointRend"):
                    outputs = model(images)
                else:
                    _, outputs = model(images)
                # outputs = model(images)
                chromosome_preds = torch.argmax(outputs[0], dim=1)
                overlap_preds = (torch.sigmoid(outputs[1]) > 0.85).long().squeeze(1)

                self.chromosome_evaluator.update(chromosome_preds, labels[0])
                self.overlap_evaluator.update(overlap_preds, labels[1].squeeze(1))

        chromosome_results = self.chromosome_evaluator.compute_all_metrics()
        overlap_results = self.overlap_evaluator.compute_all_metrics()

        return chromosome_results, overlap_results

    def _post_process(self, data):
        chromosome_results, overlap_results = data
        return chromosome_results, overlap_results
    
    def process(self, data):
        data = self._pre_process(data)
        data = self._main_process(data)
        data = self._post_process(data)
        return data
    
    def handle(self, data):
        chromosome_results, overlap_results = self.process(data)
        data.update(
            {
                "mochromosome_resultsdel": chromosome_results,
                "overlap_results": overlap_results
            }
        )
        return super().handle(data)

@evaluate_segmentation_handler_register.register("EvaluateMetaphaseSegmentation")
class EvaluateMetaphaseSegmentation(AbsAlgorithmProcessHandler):
    """
    Evaluation handler for metaphase chromosome segmentation with 47 classes
    (1 background + 46 chromosome classes).
    """
    def __init__(self, config: dict, **kwargs) -> None:
        self.config = config
        # Initialize metrics for 47-class segmentation
        ignore_bg = self.config.get("ignore_background", True)
        self.evaluator = SegmentationMetrics(num_classes=47, ignore_background=ignore_bg)

        # Generate colormap for visualization
        self.colormap = generate_tab_colormap(num_classes=47)

        # Visualization settings
        self.enable_viz = self.config.get("enable_visualization", False)
        self.viz_freq = self.config.get("visualization_frequency", 10)  # Visualize every N batches
        self.viz_save_dir = self.config.get("visualization_save_dir", "visualizations")

    def _pre_process(self, data):
        """Load model checkpoint and prepare for evaluation."""
        # Validate inputs
        model = data.get("model", None)
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
        return model.to(device)

    def _main_process(self, data):
        """Run evaluation on the dataset."""
        model = data
        model.eval()

        # Build dataset
        dataset_class = getattr(build_dataset_class, self.config.get("dataset"))
        transformation = build_transforms(self.config.get("transformation"))
        batch_size = self.config.get("batch_size")
        dataset = dataset_class(
            image_dir=self.config.get("data_dir", None),
            transformation=transformation,
            equalization_method_config=self.config.get("equalization_method")
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=self.config.get("num_workers", 10),
            pin_memory=True
        )

        # Create visualization directory if enabled
        if self.enable_viz:
            import os
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend for saving
            import matplotlib.pyplot as plt

            os.makedirs(self.viz_save_dir, exist_ok=True)
            print(f"Visualization enabled: saving to {self.viz_save_dir}/")

        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Evaluating Metaphase Segmentation")
        with torch.no_grad():
            for batch_idx, batches in pbar:
                images, labels = batches
                images = images.to(self.config.get("device"))

                # Handle labels - for single-head model, labels is a single tensor
                if isinstance(labels, (list, tuple)):
                    labels = labels[0].to(self.config.get("device"))
                else:
                    labels = labels.to(self.config.get("device"))

                # Get model predictions
                # Model returns (features, segmentation_output) for metaphase
                _, outputs = model(images)

                # Get class predictions (argmax over class dimension)
                preds = torch.argmax(outputs, dim=1)

                # Visualize predictions (every N batches)
                if self.enable_viz and batch_idx % self.viz_freq == 0:
                    save_path = f"{self.viz_save_dir}/batch_{batch_idx:04d}.png"
                    fig = visualize_segmentation_batch(
                        images=images,
                        predictions=preds,
                        labels=labels,
                        num_samples=min(4, len(images)),  # Show up to 4 samples
                        colormap=self.colormap,
                        figsize=(16, 10),
                        save_path=save_path,
                        show_legend=True,
                        title=f"Evaluation Batch {batch_idx}"
                    )
                    plt.close(fig)  # Close to prevent memory leak

                # Update metrics
                self.evaluator.update(preds, labels)

        # Compute all metrics
        results = self.evaluator.compute_all_metrics()

        # Also get per-class metrics for detailed analysis
        classwise_iou = self.evaluator.get_classwise_iou()
        classwise_f1 = self.evaluator.get_classwise_f1()

        results['Classwise IoU'] = classwise_iou
        results['Classwise F1'] = classwise_f1

        return results

    def _post_process(self, data):
        """Format and return results."""
        results = data

        # Print summary
        print("\n" + "="*60)
        print("Metaphase Segmentation Evaluation Results")
        print("="*60)
        print(f"Pixel Accuracy: {results['Pixel Accuracy']:.4f}")
        print(f"Mean IoU: {results['Mean IoU']:.4f}")
        print(f"Mean F1-Score: {results['Mean F1-Score']:.4f}")

        # Print per-class metrics for non-background classes
        print("\nPer-Class Metrics (Chromosomes 1-46):")
        print("-"*60)
        classwise_iou = results['Classwise IoU']
        classwise_f1 = results['Classwise F1']

        # Show top 5 and bottom 5 performing classes
        iou_sorted = np.argsort(classwise_iou[1:])[::-1]  # Exclude background
        print("\nTop 5 Performing Classes (by IoU):")
        for i in range(min(5, len(iou_sorted))):
            cls_idx = iou_sorted[i] + 1  # +1 because we excluded background
            print(f"  Chromosome {cls_idx}: IoU={classwise_iou[cls_idx]:.4f}, F1={classwise_f1[cls_idx]:.4f}")

        print("\nBottom 5 Performing Classes (by IoU):")
        for i in range(min(5, len(iou_sorted))):
            cls_idx = iou_sorted[-(i+1)] + 1
            print(f"  Chromosome {cls_idx}: IoU={classwise_iou[cls_idx]:.4f}, F1={classwise_f1[cls_idx]:.4f}")

        print("="*60)

        return results

    def process(self, data):
        """Run the full evaluation pipeline."""
        data = self._pre_process(data)
        data = self._main_process(data)
        data = self._post_process(data)
        return data

    def handle(self, data):
        """Handle the evaluation and pass data to next handler."""
        results = self.process(data)
        data.update({
            "metaphase_results": results
        })
        return super().handle(data)


@evaluate_segmentation_handler_register.register("EvaluateDomainAdapt")
class EvaluateDomainAdapt(AbsAlgorithmProcessHandler):
    def __init__(self, config: dict, **kwargs) -> None:
        # Accept and ignore extra parameters like is_evaluation
        self.config = config
        ignore_bg = self.config.get("ignore_background", True)
        self.chromosome_evaluator = SegmentationMetrics(num_classes=3, ignore_background=ignore_bg)
        self.overlap_evaluator = BinarySegmentationMetrics()

        self._transformation = self.build_transformation()
        self._source_dataset = self.source_dataset
        self._target_dataset = self.target_dataset
    
    @property
    def source_dataset(self):
        dataset_class = getattr(build_dataset_class, self.config.get("source_dataclass").get("dataset_name"))
        source_data = dataset_class(
            image_dir = self.config.get("source_dataclass").get("data_dir"), 
            transformation=self._transformation, 
            equalization_method_config = self.config.get("source_dataclass").get("equalization_method")
        )
        
        return source_data
        
    @property
    def target_dataset(self):
        dataset_class = getattr(build_dataset_class, self.config.get("target_dataclass").get("dataset_name"))
        target_data = dataset_class(
            image_dir=self.config.get("target_dataclass").get("data_dir", None), 
            source_dataset = self._source_dataset,
            transformation=self._transformation, 
            is_matching_features = self.config.get("target_dataclass").get("is_matching_features"),
            equalization_method_config = self.config.get("target_dataclass").get("equalization_method"),
        )
        return target_data
    
    def build_transformation(self):
        transforms = []
        for transform_method, params in self.config.get("transformation").items():
            if transform_method == "OneOf":
                choice_methods = []
                method_list = params.pop("transforms", [])
                for method in method_list:
                    choice_methods.append(getattr(A, next(iter(method.keys())))(**next(iter(method.values()))))
                transforms.append(A.OneOf(choice_methods, **params))
            elif transform_method == "ToTensorV2":
                transforms.append(ToTensorV2())
            elif hasattr(A, transform_method):
                transforms.append(getattr(A, transform_method)(**params))
            else:
                raise ValueError(f"Unknown transform: {transform_method}")
        return A.Compose(transforms)
        
    def _pre_process(self, data):
        model: nn.Module = data.get("model", None)
        msg = torch.load(self.config.get("model_path", None), map_location = self.config.get("device", None))
        # non_parallel_state_dict = {k.replace("module.", ""): v for k, v in msg["best_model"].items()}
        
        non_parallel_state_dict = {}
        for k, v in msg["best_model"].items():
            new_k = k

            # Remove 'module.' if it exists
            if new_k.startswith("module."):
                new_k = new_k.replace("module.", "", 1)

            # Fix 'pytorch_' to 'pytorch_module.' if needed
            if new_k.startswith("domain_classifier.pytorch_weight"):
                new_k = new_k.replace("pytorch_weight", "pytorch_module.weight")

            non_parallel_state_dict[new_k] = v

        status = model.load_state_dict(non_parallel_state_dict, strict = False)
        print(status)
        del msg
        
        source_dataloader = DataLoader(self._source_dataset, batch_size=self.config.get("batch_size"), num_workers=self.config.get("num_workers", 10), pin_memory=True)
        
        target_dataloader = DataLoader(self._target_dataset, batch_size=self.config.get("batch_size"), num_workers=self.config.get("num_workers", 10), pin_memory=True)
        
        return model.to(self.config.get("device")), source_dataloader, target_dataloader
    
    def _main_process(self, data):
        model, source_dataloader, target_dataloader = data
        model.eval()
        num_iters = min(len(source_dataloader), len(target_dataloader))
        
        pbar = tqdm(enumerate(zip(source_dataloader, target_dataloader)), total=num_iters, desc="Evaluation Segmentation model")
        with torch.no_grad():
            for it, (source_batches, target_batches) in pbar:
                source_images, source_labels = source_batches
                source_images = source_images.to(self.config.get("device"))
                source_labels = [lbl.to(self.config.get("device")) for lbl in source_labels]
                
                target_images = target_batches
                target_images = target_images.to(self.config.get("device"))
                
                source_segment_output, source_domain_output = model(source_images, alpha = 0)
                target_segment_output, target_domain_output = model(target_images, alpha = 0)
                
                chromosome_preds = torch.argmax(source_segment_output[0], dim=1)
                overlap_preds = (torch.sigmoid(source_segment_output[1]) > 0.85).long().squeeze(1)
                
                self.chromosome_evaluator.update(chromosome_preds, source_labels[0])
                self.overlap_evaluator.update(overlap_preds, source_labels[1].squeeze(1))
            
        chromosome_results = self.chromosome_evaluator.compute_all_metrics()
        overlap_results = self.overlap_evaluator.compute_all_metrics()
    
        return chromosome_results, overlap_results

    def _post_process(self, data):
        chromosome_results, overlap_results = data
        return chromosome_results, overlap_results
    
    def process(self, data):
        data = self._pre_process(data)
        data = self._main_process(data)
        data = self._post_process(data)
        return data
    
    def handle(self, data):
        chromosome_results, overlap_results = self.process(data)
        data.update(
            {
                "mochromosome_resultsdel": chromosome_results,
                "overlap_results": overlap_results
            }
        )
        return super().handle(data)