import os
import re
from glob import glob
from typing import Any, Dict, Tuple

import cv2
import numpy as np
import torch
from colorama import Fore, Style
from skimage import exposure
from torch import Tensor
from torch.utils.data import Dataset

from .data_utils import DataPreProcessing

red = Fore.RED
reset = Style.RESET_ALL

__all__ = [
    "RealDataset",
    "UDARealDataset",
    "DANNRealDataset"
]

class RealDataset(DataPreProcessing):
    def __init__(self, image_dir, transformation, equalization_method_config = None):
        super().__init__(equalization_method_config)
        self._image_dir = image_dir
        self.transformation = transformation
        
    @property
    def image_id(self) -> list[str]:
        return glob(f"{self._image_dir}/*.png")
    
    def load_data(self, index):
        image = cv2.imread(self.image_id[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if self.is_grayscale else cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.equalization_method is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if self.is_grayscale else cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = self.equalization_method(image, **self.equalization_method_config.get("params")) if self.equalization_method_config.get("params") is not None else self.equalization_method(image)
            image = np.expand_dims(image, axis=-1)

        return image
    
    def __len__(self):
        return len(self.image_id)
    
    def __getitem__(self, index) -> Tuple[Tensor, str]:
        image = self.load_data(index=index)
        if self.transformation:
            data = self.transformation(image=image, mask=None)
            image = data['image']
                    
        return image, torch.tensor(1)

class UDARealDataset(DataPreProcessing):
    """
    Real chromosome dataset for UDA target domain (no ground truth labels)
    """
    
    def __init__(self, image_dir: str, transformation=None, equalization_method_config=None):
        super().__init__(equalization_method_config)
        self._image_dir = image_dir
        self.transformation = transformation
        self._inference_mode = False  # Initialize to training mode by default

        # Verify directory exists
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
        
    def inference(self):
        """Set the dataset to inference mode. In this mode, __getitem__ will also return image_id."""
        self._inference_mode = True
        return self

    def train(self):
        """Set the dataset to training mode. In this mode, __getitem__ will not return image_id."""
        self._inference_mode = False
        return self
        
    @property
    def image_id(self) -> list[str]:
        """Get list of image file paths"""
        # Look for common image extensions
        extensions = ['*.png', '*.jpg', '*.jpeg', '*.tiff', '*.tif']
        image_files = []
        for ext in extensions:
            image_files.extend(glob(os.path.join(self._image_dir, ext)))
        return sorted(image_files)
    
    def __len__(self):
        return len(self.image_id)
    
    def load_data(self, index) -> Tuple[np.ndarray, str]:
        """Load image data (no masks for target domain)"""
        image_path = self.image_id[index]
        
        # Load real image
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply histogram equalization if specified
        if self.equalization_method is not None:
            if self.is_grayscale:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            params = self.equalization_method_config.get("params", {})
            image = self.equalization_method(image, **params)
            if len(image.shape) == 2:
                image = np.expand_dims(image, axis=-1)
                image = np.repeat(image, 3, axis=-1)  # Convert back to RGB
        
        return image, image_path
    
    def __getitem__(self, index) -> Dict[str, Any]:
        """Get item in detectron2 format for target domain"""
        image, image_path = self.load_data(index=index)
        
        # Apply transformations
        if self.transformation:
            try:
                data = self.transformation(image=image, mask=None)
                image = data['image']
            except Exception as e:
                print(f"Warning: Transformation failed: {e}")
        
        # Convert to tensor if not already
        if not isinstance(image, torch.Tensor):
            if len(image.shape) == 3:
                image = torch.from_numpy(image).permute(2, 0, 1).float()
            else:
                image = torch.from_numpy(image).unsqueeze(0).float()
        
        # Ensure image is 3-channel
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        
        # Return in detectron2 format for TARGET data (no instances)
        return {
            'file_name': image_path,
            'image_id': os.path.basename(image_path),
            'image': image,
            'height': image.shape[1],
            'width': image.shape[2],
            'template_img': torch.ones(1, image.shape[1], image.shape[2]) * 255,  # All white = all valid
            'depth': np.zeros((image.shape[1], image.shape[2]), dtype=np.float32)  # Dummy depth map
            # No 'instances' field for unlabeled target data
        }

class DANNRealDataset(UDARealDataset):
    def __init__(self, image_dir: str, transformation=None, equalization_method_config=None):
        super().__init__(image_dir, transformation, equalization_method_config)
    
    def __getitem__(self, index) -> Dict[str, Any]:
        """Get item in detectron2 format for target domain"""
        image, _ = self.load_data(index=index)
        
        # Apply transformations
        if self.transformation:
            try:
                data = self.transformation(image=image, mask=None)
                image = data['image']
            except Exception as e:
                print(f"Warning: Transformation failed: {e}")
        
        # Convert to tensor if not already
        if not isinstance(image, torch.Tensor):
            if len(image.shape) == 3:
                image = torch.from_numpy(image).permute(2, 0, 1).float()
            else:
                image = torch.from_numpy(image).unsqueeze(0).float()
        
        # Ensure image is 3-channel
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        
        # Return domain label 1 for real/target domain
        domain_label = torch.tensor(1, dtype=torch.long)
        
        if self._inference_mode:
            return image, self.image_id[index]
        else:
            return image, domain_label