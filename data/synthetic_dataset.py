import os
import re
from glob import glob
from typing import Tuple

import cv2
import numpy as np
import torch
from colorama import Fore, Style
from skimage import exposure
from torch import Tensor
from torch.utils.data import Dataset

from .data_utils import DataPreProcessing
from .dataset import Instance_Segmentation

red = Fore.RED
reset = Style.RESET_ALL

__all__ = [
    "SyntheticDataset",
    "Chromosome_DANNSyntheticDataset",
]

class SyntheticDataset(DataPreProcessing):
    def __init__(self, image_dir, transformation, equalization_method_config = None):
        super().__init__(equalization_method_config)
        self._image_dir = image_dir
        self.transformation = transformation
        self.overlap_label = 3
        
    @property
    def image_id(self) -> list[str]:
        return glob(f"{self._image_dir}/*.png")
    
    def __len__(self):
        return len(self.image_id)
    
    @staticmethod
    def find_overlapping_area(mask1: np.ndarray, mask2: np.ndarray) -> np.ndarray:
        overlap = np.logical_and(mask1, mask2)
        return overlap.astype(np.float32)
    
    def load_data(self, index):
        image_path = self.image_id[index]
        mask_path = image_path.replace('/images/', '/masks/')
        
        image = cv2.imread(image_path)
        if self.equalization_method is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if self.is_grayscale else cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = self.equalization_method(image, **self.equalization_method_config.get("params")) if self.equalization_method_config.get("params") is not None else self.equalization_method(image)
            image = np.expand_dims(image, axis=-1)

        mask = (cv2.imread(mask_path) / 255).astype(np.uint8)    
        overlap = self.find_overlapping_area(mask[..., 1], mask[..., 2]) 
        
        final_mask = np.zeros(mask.shape[:2], dtype = np.float32)
        final_mask = final_mask + (mask[..., 1] - overlap) * 1  
        final_mask = final_mask + (mask[..., 2] - overlap) * 2
        
        stacked_mask = np.stack([final_mask, overlap], axis = -1)
        
        return image, stacked_mask.astype(np.float32)
    
    def __getitem__(self, index) -> Tuple[Tensor, Tuple[Tensor], str]:
        image, stacked_mask = self.load_data(index=index)
        if self.transformation:
            data = self.transformation(image=image, mask=stacked_mask)
            image = data['image']
            stacked_mask = data['mask']
        
        chromosome_mask: Tensor = stacked_mask[..., 0]
        overlap_mask: Tensor = stacked_mask[..., 1]
        
        return image, (chromosome_mask.long(), overlap_mask.unsqueeze(0).long()), torch.tensor(0)
    
#^ synthetic: 0
#^ real: 1


class Chromosome_DANNSyntheticDataset(Instance_Segmentation):
    def __init__(self, image_dir, transformation, equalization_method_config = None):
        super().__init__(image_dir, transformation, equalization_method_config)
    
    @property
    def image_id(self) -> list[str]:
        """Get list of image file paths"""
        # Look for common image extensions
        extensions = ['*.png', '*.jpg', '*.jpeg', '*.tiff', '*.tif']
        image_files = []
        for ext in extensions:
            image_files.extend(glob(os.path.join(self._image_dir, "OVERLAP", ext)))
            image_files.extend(glob(os.path.join(self._image_dir, "TOUCHING", ext)))
        return sorted(image_files)
    
    def load_data(self, index):
        image_path = self.image_id[index]
        mask_path = image_path.replace('/images/', '/masks/')
        
        image = cv2.imread(image_path)
        if self.equalization_method is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if self.is_grayscale else cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = self.equalization_method(image, **self.equalization_method_config.get("params")) if self.equalization_method_config.get("params") is not None else self.equalization_method(image)
            image = np.expand_dims(image, axis=-1)

        mask = cv2.imread(mask_path).astype(np.uint8)        
        match = re.search(r"_(\d+)_(\d+)\.png", image_path)
        
        overlap = self.find_overlapping_area(mask[..., 1], mask[..., 2]) 
        
        final_mask = np.zeros(mask.shape[:2], dtype = np.float32)
        final_mask = final_mask + (mask[..., 1] - overlap) * 1  
        final_mask = final_mask + (mask[..., 2] - overlap) * 2
        
        stacked_mask = np.stack([final_mask, overlap], axis = -1)
        
        # return image, stacked_mask.astype(np.float32), torch.tensor([int(match.group(1)) - 1, int(match.group(2)) - 1], dtype = torch.float32)
        return image, stacked_mask.astype(np.float32)
    
    def __len__(self):
        return super().__len__()
    
    def __getitem__(self, index):
        image, stacked_mask = self.load_data(index=index)
        if self.transformation:
            data = self.transformation(image=image, mask=stacked_mask)
            image: Tensor = data['image']
            stacked_mask: Tensor = data['mask']
        
        chromosome_mask: Tensor = stacked_mask[..., 0]
        overlap_mask: Tensor = stacked_mask[..., 1]
        
        domain_label = torch.tensor(0, dtype=torch.long)
        return image, (chromosome_mask.long(), overlap_mask.unsqueeze(0).long()), domain_label
    
