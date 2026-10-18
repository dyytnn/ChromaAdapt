import os
import re
from glob import glob
from typing import Tuple

import cv2
import numpy as np
import torch
from colorama import Fore, Style
from torch import Tensor

from .data_utils import DataPreProcessing

red = Fore.RED
reset = Style.RESET_ALL

class Instance_Segmentation(DataPreProcessing):
    def __init__(self, image_dir, transformation, equalization_method_config = None):
        super().__init__(equalization_method_config)
        self._image_dir = image_dir
        self.transformation = transformation
        self.overlap_label = 25
        
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
        mask = cv2.resize(mask, (224, 224))
        
        match = re.search(r"_(\d+)_(\d+)\.png", image_path)
        
        overlap = self.find_overlapping_area(mask[..., 1], mask[..., 2]) 
        
        final_mask = np.zeros(mask.shape[:2], dtype = np.float32)
        final_mask = final_mask + (mask[..., 1] - overlap) * 1  
        final_mask = final_mask + (mask[..., 2] - overlap) * 2
        
        stacked_mask = np.stack([final_mask, overlap], axis = -1)
        
        return image, stacked_mask.astype(np.float32), torch.tensor([int(match.group(1)) - 1, int(match.group(2)) - 1], dtype = torch.float32)
    
    def __getitem__(self, index) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        image, stacked_mask, class_label = self.load_data(index=index)
        if self.transformation:
            data = self.transformation(image=image, mask=stacked_mask)
            image = data['image']
            stacked_mask = data['mask']
        
        chromosome_mask: Tensor = stacked_mask[..., 0]
        overlap_mask: Tensor = stacked_mask[..., 1]
        
        return image, (chromosome_mask.long(), overlap_mask.unsqueeze(0).long(), class_label.long())
    
class testset_Instance_Segmentation(Instance_Segmentation):
    def __init__(self, image_dir, transformation, equalization_method_config = None):
        super().__init__(image_dir, transformation, equalization_method_config)
    
    @property
    def image_id(self) -> list[str]:
        return glob(f"{self._image_dir}/images/*.png") 
    
    def load_data(self, index):
        image_path = self.image_id[index]
        mask_path = image_path.replace('/images/', '/masks/')
        
        image = cv2.imread(image_path)
        if self.equalization_method is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = self.equalization_method(image, **self.equalization_method_config.get("params")) if self.equalization_method_config.get("params") is not None else self.equalization_method(image)
            image = np.expand_dims(image, axis=-1)

        mask = (cv2.imread(mask_path) / 255).astype(np.uint8)
        overlap = self.find_overlapping_area(mask[..., 1], mask[..., 2]) 
        
        final_mask = np.zeros(mask.shape[:2], dtype = np.float32)
        final_mask = final_mask + (mask[..., 1] - overlap) * 1  
        final_mask = final_mask + (mask[..., 2] - overlap) * 2
        
        stacked_mask = np.stack([final_mask, overlap], axis = -1)
        
        return image, stacked_mask.astype(np.float32), torch.tensor([0, 0], dtype = torch.float32)

class Instance_Segmentation_Non_Classify(Instance_Segmentation):
    """This dataset used for the data for worker dataset"""
    def __init__(self, image_dir, transformation, equalization_method_config = None):
        super().__init__(image_dir, transformation, equalization_method_config)
        self._inference_mode = False

    def inference(self):
        """Set the dataset to inference mode. In this mode, __getitem__ will also return image_id."""
        self._inference_mode = True
        return self

    def train(self):
        """Set the dataset to training mode. In this mode, __getitem__ will not return image_id."""
        self._inference_mode = False
        return self

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
                
        final_mask = np.zeros(mask.shape[:2], dtype = np.float32)
        final_mask = final_mask + mask[..., 0] * 1  
        final_mask = final_mask + mask[..., 1] * 2
        
        stacked_mask = np.stack([final_mask, mask[..., 2]], axis = -1)
        
        return image, stacked_mask.astype(np.float32)
    
    def __getitem__(self, index) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        image, stacked_mask = self.load_data(index=index)
        if self.transformation:
            data = self.transformation(image=image, mask=stacked_mask)
            image = data['image']
            stacked_mask = data['mask']

        chromosome_mask: Tensor = stacked_mask[..., 0]
        overlap_mask: Tensor = stacked_mask[..., 1]

        if self._inference_mode:
            return image, (chromosome_mask.long(), overlap_mask.unsqueeze(0).long()), self.image_id[index]
        else:
            return image, (chromosome_mask.long(), overlap_mask.unsqueeze(0).long())
    
class Instance_segmentation_touch_overlap(Instance_Segmentation):
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
        
        return image, (chromosome_mask.long(), overlap_mask.unsqueeze(0).long())