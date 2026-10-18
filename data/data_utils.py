from typing import Literal

import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2
from skimage import exposure
from torch.utils.data import Dataset


def _init_binary(image: np.ndarray) -> np.ndarray:
    binary = np.ones(image.shape[:2]) * 255
    binary[np.where(image[..., -1] < 225)] = 1 
    
    return binary

def find_sudden_drop_index(sorted_combined, drop_factor=0.3):
    for i in range(1, len(sorted_combined)):
        if np.sum(sorted_combined[i].area) < np.sum(sorted_combined[i-1].area) * drop_factor:
            return i
    return len(sorted_combined)

def get_centroid(bbox):
    y_min, x_min, y_max, x_max = bbox
    return ((y_min + y_max) // 2, (x_min + x_max) // 2)

def extract_chromosome_from_area_props(image, bounding_box, segmented, padding):
    y1, x1, y2, x2 = bounding_box
    image_height, image_width = image.shape[:2]

    y1_padded = max(0, y1 - padding)
    x1_padded = max(0, x1 - padding)
    y2_padded = min(image_height, y2 + padding)
    x2_padded = min(image_width, x2 + padding)

    top_padding = y1 - y1_padded
    left_padding = x1 - x1_padded
    bottom_padding = y2_padded - y2
    right_padding = x2_padded - x2

    if top_padding > 0 or left_padding > 0 or bottom_padding > 0 or right_padding > 0:
        segmented_padded = np.pad(segmented, ((top_padding, bottom_padding), (left_padding, right_padding)), mode='constant', constant_values=0)
    else:
        segmented_padded = segmented

    white_background = np.ones_like(image[y1_padded:y2_padded, x1_padded:x2_padded]) * 255
    object_with_mask = (np.stack([segmented_padded] * 3, axis=-1) * image[y1_padded:y2_padded, x1_padded:x2_padded]).astype(np.uint8)
    applied = object_with_mask + ((1 - np.stack([segmented_padded] * 3, axis=-1)) * white_background).astype(np.uint8)

    return applied

class DataPreProcessing(Dataset):
    def __init__(self, equalization_method_config: dict = None):
        super().__init__()
        if equalization_method_config is None:
            # If no config is provided, set equalization method to None
            self.equalization_method = None
            self.equalization_method_config = None
            self.is_grayscale = False
        else:
            # Existing logic for specific equalization methods
            if equalization_method_config.get("name", None) not in ["histogram equalization", "adaptive histogram equalization"]:
                raise ValueError(f"Invalid equalization method: {equalization_method_config.get('name')}")
            
            self.equalization_method_config = equalization_method_config
            self.equalization_method = self._equalization_method
            self.is_grayscale = True
    
    @property
    def _equalization_method(self):
        if self.equalization_method_config is None:
            return None
        
        method_name = self.equalization_method_config.get("name", None)
        if method_name == "histogram equalization":
            return exposure.equalize_hist
        elif method_name == "adaptive histogram equalization":
            return exposure.equalize_adapthist
    
    def __len__(self):
        raise NotImplementedError
    
    def __getitem__(self, index):
        raise NotImplementedError


class DataAugmentation(object):
    def __init__(self, config, mode: Literal["train", "val"] = "train"):
        normalize = A.Compose([
            A.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
        
        resize = A.Resize(config.img_size, config.img_size)
        
        flip_and_color_jitter = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=.5),
            A.OneOf([
                    A.GaussNoise(var_limit=[10, 50]),
                    A.GaussianBlur(),
                    A.MotionBlur(),
                    ], p=0.25),
            A.CoarseDropout(max_holes=1, max_width=int(config.img_size * 0.05), max_height=int(config.img_size * 0.05), p=0.25), # TODO: randomly drops out rectangular region from the image and optionally, the corresponding regions in a associated mask to simulate occlusion and varied object sizes found in real-world setting
        ])
        
        if mode == "train":
            self.transforms = A.Compose([
                resize,
                flip_and_color_jitter,
                normalize
            ])
        else:
            self.transforms = A.Compose([
                resize,
                normalize
            ])
        
    def __call__(self, image, mask):
        return self.transforms(image=image, mask=mask)
    
def yolo_mask_to_binary_mask(txt_path, img_width, img_height):
    """
    Converts YOLO polygon segmentation (from .txt file) into a binary mask.
    """
    mask = np.zeros((img_height, img_width), dtype=np.uint16)  # Use uint16 for more instance range

    with open(txt_path, 'r') as file:
        lines = file.readlines()

    instance_id = 1  # Start from 1 (0 = background)

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 7:
            continue  # Skip invalid polygons (at least 3 points = 6 coords + class_id)

        coords = list(map(float, parts[1:]))

        polygon = np.array([
            [int(coords[i] * img_width), int(coords[i + 1] * img_height)]
            for i in range(0, len(coords), 2)
        ], np.int32)

        if polygon.shape[0] > 2:
            polygon = polygon.reshape((-1, 1, 2))
            cv2.fillPoly(mask, [polygon], instance_id)
            instance_id += 1
            
    assert mask.shape[-1] == 3, "Mask should have 3 channels"
    return mask