import os
import glob
import numpy as np
from PIL import Image, ImageEnhance
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, label_mapping=None, augment=False, image_list=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        
        # Use provided image list or load all images
        if image_list is not None:
            self.images = sorted(image_list)
        else:
            self.images = sorted(os.listdir(image_dir))
        
        self.label_mapping = label_mapping  # Dict mapping sparse IDs to dense IDs
        self.augment = augment  # Apply augmentation to all images if True
        if transform is None:
            self.transform = T.Compose([
                T.Resize((256, 256)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
    
    def augment_image_mask(self, image, mask):
        # Color jitter
        if random.random() > 0.5:
            brightness_factor = random.uniform(0.8, 1.2)
            image = ImageEnhance.Brightness(image).enhance(brightness_factor)
        
        if random.random() > 0.5:
            contrast_factor = random.uniform(0.8, 1.2)
            image = ImageEnhance.Contrast(image).enhance(contrast_factor)
        
        return image, mask
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])

        # Load image and mask
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        
        # Resize mask to 256x256 using nearest neighbor interpolation
        mask = mask.resize((256, 256), Image.NEAREST)
        
        # Get mask array
        mask_array = np.array(mask)
        
        # Apply augmentation if enabled
        if self.augment:
            image, mask = self.augment_image_mask(image, mask)
            mask_array = np.array(mask)
        
        # Convert image to tensor
        image = self.transform(image)
        
        # Remap sparse labels to dense if mapping provided
        if self.label_mapping is not None:
            remapped_mask = np.zeros_like(mask_array)
            for sparse_id, dense_id in self.label_mapping.items():
                remapped_mask[mask_array == sparse_id] = dense_id
            mask_array = remapped_mask
        
        mask = torch.from_numpy(mask_array).long() 
        
        return image, mask