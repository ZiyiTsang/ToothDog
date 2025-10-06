import os
import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import lightning as L
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.models.segmentation import deeplabv3_resnet50
from typing import Optional
import matplotlib.pyplot as plt
from matplotlib import cm


# Define mapping from tooth ID to class index
TOOTH_ID_TO_LABEL = {
    "11": 0, "12": 1, "13": 2, "14": 3, "15": 4,
    "16": 5, "17": 6, "21": 8, "22": 9, "23": 10,
    "24": 11, "25": 12, "26": 13, "27": 14,
    "34": 19, "35": 20, "36": 21, "37": 22,
    "45": 28, "46": 29, "47": 30
}

# Generate 32 RGB colors for visualization
def generate_32_rgb_colors(cmap_name='hsv'):
    """Generate 32 distinct RGB colors for visualization"""
    num_colors = 32
    samples = np.linspace(0, 1, num_colors, endpoint=False)
    cmap = cm.get_cmap(cmap_name, num_colors)
    rgba_colors = cmap(samples)
    rgb_colors = (rgba_colors[:, :3] * 255).astype(np.uint8)
    return rgb_colors

COLOR_32 = generate_32_rgb_colors()




class ToothSegmentationDataset(Dataset):
    """Tooth segmentation dataset class"""
    
    def __init__(self, root_dir, split='train', transform=None, tooth_ids=None):
        """
        Initialize dataset
        
        Args:
            root_dir: Root directory path of the dataset
            split: Dataset split type ('train', 'val', 'test')
            transform: Data augmentation transformations
            tooth_ids: List of tooth IDs to include
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.tooth_ids = tooth_ids or list(TOOTH_ID_TO_LABEL.keys())
        
        # Initialize lists to store data paths and labels
        self.image_paths = []  # Store image paths
        self.mask_paths = []   # Store mask paths
        self.tooth_labels = [] # Store tooth labels
        
        # Determine base path based on split type
        if split == 'train':
            base_dir = os.path.join(root_dir, "trainset_valset")
        elif split == 'val':
            base_dir = os.path.join(root_dir, "trainset_valset")
        elif split == 'test':
            base_dir = os.path.join(root_dir, "testset")
        else:
            raise ValueError("Split must be 'train', 'val', or 'test'")
        
        # Use glob to collect all image and mask paths
        if split in ['train', 'val']:
            # For train/val sets: search all number directories
            number_dirs = [d for d in os.listdir(base_dir)
                          if os.path.isdir(os.path.join(base_dir, d)) and d.isdigit()]
            
            for number_dir in number_dirs:
                # Determine split-specific path
                if split == 'train':
                    split_path = os.path.join(base_dir, number_dir, "train")
                else:  # val
                    split_path = os.path.join(base_dir, number_dir, "val")
                
                if not os.path.exists(split_path):
                    continue
                
                # Pattern is .../number_toothID/000_rgb.jpg
                pattern = os.path.join(split_path, "*_*")
                tooth_folders = glob.glob(pattern)
                
                for folder in tooth_folders:
                    # Extract tooth ID from folder name
                    folder_name = os.path.basename(folder)
                    tooth_id = folder_name.split('_')[-1]
                    
                    # Only include specified tooth IDs
                    if tooth_id not in self.tooth_ids:
                        continue
                    
                    # Find all RGB images in the folder
                    rgb_pattern = os.path.join(folder, "*_rgb.jpg")
                    rgb_images = glob.glob(rgb_pattern)
                    
                    for rgb_path in rgb_images:
                        # Construct corresponding mask path
                        base_name = os.path.basename(rgb_path).replace('_rgb.jpg', '')
                        mask_path = os.path.join(folder, f"{base_name}_mask.jpg")
                        
                        if os.path.exists(mask_path):
                            self.image_paths.append(rgb_path)
                            self.mask_paths.append(mask_path)
                            self.tooth_labels.append(TOOTH_ID_TO_LABEL[tooth_id])
        
        else:  # Test set
            # For test set: pattern is .../number/000_rgb.jpg
            pattern = os.path.join(base_dir, "*")
            test_folders = glob.glob(pattern)
            
            for folder in test_folders:
                # Find all RGB images in the folder
                rgb_pattern = os.path.join(folder, "*_rgb.jpg")
                rgb_images = glob.glob(rgb_pattern)
                
                for rgb_path in rgb_images:
                    # Construct corresponding mask path
                    base_name = os.path.basename(rgb_path).replace('_rgb.jpg', '')
                    mask_path = os.path.join(folder, f"{base_name}_mask.jpg")
                    
                    # For test set, include samples even without masks for inference
                    # Create placeholder masks for test set
                    self.image_paths.append(rgb_path)
                    self.mask_paths.append(mask_path)  # This path may not exist, but we'll handle it in __getitem__
                    # Test set doesn't have explicit tooth ID, set to -1
                    self.tooth_labels.append(-1)
        
        print(f"Loaded {len(self.image_paths)} {split} samples")
    
    def __len__(self):
        """Return the total number of samples in the dataset"""
        return len(self.image_paths)
    def __getitem__(self, idx):
        """Get a sample from the dataset"""
        # Read image
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Read mask or create placeholder for test set
        tooth_id = self.tooth_labels[idx]
        if tooth_id != -1:  # For train/val sets
            # Read mask for train/val sets - keep original multi-class format
            mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
            # Keep mask as multi-class (values: 0=background, 1=upper teeth, 2=lower teeth, 3=specific tooth)
            tooth_mask = mask.astype(np.uint8)
            
            # Filter out classes 5 and 6, only keep classes 0-4
            # Set classes 5 and 6 to background (class 0)
            tooth_mask = np.where((tooth_mask == 5) | (tooth_mask == 6), 0, tooth_mask)
        else:  # For test set
            # For test set, check if mask exists
            if os.path.exists(self.mask_paths[idx]):
                mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
                # Keep mask as multi-class
                tooth_mask = mask.astype(np.uint8)
                # Filter out classes 5 and 6, only keep classes 0-4
                tooth_mask = np.where((tooth_mask == 5) | (tooth_mask == 6), 0, tooth_mask)
            else:
                # Create placeholder mask (all zeros = background) for test set without masks
                tooth_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        # Apply transformations
        if self.transform:
            augmented = self.transform(image=image, mask=tooth_mask)
            image = augmented['image']
            tooth_mask = augmented['mask']
        
        # Ensure mask is long tensor
        if isinstance(tooth_mask, torch.Tensor):
            tooth_mask = tooth_mask.long()
        else:
            tooth_mask = torch.from_numpy(tooth_mask).long()
        
        # Create one-hot encoding for tooth ID
        if tooth_id != -1:
            # Find the maximum index in TOOTH_ID_TO_LABEL and create tensor of size max_index+1
            max_index = max(TOOTH_ID_TO_LABEL.values())
            tooth_id_tensor = torch.zeros(max_index + 1)
            if tooth_id < tooth_id_tensor.size(0):  # Check bounds
                tooth_id_tensor[tooth_id] = 1
            else:
                # If tooth_id is out of bounds, use all zeros
                tooth_id_tensor = torch.zeros(max_index + 1)
        else:
            # For test set, use all zeros as placeholder
            max_index = max(TOOTH_ID_TO_LABEL.values())
            tooth_id_tensor = torch.zeros(max_index + 1)
        
        return image, tooth_mask, tooth_id_tensor

# Data augmentation transformations
def get_transform(split='train'):
    """Get data augmentation transformations for the specified split"""
    if split == 'train':
        return A.Compose([
            A.Resize(256, 256),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(256, 256),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])


class ToothDataModule(L.LightningDataModule):
    """PyTorch Lightning data module for tooth segmentation"""
    
    def __init__(self, data_dir, batch_size=4, num_workers=4, tooth_ids=None):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tooth_ids = tooth_ids
    
    def setup(self, stage: Optional[str] = None):
        """Setup datasets for training, validation, and testing"""
        # Assign train/val datasets
        if stage == "fit" or stage is None:
            self.train_dataset = ToothSegmentationDataset(
                self.data_dir, split='train', 
                transform=get_transform('train'), tooth_ids=self.tooth_ids
            )
            self.val_dataset = ToothSegmentationDataset(
                self.data_dir, split='val', 
                transform=get_transform('val'), tooth_ids=self.tooth_ids
            )
        
        # Assign test dataset
        if stage == "test" or stage is None:
            self.test_dataset = ToothSegmentationDataset(
                self.data_dir, split='test', 
                transform=get_transform('val'), tooth_ids=self.tooth_ids
            )
    
    def train_dataloader(self):
        """Return training dataloader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True  # Drop last incomplete batch to avoid BatchNorm issues
        )
    
    def val_dataloader(self):
        """Return validation dataloader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True  # Drop last incomplete batch to avoid BatchNorm issues
        )
    
    def test_dataloader(self):
        """Return test dataloader"""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True  # Drop last incomplete batch to avoid BatchNorm issues
        )


if __name__=="__main__":
    dm=ToothDataModule(data_dir='../data/ToothSegmDataset', batch_size=8, num_workers=4, tooth_ids=list(TOOTH_ID_TO_LABEL.keys()))
    dm.setup()

    # Check train dataloader
    train_loader = dm.train_dataloader()
    for images, masks, tooth_ids in train_loader:
        print(f"Train batch - images: {images.shape}, masks: {masks.shape}, tooth_ids: {tooth_ids.shape}")
        break
    # Check val dataloader
    val_loader = dm.val_dataloader()
    for images, masks, tooth_ids in val_loader:
        print(f"Val batch - images: {images.shape}, masks: {masks.shape}, tooth_ids: {tooth_ids.shape}")
        break
    # Check test dataloader
    test_loader = dm.test_dataloader()
    for images, masks, tooth_ids in test_loader:
        print(f"Test batch - images: {images.shape}, masks: {masks.shape}, tooth_ids: {tooth_ids.shape}")
        break

