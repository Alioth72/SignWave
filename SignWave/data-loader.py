import os
from typing import List, Tuple, Dict
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
from pathlib import Path

class SignLanguageDataset(Dataset):
    """
    Dataset class for sign language images/videos
    """
    def __init__(self, 
                 data_dir: str,
                 transform = None,
                 target_size: Tuple[int, int] = (224, 224)):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.target_size = target_size
        
        # Get all classes (folders) in the data directory
        self.classes = sorted([d for d in os.listdir(data_dir) 
                             if os.path.isdir(os.path.join(data_dir, d))])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Get all image paths and labels
        self.samples = self._get_samples()
        
    def _get_samples(self) -> List[Tuple[str, int]]:
        """Get all image paths and their corresponding labels"""
        samples = []
        for class_name in self.classes:
            class_dir = self.data_dir / class_name
            class_idx = self.class_to_idx[class_name]
            
            for img_path in class_dir.glob('*.jpg'):
                samples.append((str(img_path), class_idx))
                
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a single sample from the dataset"""
        img_path, label = self.samples[idx]
        
        # Load and preprocess image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.target_size)
        
        # Apply transforms if any
        if self.transform:
            image = self.transform(image=image)['image']
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            
        return image, label

class SignLanguageDataLoader:
    """
    Data loader class that manages loading and batching of the dataset
    """
    def __init__(self,
                 data_dir: str,
                 batch_size: int = 32,
                 split_ratio: float = 0.8,
                 num_workers: int = 4):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.split_ratio = split_ratio
        self.num_workers = num_workers
        
        self.train_dataset = None
        self.val_dataset = None
        self.train_loader = None
        self.val_loader = None
        
    def setup(self, transform = None):
        """Setup train and validation datasets and dataloaders"""
        # Create full dataset
        full_dataset = SignLanguageDataset(self.data_dir, transform)
        
        # Split into train and validation
        train_size = int(self.split_ratio * len(full_dataset))
        val_size = len(full_dataset) - train_size
        
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size])
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
    @property
    def num_classes(self) -> int:
        """Get number of classes in the dataset"""
        return len(self.train_dataset.dataset.classes)
    
    def get_class_names(self) -> List[str]:
        """Get list of class names"""
        return self.train_dataset.dataset.classes
