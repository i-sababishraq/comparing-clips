#!/usr/bin/env python3
"""
HAR Dataset class for CLIP fine-tuning
Clean, standalone dataset implementation
"""

import logging
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset
from PIL import Image

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HARDataset(Dataset):
    """Human Action Recognition Dataset for CLIP fine-tuning"""
    
    def __init__(self, data_dir: str, split: str = 'train', preprocess=None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.preprocess = preprocess
        
        # Load dataset
        self.load_dataset()
        
    def load_dataset(self):
        """Load HAR dataset and create image-text pairs"""
        # Try different possible split directory names
        possible_splits = [self.split, f'{self.split}ing', f'{self.split}set']
        split_dir = None
        
        for split_name in possible_splits:
            potential_dir = self.data_dir / split_name
            if potential_dir.exists():
                split_dir = potential_dir
                break
        
        if split_dir is None:
            # If no split directory found, assume flat structure with class folders
            split_dir = self.data_dir
        
        self.samples = []
        self.class_names = []
        
        if not split_dir.exists():
            raise ValueError(f"Split directory {split_dir} does not exist")
        
        # Get class names (action categories)
        for class_dir in sorted(split_dir.iterdir()):
            if class_dir.is_dir():
                class_name = class_dir.name
                self.class_names.append(class_name)
                
                # Get all images in this class
                image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']
                for ext in image_extensions:
                    for img_path in class_dir.glob(ext):
                        self.samples.append({
                            'image_path': str(img_path),
                            'class_name': class_name,
                            'class_id': len(self.class_names) - 1,
                            'text': self.create_text_description(class_name)
                        })
        
        logger.info(f"Loaded {len(self.samples)} samples for {self.split} split")
        logger.info(f"Found {len(self.class_names)} action classes: {self.class_names}")
        
        if len(self.samples) == 0:
            logger.warning(f"No samples found in {split_dir}. Please check the dataset structure.")
    
    def create_text_description(self, class_name: str) -> str:
        """Create text descriptions for action classes"""
        # Convert class name to natural language description
        clean_name = class_name.replace('_', ' ').replace('-', ' ').lower()
        templates = [
            f"a photo of a person {clean_name}",
            f"someone {clean_name}",
            f"a person performing {clean_name}",
            f"{clean_name} action",
            f"human {clean_name}"
        ]
        # Return the first template for consistency
        return templates[0]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        try:
            # Load and transform image
            image = Image.open(sample['image_path']).convert('RGB')
            if self.preprocess:
                image = self.preprocess(image)
        except Exception as e:
            logger.warning(f"Could not load image {sample['image_path']}: {e}")
            # Create a black image as fallback
            image = Image.new('RGB', (224, 224), color='black')
            if self.preprocess:
                image = self.preprocess(image)
        
        return {
            'image': image,
            'text': sample['text'],
            'class_id': sample['class_id'],
            'class_name': sample['class_name']
        }
