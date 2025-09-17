#!/usr/bin/env python3
"""
Fine-tune CLIP model on Human Action Recognition (HAR) dataset
Using the existing OpenAI CLIP implementation in this repository
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

# Import CLIP from the local repository
import clip
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report

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
        # Assuming HAR dataset structure:
        # data_dir/
        #   ├── train/
        #   │   ├── action1/
        #   │   ├── action2/
        #   │   └── ...
        #   ├── test/
        #   └── val/ (or validation/)
        
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
        
        action_templates = [
            f"a photo of a person {clean_name}",
            f"someone {clean_name}",
            f"a person performing {clean_name}",
            f"{clean_name} action",
            f"human {clean_name}",
            f"a video frame of someone {clean_name}",
            f"footage of a person {clean_name}"
        ]
        # Return the first template for consistency, could randomize during training
        return action_templates[0]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load and transform image
        try:
            image = Image.open(sample['image_path']).convert('RGB')
        except Exception as e:
            logger.warning(f"Error loading image {sample['image_path']}: {e}")
            # Create a dummy image
            image = Image.new('RGB', (224, 224), color='black')
        
        if self.preprocess:
            image = self.preprocess(image)
        
        return {
            'image': image,
            'text': sample['text'],
            'class_id': sample['class_id'],
            'class_name': sample['class_name']
        }

class CLIPTrainer:
    """CLIP Fine-tuning Trainer using OpenAI CLIP implementation"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize model and preprocessing
        self.setup_model()
        
        # Setup data loaders
        self.setup_data()
        
        # Setup optimizer and scheduler
        self.setup_optimizer()
        
        # Create checkpoint directory
        self.checkpoint_dir = Path(config.get('checkpoint_dir', './checkpoints'))
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    def setup_model(self):
        """Initialize CLIP model"""
        model_name = self.config.get('model_name', 'ViT-B/32')
        
        # Load pre-trained CLIP model
        self.model, self.preprocess = clip.load(model_name, device=self.device, jit=False)
        
        # Enable training mode and gradients
        self.model.train()
        for param in self.model.parameters():
            param.requires_grad = True
        
        logger.info(f"Loaded model: {model_name}")
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
    
    def setup_data(self):
        """Setup data loaders"""
        data_dir = self.config['data_dir']
        batch_size = self.config.get('batch_size', 32)
        num_workers = self.config.get('num_workers', 4)
        
        # Create datasets
        self.train_dataset = HARDataset(data_dir, 'train', preprocess=self.preprocess)
        
        # Try to load validation dataset, fall back to test if not available
        try:
            self.val_dataset = HARDataset(data_dir, 'val', preprocess=self.preprocess)
        except ValueError:
            try:
                self.val_dataset = HARDataset(data_dir, 'validation', preprocess=self.preprocess)
            except ValueError:
                logger.warning("No validation split found. Using test split for validation.")
                self.val_dataset = HARDataset(data_dir, 'test', preprocess=self.preprocess)

        # If val dataset has no samples, create a split from train
        if len(self.val_dataset) == 0:
            val_ratio = float(self.config.get('val_ratio', 0.1))
            num_train = len(self.train_dataset)
            num_val = max(1, int(num_train * val_ratio))
            num_train_new = max(1, num_train - num_val)
            logger.info(f"No validation data available. Creating a {int(val_ratio*100)}% split from train: train={num_train_new}, val={num_val}")
            train_subset, val_subset = torch.utils.data.random_split(
                self.train_dataset, [num_train_new, num_val],
                generator=torch.Generator().manual_seed(42)
            )
            self.train_dataset = train_subset
            self.val_dataset = val_subset
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True  # For stable contrastive learning
        )
        
        self.val_loader = DataLoader(
            self.val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=True
        )
        
        logger.info(f"Train samples: {len(self.train_dataset)}")
        logger.info(f"Val samples: {len(self.val_dataset)}")
        # Handle datasets wrapped in torch.utils.data.Subset after random split
        base_train_ds = self.train_dataset.dataset if hasattr(self.train_dataset, 'dataset') else self.train_dataset
        train_class_names = getattr(base_train_ds, 'class_names', [])
        logger.info(f"Number of classes: {len(train_class_names)}")
        logger.info(f"Classes: {train_class_names}")
    
    def setup_optimizer(self):
        """Setup optimizer and scheduler"""
        lr = self.config.get('learning_rate', 1e-5)
        weight_decay = self.config.get('weight_decay', 0.1)
        epochs = self.config.get('epochs', 10)
        warmup_epochs = self.config.get('warmup_epochs', 1)
        
        # Optimizer - use different learning rates for different parts
        vision_lr = lr
        text_lr = lr * 0.1  # Lower learning rate for text encoder
        
        vision_params = list(self.model.visual.parameters())
        text_params = list(self.model.transformer.parameters()) + \
                     [self.model.token_embedding.weight, 
                      self.model.positional_embedding,
                      self.model.ln_final.weight,
                      self.model.ln_final.bias]
        projection_params = [self.model.text_projection, self.model.logit_scale]
        
        param_groups = [
            {'params': vision_params, 'lr': vision_lr, 'name': 'vision'},
            {'params': text_params, 'lr': text_lr, 'name': 'text'},
            {'params': projection_params, 'lr': lr, 'name': 'projection'}
        ]
        
        self.optimizer = AdamW(
            param_groups,
            weight_decay=weight_decay,
            betas=(0.9, 0.98),
            eps=1e-6
        )
        
        # Scheduler with warmup
        warmup_steps = int(len(self.train_loader) * warmup_epochs)
        total_steps = int(len(self.train_loader) * epochs)
        
        def lr_lambda(step):
            # Handle edge cases to avoid division by zero
            if warmup_steps <= 0 and total_steps <= 0:
                return 1.0
            if step < warmup_steps:
                return (step / warmup_steps) if warmup_steps > 0 else 1.0
            # After warmup
            denom = (total_steps - warmup_steps)
            if denom <= 0:
                return 1.0
            return 0.5 * (1 + np.cos(np.pi * (step - warmup_steps) / denom))
        
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
        logger.info(f"Learning rate: {lr}")
        logger.info(f"Warmup steps: {warmup_steps}")
        logger.info(f"Total steps: {total_steps}")
    
    def compute_contrastive_loss(self, logits_per_image, logits_per_text):
        """Compute contrastive loss (InfoNCE)"""
        batch_size = logits_per_image.shape[0]
        labels = torch.arange(batch_size, device=logits_per_image.device)
        
        loss_img = F.cross_entropy(logits_per_image, labels)
        loss_txt = F.cross_entropy(logits_per_text, labels)
        
        return (loss_img + loss_txt) / 2
    
    def train_epoch(self, epoch: int):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            texts = batch['text']
            
            # Tokenize texts
            text_tokens = clip.tokenize(texts, truncate=True).to(self.device)
            
            # Forward pass
            logits_per_image, logits_per_text = self.model(images, text_tokens)
            
            # Compute loss
            loss = self.compute_contrastive_loss(logits_per_image, logits_per_text)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{avg_loss:.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
        
        return total_loss / num_batches
    
    def evaluate(self, epoch: int):
        """Evaluate model on validation set"""
        if len(self.val_loader) == 0:
            logger.info("No validation data available, skipping evaluation")
            return 0.0, 0.0
            
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                images = batch['image'].to(self.device)
                texts = batch['text']
                labels = batch['class_id'].cpu().numpy()
                
                # Tokenize texts
                text_tokens = clip.tokenize(texts, truncate=True).to(self.device)
                
                # Forward pass
                logits_per_image, logits_per_text = self.model(images, text_tokens)
                
                # Compute loss
                loss = self.compute_contrastive_loss(logits_per_image, logits_per_text)
                total_loss += loss.item()
                
                # Get predictions
                predictions = logits_per_image.argmax(dim=1).cpu().numpy()
                
                all_predictions.extend(predictions)
                all_labels.extend(labels)
        
        # Compute metrics
        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        
        logger.info(f"Validation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        return avg_loss, accuracy
    
    def save_checkpoint(self, epoch: int, loss: float, accuracy: float, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'accuracy': accuracy,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'clip_har_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / 'clip_har_best.pt'
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model: {best_path}")
    
    def train(self):
        """Main training loop"""
        epochs = self.config.get('epochs', 10)
        logger.info(f"Starting training for {epochs} epochs...")
        
        best_accuracy = 0.0
        
        for epoch in range(1, epochs + 1):
            logger.info(f"\nEpoch {epoch}/{epochs}")
            
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Evaluate
            val_loss, val_accuracy = self.evaluate(epoch)
            
            # Check if this is the best model
            is_best = val_accuracy > best_accuracy
            if is_best:
                best_accuracy = val_accuracy
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_loss, val_accuracy, is_best)
            
            logger.info(f"Epoch {epoch} Summary:")
            logger.info(f"  Train Loss: {train_loss:.4f}")
            logger.info(f"  Val Loss: {val_loss:.4f}")
            logger.info(f"  Val Accuracy: {val_accuracy:.4f}")
            logger.info(f"  Best Accuracy: {best_accuracy:.4f}")
        
        logger.info("\nTraining completed!")
        logger.info(f"Best validation accuracy: {best_accuracy:.4f}")

def main():
    parser = argparse.ArgumentParser(description='Fine-tune CLIP on HAR dataset')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to HAR dataset directory')
    parser.add_argument('--config', type=str, 
                        help='Path to config file (JSON)')
    parser.add_argument('--model_name', type=str, default='ViT-B/32',
                        help='CLIP model name (ViT-B/32, ViT-B/16, ViT-L/14, RN50, etc.)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                        help='Learning rate')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    
    args = parser.parse_args()
    
    # Load config from file if provided
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = {}
    
    # Override config with command line arguments
    config.update({
        'data_dir': args.data_dir,
        'model_name': args.model_name,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'checkpoint_dir': args.checkpoint_dir
    })
    
    logger.info("Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    
    # Create trainer and start training
    trainer = CLIPTrainer(config)
    trainer.train()

if __name__ == '__main__':
    main()


