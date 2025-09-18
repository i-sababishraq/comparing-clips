#!/usr/bin/env python3
"""
SigLIP fine-tuning on HAR dataset using Hugging Face Transformers.
Based on the merveenoyan/siglip repository approach.
"""

import argparse
import logging
import time
import psutil
import os
from pathlib import Path
from typing import Dict, List, Tuple
import json
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Hugging Face imports
from transformers import AutoModel, AutoProcessor, SiglipModel, SiglipProcessor
from PIL import Image
import requests

# Import the HAR dataset
import sys
sys.path.append('/anvil/projects/x-soc250046/x-sishraq/CLIP/CLIP_model')
from har_dataset import HARDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SigLIPTrainer:
    """SigLIP trainer using Hugging Face Transformers"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load SigLIP model and processor
        model_name = config.get('model_name', 'google/siglip-base-patch16-224')
        logger.info(f"Loading SigLIP model: {model_name}")
        
        self.model = SiglipModel.from_pretrained(model_name)
        self.processor = SiglipProcessor.from_pretrained(model_name)
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Add classification head
        self.classifier = nn.Linear(self.model.config.vision_config.hidden_size, 15).to(self.device)
        
        # Load dataset
        self.train_dataset = HARDataset(config['data_dir'], 'train')
        self.test_dataset = HARDataset(config['data_dir'], 'test')
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=config['batch_size'], 
            shuffle=True, 
            num_workers=config['num_workers'],
            collate_fn=self.custom_collate_fn
        )
        self.test_loader = DataLoader(
            self.test_dataset, 
            batch_size=config['batch_size'], 
            shuffle=False, 
            num_workers=config['num_workers'],
            collate_fn=self.custom_collate_fn
        )
        
        # Optimizer - only train the classifier head initially
        self.optimizer = torch.optim.AdamW(
            self.classifier.parameters(), 
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Setup checkpointing
        self.checkpoint_dir = Path(config['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Performance monitoring
        self.start_time = None
        self.memory_usage = []
        self.speed_metrics = []
        
        logger.info(f"Model initialized on {self.device}")
        logger.info(f"Train samples: {len(self.train_dataset)}")
        logger.info(f"Test samples: {len(self.test_dataset)}")
    
    def custom_collate_fn(self, batch):
        """Custom collate function for HAR dataset"""
        images = [item['image'] for item in batch]
        texts = [item['text'] for item in batch]
        labels = torch.tensor([item['class_id'] for item in batch])
        
        return {
            'images': images,
            'texts': texts,
            'labels': labels
        }
    
    def process_batch(self, images, texts):
        """Process batch using SigLIP processor"""
        # Process images and text with SigLIP processor
        inputs = self.processor(
            images=images, 
            text=texts, 
            return_tensors="pt", 
            padding=True
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        return inputs
    
    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.eval()  # Keep SigLIP frozen
        self.classifier.train()  # Only train classifier
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Start timing
        epoch_start_time = time.time()
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Memory monitoring
            if batch_idx % 10 == 0:
                memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
                self.memory_usage.append(memory_mb)
            
            images = batch['images']
            texts = batch['texts']
            labels = batch['labels'].to(self.device)
            
            # Process batch
            inputs = self.process_batch(images, texts)
            
            # Forward pass through SigLIP
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use pooled image embeddings returned by SigLIP
                image_features = outputs.image_embeds
            
            # Classification
            class_logits = self.classifier(image_features)
            
            # Loss
            loss = self.criterion(class_logits, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(class_logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{correct/total:.4f}'
            })
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total
        epoch_time = time.time() - epoch_start_time
        
        # Speed metrics
        samples_per_sec = len(self.train_dataset) / epoch_time
        self.speed_metrics.append(samples_per_sec)
        
        return avg_loss, accuracy
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model"""
        self.model.eval()
        self.classifier.eval()
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Evaluating"):
                images = batch['images']
                texts = batch['texts']
                labels = batch['labels'].to(self.device)
                
                # Process batch
                inputs = self.process_batch(images, texts)
                
                # Forward pass
                outputs = self.model(**inputs)
                image_features = outputs.image_embeds
                
                # Classification
                class_logits = self.classifier(image_features)
                
                # Get predictions
                _, predicted = torch.max(class_logits, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        f1_macro = f1_score(all_labels, all_predictions, average='macro')
        f1_weighted = f1_score(all_labels, all_predictions, average='weighted')
        
        return {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'predictions': all_predictions,
            'labels': all_labels
        }
    
    def train(self):
        """Main training loop"""
        logger.info("Starting SigLIP training...")
        self.start_time = time.time()
        
        best_accuracy = 0.0
        
        for epoch in range(1, self.config['epochs'] + 1):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Evaluate
            eval_metrics = self.evaluate()
            test_acc = eval_metrics['accuracy']
            
            logger.info(f"Epoch {epoch}: TrainLoss={train_loss:.4f}, TrainAcc={train_acc:.4f}, TestAcc={test_acc:.4f}")
            
            # Save best model
            if test_acc > best_accuracy:
                best_accuracy = test_acc
                self.save_checkpoint(epoch, is_best=True)
                logger.info(f"New best model saved with accuracy: {test_acc:.4f}")
            
            # Save regular checkpoint
            if epoch % 5 == 0:
                self.save_checkpoint(epoch, is_best=False)
        
        # Final evaluation and performance report
        self.generate_performance_report()
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'classifier_state_dict': self.classifier.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }
        
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / 'siglip_best.pt')
        else:
            torch.save(checkpoint, self.checkpoint_dir / f'siglip_epoch_{epoch}.pt')
    
    def generate_performance_report(self):
        """Generate performance, speed, and memory report"""
        total_time = time.time() - self.start_time
        
        # Calculate average memory usage
        avg_memory = np.mean(self.memory_usage) if self.memory_usage else 0
        max_memory = np.max(self.memory_usage) if self.memory_usage else 0
        
        # Calculate average speed
        avg_speed = np.mean(self.speed_metrics) if self.speed_metrics else 0
        
        # Final evaluation
        final_metrics = self.evaluate()
        
        # Create performance report
        report = {
            'model': 'SigLIP_HF',
            'total_training_time': total_time,
            'average_memory_usage_mb': avg_memory,
            'peak_memory_usage_mb': max_memory,
            'average_speed_samples_per_sec': avg_speed,
            'final_accuracy': final_metrics['accuracy'],
            'final_f1_macro': final_metrics['f1_macro'],
            'final_f1_weighted': final_metrics['f1_weighted'],
            'batch_size': self.config['batch_size'],
            'epochs': self.config['epochs']
        }
        
        # Save report
        with open(self.checkpoint_dir / 'performance_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info("Performance Report:")
        logger.info(f"  Total Training Time: {total_time:.2f}s")
        logger.info(f"  Average Memory Usage: {avg_memory:.2f} MB")
        logger.info(f"  Peak Memory Usage: {max_memory:.2f} MB")
        logger.info(f"  Average Speed: {avg_speed:.2f} samples/sec")
        logger.info(f"  Final Accuracy: {final_metrics['accuracy']:.4f}")
        logger.info(f"  Final F1 Macro: {final_metrics['f1_macro']:.4f}")
        
        return report


def main():
    parser = argparse.ArgumentParser(description='SigLIP fine-tuning on HAR dataset')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to HAR dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/siglip_hf', help='Checkpoint directory')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--model_name', type=str, default='google/siglip-base-patch16-224', help='SigLIP model name')
    
    args = parser.parse_args()
    
    config = {
        'data_dir': args.data_dir,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'checkpoint_dir': args.checkpoint_dir,
        'num_workers': args.num_workers,
        'model_name': args.model_name,
        'weight_decay': 0.01
    }
    
    trainer = SigLIPTrainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
