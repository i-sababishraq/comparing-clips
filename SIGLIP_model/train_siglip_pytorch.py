#!/usr/bin/env python3
"""
SigLIP fine-tuning on HAR dataset using PyTorch implementation.
Based on the original SigLIP paper and implementation.
"""

import argparse
import logging
import time
import psutil
import os
from pathlib import Path
from typing import Dict, List, Tuple
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Import the HAR dataset
import sys
sys.path.append('/anvil/projects/x-soc250046/x-sishraq/CLIP/CLIP_model')
from har_dataset import HARDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SigLIPModel(nn.Module):
    """SigLIP model implementation for PyTorch"""
    
    def __init__(self, vision_model_name: str = "vit_base_patch16_224", 
                 text_model_name: str = "bert_base", 
                 embed_dim: int = 512,
                 temperature_init: float = 1.0):
        super().__init__()
        
        # Vision encoder (ViT)
        self.vision_encoder = self._create_vision_encoder(vision_model_name, embed_dim)
        
        # Text encoder (BERT-style)
        self.text_encoder = self._create_text_encoder(text_model_name, embed_dim)
        
        # Projection layers
        self.vision_proj = nn.Linear(embed_dim, embed_dim)
        self.text_proj = nn.Linear(embed_dim, embed_dim)
        
        # Temperature parameter
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(temperature_init)))
        
        # Classification head
        self.classifier = nn.Linear(embed_dim, 15)  # 15 action classes
        
    def _create_vision_encoder(self, model_name: str, embed_dim: int):
        """Create vision encoder (simplified ViT)"""
        if model_name == "vit_base_patch16_224":
            # Simplified ViT implementation
            return SimpleViT(
                image_size=224,
                patch_size=16,
                num_classes=embed_dim,
                dim=768,
                depth=12,
                heads=12,
                mlp_dim=3072
            )
        else:
            raise ValueError(f"Unsupported vision model: {model_name}")
    
    def _create_text_encoder(self, model_name: str, embed_dim: int):
        """Create text encoder (simplified BERT)"""
        if model_name == "bert_base":
            return SimpleBERT(vocab_size=30522, embed_dim=embed_dim)
        else:
            raise ValueError(f"Unsupported text model: {model_name}")
    
    def encode_image(self, images):
        """Encode images to features"""
        features = self.vision_encoder(images)
        return self.vision_proj(features)
    
    def encode_text(self, text_tokens):
        """Encode text to features"""
        features = self.text_encoder(text_tokens)
        return self.text_proj(features)
    
    def forward(self, images, text_tokens):
        """Forward pass"""
        # Encode images and text
        image_features = self.encode_image(images)
        text_features = self.encode_text(text_tokens)
        
        # Normalize features
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        # Compute similarity
        logit_scale = self.logit_scale.exp()
        logits_per_image = torch.matmul(image_features, text_features.t()) * logit_scale
        logits_per_text = logits_per_image.t()
        
        # Classification
        class_logits = self.classifier(image_features)
        
        return logits_per_image, logits_per_text, class_logits


class SimpleViT(nn.Module):
    """Simplified Vision Transformer"""
    
    def __init__(self, image_size=224, patch_size=16, num_classes=1000, 
                 dim=768, depth=12, heads=12, mlp_dim=3072):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        
        # Transformer blocks
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim, heads, mlp_dim, batch_first=True),
            depth
        )
        
        # Head
        self.head = nn.Linear(dim, num_classes)
        
    def forward(self, x):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, dim, H/patch_size, W/patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, dim)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Transformer
        x = self.transformer(x)
        
        # Return CLS token
        return x[:, 0]  # (B, dim)


class SimpleBERT(nn.Module):
    """Simplified BERT encoder"""
    
    def __init__(self, vocab_size=30522, embed_dim=512, max_length=77):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_length = max_length
        
        # Embeddings
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_length, embed_dim)
        
        # Transformer
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, 8, 2048, batch_first=True),
            6
        )
        
    def forward(self, input_ids):
        B, seq_len = input_ids.shape
        
        # Token embeddings
        x = self.token_embed(input_ids)
        
        # Position embeddings
        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = x + self.pos_embed(pos_ids)
        
        # Transformer
        x = self.transformer(x)
        
        # Return [CLS] token (first token)
        return x[:, 0]  # (B, embed_dim)


class SigLIPTrainer:
    """SigLIP trainer with performance monitoring"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = SigLIPModel(
            vision_model_name=config['vision_model'],
            text_model_name=config['text_model'],
            embed_dim=config['embed_dim'],
            temperature_init=config['temperature_init']
        ).to(self.device)
        
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
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Loss function (SigLIP uses BCE with sigmoid)
        self.criterion = nn.BCEWithLogitsLoss()
        
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
    
    def sigmoid_contrastive_loss(self, logits_per_image, logits_per_text):
        """SigLIP contrastive loss with sigmoid"""
        batch_size = logits_per_image.size(0)
        labels = torch.eye(batch_size, device=logits_per_image.device)
        
        loss_i = self.criterion(logits_per_image, labels)
        loss_t = self.criterion(logits_per_text, labels)
        
        return 0.5 * (loss_i + loss_t)
    
    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
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
            
            # Simple text tokenization (for demo purposes)
            # In practice, you'd use a proper tokenizer
            text_tokens = self.simple_tokenize(texts).to(self.device)
            
            # Forward pass
            logits_per_image, logits_per_text, class_logits = self.model(images, text_tokens)
            
            # Losses
            contrastive_loss = self.sigmoid_contrastive_loss(logits_per_image, logits_per_text)
            classification_loss = F.cross_entropy(class_logits, labels)
            total_loss_batch = contrastive_loss + classification_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss_batch.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += total_loss_batch.item()
            _, predicted = torch.max(class_logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{total_loss_batch.item():.4f}',
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
    
    def simple_tokenize(self, texts):
        """Simple tokenization for demo purposes"""
        # This is a very basic tokenization
        # In practice, you'd use a proper tokenizer
        max_length = 77
        vocab_size = 30522
        
        # Simple word-based tokenization
        token_ids = []
        for text in texts:
            # Convert to lowercase and split
            words = text.lower().split()[:max_length-1]
            # Simple hash-based tokenization
            tokens = [hash(word) % vocab_size for word in words]
            # Pad to max_length
            while len(tokens) < max_length:
                tokens.append(0)  # PAD token
            token_ids.append(tokens[:max_length])
        
        return torch.tensor(token_ids, dtype=torch.long)
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model"""
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Evaluating"):
                images = batch['images']
                texts = batch['texts']
                labels = batch['labels'].to(self.device)
                
                text_tokens = self.simple_tokenize(texts).to(self.device)
                _, _, class_logits = self.model(images, text_tokens)
                
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
            'model_state_dict': self.model.state_dict(),
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
            'model': 'SigLIP',
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
    parser.add_argument('--embed_dim', type=int, default=512, help='Embedding dimension')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/siglip_pytorch', help='Checkpoint directory')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    
    args = parser.parse_args()
    
    config = {
        'data_dir': args.data_dir,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'embed_dim': args.embed_dim,
        'checkpoint_dir': args.checkpoint_dir,
        'num_workers': args.num_workers,
        'vision_model': 'vit_base_patch16_224',
        'text_model': 'bert_base',
        'temperature_init': 1.0,
        'weight_decay': 0.01
    }
    
    trainer = SigLIPTrainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
