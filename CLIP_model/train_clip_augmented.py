#!/usr/bin/env python3
"""
CLIP fine-tuning with comprehensive data augmentation
Includes image augmentation, text augmentation, and hard pair mining
"""

import argparse
import logging
from pathlib import Path
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image

import clip
from train_clip_har_clean import HARDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AugmentedHARDataset(Dataset):
    """HAR Dataset with comprehensive augmentation"""
    
    def __init__(self, data_dir: str, split: str = 'train', preprocess=None, augment=True):
        self.data_dir = Path(data_dir)
        self.split = split
        self.preprocess = preprocess
        self.augment = augment
        
        # Load base dataset
        self.base_dataset = HARDataset(data_dir, split, preprocess=None)
        
        # Image augmentation pipeline
        if augment and split == 'train':
            self.image_augment = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomRotation(degrees=10),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.image_augment = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        # Text augmentation templates
        self.text_templates = [
            "a photo of a person {}",
            "someone {}",
            "a person performing {}",
            "{} action",
            "human {}",
            "a video frame of someone {}",
            "footage of a person {}",
            "an image showing {}",
            "a picture of {}",
            "{} in progress"
        ]
        
        # Hard pair mining setup
        self.hard_pairs = []
        self.mining_epoch = 0
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        sample = self.base_dataset[idx]
        
        # Load and augment image
        try:
            image = Image.open(sample['image_path']).convert('RGB')
            if self.augment and self.split == 'train':
                image = self.image_augment(image)
            else:
                image = self.image_augment(image)
        except Exception as e:
            logger.warning(f"Could not load image {sample['image_path']}: {e}")
            image = torch.zeros(3, 224, 224)
        
        # Augment text
        if self.augment and self.split == 'train':
            text = self.augment_text(sample['class_name'])
        else:
            text = sample['text']
        
        return {
            'image': image,
            'text': text,
            'class_id': sample['class_id'],
            'class_name': sample['class_name']
        }
    
    def augment_text(self, class_name):
        """Augment text with multiple templates"""
        clean_name = class_name.replace('_', ' ').replace('-', ' ').lower()
        
        # Randomly select template
        template = random.choice(self.text_templates)
        return template.format(clean_name)
    
    def update_hard_pairs(self, similarities, batch_indices):
        """Update hard pairs for mining"""
        # Find hardest negatives (highest similarity with wrong class)
        batch_size = similarities.shape[0]
        labels = torch.arange(batch_size, device=similarities.device)
        
        # Mask out correct pairs
        mask = torch.eye(batch_size, device=similarities.device).bool()
        similarities_masked = similarities.masked_fill(mask, -float('inf'))
        
        # Find hardest negatives
        hard_negatives = similarities_masked.max(dim=1)[1]
        
        # Store hard pairs
        for i, (pos_idx, neg_idx) in enumerate(zip(batch_indices, hard_negatives)):
            if pos_idx < len(self.base_dataset) and neg_idx < len(self.base_dataset):
                self.hard_pairs.append((pos_idx, neg_idx.item()))


class HardPairMiner:
    """Hard pair mining for contrastive learning"""
    
    def __init__(self, dataset, mining_ratio=0.3):
        self.dataset = dataset
        self.mining_ratio = mining_ratio
        self.hard_pairs = []
    
    def mine_hard_pairs(self, model, dataloader, device):
        """Mine hard pairs from current batch"""
        model.eval()
        hard_pairs = []
        
        with torch.no_grad():
            for batch in dataloader:
                images = batch['image'].to(device)
                texts = batch['text']
                text_tokens = clip.tokenize(texts, truncate=True).to(device)
                
                # Get features
                image_features = model.encode_image(images)
                text_features = model.encode_text(text_tokens)
                
                # Compute similarities
                similarities = image_features @ text_features.T
                
                # Find hard negatives
                batch_size = similarities.shape[0]
                labels = torch.arange(batch_size, device=device)
                
                # Mask correct pairs
                mask = torch.eye(batch_size, device=device).bool()
                similarities_masked = similarities.masked_fill(mask, -float('inf'))
                
                # Get hardest negatives
                hard_negatives = similarities_masked.argmax(dim=1)
                
                # Store hard pairs
                for i, neg_idx in enumerate(hard_negatives):
                    hard_pairs.append((i, neg_idx.item()))
        
        # Update dataset with hard pairs
        self.dataset.hard_pairs = hard_pairs[:int(len(hard_pairs) * self.mining_ratio)]
        model.train()


def compute_contrastive_loss(logits_per_image, logits_per_text, hard_pairs=None):
    """Compute contrastive loss with optional hard pair weighting"""
    batch_size = logits_per_image.shape[0]
    labels = torch.arange(batch_size, device=logits_per_image.device)
    
    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    
    base_loss = (loss_i + loss_t) / 2
    
    # Add hard pair loss if available
    if hard_pairs and len(hard_pairs) > 0:
        hard_loss = 0.0
        for pos_idx, neg_idx in hard_pairs:
            if pos_idx < batch_size and neg_idx < batch_size:
                # Increase loss for hard negatives
                hard_loss += F.cross_entropy(
                    logits_per_image[pos_idx:pos_idx+1], 
                    labels[neg_idx:neg_idx+1]
                )
        base_loss += 0.1 * hard_loss / len(hard_pairs)
    
    return base_loss


def main():
    parser = argparse.ArgumentParser(description='Augmented CLIP fine-tuning on HAR')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='ViT-B/32')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/clip_augmented')
    parser.add_argument('--mining_ratio', type=float, default=0.3)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load CLIP
    model, preprocess = clip.load(args.model_name, device=device, jit=False)
    
    # Freeze some layers to prevent catastrophic forgetting
    for name, param in model.named_parameters():
        if 'transformer' in name or 'token_embedding' in name:
            param.requires_grad = False
    
    # Data with augmentation
    train_ds = AugmentedHARDataset(args.data_dir, 'train', preprocess=preprocess, augment=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    # Hard pair miner
    miner = HardPairMiner(train_ds, mining_ratio=args.mining_ratio)

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.98), eps=1e-6
    )

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    best_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        num_batches = 0
        correct_diag = 0
        total_samples = 0

        # Mine hard pairs every 2 epochs
        if epoch % 2 == 0:
            miner.mine_hard_pairs(model, train_loader, device)

        for batch in train_loader:
            images = batch['image'].to(device)
            texts = batch['text']
            text_tokens = clip.tokenize(texts, truncate=True).to(device)

            logits_per_image, logits_per_text = model(images, text_tokens)
            
            # Get hard pairs for this batch
            hard_pairs = train_ds.hard_pairs if hasattr(train_ds, 'hard_pairs') else None
            
            loss = compute_contrastive_loss(logits_per_image, logits_per_text, hard_pairs)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
            optimizer.step()

            # Metrics
            total_loss += loss.item()
            num_batches += 1
            preds = logits_per_image.argmax(dim=1)
            labels = torch.arange(images.size(0), device=preds.device)
            correct_diag += (preds == labels).sum().item()
            total_samples += images.size(0)

        avg_loss = total_loss / max(1, num_batches)
        diag_acc = correct_diag / max(1, total_samples)
        logger.info(f"Epoch {epoch}: TrainLoss={avg_loss:.4f} DiagAcc={diag_acc:.4f}")

        # Save best by loss
        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss
            state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_loss,
                'config': vars(args),
            }
            torch.save(state, ckpt_dir / 'clip_augmented_best.pt')

    logger.info(f"Best TrainLoss: {best_loss:.4f}")


if __name__ == '__main__':
    main()
