#!/usr/bin/env python3
"""
LoRA (Low-Rank Adaptation) fine-tuning for CLIP on HAR dataset
Parameter-efficient fine-tuning that should avoid catastrophic forgetting
"""

import argparse
import logging
from pathlib import Path
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import clip
from train_clip_har_clean import HARDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LoRALayer(nn.Module):
    """LoRA layer for parameter-efficient fine-tuning"""
    
    def __init__(self, in_features, out_features, rank=16, alpha=32, dropout=0.1):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: [batch_size, in_features]
        # Apply LoRA: x @ A^T @ B^T
        x = x.float()  # Ensure consistent dtype
        result = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        return result * self.scaling


class LoRACLIP(nn.Module):
    """CLIP with LoRA adapters for fine-tuning"""
    
    def __init__(self, clip_model, rank=16, alpha=32, dropout=0.1):
        super().__init__()
        self.clip = clip_model
        
        # Freeze original CLIP parameters
        for param in self.clip.parameters():
            param.requires_grad = False
        
        # Add LoRA adapters to key layers
        self.visual_lora = LoRALayer(
            self.clip.visual.output_dim, 
            self.clip.visual.output_dim,
            rank=rank, alpha=alpha, dropout=dropout
        )
        
        self.text_lora = LoRALayer(
            self.clip.text_projection.shape[1],
            self.clip.text_projection.shape[0], 
            rank=rank, alpha=alpha, dropout=dropout
        )
        
        # Learnable temperature scaling
        self.temp_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))
        
    def encode_image(self, images):
        """Encode images with LoRA adaptation"""
        with torch.no_grad():
            # Get frozen features
            image_features = self.clip.encode_image(images)
            image_features = F.normalize(image_features, dim=-1)
        
        # Apply LoRA adaptation
        adapted_features = self.visual_lora(image_features)
        return image_features + adapted_features
    
    def encode_text(self, text_tokens):
        """Encode text with LoRA adaptation"""
        with torch.no_grad():
            # Get frozen features
            text_features = self.clip.encode_text(text_tokens)
            text_features = F.normalize(text_features, dim=-1)
        
        # Apply LoRA adaptation
        adapted_features = self.text_lora(text_features)
        return text_features + adapted_features
    
    def forward(self, images, text_tokens):
        """Forward pass with LoRA adaptations"""
        image_features = self.encode_image(images)
        text_features = self.encode_text(text_tokens)
        
        # Compute logits with learnable temperature
        logit_scale = self.temp_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logits_per_image.T
        
        return logits_per_image, logits_per_text


def compute_contrastive_loss(logits_per_image, logits_per_text):
    """Compute contrastive loss (InfoNCE)"""
    batch_size = logits_per_image.shape[0]
    labels = torch.arange(batch_size, device=logits_per_image.device)
    
    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    
    return (loss_i + loss_t) / 2


def main():
    parser = argparse.ArgumentParser(description='LoRA fine-tuning for CLIP on HAR')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='ViT-B/32')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--rank', type=int, default=16)
    parser.add_argument('--alpha', type=float, default=32.0)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/clip_lora')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load CLIP
    clip_model, preprocess = clip.load(args.model_name, device=device, jit=False)
    
    # Create LoRA model
    model = LoRACLIP(clip_model, rank=args.rank, alpha=args.alpha, dropout=args.dropout).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Trainable ratio: {trainable_params/total_params:.2%}")

    # Data (train only)
    train_ds = HARDataset(args.data_dir, 'train', preprocess=preprocess)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    # Optimizer (only trainable parameters)
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

        for batch in train_loader:
            images = batch['image'].to(device)
            texts = batch['text']
            text_tokens = clip.tokenize(texts, truncate=True).to(device)

            logits_per_image, logits_per_text = model(images, text_tokens)
            loss = compute_contrastive_loss(logits_per_image, logits_per_text)

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
            torch.save(state, ckpt_dir / 'clip_lora_best.pt')

    logger.info(f"Best TrainLoss: {best_loss:.4f}")


if __name__ == '__main__':
    main()
