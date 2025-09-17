#!/usr/bin/env python3
"""
Linear probe on CLIP image encoder for HAR dataset (encoders frozen)
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import clip
from train_clip_har_clean import HARDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClipImageLinearProbe(nn.Module):
    def __init__(self, clip_model: nn.Module, num_classes: int):
        super().__init__()
        self.clip = clip_model
        # freeze all CLIP params
        for p in self.clip.parameters():
            p.requires_grad = False
        self.clip.eval()

        # Determine feature dim from visual projection
        feature_dim = self.clip.visual.output_dim if hasattr(self.clip.visual, 'output_dim') else self.clip.visual.proj.shape[1]
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feats = self.clip.encode_image(images)
            feats = F.normalize(feats, dim=-1)
        # Ensure consistent dtype
        feats = feats.float()
        logits = self.classifier(feats)
        return logits


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device)
            labels = batch['class_id'].to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.numel()
    return correct / max(1, total)


def main():
    p = argparse.ArgumentParser(description='CLIP linear probe on HAR (encoders frozen)')
    p.add_argument('--data_dir', type=str, required=True)
    p.add_argument('--model_name', type=str, default='ViT-B/32')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--checkpoint_dir', type=str, default='./checkpoints/clip_linear_probe')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load CLIP and preprocess
    clip_model, preprocess = clip.load(args.model_name, device=device, jit=False)

    # Dataset: train only (no validation split)
    train_ds = HARDataset(args.data_dir, 'train', preprocess=preprocess)

    # Get num_classes from train_ds
    if hasattr(train_ds, 'class_names'):
        num_classes = len(train_ds.class_names)
    else:
        num_classes = len({s['class_id'] for s in train_ds.samples})

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    # Model
    model = ClipImageLinearProbe(clip_model, num_classes).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))

    # Training
    best_acc = 0.0
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        correct = 0
        for batch in train_loader:
            images = batch['image'].to(device)
            labels = batch['class_id'].to(device)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            total += labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()

        train_loss = total_loss / max(1, total)
        train_acc = correct / max(1, total)

        logger.info(f"Epoch {epoch}: TrainLoss={train_loss:.4f} TrainAcc={train_acc:.4f}")

        # Save best
        is_best = train_acc > best_acc
        if is_best:
            best_acc = train_acc
            state = {
                'epoch': epoch,
                'model_name': args.model_name,
                'num_classes': num_classes,
                'state_dict': model.state_dict(),
                'train_acc': train_acc,
                'config': vars(args),
            }
            torch.save(state, ckpt_dir / 'clip_linear_best.pt')

    logger.info(f"Best TrainAcc: {best_acc:.4f}")


if __name__ == '__main__':
    main()


