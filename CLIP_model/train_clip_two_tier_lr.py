#!/usr/bin/env python3
"""
Conservative two-tier LR fine-tuning for CLIP on HAR

- Freeze text encoder entirely
- Tiny LR for visual encoder (default 1e-6)
- Clamp logit_scale and keep it frozen
- Train with contrastive loss (InfoNCE) on train split only
"""

import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import clip
from train_clip_har_clean import HARDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_contrastive_loss(logits_per_image, logits_per_text):
    batch_size = logits_per_image.shape[0]
    labels = torch.arange(batch_size, device=logits_per_image.device)
    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    return 0.5 * (loss_i + loss_t)


def main():
    p = argparse.ArgumentParser(description='Two-tier LR CLIP fine-tuning on HAR')
    p.add_argument('--data_dir', type=str, required=True)
    p.add_argument('--model_name', type=str, default='ViT-B/32')
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--lr_visual', type=float, default=1e-6)
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--checkpoint_dir', type=str, default='./checkpoints/clip_two_tier_lr_3e')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load CLIP
    model, preprocess = clip.load(args.model_name, device=device, jit=False)

    # Freeze text encoder and logit_scale
    for n, p_ in model.named_parameters():
        if n.startswith('transformer.') or n.startswith('token_embedding') or n.startswith('positional_embedding') or n.startswith('text_projection'):
            p_.requires_grad = False
    # logit_scale is a parameter; keep stable (no grad) and clamp
    if isinstance(model.logit_scale, torch.Tensor):
        model.logit_scale.requires_grad = False
        with torch.no_grad():
            max_val = torch.log(torch.tensor(100.0, device=model.logit_scale.device))
            model.logit_scale.clamp_(max=max_val)

    # Build optimizer for visual params only
    visual_params = []
    for n, p_ in model.named_parameters():
        if n.startswith('visual') and p_.requires_grad:
            visual_params.append(p_)

    optimizer = torch.optim.AdamW(
        [
            { 'params': visual_params, 'lr': args.lr_visual, 'weight_decay': args.weight_decay },
        ],
        betas=(0.9, 0.98), eps=1e-6
    )

    # Data (train only)
    train_ds = HARDataset(args.data_dir, 'train', preprocess=preprocess)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

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
            torch.nn.utils.clip_grad_norm_(visual_params, max_norm=1.0)
            optimizer.step()

            # Clamp logit_scale for safety
            with torch.no_grad():
                if isinstance(model.logit_scale, torch.Tensor):
                    max_val = torch.log(torch.tensor(100.0, device=model.logit_scale.device))
                    model.logit_scale.clamp_(max=max_val)

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
            torch.save(state, ckpt_dir / 'clip_two_tier_lr_best.pt')

    logger.info(f"Best TrainLoss: {best_loss:.4f}")


if __name__ == '__main__':
    main()


