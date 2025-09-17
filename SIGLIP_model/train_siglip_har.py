#!/usr/bin/env python3
"""
Minimal SigLIP-style fine-tuning on the HAR dataset using OpenAI CLIP encoders.

Loss: symmetric sigmoid BCE over image-text similarity matrix (diagonal positives).
"""

import argparse
from pathlib import Path
import logging
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import clip

# Reuse the dataset from the CLIP trainer
import sys
from pathlib import Path as _Path
_clip_model_dir = _Path(__file__).resolve().parents[1] / 'CLIP_model'
sys.path.append(str(_clip_model_dir))
from train_clip_har import HARDataset  # noqa: E402


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='SigLIP-style fine-tuning on HAR')
    p.add_argument('--data_dir', type=str, required=True)
    p.add_argument('--model_name', type=str, default='ViT-B/32')
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--learning_rate', type=float, default=1e-5)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--checkpoint_dir', type=str, default='./checkpoints/siglip_har')
    return p.parse_args()


class SigLIPTrainer:
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Model
        self.model, self.preprocess = clip.load(cfg['model_name'], device=self.device, jit=False)
        self.model.train()

        # Data
        self.train_dataset = HARDataset(cfg['data_dir'], 'train', preprocess=self.preprocess)
        self.val_dataset = None
        try:
            self.val_dataset = HARDataset(cfg['data_dir'], 'val', preprocess=self.preprocess)
        except ValueError:
            try:
                self.val_dataset = HARDataset(cfg['data_dir'], 'test', preprocess=self.preprocess)
            except ValueError:
                self.val_dataset = None

        # If no val set, create a small split from train
        if (self.val_dataset is None) or (len(self.val_dataset) == 0):
            val_ratio = 0.1
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

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=cfg['batch_size'], shuffle=True,
            num_workers=cfg['num_workers'], pin_memory=True, drop_last=True
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=cfg['batch_size'], shuffle=False,
            num_workers=cfg['num_workers'], pin_memory=True
        ) if (self.val_dataset is not None and len(self.val_dataset) > 0) else None

        # Optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg['learning_rate'], betas=(0.9, 0.98), eps=1e-6, weight_decay=0.1)

        # Loss
        self.bce_logits = nn.BCEWithLogitsLoss()

        # IO
        self.ckpt_dir = Path(cfg['checkpoint_dir'])
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        logger.info('Configuration:')
        for k, v in cfg.items():
            logger.info(f'  {k}: {v}')
        logger.info(f"Using device: {self.device}")
        logger.info(f"Train samples: {len(self.train_dataset)}")
        logger.info(f"Val samples: {len(self.val_dataset) if self.val_dataset is not None else 0}")

    def sigmoid_contrastive_loss(self, logits_ij: torch.Tensor, logits_ti: torch.Tensor) -> torch.Tensor:
        """SigLIP-style symmetric BCE over similarity matrices.

        Targets are identity matrices (diagonal positives). Shape: [B, B].
        """
        bsz = logits_ij.size(0)
        target = torch.eye(bsz, device=logits_ij.device)
        loss_i = self.bce_logits(logits_ij, target)
        loss_t = self.bce_logits(logits_ti, target)
        return 0.5 * (loss_i + loss_t)

    @torch.no_grad()
    def evaluate(self) -> float:
        """Evaluate accuracy by class-prompt zero-shot classification.

        Build one prompt per class name (from the base dataset) and classify
        each validation image by argmax over class logits.
        """
        if self.val_loader is None:
            logger.info('No validation data; skipping eval.')
            return 0.0

        # Recover underlying dataset to access class names when using Subset
        base_val_ds = self.val_dataset.dataset if hasattr(self.val_dataset, 'dataset') else self.val_dataset
        class_names = getattr(base_val_ds, 'class_names', None)
        if not class_names:
            logger.info('No class names available; skipping eval.')
            return 0.0

        # Prepare class prompts and encode once
        prompts = [f"a photo of {name}" for name in class_names]
        text_tokens = clip.tokenize(prompts, truncate=True).to(self.device)

        self.model.eval()
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            logit_scale = self.model.logit_scale.exp()

        total, correct = 0, 0
        for batch in self.val_loader:
            images = batch['image'].to(self.device)
            labels = batch['class_id'].to(self.device)
            with torch.no_grad():
                image_features = self.model.encode_image(images)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                logits = logit_scale * image_features @ text_features.t()
                preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.numel()

        acc = (correct / total) if total > 0 else 0.0
        self.model.train()
        return acc

    def train(self):
        for epoch in range(1, self.cfg['epochs'] + 1):
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.cfg['epochs']}")
            running = 0.0
            for i, batch in enumerate(pbar):
                images = batch['image'].to(self.device)
                texts = batch['text']
                text_tokens = clip.tokenize(texts, truncate=True).to(self.device)

                logits_ij, logits_ti = self.model(images, text_tokens)
                loss = self.sigmoid_contrastive_loss(logits_ij, logits_ti)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                running += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{running/(i+1):.4f}")

            # quick eval metric (optional)
            acc = self.evaluate()
            logger.info(f"Epoch {epoch} done. TrainLoss={running/len(self.train_loader):.4f} ValDiagAcc={acc:.4f}")

            ckpt_path = self.ckpt_dir / f"siglip_har_epoch_{epoch}.pt"
            torch.save({'epoch': epoch, 'state_dict': self.model.state_dict()}, ckpt_path)
            logger.info(f"Saved checkpoint: {ckpt_path}")


def main():
    args = parse_args()
    cfg = {
        'data_dir': args.data_dir,
        'model_name': args.model_name,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'num_workers': args.num_workers,
        'checkpoint_dir': args.checkpoint_dir,
    }
    trainer = SigLIPTrainer(cfg)
    trainer.train()


if __name__ == '__main__':
    main()


