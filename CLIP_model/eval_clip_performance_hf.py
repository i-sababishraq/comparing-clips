#!/usr/bin/env python3
"""
CLIP zero-shot performance evaluation with large batch size using Hugging Face Transformers.
Reports accuracy, throughput (samples/sec), and memory usage (MB).
"""

import argparse
import logging
import time
import psutil
from pathlib import Path
from typing import Dict, List
import json
import csv

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

from transformers import CLIPModel, CLIPProcessor

import sys
sys.path.append('/anvil/projects/x-soc250046/x-sishraq/CLIP/CLIP_model')
from har_dataset import HARDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CLIPPerformanceEvaluator:
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model_name = config.get('model_name', 'openai/clip-vit-base-patch32')
        logger.info(f"Loading CLIP model: {model_name}")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)

        self.test_dataset = HARDataset(config['data_dir'], 'test')
        self.class_names = self.test_dataset.class_names
        self.class_prompts = [self._make_prompt(cn) for cn in self.class_names]

        self.memory_usage = []

    def _make_prompt(self, class_name: str) -> str:
        clean_name = class_name.replace('_', ' ').replace('-', ' ').lower()
        return f"a photo of a person {clean_name}"

    def collate_fn(self, batch):
        images = [item['image'] for item in batch]
        labels = torch.tensor([item['class_id'] for item in batch])
        return {'images': images, 'labels': labels}

    def process_batch(self, images):
        inputs = self.processor(images=images, text=self.class_prompts, return_tensors='pt', padding=True)
        return {k: v.to(self.device) for k, v in inputs.items()}

    def evaluate_bs(self, batch_size: int) -> Dict[str, float]:
        loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=self.collate_fn)
        all_preds, all_labels = [], []
        total_time = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Batch size {batch_size}"):
                self.memory_usage.append(psutil.Process().memory_info().rss / 1024 / 1024)
                images = batch['images']
                labels = batch['labels'].to(self.device)
                inputs = self.process_batch(images)

                start = time.time()
                outputs = self.model(**inputs)
                logits = outputs.logits_per_image
                probs = torch.softmax(logits, dim=-1)
                infer_t = time.time() - start

                total_time += infer_t
                total_samples += len(images)

                preds = probs.argmax(dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        f1_macro = f1_score(all_labels, all_preds, average='macro')
        throughput = total_samples / total_time if total_time > 0 else 0.0
        avg_mem = float(np.mean(self.memory_usage)) if self.memory_usage else 0.0
        peak_mem = float(np.max(self.memory_usage)) if self.memory_usage else 0.0

        return {
            'batch_size': batch_size,
            'accuracy': acc,
            'f1_macro': f1_macro,
            'throughput_samples_per_sec': throughput,
            'avg_memory_mb': avg_mem,
            'peak_memory_mb': peak_mem,
            'total_samples': total_samples,
            'total_time_sec': total_time,
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--batch_size', type=int, default=4096)
    ap.add_argument('--model_name', type=str, default='openai/clip-vit-base-patch32')
    ap.add_argument('--output_dir', type=str, default='/anvil/projects/x-soc250046/x-sishraq/CLIP/results/clip')
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    evaluator = CLIPPerformanceEvaluator({
        'data_dir': args.data_dir,
        'model_name': args.model_name,
    })

    logger.info("Starting CLIP zero-shot performance evaluation...")
    results = evaluator.evaluate_bs(args.batch_size)

    csv_path = out_dir / 'clip_performance_results.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(results.keys()))
        writer.writeheader()
        writer.writerow(results)
    logger.info(f"Saved results to {csv_path}")

    logger.info("\nCLIP ZERO-SHOT SUMMARY")
    for k, v in results.items():
        if isinstance(v, float):
            logger.info(f"{k}: {v:.4f}")
        else:
            logger.info(f"{k}: {v}")


if __name__ == '__main__':
    main()


