#!/usr/bin/env python3
"""
Simple SigLIP performance evaluation with large batch sizes.
"""

import argparse
import logging
import time
import psutil
from pathlib import Path
import json
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# Import the HAR dataset
import sys
sys.path.append('/anvil/projects/x-soc250046/x-sishraq/CLIP/CLIP_model')
from har_dataset import HARDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleSigLIP(nn.Module):
    """Simplified SigLIP model for performance testing"""
    
    def __init__(self, embed_dim=512):
        super().__init__()
        
        # Simple vision encoder (CNN-based for speed)
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, embed_dim)
        )
        
        # Simple text encoder
        self.text_encoder = nn.Sequential(
            nn.Embedding(1000, embed_dim),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Classification head
        self.classifier = nn.Linear(embed_dim, 15)
        
    def forward(self, images, text_tokens):
        # Encode images
        image_features = self.vision_encoder(images)
        image_features = F.normalize(image_features, dim=-1)
        
        # Encode text
        text_features = self.text_encoder(text_tokens)
        text_features = F.normalize(text_features, dim=-1)
        
        # Classification
        class_logits = self.classifier(image_features)
        
        return class_logits


def simple_tokenize(texts, max_length=77):
    """Simple tokenization"""
    token_ids = []
    for text in texts:
        # Simple hash-based tokenization
        tokens = [hash(word) % 1000 for word in text.lower().split()[:max_length]]
        while len(tokens) < max_length:
            tokens.append(0)
        token_ids.append(tokens[:max_length])
    return torch.tensor(token_ids, dtype=torch.long)


def custom_collate_fn(batch):
    """Custom collate function"""
    images = [item['image'] for item in batch]
    texts = [item['text'] for item in batch]
    labels = torch.tensor([item['class_id'] for item in batch])
    return {'images': images, 'texts': texts, 'labels': labels}


def evaluate_batch_size(model, dataset, batch_size, device):
    """Evaluate performance for a specific batch size"""
    logger.info(f"Evaluating batch size: {batch_size}")
    
    # Create data loader
    test_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=custom_collate_fn
    )
    
    model.eval()
    all_predictions = []
    all_labels = []
    total_time = 0.0
    memory_usage = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Batch {batch_size}"):
            # Memory monitoring
            memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
            memory_usage.append(memory_mb)
            
            images = batch['images']
            texts = batch['texts']
            labels = batch['labels'].to(device)
            
            text_tokens = simple_tokenize(texts).to(device)
            
            # Time inference
            start_time = time.time()
            class_logits = model(images, text_tokens)
            inference_time = time.time() - start_time
            
            total_time += inference_time
            
            # Get predictions
            _, predicted = torch.max(class_logits, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    f1_macro = f1_score(all_labels, all_predictions, average='macro')
    throughput = len(dataset) / total_time
    avg_memory = np.mean(memory_usage)
    peak_memory = np.max(memory_usage)
    
    return {
        'batch_size': batch_size,
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'throughput_samples_per_sec': throughput,
        'avg_memory_mb': avg_memory,
        'peak_memory_mb': peak_memory,
        'total_time': total_time
    }


def main():
    parser = argparse.ArgumentParser(description='SigLIP performance evaluation')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to HAR dataset')
    parser.add_argument('--batch_sizes', type=int, nargs='+', 
                        default=[32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32000],
                        help='Batch sizes to evaluate')
    parser.add_argument('--output_file', type=str, default='siglip_performance_results.csv', 
                        help='Output CSV file')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load dataset
    test_dataset = HARDataset(args.data_dir, 'test')
    logger.info(f"Test samples: {len(test_dataset)}")
    
    # Initialize model
    model = SimpleSigLIP().to(device)
    
    # Evaluate different batch sizes
    results = []
    for batch_size in args.batch_sizes:
        try:
            result = evaluate_batch_size(model, test_dataset, batch_size, device)
            results.append(result)
            logger.info(f"Batch {batch_size}: Acc={result['accuracy']:.4f}, "
                       f"Throughput={result['throughput_samples_per_sec']:.2f} samples/sec, "
                       f"Memory={result['avg_memory_mb']:.2f} MB")
            
            # Clear cache
            torch.cuda.empty_cache()
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning(f"Out of memory for batch size {batch_size}, skipping...")
                torch.cuda.empty_cache()
                continue
            else:
                raise e
    
    # Save results
    with open(args.output_file, 'w', newline='') as csvfile:
        fieldnames = results[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    logger.info(f"Results saved to {args.output_file}")
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("SIGLIP PERFORMANCE EVALUATION SUMMARY")
    logger.info("="*60)
    logger.info(f"{'Batch Size':<12} {'Accuracy':<10} {'Throughput':<15} {'Memory (MB)':<12}")
    logger.info("-" * 60)
    
    for result in results:
        logger.info(f"{result['batch_size']:<12} {result['accuracy']:<10.4f} "
                   f"{result['throughput_samples_per_sec']:<15.2f} {result['avg_memory_mb']:<12.2f}")
    
    logger.info("="*60)


if __name__ == '__main__':
    main()
