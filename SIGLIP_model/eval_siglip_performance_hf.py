#!/usr/bin/env python3
"""
SigLIP performance evaluation with large batch sizes using Hugging Face Transformers.
Based on the merveenoyan/siglip repository approach.
"""

import argparse
import logging
import time
import psutil
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
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Hugging Face imports
from transformers import SiglipModel, SiglipProcessor

# Import the HAR dataset
import sys
sys.path.append('/anvil/projects/x-soc250046/x-sishraq/CLIP/CLIP_model')
from har_dataset import HARDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SigLIPPerformanceEvaluator:
    """SigLIP performance evaluator using Hugging Face Transformers"""
    
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
        self.model.eval()
        
        # Build zero-shot class prompts from dataset class names
        self.class_names = None
        self.class_prompts = None
        
        # Load dataset
        self.test_dataset = HARDataset(config['data_dir'], 'test')
        self.class_names = self.test_dataset.class_names
        # Simple prompt as used in repo/notebook
        self.class_prompts = [self._make_prompt(cn) for cn in self.class_names]
        
        # Performance tracking
        self.memory_usage = []
        self.inference_times = []
        self.throughput_metrics = []
        
        logger.info(f"Model initialized on {self.device}")
        logger.info(f"Test samples: {len(self.test_dataset)}")
    
    def custom_collate_fn(self, batch):
        """Custom collate function for HAR dataset"""
        images = [item['image'] for item in batch]
        labels = torch.tensor([item['class_id'] for item in batch])
        
        return {
            'images': images,
            'labels': labels
        }
    
    def process_batch(self, images):
        """Process batch using SigLIP processor"""
        # Process images and text with SigLIP processor
        inputs = self.processor(
            images=images, 
            text=self.class_prompts, 
            return_tensors="pt", 
            padding=True
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        return inputs
    
    def evaluate_batch_size(self, batch_size: int) -> Dict[str, float]:
        """Evaluate performance for a specific batch size"""
        logger.info(f"Evaluating batch size: {batch_size}")
        
        # Create data loader
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,  # Use 0 for accurate timing
            collate_fn=self.custom_collate_fn
        )
        
        # Warm up
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                if i >= 3:  # Warm up for 3 batches
                    break
                images = batch['images']
                inputs = self.process_batch(images)
                outputs = self.model(**inputs)
                _ = torch.sigmoid(outputs.logits_per_image)
        
        # Clear cache
        torch.cuda.empty_cache()
        
        # Actual evaluation
        all_predictions = []
        all_labels = []
        total_inference_time = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Batch size {batch_size}"):
                # Memory monitoring
                memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
                self.memory_usage.append(memory_mb)
                
                images = batch['images']
                labels = batch['labels'].to(self.device)
                
                # Process batch
                inputs = self.process_batch(images)
                
                # Time inference
                start_time = time.time()
                outputs = self.model(**inputs)
                logits = outputs.logits_per_image  # [batch_size, num_classes]
                probs = torch.sigmoid(logits)
                inference_time = time.time() - start_time
                
                total_inference_time += inference_time
                total_samples += len(images)
                
                # Get predictions
                _, predicted = torch.max(probs, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Store timing
                self.inference_times.append(inference_time)
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        f1_macro = f1_score(all_labels, all_predictions, average='macro')
        f1_weighted = f1_score(all_labels, all_predictions, average='weighted')
        
        # Performance metrics
        avg_inference_time = total_inference_time / len(test_loader)
        throughput = total_samples / total_inference_time  # samples per second
        avg_memory = np.mean(self.memory_usage) if self.memory_usage else 0
        peak_memory = np.max(self.memory_usage) if self.memory_usage else 0
        
        results = {
            'batch_size': batch_size,
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'avg_inference_time': avg_inference_time,
            'throughput_samples_per_sec': throughput,
            'avg_memory_mb': avg_memory,
            'peak_memory_mb': peak_memory,
            'total_samples': total_samples,
            'total_time': total_inference_time
        }
        
        logger.info(f"Batch size {batch_size} results:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  F1 Macro: {f1_macro:.4f}")
        logger.info(f"  Throughput: {throughput:.2f} samples/sec")
        logger.info(f"  Avg Memory: {avg_memory:.2f} MB")
        logger.info(f"  Peak Memory: {peak_memory:.2f} MB")
        
        return results

    def _make_prompt(self, class_name: str) -> str:
        clean_name = class_name.replace('_', ' ').replace('-', ' ').lower()
        return f"a photo of a person {clean_name}"
    
    def evaluate_all_batch_sizes(self) -> List[Dict[str, float]]:
        """Evaluate performance across different batch sizes"""
        batch_sizes = self.config['batch_sizes']
        all_results = []
        
        for batch_size in batch_sizes:
            try:
                results = self.evaluate_batch_size(batch_size)
                all_results.append(results)
                
                # Clear memory between evaluations
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning(f"Out of memory for batch size {batch_size}, skipping...")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
        
        return all_results
    
    def save_results(self, results: List[Dict[str, float]], output_file: str):
        """Save results to CSV file"""
        with open(output_file, 'w', newline='') as csvfile:
            fieldnames = results[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        logger.info(f"Results saved to {output_file}")
    
    def generate_performance_plots(self, results: List[Dict[str, float]], output_dir: str):
        """Generate performance visualization plots"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract data
        batch_sizes = [r['batch_size'] for r in results]
        accuracies = [r['accuracy'] for r in results]
        throughputs = [r['throughput_samples_per_sec'] for r in results]
        memory_usage = [r['avg_memory_mb'] for r in results]
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy vs Batch Size
        axes[0, 0].plot(batch_sizes, accuracies, 'bo-')
        axes[0, 0].set_xlabel('Batch Size')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_title('SigLIP: Accuracy vs Batch Size')
        axes[0, 0].grid(True)
        
        # Throughput vs Batch Size
        axes[0, 1].plot(batch_sizes, throughputs, 'ro-')
        axes[0, 1].set_xlabel('Batch Size')
        axes[0, 1].set_ylabel('Throughput (samples/sec)')
        axes[0, 1].set_title('SigLIP: Throughput vs Batch Size')
        axes[0, 1].grid(True)
        
        # Memory Usage vs Batch Size
        axes[1, 0].plot(batch_sizes, memory_usage, 'go-')
        axes[1, 0].set_xlabel('Batch Size')
        axes[1, 0].set_ylabel('Memory Usage (MB)')
        axes[1, 0].set_title('SigLIP: Memory Usage vs Batch Size')
        axes[1, 0].grid(True)
        
        # Throughput vs Memory Trade-off
        axes[1, 1].scatter(memory_usage, throughputs, c=batch_sizes, cmap='viridis')
        axes[1, 1].set_xlabel('Memory Usage (MB)')
        axes[1, 1].set_ylabel('Throughput (samples/sec)')
        axes[1, 1].set_title('SigLIP: Throughput vs Memory Trade-off')
        axes[1, 1].grid(True)
        
        # Add colorbar for batch size
        cbar = plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1])
        cbar.set_label('Batch Size')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'siglip_performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Performance plots saved to {output_dir / 'siglip_performance_analysis.png'}")


def main():
    parser = argparse.ArgumentParser(description='SigLIP performance evaluation')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to HAR dataset')
    parser.add_argument('--batch_sizes', type=int, nargs='+', 
                        default=[32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32000],
                        help='Batch sizes to evaluate')
    parser.add_argument('--output_dir', type=str, default='./siglip_performance_results', 
                        help='Output directory')
    parser.add_argument('--model_name', type=str, default='google/siglip-base-patch16-224', 
                        help='SigLIP model name')
    
    args = parser.parse_args()
    
    config = {
        'data_dir': args.data_dir,
        'batch_sizes': args.batch_sizes,
        'model_name': args.model_name
    }
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize evaluator
    evaluator = SigLIPPerformanceEvaluator(config)
    
    # Run evaluation
    logger.info("Starting SigLIP performance evaluation...")
    results = evaluator.evaluate_all_batch_sizes()
    
    # Save results
    csv_file = output_dir / 'siglip_performance_results.csv'
    evaluator.save_results(results, str(csv_file))
    
    # Generate plots
    evaluator.generate_performance_plots(results, str(output_dir))
    
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
