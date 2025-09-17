#!/usr/bin/env python3
"""
Clean evaluation script for CLIP fine-tuned on HAR dataset
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import clip
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    top_k_accuracy_score
)

# Import training modules
from train_clip_har_clean import HARDataset

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CLIPHAREvaluator:
    """CLIP Model Evaluator for HAR dataset"""
    
    def __init__(self, checkpoint_path: str, data_dir: str, model_name: str = 'ViT-B/32'):
        self.checkpoint_path = checkpoint_path
        self.data_dir = Path(data_dir)
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load models
        self.load_models()
        
        # Setup test data
        self.setup_test_data()
    
    def load_models(self):
        """Load both fine-tuned and baseline CLIP models"""
        logger.info(f"Loading models...")
        
        # Load baseline model
        self.baseline_model, self.preprocess = clip.load(self.model_name, device=self.device, jit=False)
        self.baseline_model.eval()
        
        # Load fine-tuned model
        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            logger.info(f"Loading fine-tuned model from {self.checkpoint_path}")
            
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
            
            # Initialize fine-tuned model with same architecture
            self.finetuned_model, _ = clip.load(self.model_name, device=self.device, jit=False)
            
            # Load fine-tuned weights
            self.finetuned_model.load_state_dict(checkpoint['model_state_dict'])
            self.finetuned_model.eval()
            
            logger.info(f"Loaded fine-tuned model from epoch {checkpoint['epoch']}")
            logger.info(f"Checkpoint accuracy: {checkpoint.get('accuracy', 'N/A')}")
        else:
            logger.warning(f"Checkpoint not found at {self.checkpoint_path}")
            self.finetuned_model = None
    
    def setup_test_data(self):
        """Setup test dataset and data loader"""
        # Try different test split names
        test_splits = ['test', 'testing', 'testset']
        self.test_dataset = None
        
        for split_name in test_splits:
            try:
                self.test_dataset = HARDataset(self.data_dir, split_name, preprocess=self.preprocess)
                logger.info(f"Using {split_name} split for evaluation")
                break
            except ValueError:
                continue
        
        if self.test_dataset is None:
            # Fall back to validation split or train split for testing
            for split_name in ['val', 'train']:
                try:
                    self.test_dataset = HARDataset(self.data_dir, split_name, preprocess=self.preprocess)
                    logger.info(f"Using {split_name} split for evaluation (no test split found)")
                    break
                except ValueError:
                    continue
        
        if self.test_dataset is None:
            raise ValueError("No test, validation, or training split found in the dataset")
        
        self.test_loader = DataLoader(
            self.test_dataset, 
            batch_size=32, 
            shuffle=False, 
            num_workers=4
        )
        
        self.class_names = self.test_dataset.class_names
        self.num_classes = len(self.class_names)
        
        logger.info(f"Test samples: {len(self.test_dataset)}")
        logger.info(f"Number of classes: {self.num_classes}")
        logger.info(f"Classes: {self.class_names}")
    
    def zero_shot_classification(self, model, model_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Perform zero-shot classification using text templates"""
        logger.info(f"Performing zero-shot classification with {model_name}...")
        
        # Create text templates for each class
        templates = [
            "a photo of a person {}",
            "someone {}",
            "a person performing {}",
            "{} action",
            "human {}",
            "a video frame of someone {}",
            "footage of a person {}"
        ]
        
        # Generate text descriptions for all classes
        text_descriptions = []
        for class_name in self.class_names:
            clean_name = class_name.replace('_', ' ').replace('-', ' ').lower()
            class_texts = [template.format(clean_name) for template in templates]
            text_descriptions.extend(class_texts)
        
        # Encode text descriptions
        text_tokens = clip.tokenize(text_descriptions, truncate=True).to(self.device)
        with torch.no_grad():
            text_features = model.encode_text(text_tokens)
            text_features = F.normalize(text_features, dim=-1)
        
        # Reshape text features: [num_classes, num_templates, feature_dim]
        text_features = text_features.reshape(self.num_classes, len(templates), -1)
        # Average across templates
        text_features = text_features.mean(dim=1)  # [num_classes, feature_dim]
        
        # Evaluate on test set
        all_predictions = []
        all_labels = []
        all_similarities = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc=f"Zero-shot {model_name}"):
                images = batch['image'].to(self.device)
                labels = batch['class_id'].cpu().numpy()
                
                # Encode images
                image_features = model.encode_image(images)
                image_features = F.normalize(image_features, dim=-1)
                
                # Compute similarities
                similarities = (image_features @ text_features.T) * model.logit_scale.exp()
                predictions = similarities.argmax(dim=1).cpu().numpy()
                
                all_predictions.extend(predictions)
                all_labels.extend(labels)
                all_similarities.extend(similarities.cpu().numpy())
        
        return np.array(all_predictions), np.array(all_labels), np.array(all_similarities)
    
    def compute_metrics(self, predictions: np.ndarray, labels: np.ndarray, 
                       similarities: np.ndarray, model_name: str) -> Dict:
        """Compute evaluation metrics"""
        # Basic accuracy
        accuracy = accuracy_score(labels, predictions)
        
        # Top-k accuracy
        top3_accuracy = top_k_accuracy_score(labels, similarities, k=min(3, self.num_classes))
        top5_accuracy = top_k_accuracy_score(labels, similarities, k=min(5, self.num_classes))
        
        # Per-class metrics
        report = classification_report(
            labels, predictions, 
            target_names=self.class_names, 
            output_dict=True, 
            zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        
        metrics = {
            'model_name': model_name,
            'accuracy': accuracy,
            'top3_accuracy': top3_accuracy,
            'top5_accuracy': top5_accuracy,
            'macro_precision': report['macro avg']['precision'],
            'macro_recall': report['macro avg']['recall'],
            'macro_f1': report['macro avg']['f1-score'],
            'weighted_precision': report['weighted avg']['precision'],
            'weighted_recall': report['weighted avg']['recall'],
            'weighted_f1': report['weighted avg']['f1-score'],
            'per_class_metrics': {
                class_name: {
                    'precision': report[class_name]['precision'],
                    'recall': report[class_name]['recall'],
                    'f1-score': report[class_name]['f1-score'],
                    'support': report[class_name]['support']
                }
                for class_name in self.class_names if class_name in report
            },
            'confusion_matrix': cm.tolist()
        }
        
        return metrics
    
    def plot_confusion_matrix(self, cm: np.ndarray, model_name: str, save_path: str = None):
        """Plot confusion matrix"""
        plt.figure(figsize=(max(8, self.num_classes * 0.5), max(6, self.num_classes * 0.4)))
        
        # Use percentage if there are many classes
        if self.num_classes > 10:
            cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
            sns.heatmap(
                cm_percent, 
                annot=True, 
                fmt='.1f', 
                cmap='Blues',
                xticklabels=self.class_names,
                yticklabels=self.class_names,
                cbar_kws={'label': 'Percentage'}
            )
        else:
            sns.heatmap(
                cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=self.class_names,
                yticklabels=self.class_names
            )
        
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved confusion matrix to {save_path}")
        
        plt.close()  # Don't show, just save
    
    def plot_comparison(self, baseline_metrics: Dict, finetuned_metrics: Dict, save_path: str = None):
        """Plot comparison between baseline and fine-tuned models"""
        metrics_to_compare = ['accuracy', 'top3_accuracy', 'top5_accuracy', 'macro_f1', 'weighted_f1']
        
        baseline_values = [baseline_metrics[metric] for metric in metrics_to_compare]
        finetuned_values = [finetuned_metrics[metric] for metric in metrics_to_compare]
        
        x = np.arange(len(metrics_to_compare))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        rects1 = ax.bar(x - width/2, baseline_values, width, label='Baseline CLIP', alpha=0.8)
        rects2 = ax.bar(x + width/2, finetuned_values, width, label='Fine-tuned CLIP', alpha=0.8)
        
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels([metric.replace('_', ' ').title() for metric in metrics_to_compare])
        ax.legend()
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.3f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom')
        
        autolabel(rects1)
        autolabel(rects2)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved comparison plot to {save_path}")
        
        plt.close()  # Don't show, just save
    
    def evaluate(self, save_dir: str = './evaluation_results'):
        """Run complete evaluation"""
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        
        logger.info("Starting evaluation...")
        
        # Baseline evaluation
        baseline_predictions, baseline_labels, baseline_similarities = self.zero_shot_classification(
            self.baseline_model, "Baseline CLIP"
        )
        baseline_metrics = self.compute_metrics(
            baseline_predictions, baseline_labels, baseline_similarities, "Baseline CLIP"
        )
        
        results = {
            'baseline_metrics': baseline_metrics
        }
        
        # Fine-tuned evaluation (if available)
        if self.finetuned_model is not None:
            finetuned_predictions, finetuned_labels, finetuned_similarities = self.zero_shot_classification(
                self.finetuned_model, "Fine-tuned CLIP"
            )
            finetuned_metrics = self.compute_metrics(
                finetuned_predictions, finetuned_labels, finetuned_similarities, "Fine-tuned CLIP"
            )
            
            results['finetuned_metrics'] = finetuned_metrics
            results['improvement'] = {
                'accuracy': finetuned_metrics['accuracy'] - baseline_metrics['accuracy'],
                'macro_f1': finetuned_metrics['macro_f1'] - baseline_metrics['macro_f1'],
                'weighted_f1': finetuned_metrics['weighted_f1'] - baseline_metrics['weighted_f1']
            }
        
        # Print results
        logger.info("\n" + "="*60)
        logger.info("EVALUATION RESULTS")
        logger.info("="*60)
        logger.info(f"Baseline CLIP Accuracy: {baseline_metrics['accuracy']:.4f}")
        logger.info(f"Baseline Top-3 Accuracy: {baseline_metrics['top3_accuracy']:.4f}")
        logger.info(f"Baseline Top-5 Accuracy: {baseline_metrics['top5_accuracy']:.4f}")
        logger.info(f"Baseline Macro F1: {baseline_metrics['macro_f1']:.4f}")
        logger.info(f"Baseline Weighted F1: {baseline_metrics['weighted_f1']:.4f}")
        
        if self.finetuned_model is not None:
            logger.info(f"\nFine-tuned CLIP Accuracy: {finetuned_metrics['accuracy']:.4f}")
            logger.info(f"Fine-tuned Top-3 Accuracy: {finetuned_metrics['top3_accuracy']:.4f}")
            logger.info(f"Fine-tuned Top-5 Accuracy: {finetuned_metrics['top5_accuracy']:.4f}")
            logger.info(f"Fine-tuned Macro F1: {finetuned_metrics['macro_f1']:.4f}")
            logger.info(f"Fine-tuned Weighted F1: {finetuned_metrics['weighted_f1']:.4f}")
            
            logger.info(f"\nIMPROVEMENTS:")
            logger.info(f"Accuracy: {results['improvement']['accuracy']:+.4f}")
            logger.info(f"Macro F1: {results['improvement']['macro_f1']:+.4f}")
            logger.info(f"Weighted F1: {results['improvement']['weighted_f1']:+.4f}")
        
        # Save results
        with open(save_dir / 'evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Plot confusion matrices
        self.plot_confusion_matrix(
            np.array(baseline_metrics['confusion_matrix']),
            "Baseline CLIP",
            save_path=save_dir / 'confusion_matrix_baseline.png'
        )
        
        if self.finetuned_model is not None:
            self.plot_confusion_matrix(
                np.array(finetuned_metrics['confusion_matrix']),
                "Fine-tuned CLIP",
                save_path=save_dir / 'confusion_matrix_finetuned.png'
            )
            
            # Plot comparison
            self.plot_comparison(
                baseline_metrics, finetuned_metrics,
                save_path=save_dir / 'model_comparison.png'
            )
        
        # Save detailed classification reports
        with open(save_dir / 'baseline_classification_report.txt', 'w') as f:
            f.write("Baseline CLIP Classification Report\n")
            f.write("="*50 + "\n\n")
            f.write(classification_report(
                baseline_labels, baseline_predictions, 
                target_names=self.class_names
            ))
        
        if self.finetuned_model is not None:
            with open(save_dir / 'finetuned_classification_report.txt', 'w') as f:
                f.write("Fine-tuned CLIP Classification Report\n")
                f.write("="*50 + "\n\n")
                f.write(classification_report(
                    finetuned_labels, finetuned_predictions, 
                    target_names=self.class_names
                ))
        
        logger.info(f"\nResults saved to {save_dir}")
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Evaluate CLIP models on HAR dataset')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to HAR dataset directory')
    parser.add_argument('--checkpoint', type=str,
                        help='Path to fine-tuned CLIP checkpoint (optional)')
    parser.add_argument('--model_name', type=str, default='ViT-B/32',
                        help='CLIP model name')
    parser.add_argument('--save_dir', type=str, default='./evaluation_results',
                        help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    # Create evaluator and run evaluation
    evaluator = CLIPHAREvaluator(args.checkpoint, args.data_dir, args.model_name)
    results = evaluator.evaluate(args.save_dir)
    
    print("\nEvaluation completed!")
    print(f"Results saved to: {args.save_dir}")

if __name__ == '__main__':
    main()
