#!/usr/bin/env python3
"""
Evaluate LoRA fine-tuned CLIP model using zero-shot evaluation
"""

import argparse
import logging
import pandas as pd
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import clip
from har_dataset import HARDataset
from train_clip_lora import LoRACLIP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def zero_shot_classification(model, test_loader, class_names, device):
    """Perform zero-shot classification using text templates"""
    logger.info("Performing zero-shot classification...")
    
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
    for class_name in class_names:
        clean_name = class_name.replace('_', ' ').replace('-', ' ').lower()
        class_texts = [template.format(clean_name) for template in templates]
        text_descriptions.extend(class_texts)
    
    # Encode text descriptions
    text_tokens = clip.tokenize(text_descriptions, truncate=True).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = F.normalize(text_features, dim=-1)
    
    # Reshape text features: [num_classes, num_templates, feature_dim]
    num_classes = len(class_names)
    text_features = text_features.reshape(num_classes, len(templates), -1)
    # Average across templates
    text_features = text_features.mean(dim=1)  # [num_classes, feature_dim]
    
    # Evaluate on test set
    all_predictions = []
    all_labels = []
    all_similarities = []
    
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            labels = batch['class_id'].cpu().numpy()
            
            # Encode images
            image_features = model.encode_image(images)
            image_features = F.normalize(image_features, dim=-1)
            
            # Compute similarities
            similarities = (image_features @ text_features.T) * model.temp_scale.exp()
            predictions = similarities.argmax(dim=1).cpu().numpy()
            
            all_predictions.extend(predictions)
            all_labels.extend(labels)
            all_similarities.extend(similarities.cpu().numpy())
    
    return all_predictions, all_labels, all_similarities


def compute_metrics(predictions, labels, similarities, class_names):
    """Compute evaluation metrics"""
    from sklearn.metrics import (
        accuracy_score, classification_report, confusion_matrix,
        top_k_accuracy_score
    )
    
    # Basic accuracy
    accuracy = accuracy_score(labels, predictions)
    
    # Top-k accuracy
    num_classes = len(class_names)
    top3_accuracy = top_k_accuracy_score(labels, similarities, k=min(3, num_classes))
    top5_accuracy = top_k_accuracy_score(labels, similarities, k=min(5, num_classes))
    
    # Per-class metrics
    report = classification_report(
        labels, predictions, 
        target_names=class_names, 
        output_dict=True, 
        zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    
    return {
        'accuracy': accuracy,
        'top3_accuracy': top3_accuracy,
        'top5_accuracy': top5_accuracy,
        'macro_precision': report['macro avg']['precision'],
        'macro_recall': report['macro avg']['recall'],
        'macro_f1': report['macro avg']['f1-score'],
        'weighted_precision': report['weighted avg']['precision'],
        'weighted_recall': report['weighted avg']['recall'],
        'weighted_f1': report['weighted avg']['f1-score'],
        'confusion_matrix': cm
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate LoRA fine-tuned CLIP')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='ViT-B/32')
    parser.add_argument('--rank', type=int, default=16)
    parser.add_argument('--alpha', type=float, default=32.0)
    parser.add_argument('--output_csv', type=str, default='lora_clip_results.csv')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load baseline CLIP
    clip_model, preprocess = clip.load(args.model_name, device=device, jit=False)
    
    # Create LoRA model
    model = LoRACLIP(clip_model, rank=args.rank, alpha=args.alpha, dropout=0.1).to(device)
    
    # Load LoRA checkpoint
    logger.info(f"Loading LoRA checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logger.info(f"Loaded LoRA model from epoch {checkpoint['epoch']}")
    
    # Setup test data
    test_dataset = HARDataset(args.data_dir, 'test', preprocess=preprocess)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    class_names = test_dataset.class_names
    logger.info(f"Test samples: {len(test_dataset)}")
    logger.info(f"Classes: {class_names}")
    
    # Evaluate
    predictions, labels, similarities = zero_shot_classification(
        model, test_loader, class_names, device
    )
    
    # Compute metrics
    metrics = compute_metrics(predictions, labels, similarities, class_names)
    
    # Print results
    logger.info("\n" + "="*50)
    logger.info("LoRA CLIP EVALUATION RESULTS")
    logger.info("="*50)
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Top-3 Accuracy: {metrics['top3_accuracy']:.4f}")
    logger.info(f"Top-5 Accuracy: {metrics['top5_accuracy']:.4f}")
    logger.info(f"Macro F1: {metrics['macro_f1']:.4f}")
    logger.info(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    
    # Create detailed results DataFrame
    results_data = []
    
    # Overall metrics
    results_data.append({
        'Metric': 'Accuracy',
        'Value': round(metrics['accuracy'], 4),
        'Type': 'Overall'
    })
    results_data.append({
        'Metric': 'Top-3 Accuracy',
        'Value': round(metrics['top3_accuracy'], 4),
        'Type': 'Overall'
    })
    results_data.append({
        'Metric': 'Top-5 Accuracy',
        'Value': round(metrics['top5_accuracy'], 4),
        'Type': 'Overall'
    })
    results_data.append({
        'Metric': 'Macro Precision',
        'Value': round(metrics['macro_precision'], 4),
        'Type': 'Overall'
    })
    results_data.append({
        'Metric': 'Macro Recall',
        'Value': round(metrics['macro_recall'], 4),
        'Type': 'Overall'
    })
    results_data.append({
        'Metric': 'Macro F1',
        'Value': round(metrics['macro_f1'], 4),
        'Type': 'Overall'
    })
    results_data.append({
        'Metric': 'Weighted Precision',
        'Value': round(metrics['weighted_precision'], 4),
        'Type': 'Overall'
    })
    results_data.append({
        'Metric': 'Weighted Recall',
        'Value': round(metrics['weighted_recall'], 4),
        'Type': 'Overall'
    })
    results_data.append({
        'Metric': 'Weighted F1',
        'Value': round(metrics['weighted_f1'], 4),
        'Type': 'Overall'
    })
    
    # Per-class metrics
    from sklearn.metrics import classification_report
    report = classification_report(
        labels, predictions, 
        target_names=class_names, 
        output_dict=True, 
        zero_division=0
    )
    
    for class_name in class_names:
        if class_name in report:
            results_data.append({
                'Metric': 'Precision',
                'Value': round(report[class_name]['precision'], 4),
                'Type': f'Class_{class_name}'
            })
            results_data.append({
                'Metric': 'Recall',
                'Value': round(report[class_name]['recall'], 4),
                'Type': f'Class_{class_name}'
            })
            results_data.append({
                'Metric': 'F1-Score',
                'Value': round(report[class_name]['f1-score'], 4),
                'Type': f'Class_{class_name}'
            })
            results_data.append({
                'Metric': 'Support',
                'Value': int(report[class_name]['support']),
                'Type': f'Class_{class_name}'
            })
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(results_data)
    df.to_csv(args.output_csv, index=False)
    
    logger.info(f"\nResults saved to: {args.output_csv}")
    logger.info(f"Total rows: {len(df)}")
    
    # Also save confusion matrix as separate CSV
    cm_df = pd.DataFrame(metrics['confusion_matrix'], 
                        index=class_names, 
                        columns=class_names)
    cm_csv = args.output_csv.replace('.csv', '_confusion_matrix.csv')
    cm_df.to_csv(cm_csv)
    logger.info(f"Confusion matrix saved to: {cm_csv}")


if __name__ == '__main__':
    main()
