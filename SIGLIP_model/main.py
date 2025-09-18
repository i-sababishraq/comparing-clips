#!/usr/bin/env python3
"""
Main orchestrator for SigLIP on HAR:
- Fine-tunes a linear classifier on frozen SigLIP features
- Evaluates on the test set
- Saves standardized CSV metrics and plots to the results directory
"""

import argparse
import json
from pathlib import Path
from typing import Dict

import torch
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Local imports
from train_siglip_hf import SigLIPTrainer


def write_standardized_csv(perf_report_path: Path, output_csv_path: Path) -> None:
    data = json.loads(Path(perf_report_path).read_text())
    rows = [
        {"Metric": "Accuracy", "Value": round(float(data["final_accuracy"]), 4), "Type": "Overall"},
        {"Metric": "F1 Macro", "Value": round(float(data["final_f1_macro"]), 4), "Type": "Overall"},
        {"Metric": "F1 Weighted", "Value": round(float(data.get("final_f1_weighted", 0.0)), 4), "Type": "Overall"},
        {"Metric": "Throughput (samples/sec)", "Value": round(float(data["average_speed_samples_per_sec"]), 2), "Type": "Performance"},
        {"Metric": "Avg Memory (MB)", "Value": round(float(data["average_memory_usage_mb"]), 2), "Type": "Performance"},
        {"Metric": "Peak Memory (MB)", "Value": round(float(data["peak_memory_usage_mb"]), 2), "Type": "Performance"},
        {"Metric": "Total Time (sec)", "Value": round(float(data["total_training_time"]), 2), "Type": "Performance"},
        {"Metric": "Batch Size", "Value": int(data["batch_size"]), "Type": "RunConfig"},
        {"Metric": "Epochs", "Value": int(data["epochs"]), "Type": "RunConfig"},
    ]
    df = pd.DataFrame(rows)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv_path, index=False)


def evaluate_best_and_save_artifacts(config: Dict, checkpoint_dir: Path, results_dir: Path) -> None:
    """Reload best checkpoint, evaluate, and save confusion matrix and per-class report."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Recreate trainer (datasets, model, classifier shapes)
    trainer = SigLIPTrainer(config)
    trainer.model.eval()
    trainer.classifier.eval()

    # Load best checkpoint weights (classifier-only)
    best_path = checkpoint_dir / 'siglip_best.pt'
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    trainer.classifier.load_state_dict(ckpt['classifier_state_dict'])

    # Evaluate to get predictions/labels
    eval_out = trainer.evaluate()
    y_true = np.array(eval_out['labels'])
    y_pred = np.array(eval_out['predictions'])

    # Classification report (per-class CSV)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    per_class_csv = results_dir / 'siglip_finetuned_classification_report.csv'
    pd.DataFrame(report).to_csv(per_class_csv)

    # Confusion matrix plot
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, cmap='Blues', ax=ax)
    ax.set_title('SigLIP (finetuned) Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    fig.tight_layout()
    cm_png = results_dir / 'siglip_finetuned_confusion_matrix.png'
    fig.savefig(cm_png, dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Also save raw confusion matrix CSV
    cm_csv = results_dir / 'siglip_finetuned_confusion_matrix.csv'
    pd.DataFrame(cm).to_csv(cm_csv, index=False)


def main():
    ap = argparse.ArgumentParser(description='SigLIP HAR: train + evaluate + export artifacts')
    ap.add_argument('--data_dir', required=True, type=str)
    ap.add_argument('--batch_size', default=64, type=int)
    ap.add_argument('--epochs', default=5, type=int)
    ap.add_argument('--learning_rate', default=1e-4, type=float)
    ap.add_argument('--num_workers', default=4, type=int)
    ap.add_argument('--model_name', default='google/siglip-base-patch16-224', type=str)
    ap.add_argument('--checkpoint_dir', default='./checkpoints/siglip_5e_classprobe', type=str)
    ap.add_argument('--results_dir', default='/anvil/projects/x-soc250046/x-sishraq/CLIP/results/siglip', type=str)
    args = ap.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    config = {
        'data_dir': args.data_dir,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'checkpoint_dir': str(checkpoint_dir),
        'num_workers': args.num_workers,
        'model_name': args.model_name,
        'weight_decay': 0.01,
    }

    # Train + internal test evaluations (best checkpoint saved)
    trainer = SigLIPTrainer(config)
    trainer.train()

    # Convert performance report to standardized CSV
    perf_json = checkpoint_dir / 'performance_report.json'
    out_csv = results_dir / 'siglip_finetuned_results.csv'
    write_standardized_csv(perf_json, out_csv)

    # Evaluate best checkpoint and save confusion matrix + per-class metrics
    evaluate_best_and_save_artifacts(config, checkpoint_dir, results_dir)

    print(f"Saved results to {results_dir}")


if __name__ == '__main__':
    main()


