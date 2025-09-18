#!/usr/bin/env python3
"""
Comprehensive SigLIP evaluation with performance, speed, and memory monitoring.
Uses 32k batch size as specified in the requirements.
"""

import os
import argparse
import time
import psutil
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import List, Dict, Tuple
import json
import csv

from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, SiglipModel


def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def list_classes(split_dir: Path) -> List[str]:
    classes = [d.name for d in split_dir.iterdir() if d.is_dir()]
    classes.sort()
    return classes


def iter_images(split_dir: Path):
    for cls in sorted([d for d in split_dir.iterdir() if d.is_dir()], key=lambda p: p.name):
        for f in cls.iterdir():
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                yield f, cls.name


@torch.no_grad()
def evaluate_siglip_performance(data_dir: str, split: str = "test", 
                               model_name: str = "google/siglip-base-patch16-224",
                               batch_size: int = 32000):
    """Evaluate SigLIP with comprehensive performance monitoring"""
    
    print(f"Evaluating SigLIP with batch size: {batch_size}")
    print(f"Model: {model_name}")
    print(f"Data: {data_dir}/{split}")
    
    # Set up environment
    os.environ.setdefault("TRANSFORMERS_CACHE", str(Path(__file__).parent / "hf_cache"))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Memory before model loading
    memory_before = get_memory_usage()
    print(f"Memory before model loading: {memory_before:.2f} MB")
    
    # Load model and processor
    print("Loading model and processor...")
    start_time = time.time()
    
    processor = AutoProcessor.from_pretrained(model_name)
    model = SiglipModel.from_pretrained(model_name).to(device)
    model.eval()
    
    model_load_time = time.time() - start_time
    memory_after_model = get_memory_usage()
    print(f"Model loaded in {model_load_time:.2f}s")
    print(f"Memory after model loading: {memory_after_model:.2f} MB")
    print(f"Model memory usage: {memory_after_model - memory_before:.2f} MB")
    
    # Prepare data
    split_dir = Path(data_dir) / split
    classes = list_classes(split_dir)
    print(f"Classes: {classes}")
    
    # Collect all images and labels
    images = []
    labels = []
    image_paths = []
    
    print("Loading images...")
    for img_path, class_name in tqdm(iter_images(split_dir), desc="Loading images"):
        images.append(img_path)
        labels.append(class_name)
        image_paths.append(str(img_path))
    
    print(f"Total images: {len(images)}")
    
    # Create text prompts
    text_prompts = [f"a photo of {cls}" for cls in classes]
    print(f"Text prompts: {text_prompts}")
    
    # Process text
    print("Processing text prompts...")
    text_start = time.time()
    text_inputs = processor(text=text_prompts, return_tensors="pt", padding=True, truncation=True)
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    text_time = time.time() - text_start
    print(f"Text processing time: {text_time:.2f}s")
    
    # Get text embeddings
    print("Computing text embeddings...")
    text_start = time.time()
    text_outputs = model.get_text_features(**text_inputs)
    text_embeddings = F.normalize(text_outputs, dim=-1)
    text_time = time.time() - text_start
    print(f"Text embedding time: {text_time:.2f}s")
    
    # Process images in batches
    correct = 0
    total = 0
    total_inference_time = 0
    total_image_processing_time = 0
    
    print(f"Processing images in batches of {batch_size}...")
    
    for i in tqdm(range(0, len(images), batch_size), desc="Processing batches"):
        batch_images = images[i:i + batch_size]
        batch_labels = labels[i:i + batch_size]
        
        # Process images
        img_start = time.time()
        try:
            pil_images = [Image.open(img_path).convert('RGB') for img_path in batch_images]
            image_inputs = processor(images=pil_images, return_tensors="pt", padding=True)
            image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
        except Exception as e:
            print(f"Error processing batch {i//batch_size}: {e}")
            continue
            
        img_processing_time = time.time() - img_start
        total_image_processing_time += img_processing_time
        
        # Get image embeddings
        inf_start = time.time()
        image_outputs = model.get_image_features(**image_inputs)
        image_embeddings = F.normalize(image_outputs, dim=-1)
        inf_time = time.time() - inf_start
        total_inference_time += inf_time
        
        # Compute similarities
        similarities = image_embeddings @ text_embeddings.T
        
        # Get predictions
        predictions = similarities.argmax(dim=1)
        
        # Calculate accuracy for this batch
        batch_correct = sum(1 for pred, true_label in zip(predictions, batch_labels) 
                           if classes[pred] == true_label)
        correct += batch_correct
        total += len(batch_images)
        
        # Memory monitoring
        current_memory = get_memory_usage()
        if i % (batch_size * 5) == 0:  # Print every 5 batches
            print(f"Batch {i//batch_size}: Memory: {current_memory:.2f} MB, "
                  f"Batch accuracy: {batch_correct/len(batch_images)*100:.2f}%")
    
    # Final results
    accuracy = correct / total if total > 0 else 0
    total_time = time.time() - start_time
    
    print(f"\n=== SigLIP Performance Results ===")
    print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"Total time: {total_time:.2f}s")
    print(f"Model loading time: {model_load_time:.2f}s")
    print(f"Text processing time: {text_time:.2f}s")
    print(f"Image processing time: {total_image_processing_time:.2f}s")
    print(f"Inference time: {total_inference_time:.2f}s")
    print(f"Images per second: {len(images)/total_time:.2f}")
    print(f"Final memory usage: {get_memory_usage():.2f} MB")
    print(f"Peak memory usage: {memory_after_model:.2f} MB")
    
    # Save results to CSV
    results = {
        'model': 'SigLIP',
        'model_name': model_name,
        'batch_size': batch_size,
        'accuracy': accuracy,
        'total_images': total,
        'correct_predictions': correct,
        'total_time': total_time,
        'model_load_time': model_load_time,
        'text_processing_time': text_time,
        'image_processing_time': total_image_processing_time,
        'inference_time': total_inference_time,
        'images_per_second': len(images)/total_time,
        'final_memory_mb': get_memory_usage(),
        'peak_memory_mb': memory_after_model,
        'model_memory_mb': memory_after_model - memory_before
    }
    
    # Save to CSV
    csv_path = f"siglip_performance_batch_{batch_size}.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results.keys())
        writer.writeheader()
        writer.writerow(results)
    
    print(f"Results saved to: {csv_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='SigLIP performance evaluation')
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--model_name", type=str, default="google/siglip-base-patch16-224")
    parser.add_argument("--batch_size", type=int, default=32000)
    args = parser.parse_args()
    
    results = evaluate_siglip_performance(
        data_dir=args.data_dir,
        split=args.split,
        model_name=args.model_name,
        batch_size=args.batch_size
    )
    
    return results


if __name__ == "__main__":
    main()
