#!/usr/bin/env python3
"""
Main script to run SigLIP performance evaluation using Hugging Face Transformers.
Based on the merveenoyan/siglip repository approach.
"""

import subprocess
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_siglip_evaluation():
    """Run SigLIP performance evaluation using Hugging Face Transformers"""
    
    # Set up paths
    data_dir = "/anvil/projects/x-soc250046/x-sishraq/CLIP/data"
    output_dir = "./siglip_performance_results"
    
    # Define batch sizes to test (focusing on large batch sizes as requested)
    batch_sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32000]
    
    logger.info("Starting SigLIP Performance Evaluation (Hugging Face)")
    logger.info("="*60)
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Batch sizes to test: {batch_sizes}")
    logger.info("="*60)
    
    # Run the evaluation
    cmd = [
        sys.executable, "eval_siglip_performance_hf.py",
        "--data_dir", data_dir,
        "--batch_sizes"] + [str(bs) for bs in batch_sizes] + [
        "--output_dir", output_dir,
        "--model_name", "google/siglip-base-patch16-224"
    ]
    
    try:
        logger.info("Running SigLIP evaluation...")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        logger.info("Evaluation completed successfully!")
        logger.info("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            logger.info("STDERR:")
            print(result.stderr)
            
    except subprocess.CalledProcessError as e:
        logger.error(f"Evaluation failed with return code {e.returncode}")
        logger.error("STDOUT:")
        print(e.stdout)
        logger.error("STDERR:")
        print(e.stderr)
        return False
    
    # Check if output file was created
    output_file = Path(output_dir) / "siglip_performance_results.csv"
    if output_file.exists():
        logger.info(f"Results saved to {output_file}")
        
        # Read and display summary
        try:
            import pandas as pd
            df = pd.read_csv(output_file)
            logger.info("\nResults Summary:")
            logger.info(df.to_string(index=False))
        except ImportError:
            logger.info("pandas not available, skipping summary display")
    else:
        logger.warning(f"Output file {output_file} not found")
    
    return True


def run_siglip_training():
    """Run SigLIP training"""
    
    # Set up paths
    data_dir = "/anvil/projects/x-soc250046/x-sishraq/CLIP/data"
    checkpoint_dir = "./checkpoints/siglip_hf"
    
    logger.info("Starting SigLIP Training (Hugging Face)")
    logger.info("="*50)
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Checkpoint directory: {checkpoint_dir}")
    logger.info("="*50)
    
    # Run the training
    cmd = [
        sys.executable, "train_siglip_hf.py",
        "--data_dir", data_dir,
        "--batch_size", "64",
        "--epochs", "5",
        "--learning_rate", "1e-4",
        "--checkpoint_dir", checkpoint_dir,
        "--model_name", "google/siglip-base-patch16-224"
    ]
    
    try:
        logger.info("Running SigLIP training...")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        logger.info("Training completed successfully!")
        logger.info("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            logger.info("STDERR:")
            print(result.stderr)
            
    except subprocess.CalledProcessError as e:
        logger.error(f"Training failed with return code {e.returncode}")
        logger.error("STDOUT:")
        print(e.stdout)
        logger.error("STDERR:")
        print(e.stderr)
        return False
    
    return True


def main():
    """Main function to run SigLIP training and evaluation"""
    
    # First run training
    logger.info("Step 1: Training SigLIP model...")
    training_success = run_siglip_training()
    
    if not training_success:
        logger.error("Training failed, skipping evaluation")
        return False
    
    # Then run evaluation
    logger.info("\nStep 2: Evaluating SigLIP performance...")
    evaluation_success = run_siglip_evaluation()
    
    if not evaluation_success:
        logger.error("Evaluation failed")
        return False
    
    logger.info("\n" + "="*60)
    logger.info("SIGLIP TRAINING AND EVALUATION COMPLETED SUCCESSFULLY!")
    logger.info("="*60)
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
