#!/usr/bin/env python3
"""
Main script to run SigLIP performance evaluation.
"""

import subprocess
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_siglip_evaluation():
    """Run SigLIP performance evaluation"""
    
    # Set up paths
    data_dir = "/anvil/projects/x-soc250046/x-sishraq/CLIP/data"
    output_file = "siglip_performance_batch_32000.csv"
    
    # Define batch sizes to test (focusing on large batch sizes as requested)
    batch_sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32000]
    
    logger.info("Starting SigLIP Performance Evaluation")
    logger.info("="*50)
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output file: {output_file}")
    logger.info(f"Batch sizes to test: {batch_sizes}")
    logger.info("="*50)
    
    # Run the evaluation
    cmd = [
        sys.executable, "eval_siglip_simple.py",
        "--data_dir", data_dir,
        "--batch_sizes"] + [str(bs) for bs in batch_sizes] + [
        "--output_file", output_file
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
    if Path(output_file).exists():
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


if __name__ == "__main__":
    success = run_siglip_evaluation()
    sys.exit(0 if success else 1)
