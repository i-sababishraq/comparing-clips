#!/usr/bin/env python3
"""
Main script to run CLIP LoRA training
Executes the training command with predefined parameters
"""

import subprocess
import sys
import os
from pathlib import Path

def run_lora_training():
    """Run LoRA CLIP training with predefined parameters"""
    
    # Change to the correct directory
    os.chdir('/anvil/projects/x-soc250046/x-sishraq/CLIP/CLIP_model')
    
    # Command to run
    cmd = [
        'conda', 'run', '--prefix', '/anvil/projects/x-soc250046/x-sishraq/CLIP/CLIP_model/clip_har_env',
        'python', 'train_clip_lora.py',
        '--data_dir', '../data',
        '--epochs', '1',
        '--batch_size', '64',
        '--lr', '1e-3',
        '--rank', '16',
        '--alpha', '32',
        '--checkpoint_dir', './checkpoints/clip_lora_1e'
    ]
    
    print("Starting LoRA CLIP training for 1 epoch...")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)
    
    try:
        # Run the command
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("=" * 60)
        print("Training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print("=" * 60)
        print(f"Training failed with error code: {e.returncode}")
        return False
    except Exception as e:
        print("=" * 60)
        print(f"Error running training: {e}")
        return False

def run_evaluation():
    """Run evaluation on the trained model"""
    
    # Change to the correct directory
    os.chdir('/anvil/projects/x-soc250046/x-sishraq/CLIP/CLIP_model')
    
    # Command to run evaluation
    cmd = [
        'conda', 'run', '--prefix', '/anvil/projects/x-soc250046/x-sishraq/CLIP/CLIP_model/clip_har_env',
        'python', 'evaluate_lora_clip.py',
        '--data_dir', '../data',
        '--checkpoint', 'checkpoints/clip_lora_1e/clip_lora_best.pt',
        '--model_name', 'ViT-B/32',
        '--rank', '16',
        '--alpha', '32',
        '--output_csv', '/anvil/projects/x-soc250046/x-sishraq/CLIP/results/clip/lora_clip_1e_results.csv'
    ]
    
    print("\nRunning evaluation on trained model...")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)
    
    try:
        # Run the evaluation
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("=" * 60)
        print("Evaluation completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print("=" * 60)
        print(f"Evaluation failed with error code: {e.returncode}")
        return False
    except Exception as e:
        print("=" * 60)
        print(f"Error running evaluation: {e}")
        return False

def main():
    """Main function"""
    print("CLIP LoRA Training Pipeline")
    print("=" * 60)
    
    # Check if we're in the right directory
    current_dir = Path.cwd()
    expected_dir = Path('/anvil/projects/x-soc250046/x-sishraq/CLIP/CLIP_model')
    
    if current_dir != expected_dir:
        print(f"Warning: Expected to be in {expected_dir}")
        print(f"   Current directory: {current_dir}")
        print("   Changing to correct directory...")
    
    # Run training
    training_success = run_lora_training()
    
    if training_success:
        # Run evaluation
        evaluation_success = run_evaluation()
        
        if evaluation_success:
            print("\nComplete pipeline finished successfully!")
            print("Results saved to: lora_clip_1e_results.csv")
        else:
            print("\nTraining completed but evaluation failed")
    else:
        print("\nTraining failed, skipping evaluation")

if __name__ == '__main__':
    main()
