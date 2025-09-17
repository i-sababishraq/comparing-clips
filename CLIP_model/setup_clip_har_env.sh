#!/bin/bash

# Setup script for CLIP HAR fine-tuning environment
echo "Setting up CLIP HAR fine-tuning environment..."

# Create conda environment for CLIP
echo "Creating conda environment: clip_har_env"
conda create -n clip_har_env python=3.9 -y

# Activate environment
echo "Activating clip_har_env..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate clip_har_env

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA support..."
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install base CLIP requirements
echo "Installing base CLIP requirements..."
cd /anvil/projects/x-soc250046/x-sishraq/CLIP/CLIP_model
pip install -r requirements.txt

# Install additional requirements for HAR fine-tuning
echo "Installing additional HAR requirements..."
pip install -r requirements_har.txt

# Install the CLIP package from current directory
echo "Installing CLIP package from source..."
pip install -e .

# Create directory structure for experiments
mkdir -p checkpoints
mkdir -p data
mkdir -p results
mkdir -p logs

echo "CLIP HAR environment setup completed!"
echo ""
echo "To activate the environment:"
echo "  conda activate clip_har_env"
echo ""
echo "To run training:"
echo "  cd /anvil/projects/x-soc250046/x-sishraq/CLIP/CLIP_model"
echo "  python train_clip_har.py --data_dir /path/to/har/dataset --epochs 20"


