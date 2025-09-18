#!/bin/bash

# Setup script for CLIP HAR fine-tuning environment
# This script creates a complete environment with all necessary packages
echo "Setting up CLIP HAR fine-tuning environment..."

# Create conda environment for CLIP
echo "Creating conda environment: clip_har_env"
conda create -n clip_har_env python=3.9 -y

# Activate environment
echo "Activating clip_har_env..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /anvil/projects/x-soc250046/x-sishraq/CLIP/CLIP_model/clip_har_env

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA support..."
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install base CLIP requirements
echo "Installing base CLIP requirements..."
cd /anvil/projects/x-soc250046/x-sishraq/CLIP/CLIP_model
pip install -r requirements.txt

# Install complete HAR requirements (includes all packages from working environment)
echo "Installing complete HAR requirements..."
pip install -r requirements_har.txt

# Install the CLIP package from current directory
echo "Installing CLIP package from source..."
pip install -e .

# Create directory structure for experiments
mkdir -p checkpoints
mkdir -p data
mkdir -p results
mkdir -p logs

# Verify installation
echo "Verifying installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import clip; print('CLIP imported successfully')"
python -c "import pandas, matplotlib, seaborn, sklearn; print('Data science packages imported successfully')"

echo ""
echo "CLIP HAR environment setup completed!"
echo ""
echo "Environment includes:"
echo "  - PyTorch with CUDA support"
echo "  - CLIP model implementation"
echo "  - Complete data science stack (pandas, matplotlib, seaborn, sklearn)"
echo "  - Jupyter notebook support"
echo "  - Experiment tracking (wandb, tensorboard)"
echo "  - All necessary utilities and dependencies"
echo ""
echo "To activate the environment:"
echo "  conda activate /anvil/projects/x-soc250046/x-sishraq/CLIP/CLIP_model/clip_har_env"
echo ""
echo "To run LoRA training:"
echo "  cd /anvil/projects/x-soc250046/x-sishraq/CLIP/CLIP_model"
echo "  python main_clip.py"
echo ""
echo "To run individual training scripts:"
echo "  python train_clip_lora.py --data_dir ../data --epochs 15"
echo "  python evaluate_lora_clip.py --data_dir ../data --checkpoint checkpoints/clip_lora_15e/clip_lora_best.pt"


