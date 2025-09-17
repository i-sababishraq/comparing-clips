# CLIP Fine-tuning on Human Action Recognition Dataset

This directory contains the setup for fine-tuning the OpenAI CLIP model on the Human Action Recognition (HAR) dataset.

## Directory Structure

```
CLIP_model/
├── clip/                          # Original CLIP implementation
│   ├── __init__.py
│   ├── clip.py                    # Main CLIP interface
│   ├── model.py                   # Model architecture
│   └── simple_tokenizer.py       # Tokenization
├── train_clip_har.py              # Fine-tuning script for HAR dataset
├── evaluate_clip_har.py           # Evaluation script
├── setup_clip_har_env.sh          # Environment setup script
├── requirements.txt               # Base CLIP requirements
├── requirements_har.txt           # Additional HAR requirements
├── config_har.json               # Training configuration
├── checkpoints/                   # Model checkpoints (created during training)
├── data/                          # Dataset directory (you'll add the HAR dataset here)
├── results/                       # Evaluation results
└── logs/                          # Training logs
```

## Setup Instructions

### 1. Environment Setup

Run the setup script to create the conda environment:

```bash
cd /anvil/projects/x-soc250046/x-sishraq/CLIP/CLIP_model
chmod +x setup_clip_har_env.sh
./setup_clip_har_env.sh
```

This will:
- Create a conda environment named `clip_har_env`
- Install PyTorch with CUDA support
- Install all required dependencies
- Install the CLIP package from source
- Create necessary directories

### 2. Activate Environment

```bash
conda activate clip_har_env
```

### 3. Download HAR Dataset

Download the Human Action Recognition dataset from Kaggle:

```bash
# Install kaggle CLI if not already done
pip install kaggle

# Configure Kaggle API (you'll need your Kaggle credentials)
# Place kaggle.json in ~/.kaggle/ or set environment variables

# Download the dataset
kaggle datasets download -d shashankrapolu/human-action-recognition-dataset
unzip human-action-recognition-dataset.zip -d data/har_dataset
```

Expected dataset structure:
```
data/har_dataset/
├── train/
│   ├── action_class_1/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── action_class_2/
│   └── ...
├── test/
│   ├── action_class_1/
│   └── ...
└── val/ (or validation/)
    ├── action_class_1/
    └── ...
```

## Training

### Basic Training

```bash
python train_clip_har.py --data_dir data/har_dataset --epochs 20
```

### Training with Configuration File

```bash
python train_clip_har.py --data_dir data/har_dataset --config config_har.json
```

### Training Options

- `--data_dir`: Path to HAR dataset directory
- `--model_name`: CLIP model variant (ViT-B/32, ViT-B/16, ViT-L/14, RN50, etc.)
- `--batch_size`: Batch size for training
- `--epochs`: Number of training epochs
- `--learning_rate`: Learning rate
- `--checkpoint_dir`: Directory to save checkpoints
- `--config`: JSON configuration file

### Available CLIP Models

- `ViT-B/32`: Vision Transformer Base with 32x32 patches (default)
- `ViT-B/16`: Vision Transformer Base with 16x16 patches
- `ViT-L/14`: Vision Transformer Large with 14x14 patches
- `RN50`: ResNet-50 backbone
- `RN101`: ResNet-101 backbone

### Configuration Options

The `config_har.json` file contains various hyperparameters:

```json
{
    "model_name": "ViT-B/32",
    "batch_size": 64,
    "epochs": 20,
    "learning_rate": 5e-6,
    "weight_decay": 0.1,
    "warmup_epochs": 2,
    "optimization": {
        "vision_lr_multiplier": 1.0,
        "text_lr_multiplier": 0.1,
        "projection_lr_multiplier": 1.0
    }
}
```

## Evaluation

### Evaluate Fine-tuned Model

```bash
python evaluate_clip_har.py \
    --data_dir data/har_dataset \
    --checkpoint checkpoints/clip_har_best.pt \
    --save_dir results/clip_evaluation
```

### Evaluate Baseline Only

```bash
python evaluate_clip_har.py \
    --data_dir data/har_dataset \
    --save_dir results/baseline_evaluation
```

The evaluation script will:
- Compare fine-tuned model with baseline CLIP
- Generate confusion matrices
- Calculate accuracy, top-k accuracy, precision, recall, F1-score
- Save detailed classification reports
- Create visualization plots

## Monitoring Training

The training script provides several monitoring features:

1. **Progress Bars**: Real-time training progress with loss and learning rate
2. **Zero-shot Evaluation**: Evaluation using text templates after each epoch
3. **Checkpointing**: Automatic saving of best models and regular checkpoints
4. **Logging**: Detailed logging of training progress

### Training Output Example

```
Epoch 1/20
Training: 100%|██████████| 150/150 [02:30<00:00,  1.00it/s, loss=2.456, avg_loss=2.543, lr=1.2e-06]
Evaluating: 100%|██████████| 25/25 [00:15<00:00,  1.67it/s]
Zero-shot evaluation: 100%|██████████| 25/25 [00:12<00:00,  2.08it/s]
Validation - Loss: 2.234, Accuracy: 0.654
Zero-shot accuracy: 0.612
```

## Expected Results

### Performance Metrics

The fine-tuning should improve performance on the HAR dataset compared to zero-shot CLIP:

- **Baseline CLIP Zero-shot**: ~30-50% accuracy (depends on dataset)
- **Fine-tuned CLIP**: ~60-80% accuracy (target improvement)
- **Top-3 Accuracy**: Usually 10-15% higher than top-1
- **Top-5 Accuracy**: Usually 15-20% higher than top-1

### Training Time

- **ViT-B/32 on single GPU**: ~2-3 hours for 20 epochs (depends on dataset size)
- **ViT-L/14 on single GPU**: ~4-6 hours for 20 epochs
- **Memory Usage**: 6-12GB GPU memory (depends on model and batch size)

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce batch size in config
   - Use gradient accumulation
   - Try smaller model (ViT-B/32 instead of ViT-L/14)

2. **Dataset Not Found**:
   - Check dataset structure matches expected format
   - Ensure train/test/val splits exist
   - Verify image file extensions (.jpg, .png, etc.)

3. **Poor Performance**:
   - Check class names and text templates
   - Verify dataset quality
   - Try different learning rates
   - Increase training epochs

4. **Import Errors**:
   - Ensure environment is activated
   - Install missing packages
   - Check Python path

### Performance Tips

1. **Hyperparameter Tuning**:
   - Start with learning rate 1e-5 to 5e-6
   - Use warmup for stable training
   - Different learning rates for vision and text encoders

2. **Data Augmentation**:
   - CLIP preprocessing includes basic augmentation
   - Additional augmentation can help but use carefully

3. **Text Templates**:
   - Create diverse and natural text descriptions
   - Use multiple templates per class
   - Avoid overly specific or generic descriptions

## Next Steps

After successfully fine-tuning CLIP on HAR:

1. **Compare with SigCLIP**: Move to SigCLIP fine-tuning
2. **Compare with A-CLIP**: Set up A-CLIP fine-tuning
3. **Ablation Studies**: Try different hyperparameters
4. **Analysis**: Analyze what the model learned
5. **Deployment**: Create inference pipeline

## Citation

If you use this code for research, please cite:

```bibtex
@misc{openai2021clip,
  title={Learning Transferable Visual Representations from Natural Language Supervision},
  author={Alec Radford and Jong Wook Kim and Chris Hallacy and Aditya Ramesh and Gabriel Goh and Sandhini Agarwal and Girish Sastry and Amanda Askell and Pamela Mishkin and Jack Clark and Gretchen Krueger and Ilya Sutskever},
  year={2021},
  eprint={2103.00020},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```




