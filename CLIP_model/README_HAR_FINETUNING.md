# CLIP Fine-tuning on Human Action Recognition Dataset

This directory contains the complete setup for fine-tuning the OpenAI CLIP model on the Human Action Recognition (HAR) dataset using various robust fine-tuning methods including LoRA (Low-Rank Adaptation).

## Directory Structure

```
CLIP_model/
├── clip/                          # Original CLIP implementation
│   ├── __init__.py
│   ├── clip.py                    # Main CLIP interface
│   ├── model.py                   # Model architecture
│   └── simple_tokenizer.py       # Tokenization
├── har_dataset.py                 # HAR dataset class (reusable)
├── train_clip_lora.py             # LoRA fine-tuning script (RECOMMENDED)
├── evaluate_lora_clip.py          # LoRA evaluation script
├── main_clip.py                   # Automated training pipeline
├── train_clip_augmented.py        # Data augmentation + hard pair mining
├── setup_clip_har_env.sh          # Complete environment setup script
├── requirements.txt               # Base CLIP requirements
├── requirements_har.txt           # Complete HAR requirements (all packages)
├── checkpoints/                   # Model checkpoints (created during training)
│   ├── clip_lora_15e/            # LoRA fine-tuned model (15 epochs)
│   └── clip_lora_5e/             # LoRA fine-tuned model (5 epochs)
├── data/                          # Dataset directory
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

## Training Methods

### 1. LoRA Fine-tuning (RECOMMENDED)

LoRA (Low-Rank Adaptation) is the most effective method, achieving 89.28% accuracy while training only 0.02% of parameters.

#### Automated Training Pipeline

```bash
# Run complete LoRA training and evaluation pipeline
python main_clip.py
```

#### Manual LoRA Training

```bash
# Train LoRA model for 15 epochs
python train_clip_lora.py --data_dir ../data --epochs 15 --batch_size 64 --lr 1e-3 --rank 16 --alpha 32 --checkpoint_dir ./checkpoints/clip_lora_15e

# Evaluate the trained model
python evaluate_lora_clip.py --data_dir ../data --checkpoint checkpoints/clip_lora_15e/clip_lora_best.pt --model_name ViT-B/32 --rank 16 --alpha 32 --output_csv lora_clip_15e_results.csv
```

#### LoRA Training Options

- `--data_dir`: Path to HAR dataset directory
- `--model_name`: CLIP model variant (ViT-B/32, ViT-B/16, ViT-L/14, RN50, etc.)
- `--batch_size`: Batch size for training (default: 64)
- `--epochs`: Number of training epochs (default: 15)
- `--lr`: Learning rate (default: 1e-3)
- `--rank`: LoRA rank (default: 16)
- `--alpha`: LoRA alpha scaling (default: 32)
- `--checkpoint_dir`: Directory to save checkpoints

### 2. Data Augmentation + Hard Pair Mining

```bash
python train_clip_augmented.py --data_dir ../data --epochs 15 --batch_size 64 --lr 1e-3 --checkpoint_dir ./checkpoints/clip_augmented_15e
```

### 3. Legacy Methods (Not Recommended)

The following methods were tested but showed poor performance due to catastrophic forgetting:

- **Full Fine-tuning**: Achieved only 7.6% accuracy (vs 85.7% zero-shot)
- **Linear Probe**: Achieved 83.25% accuracy but limited flexibility
- **Two-tier Learning Rate**: Achieved 52.4% accuracy but still below zero-shot

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

#### LoRA Fine-tuning Results (15 epochs)

- **Baseline CLIP Zero-shot**: 85.70% accuracy
- **LoRA Fine-tuned CLIP**: **89.28% accuracy** (+3.58% improvement)
- **Training Parameters**: Only 0.02% of total model parameters
- **Memory Efficiency**: Significantly reduced memory usage
- **Training Time**: ~45 minutes for 15 epochs on single GPU

#### Performance Comparison

| Method | Accuracy | Parameters Trained | Memory Usage | Training Time |
|--------|----------|-------------------|--------------|---------------|
| Zero-shot CLIP | 85.70% | 0% | Low | N/A |
| LoRA (5 epochs) | 88.70% | 0.02% | Low | ~15 min |
| LoRA (15 epochs) | **89.28%** | 0.02% | Low | ~45 min |
| Full Fine-tuning | 7.60% | 100% | High | ~2 hours |
| Linear Probe | 83.25% | 0.1% | Medium | ~30 min |
| Two-tier LR | 52.40% | 100% | High | ~2 hours |

### Training Time and Resources

- **LoRA ViT-B/32**: ~3 minutes per epoch on single GPU
- **Memory Usage**: 4-6GB GPU memory (vs 8-12GB for full fine-tuning)
- **Parameter Efficiency**: 99.98% reduction in trainable parameters
- **Convergence**: Stable training without catastrophic forgetting

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




