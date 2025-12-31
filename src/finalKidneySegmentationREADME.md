# Kidney Segmentation Model - Complete Guide

## Overview
This repository contains an Enhanced U-Net model for kidney segmentation from medical images using PyTorch.

---

## Prerequisites

### System Requirements
- Python 3.8 or higher
- CUDA-capable GPU (recommended)
- 8GB+ RAM
- 10GB+ free disk space

### Required Libraries
```bash
pip install torch torchvision
pip install opencv-python
pip install albumentations
pip install numpy
pip install scikit-learn
pip install matplotlib
pip install pandas
pip install kagglehub
```

---

## Dataset Preparation

### 1. Dataset Structure
Organize your dataset in the following structure:
```
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── masks/
    ├── train/
    ├── val/
    └── test/
```

### 2. Image Requirements
- **Format**: JPG, PNG, BMP, or TIFF
- **Recommended Size**: 256x256 pixels (will be resized automatically)
- **Color**: Grayscale or RGB (will be converted to grayscale)
- **Masks**: Binary masks (0 for background, 255 for kidney)

---

## Training the Model

### Step 1: Configure Dataset Paths
Open the training notebook and modify these paths:

```python
# Change these paths to match your dataset location
TRAIN_IMAGE_DIR = '/path/to/your/dataset/images/train/'
TRAIN_MASK_DIR = '/path/to/your/dataset/masks/train/'
VAL_IMAGE_DIR = '/path/to/your/dataset/images/val/'
VAL_MASK_DIR = '/path/to/your/dataset/masks/val/'
```

### Step 2: Adjust Training Parameters
Modify these hyperparameters based on your requirements:

```python
# Training Configuration
BATCH_SIZE = 8          # Reduce if GPU memory is limited (try 4 or 2)
LEARNING_RATE = 1e-4    # Default learning rate
NUM_EPOCHS = 100        # Number of training epochs
IMAGE_SIZE = 256        # Input image size (256x256)

# Early Stopping
PATIENCE = 15           # Stop if no improvement for 15 epochs

# Model Checkpoint
CHECKPOINT_DIR = './checkpoints/'  # Where to save model weights
```

### Step 3: Hardware Configuration
```python
# Device Selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# For multiple GPUs (optional)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
```

### Step 4: Run Training
Execute all cells in the training notebook sequentially.

**Monitor Training:**
- Training loss should decrease
- Validation Dice score should increase
- Best model is automatically saved to checkpoint directory

---

## Model Inference

### Step 1: Configure Model Path
In the inference notebook, update the model path:

```python
MODEL_PATH = '/path/to/your/best_model_checkpoint.pth'
```

### Step 2: Single Image Inference
```python
# Update these paths
IMAGE_PATH = '/path/to/your/test/image.jpg'
SAVE_PATH = '/path/to/save/result.png'

# Run inference
result = inference_single_image(MODEL_PATH, IMAGE_PATH, SAVE_PATH)
```

### Step 3: Batch Inference
```python
# Update these paths
IMAGE_DIR = '/path/to/test/images/'
OUTPUT_DIR = '/path/to/save/results/'

# Run batch inference
results = inference_batch(MODEL_PATH, IMAGE_DIR, OUTPUT_DIR)
```

---

## Output Files

### Training Outputs
```
checkpoints/
├── best_model_checkpoint.pth          # Best model weights
├── training_history.png               # Training curves
└── training_log.txt                   # Detailed training log
```

### Inference Outputs
```
results/
├── result_image1.jpg                  # Visualization with all metrics
├── result_image2.jpg
├── ...
└── batch_summary.txt                  # Summary statistics
```

---

## Key Configuration Changes

### 1. **For Different Image Sizes**
```python
IMAGE_SIZE = 512  # Change from 256 to your desired size
# Note: Larger sizes require more GPU memory
```

### 2. **For Limited GPU Memory**
```python
BATCH_SIZE = 2              # Reduce batch size
model.dropout = nn.Dropout2d(0.5)  # Increase dropout
```

### 3. **For Faster Training**
```python
NUM_EPOCHS = 50             # Reduce epochs
NUM_WORKERS = 4             # Increase data loading workers
```

### 4. **For Better Accuracy**
```python
NUM_EPOCHS = 200            # More training
LEARNING_RATE = 5e-5        # Lower learning rate
BATCH_SIZE = 16             # Larger batch (if GPU allows)
```

### 5. **For Different Input Channels**
```python
# If using RGB images instead of grayscale
model = EnhancedUNet(in_channels=3, out_channels=1)
```

---

## Troubleshooting

### CUDA Out of Memory
```python
BATCH_SIZE = 2              # Reduce batch size
IMAGE_SIZE = 128            # Use smaller images
```

### Model Not Learning
- Check if masks are binary (0 and 255)
- Verify image-mask pairs are aligned
- Increase `NUM_EPOCHS`
- Try `LEARNING_RATE = 1e-3`

### Low Dice Score
- Ensure dataset quality
- Check class imbalance in masks
- Increase training epochs
- Try data augmentation parameters

### File Not Found Errors
- Verify all paths are absolute paths
- Check file extensions match (case-sensitive)
- Ensure images and masks have same filenames

---

## Performance Optimization

### Speed Up Training
1. Use `NUM_WORKERS = 4` in DataLoader
2. Enable `pin_memory=True` in DataLoader
3. Use mixed precision training (add AMP)
4. Reduce `IMAGE_SIZE`

### Improve Accuracy
1. Increase `NUM_EPOCHS`
2. Use learning rate scheduling
3. Add more data augmentation
4. Ensemble multiple models

---

## Model Metrics

### During Training
- **Loss**: Should decrease steadily
- **Dice Score**: Should increase (target >0.85)
- **IoU Score**: Should increase (target >0.75)

### During Inference
- **Confidence Score**: 0.0 to 1.0 (higher is better)
- **Inference Time**: Typically <0.1s per image on GPU

---

## Citation
If you use this model, please cite:
```
Enhanced U-Net for Kidney Segmentation
Architecture: U-Net with Residual Blocks and Deep Supervision
Framework: PyTorch
```

---

## Support
For issues and questions:
1. Check all file paths are correct
2. Verify dataset structure matches requirements
3. Ensure CUDA is properly installed for GPU training
4. Check Python package versions are compatible

---

## Quick Start Checklist

- [ ] Install all required libraries
- [ ] Prepare dataset in correct structure
- [ ] Update dataset paths in training notebook
- [ ] Adjust batch size based on GPU memory
- [ ] Run training notebook
- [ ] Wait for training to complete
- [ ] Update model path in inference notebook
- [ ] Run inference on test images
- [ ] Check output results

---

## License
This model is provided for research and educational purposes.
