# Kidney Segmentation using Deep Learning

A deep learning project for automatic kidney segmentation in medical images using an enhanced U-Net architecture with attention mechanisms and advanced training techniques.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Project Structure](#project-structure)
- [Performance Metrics](#performance-metrics)
- [Results](#results)
- [License](#license)

## Overview

This project implements an advanced medical image segmentation system for identifying kidneys in grayscale medical images. The model uses an Enhanced U-Net architecture with spatial attention mechanisms, achieving high accuracy through sophisticated loss functions and data augmentation techniques.

## Features

- **Enhanced U-Net Architecture**: Custom U-Net with spatial attention modules and residual connections
- **Advanced Loss Function**: Combination of Focal Loss (60%) and Dice Loss (40%) for handling class imbalance
- **Comprehensive Metrics Tracking**: Monitors Dice coefficient, IoU, Precision, Recall, and F1 score
- **Data Augmentation**: Extensive augmentation pipeline using Albumentations
- **Learning Rate Scheduling**: Adaptive learning rate with ReduceLROnPlateau
- **Early Stopping**: Prevents overfitting with configurable patience
- **Gradient Clipping**: Stabilizes training with gradient norm clipping
- **Automated Logging**: Saves training metrics and generates visualization plots

## Requirements

```
!pip install torch
!pip install torchvision
!pip install opencv-python
!pip install albumentations
!pip install numpy
!pip install scikit-learn
!pip install matplotlib
!pip install pandas
```
### Installation
### Option 1: Kaggle Notebook

#### Step 1: Create New Notebook
1. Go to [Kaggle](https://www.kaggle.com)
2. Click **"Create"** → **"New Notebook"**
3. Enable GPU: Settings → Accelerator → **"GPU T4 x2"** (or available GPU)

## option2: Local setup Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd kidney-segmentation
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install torch torchvision opencv-python albumentations numpy scikit-learn matplotlib pandas
```

## Dataset


### Download Dataset

1. Access the dataset from Google Drive:
   - **Link**: https://drive.google.com/drive/u/4/folders/1GUZau-k0BW-CsbKfRagkGtvoR0c0LklO

2. Download the folder containing:
   - `images/` - Original kidney images
   - `masks/` - Corresponding segmentation masks

3. Extract and organize the dataset:
```
project_root/
├── data/
│   ├── images/
│   │   ├── image_001.png
│   │   ├── image_002.png
│   │   └── ...
│   └── masks/
│       ├── image_001.png
│       ├── image_002.png
│       └── ...
```
#### Step 1: how to Upload Dataset to Kaggle
1. Go to [Kaggle](https://www.kaggle.com)
2. Navigate to the **Datasets** section
3. Click **Your Work** → **Create New Dataset**
4. Upload the downloaded `.zip` file
5. Fill in the required dataset details
6. Click **Create**

#### Step 2: Add Dataset to Your Kaggle Notebook
1. Open your Kaggle **Notebook**
2. On the right-hand side, locate the **Input** section
3. Click **Add Input**
4. Select **Your Work**
5. Find the uploaded dataset and click **Add**
   
### Dataset Format

- **Images**: Grayscale PNG format
- **Masks**: Binary masks (0 for background, 255 for kidney)
- **Resolution**: Images will be resized to 256×256 during training
- **Expected**: Matching filenames between images and masks

## Model Architecture

### Enhanced U-Net

The model features:

1. **Encoder Path** (Downsampling):
   - 4 encoding blocks with channel progression: 64 → 128 → 256 → 512
   - Each block: Conv2D → BatchNorm → ReLU → Dropout → Conv2D → BatchNorm → ReLU
   - MaxPooling between blocks

2. **Bridge**:
   - 1024 channels with spatial attention mechanism
   - Attention module focuses on relevant features

3. **Decoder Path** (Upsampling):
   - 4 decoding blocks with skip connections
   - Transposed convolutions for upsampling
   - Concatenation with corresponding encoder features

4. **Output Layer**:
   - 1×1 convolution with sigmoid activation
   - Produces probability map for kidney regions

### Key Components

- **Spatial Attention**: Enhances feature representation by focusing on important regions
- **Skip Connections**: Preserves spatial information from encoder to decoder
- **Dropout**: 0.2 dropout rate for regularization
- **Batch Normalization**: Stabilizes training and improves convergence

## Training

### Configuration

Key hyperparameters (adjustable in `RealTimeConfig`):

```python
IMAGE_SIZE = 256        # Input image size
BATCH_SIZE = 16         # Batch size
LEARNING_RATE = 1e-3    # Initial learning rate
NUM_EPOCHS = 150        # Maximum epochs
PATIENCE = 20           # Early stopping patience
```

### Running Training

1. Update dataset paths in the code:
```python
image_dir = '/path/to/your/images'
masks_dir = '/path/to/your/masks'
```

2. Run the training script:
```bash
python train.py
```

3. For Kaggle environment:
```python
# Update paths to Kaggle input
image_dir = '/kaggle/input/2kdataset/2kdataset/images'
masks_dir = '/kaggle/input/2kdataset/2kdataset/masks'
```

### Training Process

The training pipeline includes:

1. **Data Loading**: 80-20 train-validation split
2. **Augmentation**: Random flips, rotations, brightness/contrast adjustments
3. **Loss Calculation**: Combined Focal Loss and Dice Loss
4. **Optimization**: AdamW optimizer with weight decay (1e-4)
5. **Learning Rate Scheduling**: Reduces LR when validation loss plateaus
6. **Checkpointing**: Saves best model based on validation loss
7. **Metrics Logging**: Tracks 6 metrics across training

### Data Augmentation

Applied transformations:
- Resize to 256×256
- Random horizontal/vertical flips (70% probability)
- Shift, scale, rotate (±45°)
- Random brightness and contrast adjustment
- Normalization to [0, 1]

## Project Structure

```
kidney-segmentation/
├── train.py                    # Main training script
├── README.md                   # This file
├── data/                       # Dataset directory
│   ├── images/
│   └── masks/
├── outputs/                    # Training outputs
│   ├── best_kidney_segmentation_model.pth
│   ├── training_metrics.png
│   └── training_metrics.csv
└── requirements.txt            # Python dependencies
```

## Performance Metrics

The model tracks comprehensive metrics:

### Primary Metrics
- **Dice Coefficient**: Measures overlap between prediction and ground truth
- **IoU (Intersection over Union)**: Jaccard index for segmentation quality
- **Loss**: Combined Focal Loss and Dice Loss

### Secondary Metrics
- **Precision**: Accuracy of positive predictions
- **Recall**: Ability to find all positive samples
- **F1 Score**: Harmonic mean of precision and recall

### Visualization

After training, the following are generated:
- `training_metrics.png`: 6-panel plot showing all metrics over epochs
- `training_metrics.csv`: Detailed metrics data for analysis

## Results

### Expected Performance
- **Target Dice Coefficient**: > 0.90
- **Target IoU**: > 0.85
- **Training Time**: ~2-3 hours on GPU (depends on dataset size)

### Model Output
- Binary segmentation mask (kidney vs. background)
- Probability values between 0 and 1
- Threshold: 0.5 (configurable via `INFERENCE_THRESHOLD`)

### Saved Artifacts
- **Model Weights**: `best_kidney_segmentation_model.pth`
- **Training Plots**: `training_metrics.png`
- **Metrics CSV**: `training_metrics.csv`

## Advanced Features

### Loss Function
```python
loss = 0.6 * focal_loss + 0.4 * dice_loss
```
- Focal Loss: Handles class imbalance (α=0.25, γ=2.0)
- Dice Loss: Optimizes segmentation overlap

### Regularization Techniques
- Dropout (0.2) in convolution blocks
- Weight decay (1e-4) in optimizer
- Gradient clipping (max_norm=1.0)
- Batch normalization

### Image Preprocessing (Optional)
The code includes utilities for:
- White balance correction
- CLAHE enhancement
- Unsharp masking
(Currently not applied in main pipeline but available for experimentation)

## Usage Tips

1. **GPU Recommendation**: Training on GPU is highly recommended (10-20x faster)
2. **Memory**: Adjust `BATCH_SIZE` if encountering OOM errors
3. **Dataset Size**: Minimum 1000 images recommended for good generalization
4. **Validation**: Monitor validation metrics to detect overfitting
5. **Checkpoints**: Best model is automatically saved during training

## Troubleshooting

**Issue**: CUDA out of memory
- **Solution**: Reduce `BATCH_SIZE` in configuration

**Issue**: Poor segmentation quality
- **Solution**: Increase training epochs or adjust loss weights

**Issue**: Training too slow
- **Solution**: Reduce `IMAGE_SIZE` or use fewer augmentations

**Issue**: Model overfitting
- **Solution**: Increase dropout rate or add more augmentation

## 👨‍💻 Developers

This project was developed by the **Asthra Medtech Development Team**.

**Team Members:**
- **Jeeva** – AI Engineer

---

## 🛠 Support

For questions, issues, or support regarding this project, please reach out:

**Contact Information:**

- **Email:** jeeva@asthramedtech.com  
- **Office:** Asthra Medtech Private Limited  
  292/1B Sharp Nagar, Kalapatti Road, Kalapatti, Coimbatore 641 048, Tamil Nadu, India  
- **LinkedIn:** [Asthra Medtech](https://www.linkedin.com/company/asthramedtech)  
- **Website:** [www.asthramedtech.com](http://www.asthramedtech.com)

---

## 📜 License

© 2025 Asthra Medtech Private Limited. All rights reserved.  

This software is **proprietary and confidential**. Unauthorized copying, distribution, or use of this software, via any medium, is strictly prohibited without **explicit written permission** from Asthra Medtech Private Limited.

**Permissions & Licensing Inquiries:**

- **Email:** amirtha@asthramedtech.com  
- **Office:** Asthra Medtech Private Limited, Coimbatore

---

**Version:** 1.0.0  
**Last Updated:** December 2025  
**Status:** Active Development
