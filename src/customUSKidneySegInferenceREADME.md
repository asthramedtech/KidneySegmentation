# Kidney Segmentation Inference - README

## Overview
This notebook performs inference (prediction) on kidney images using a pre-trained Enhanced U-Net model. It can process single images or batch process multiple images with detailed visualizations.

---

## Quick Start

### 1. Install Dependencies
Run the first cell to install all required packages:
```python
!pip install torch torchvision opencv-python albumentations numpy scikit-learn matplotlib pandas
```

### 2. Download Pre-trained Model (Optional)
If using Kaggle's pre-trained model:
```python
import kagglehub
path = kagglehub.model_download("karthickjeeva/kidneyseg/pyTorch/default")
```

---

## Configuration Changes

### SECTION 5: Main Execution - Required Changes

#### 1. **Model Path** (REQUIRED)
```python
# Change this to your model checkpoint path
MODEL_PATH = '/path/to/your/best_model_checkpoint.pth'

# Examples:
# Local: './checkpoints/best_model.pth'
# Kaggle: '/kaggle/input/kidneyseg/pytorch/default/20/best_model_checkpoint_1MW.pth'
# Google Colab: '/content/drive/MyDrive/models/best_model.pth'
```

#### 2. **Single Image Inference Paths**
```python
# Input image path
IMAGE_PATH = '/path/to/your/test/image.jpg'

# Output result path
SAVE_PATH = '/path/to/save/result.png'

# Examples:
# Kaggle: IMAGE_PATH = '/kaggle/input/testimages/kidney_scan.jpg'
# Local: IMAGE_PATH = './test_images/kidney_001.png'
```

#### 3. **Batch Inference Paths** (Uncomment to use)
```python
# Uncomment these lines to enable batch processing
IMAGE_DIR = '/path/to/test/images/folder/'
OUTPUT_DIR = '/path/to/save/results/folder/'

# Then uncomment the batch inference call
results = inference_batch(MODEL_PATH, IMAGE_DIR, OUTPUT_DIR)
```

---

## Usage Modes

### Mode 1: Single Image Inference (Default)
**What it does:** Processes one image and shows detailed visualization

**How to use:**
1. Set `MODEL_PATH` to your trained model
2. Set `IMAGE_PATH` to your test image
3. Set `SAVE_PATH` where you want to save the result
4. Run all cells

**Output:**
- 6-panel visualization showing:
  - Original image
  - Predicted mask
  - Confidence map
  - Mask overlay
  - Contour visualization
  - Statistics panel
- Saved result image at `SAVE_PATH`
- Console output with metrics

---

### Mode 2: Batch Processing
**What it does:** Processes multiple images automatically

**How to use:**
1. Uncomment the batch inference section in SECTION 5
2. Set `IMAGE_DIR` to folder containing test images
3. Set `OUTPUT_DIR` to folder where results will be saved
4. Run all cells

**Output:**
```
batch_results/
├── result_image1.jpg
├── result_image2.jpg
├── result_image3.jpg
├── ...
└── batch_summary.txt
```

**Batch Summary Contains:**
- Individual image statistics
- Average confidence scores
- Average inference times
- Number of contours detected

---

## Image Requirements

### Supported Formats
- JPG/JPEG
- PNG
- BMP
- TIFF

### Image Specifications
- **Any size** - automatically resized to 256×256 for model input
- **Color or grayscale** - automatically converted to grayscale
- **Original dimensions preserved** in output visualizations

---

## Model Configuration

### If Using Different Model Architecture

#### Change Input Channels
```python
# In KidneyEvaluator.load_model() method
model = EnhancedUNet(in_channels=3, out_channels=1)  # For RGB input
model = EnhancedUNet(in_channels=1, out_channels=1)  # For grayscale (default)
```

#### Change Image Size
```python
# In KidneyEvaluator.preprocess_image() method
image = cv2.resize(image, (512, 512))  # Change from (256, 256)

# Also update LayerNorm sizes in model architecture accordingly
```

#### Adjust Confidence Threshold
```python
# In process_single_image() method
prediction = (probabilities > 0.3).float()  # Change from 0.5 for more/less sensitive detection
```

---

## Customization Options

### 1. **Visualization Settings**
```python
# In process_single_image() method

# Change figure size
fig, axes = plt.subplots(2, 3, figsize=(20, 12))  # Larger display

# Change DPI (resolution)
plt.savefig(save_path, dpi=150)  # Lower for smaller files, higher for better quality

# Change overlay color
mask_overlay[pred_mask > 0.5] = [0, 255, 0]  # Green instead of yellow

# Change overlay transparency
overlay = cv2.addWeighted(overlay, 0.7, mask_overlay, 0.3, 0)  # More transparent
```

### 2. **Contour Settings**
```python
# Change contour color and thickness
cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 3)  # Green, thicker lines
```

### 3. **Device Configuration**
```python
# Force CPU usage (if GPU not available or for testing)
evaluator = KidneyEvaluator(model_path, device='cpu')

# Use specific GPU
evaluator = KidneyEvaluator(model_path, device='cuda:0')  # First GPU
```

---

## Output Interpretation

### Visualization Panels Explained

1. **Original Image**: Input kidney scan
2. **Predicted Mask**: Binary segmentation (white = kidney, black = background)
3. **Confidence Map**: Color-coded confidence (red = high, blue = low)
4. **Mask Overlay**: Kidney region highlighted on original image
5. **Contour Visualization**: Kidney boundaries outlined
6. **Statistics Panel**: Numerical metrics

### Key Metrics

**Confidence Score (0.0 - 1.0)**
- > 0.8: Excellent prediction
- 0.6 - 0.8: Good prediction
- < 0.6: Review manually

**Inference Time**
- GPU: Typically 0.01 - 0.1 seconds
- CPU: Typically 0.5 - 2 seconds

**Mask Coverage**
- Percentage of image classified as kidney
- Typical range: 10-40%

---

## Troubleshooting

### Error: Model file not found
```python
# Check if path exists
import os
print(os.path.exists(MODEL_PATH))  # Should print True

# Use absolute path
MODEL_PATH = '/absolute/path/to/model.pth'
```

### Error: Image not found
```python
# Check image path
print(os.path.exists(IMAGE_PATH))

# Check file extension
print(os.path.splitext(IMAGE_PATH)[1])  # Should be .jpg, .png, etc.
```

### Error: CUDA out of memory
```python
# Use CPU instead
evaluator = KidneyEvaluator(model_path, device='cpu')
```

### Error: Checkpoint loading failed
```python
# Your model might have different keys
checkpoint = torch.load(model_path)
print(checkpoint.keys())  # Check available keys

# If keys are different, adjust loading:
model.load_state_dict(checkpoint)  # Instead of checkpoint['model_state_dict']
```

### Warning: Low confidence scores
- Check if image is actually a kidney scan
- Verify image quality (not blurry/corrupted)
- Ensure model was trained on similar images

---

## Performance Tips

### Speed Up Inference
1. **Use GPU**: Ensure CUDA is available
2. **Batch Processing**: More efficient than processing one by one
3. **Lower Resolution**: Use smaller images if speed is critical
4. **Disable Plotting**: Set `show_plot=False` for batch processing

### Improve Results
1. **Pre-process Images**: Ensure good contrast and quality
2. **Adjust Threshold**: Try different confidence thresholds
3. **Ensemble Models**: Average predictions from multiple models
4. **Post-processing**: Apply morphological operations to clean masks

---

## Function Reference

### inference_single_image()
```python
result = inference_single_image(
    model_path='/path/to/model.pth',
    image_path='/path/to/image.jpg',
    save_path='/path/to/save/result.png'  # Optional
)
```
**Returns:** Dictionary with prediction results

### inference_batch()
```python
results = inference_batch(
    model_path='/path/to/model.pth',
    image_dir='/path/to/images/',
    output_dir='/path/to/results/'
)
```
**Returns:** List of dictionaries with batch results

### KidneyEvaluator Class
```python
evaluator = KidneyEvaluator(model_path, device='cuda')
result = evaluator.process_single_image(image_path, save_path, show_plot=True)
results = evaluator.process_batch(image_paths, output_dir)
```

---

## Common Workflows

### Workflow 1: Test Single Image
1. Set `MODEL_PATH`
2. Set `IMAGE_PATH`
3. Set `SAVE_PATH`
4. Run all cells
5. Check visualization and metrics

### Workflow 2: Process Multiple Images
1. Set `MODEL_PATH`
2. Uncomment batch inference section
3. Set `IMAGE_DIR` and `OUTPUT_DIR`
4. Run all cells
5. Check `batch_summary.txt` for results

### Workflow 3: Quick Testing Without Saving
1. Set `MODEL_PATH` and `IMAGE_PATH`
2. Set `SAVE_PATH = None`
3. Run all cells
4. View results in notebook only

---

## Platform-Specific Notes

### Kaggle
- Model downloaded automatically with kagglehub
- Outputs saved to `/kaggle/working/`
- GPU available in sessions with GPU enabled




## Quick Configuration Template

```python
# ===== CHANGE THESE PATHS =====
MODEL_PATH = '/your/path/to/model.pth'
IMAGE_PATH = '/your/path/to/test_image.jpg'
SAVE_PATH = '/your/path/to/save_result.png'

# For batch processing (uncomment to use):
# IMAGE_DIR = '/your/path/to/images/'
# OUTPUT_DIR = '/your/path/to/results/'
# results = inference_batch(MODEL_PATH, IMAGE_DIR, OUTPUT_DIR)
```

---

## License
This inference notebook is provided for research and educational purposes.
