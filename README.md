# Payment Slip Splitter

This project automatically detects and splits individual payment slips from a collage image using computer vision and deep learning.

## Features

 **Fully Offline** - No API calls required  
 **CNN-Powered** - Uses pretrained YOLOv8 model for robust detection  
 **Automatic Detection** - Smart detection with confidence scores  
 **High Quality** - Preserves original image quality  
 **Multiple Methods** - Choose between CNN or traditional OpenCV

## Installation

```bash
pip install -r requirements.txt
```

**Note**: First run will download the YOLOv8 model (~6MB), then it works completely offline.

## Usage

### Method 1: CNN-Based Detection (Recommended) 🚀

Uses pretrained YOLOv8 deep learning model for accurate detection:

```bash
python cnn_slip_detector.py
```

**Advantages**:
- More robust to variations in layout
- Handles complex backgrounds better
- Provides confidence scores
- State-of-the-art accuracy

### Method 2: Traditional OpenCV Detection

Uses morphological operations and contour detection:

```bash
python split_payment_slips.py
```

**Advantages**:
- Lighter dependencies (no PyTorch)
- Faster on CPU-only systems
- Good for simple, clean layouts

Place your image as `input_image.jpg` in the project folder.


## Output

### CNN Method
Extracted slips are saved in the `output_slips_cnn/` folder as:
- `slip_1.jpg`
- `slip_2.jpg`
- `slip_3.jpg`
- etc.

### OpenCV Method
Extracted slips are saved in the `output_slips/` folder.

A visualization showing detected regions with bounding boxes is saved as `detection_visualization.jpg`.

## Files

- `cnn_slip_detector.py` - **CNN-based detection** (YOLOv8) ⭐ Recommended
- `split_payment_slips.py` - Traditional OpenCV detection
- `detect_regions.py` - Coordinate-based extraction script
- `requirements.txt` - Python dependencies

## Dependencies

### Core (Required for all methods)
- opencv-python (≥4.5.0)
- numpy (≥1.19.0)
- Pillow (≥8.0.0)

### CNN Method (Additional)
- ultralytics (≥8.0.0) - YOLOv8 framework
- torch (≥2.0.0) - PyTorch deep learning
- torchvision (≥0.15.0) - PyTorch vision utilities

## Comparison

| Feature | CNN (YOLOv8) | OpenCV |
|---------|--------------|--------|
| Accuracy | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| Speed | Fast (GPU) / Medium (CPU) | Fast |
| Robustness | High | Medium |
| Dependencies | PyTorch required | Lightweight |
| Best for | Complex layouts, varied images | Simple, clean layouts |

Payment Slip Splitter
This project automatically detects and splits individual payment slips from a collage image using computer vision.

Features Fully Offline - No API calls required Automatic Detection - Detects payment slips using OpenCV High Quality - Preserves original image quality

Installation pip install -r requirements.txt Usage Method 1: Automatic Detection python split_payment_slips.py Place your image as input_image.jpg in the project folder.

Output All extracted slips are saved in the output_slips/ folder as:

slip_1.jpg slip_2.jpg slip_3.jpg etc. A visualization showing detected regions is also saved as detection_visualization.jpg.

Files split_payment_slips.py - Automatic detection script detect_regions.py - image extraction script requirements.txt - Python dependencies Dependencies opencv-python (≥4.5.0) numpy (≥1.19.0) Pillow (≥8.0.0)
