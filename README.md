# Payment Slip Splitter

Automatically detect and split multiple payment slips from a single scanned image into separate files using computer vision.

## Features

- **Fully Offline**: No API calls required, works completely offline
- **Automatic Detection**: Uses OpenCV contour detection to identify individual slips
- **Robust Processing**: Handles various slip layouts, sizes, and orientations
- **High Quality Output**: Saves slips as high-quality JPEG images
- **Debug Mode**: Optional visualization of detection steps for troubleshooting
- **Configurable**: Easily adjust detection parameters via `config.py`

## Requirements

- Python 3.7+
- OpenCV
- NumPy
- Pillow
- imutils

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python split_payment_slips.py --input <input_image> --output <output_directory>
```

### Examples

```bash
# Split slips from an image
python split_payment_slips.py -i receipts.jpg -o output_slips/

# Enable debug mode to see intermediate processing steps
python split_payment_slips.py -i scan.png -o slips/ --debug
```

### Command Line Arguments

- `-i, --input`: Path to input image containing payment slips (required)
- `-o, --output`: Directory to save extracted payment slips (required)
- `--debug`: Enable debug mode to save intermediate processing images

## How It Works

 **Slip Extraction**
   - Extract each detected region with padding
   - Save as separate high-quality images

## Configuration

Adjust detection parameters in `config.py`:

- **MIN_SLIP_AREA**: Minimum area for valid slips (default: 10000 pixels)
- **MAX_SLIP_AREA**: Maximum area for valid slips (default: 500000 pixels)
- **MIN_ASPECT_RATIO**: Minimum width/height ratio (default: 0.3)
- **MAX_ASPECT_RATIO**: Maximum width/height ratio (default: 3.0)
- **CANNY_THRESHOLD_1**: Lower threshold for Canny edge detection (default: 50)
- **CANNY_THRESHOLD_2**: Upper threshold for Canny edge detection (default: 150)
- **OUTPUT_IMAGE_QUALITY**: JPEG quality 1-100 (default: 95)

## Output

The script creates the following structure:

```
output_directory/
├── slip_001.jpg
├── slip_002.jpg
├── slip_003.jpg
└── debug/              (if --debug flag is used)
    ├── 01_resized.jpg
    ├── 02_grayscale.jpg
    ├── 03_blurred.jpg
    ├── 04_threshold.jpg
    ├── 05_morphology.jpg
    ├── 06_edges.jpg
    └── 07_detections.jpg
```

## Troubleshooting

### No slips detected

- Try adjusting `MIN_SLIP_AREA` and `MAX_SLIP_AREA` in `config.py`
- Use `--debug` flag to see intermediate processing steps
- Check if slips have sufficient contrast with background

### Too many false detections

- Increase `MIN_SLIP_AREA` to filter out small noise
- Adjust `MIN_ASPECT_RATIO` and `MAX_ASPECT_RATIO` to match your slip dimensions

### Slips are cut off or incomplete

- Increase `PADDING` in `config.py`
- Adjust edge detection thresholds (`CANNY_THRESHOLD_1`, `CANNY_THRESHOLD_2`)

## Technical Details

The system uses traditional computer vision techniques:

- **Adaptive Thresholding**: Handles varying lighting conditions
- **Morphological Operations**: Closes gaps and connects nearby edges
- **Contour Analysis**: Identifies rectangular regions representing slips
- **Spatial Sorting**: Orders slips logically (top-to-bottom, left-to-right)


