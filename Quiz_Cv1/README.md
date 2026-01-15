Payment Slip Splitter
This project automatically detects and splits individual payment slips from a collage image using computer vision.

Features
Fully Offline - No API calls required Automatic Detection - Detects payment slips using OpenCV High Quality - Preserves original image quality

Installation
pip install -r requirements.txt
Usage
Method 1: Automatic Detection
python split_payment_slips.py
Place your image as input_image.jpg in the project folder.

Output
All extracted slips are saved in the output_slips/ folder as:

slip_1.jpg
slip_2.jpg
slip_3.jpg
etc.
A visualization showing detected regions is also saved as detection_visualization.jpg.

Files
split_payment_slips.py - Automatic detection script
detect_regions.py - image extraction script
requirements.txt - Python dependencies
Dependencies
opencv-python (≥4.5.0)
numpy (≥1.19.0)
Pillow (≥8.0.0)
