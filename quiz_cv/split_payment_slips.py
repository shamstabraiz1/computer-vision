"""
Payment Slip Splitter - Automatic detection and splitting of payment slips from images.

This script uses computer vision techniques to detect individual payment slips
in a scanned image and save them as separate files.

Usage:
    python split_payment_slips.py --input <input_image> --output <output_directory>
    python split_payment_slips.py -i image.jpg -o output_slips/
    python split_payment_slips.py -i image.jpg -o output_slips/ --debug
"""

import cv2
import numpy as np
import os
import argparse
from pathlib import Path
from typing import List, Tuple
import config
from utils import (
    resize_image, 
    four_point_transform, 
    visualize_detections,
    get_contour_area,
    get_aspect_ratio,
    sort_contours
)


class PaymentSlipDetector:
    """Detects and extracts individual payment slips from an image."""
    
    def __init__(self, debug: bool = False):
        """
        Initialize the detector.
        
        Args:
            debug: If True, save intermediate processing steps for debugging
        """
        self.debug = debug
        self.debug_images = {}
    
    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess the image for contour detection.
        
        Args:
            image: Input BGR image
            
        Returns:
            Tuple of (grayscale image, edge-detected image)
        """
        # Resize if image is too large
        resized = resize_image(image, config.MAX_IMAGE_WIDTH, config.MAX_IMAGE_HEIGHT)
        if self.debug:
            self.debug_images['01_resized'] = resized
        
        # Convert to grayscale
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        if self.debug:
            self.debug_images['02_grayscale'] = gray
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, config.GAUSSIAN_BLUR_KERNEL, 0)
        if self.debug:
            self.debug_images['03_blurred'] = blurred
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        if self.debug:
            self.debug_images['04_threshold'] = thresh
        
        # Apply morphological operations to close gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, config.MORPH_KERNEL_SIZE)
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, 
                                iterations=config.MORPH_ITERATIONS)
        if self.debug:
            self.debug_images['05_morphology'] = morph
        
        # Edge detection
        edges = cv2.Canny(blurred, config.CANNY_THRESHOLD_1, config.CANNY_THRESHOLD_2)
        
        # Dilate edges to connect nearby contours
        dilated = cv2.dilate(edges, kernel, iterations=2)
        if self.debug:
            self.debug_images['06_edges'] = dilated
        
        return gray, dilated
    
    def detect_slips(self, image: np.ndarray, edges: np.ndarray) -> List[np.ndarray]:
        """
        Detect individual payment slips using contour detection.
        
        Args:
            image: Original grayscale image
            edges: Edge-detected image
            
        Returns:
            List of contours representing detected slips
        """
        # Find contours
        contours, hierarchy = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        print(f"Found {len(contours)} total contours")
        
        # Filter contours based on area and aspect ratio
        valid_contours = []
        
        for contour in contours:
            area = get_contour_area(contour)
            
            # Filter by area
            if area < config.MIN_SLIP_AREA or area > config.MAX_SLIP_AREA:
                continue
            
            # Filter by aspect ratio
            aspect_ratio = get_aspect_ratio(contour)
            if aspect_ratio < config.MIN_ASPECT_RATIO or aspect_ratio > config.MAX_ASPECT_RATIO:
                continue
            
            # Approximate the contour to reduce number of points
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, config.APPROX_EPSILON_FACTOR * peri, True)
            
            valid_contours.append(contour)
        
        print(f"Detected {len(valid_contours)} valid payment slips")
        
        # Sort contours from top to bottom, left to right
        if valid_contours:
            sorted_contours, _ = sort_contours(valid_contours, method="top-to-bottom")
            return sorted_contours
        
        return []
    
    def extract_slips(self, image: np.ndarray, contours: List[np.ndarray]) -> List[np.ndarray]:
        """
        Extract individual slip images from the original image.
        
        Args:
            image: Original BGR image
            contours: List of contours representing slips
            
        Returns:
            List of extracted slip images
        """
        slips = []
        
        for i, contour in enumerate(contours):
            # Get bounding rectangle with padding
            x, y, w, h = cv2.boundingRect(contour)
            
            # Add padding
            x = max(0, x - config.PADDING)
            y = max(0, y - config.PADDING)
            w = min(image.shape[1] - x, w + 2 * config.PADDING)
            h = min(image.shape[0] - y, h + 2 * config.PADDING)
            
            # Extract the slip region
            slip = image[y:y+h, x:x+w]
            slips.append(slip)
        
        return slips
    
    def save_debug_images(self, output_dir: Path):
        """
        Save debug images showing intermediate processing steps.
        
        Args:
            output_dir: Directory to save debug images
        """
        debug_dir = output_dir / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        
        for name, img in self.debug_images.items():
            output_path = debug_dir / f"{name}.jpg"
            cv2.imwrite(str(output_path), img)
        
        print(f"Debug images saved to: {debug_dir}")
    
    def process(self, input_path: str, output_dir: str) -> int:
        """
        Main processing pipeline to detect and save payment slips.
        
        Args:
            input_path: Path to input image
            output_dir: Directory to save extracted slips
            
        Returns:
            Number of slips detected and saved
        """
        # Load image
        print(f"Loading image: {input_path}")
        image = cv2.imread(input_path)
        
        if image is None:
            raise ValueError(f"Failed to load image: {input_path}")
        
        print(f"Image size: {image.shape[1]}x{image.shape[0]}")
        
        # Preprocess
        print("Preprocessing image...")
        gray, edges = self.preprocess_image(image)
        
        # Detect slips
        print("Detecting payment slips...")
        contours = self.detect_slips(gray, edges)
        
        if not contours:
            print("No payment slips detected!")
            return 0
        
        # Visualize detections if debug mode
        if self.debug:
            vis_image = visualize_detections(
                resize_image(image, config.MAX_IMAGE_WIDTH, config.MAX_IMAGE_HEIGHT),
                contours,
                config.BBOX_COLOR,
                config.BBOX_THICKNESS
            )
            self.debug_images['07_detections'] = vis_image
        
        # Extract slips
        print("Extracting slip regions...")
        resized_image = resize_image(image, config.MAX_IMAGE_WIDTH, config.MAX_IMAGE_HEIGHT)
        slips = self.extract_slips(resized_image, contours)
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save each slip
        print(f"Saving {len(slips)} slips to: {output_dir}")
        for i, slip in enumerate(slips, 1):
            filename = f"slip_{i:03d}.{config.OUTPUT_FORMAT}"
            filepath = output_path / filename
            
            # Save with high quality
            if config.OUTPUT_FORMAT.lower() in ['jpg', 'jpeg']:
                cv2.imwrite(str(filepath), slip, 
                           [cv2.IMWRITE_JPEG_QUALITY, config.OUTPUT_IMAGE_QUALITY])
            else:
                cv2.imwrite(str(filepath), slip)
            
            print(f"  Saved: {filename} ({slip.shape[1]}x{slip.shape[0]})")
        
        # Save debug images if enabled
        if self.debug:
            self.save_debug_images(output_path)
        
        print(f"\n✓ Successfully extracted {len(slips)} payment slips!")
        return len(slips)


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Automatically detect and split payment slips from an image",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python split_payment_slips.py -i receipts.jpg -o output/
  python split_payment_slips.py --input scan.png --output slips/ --debug
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        required=True,
        help='Path to input image containing payment slips'
    )
    
    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Directory to save extracted payment slips'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode (saves intermediate processing images)'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return 1
    
    try:
        # Create detector and process image
        detector = PaymentSlipDetector(debug=args.debug)
        num_slips = detector.process(args.input, args.output)
        
        return 0 if num_slips > 0 else 1
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
