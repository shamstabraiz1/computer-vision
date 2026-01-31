"""
CNN-Based Payment Slip Detector using YOLOv8

This script uses a pretrained YOLOv8 model to automatically detect and extract
payment slips from images. Works completely offline after initial model download.

Author: AI Assistant
Date: 2026-02-01
"""

import cv2
import numpy as np
import os
from pathlib import Path
from ultralytics import YOLO


class CNNSlipDetector:
    """
    Payment slip detector using pretrained YOLOv8 CNN model.
    """
    
    def __init__(self, model_size='n', confidence_threshold=0.25):
        """
        Initialize the CNN detector.
        
        Args:
            model_size: YOLOv8 model size ('n', 's', 'm', 'l', 'x')
                       n=nano (fastest), s=small, m=medium, l=large, x=xlarge (most accurate)
            confidence_threshold: Minimum confidence for detections (0.0-1.0)
        """
        self.confidence_threshold = confidence_threshold
        self.model_name = f'yolov8{model_size}.pt'
        
        print(f"Loading YOLOv8 model: {self.model_name}")
        print("(First run will download the model, then it works offline)")
        
        # Load pretrained YOLOv8 model
        self.model = YOLO(self.model_name)
        
        print(f"✓ Model loaded successfully!")
        
    def detect_slips(self, image_path, min_area_ratio=0.01, max_area_ratio=0.4):
        """
        Detect payment slips in an image.
        
        Args:
            image_path: Path to input image
            min_area_ratio: Minimum area as ratio of image size (filter small detections)
            max_area_ratio: Maximum area as ratio of image size (filter large detections)
            
        Returns:
            boxes: List of bounding boxes [(x, y, w, h), ...]
            confidences: List of confidence scores
            image: Original image (BGR)
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        h, w = image.shape[:2]
        image_area = h * w
        
        print(f"\nProcessing image: {w}x{h} pixels")
        
        # Run YOLOv8 inference
        print("Running CNN detection...")
        results = self.model(image, conf=self.confidence_threshold, verbose=False)
        
        # Extract detections
        boxes = []
        confidences = []
        
        # YOLOv8 returns results for each image
        for result in results:
            # Get bounding boxes and confidences
            for box in result.boxes:
                # Get coordinates (xyxy format)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                # Convert to xywh format
                x, y, w_box, h_box = int(x1), int(y1), int(x2-x1), int(y2-y1)
                
                # Calculate area ratio
                box_area = w_box * h_box
                area_ratio = box_area / image_area
                
                # Filter by area
                if min_area_ratio <= area_ratio <= max_area_ratio:
                    # Filter by aspect ratio (receipts are usually tall or wide rectangles)
                    aspect = w_box / h_box if h_box > 0 else 0
                    if 0.2 <= aspect <= 5.0:
                        boxes.append((x, y, w_box, h_box))
                        confidences.append(conf)
                        print(f"  Detected slip: {w_box}x{h_box}px, confidence={conf:.2f}, class={cls}")
        
        print(f"\n✓ Found {len(boxes)} payment slips")
        
        return boxes, confidences, image
    
    def extract_and_save_slips(self, image_path, output_dir="output_slips_cnn", 
                               min_area_ratio=0.01, max_area_ratio=0.4):
        """
        Detect and extract payment slips, saving them to individual files.
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save extracted slips
            min_area_ratio: Minimum area ratio for filtering
            max_area_ratio: Maximum area ratio for filtering
            
        Returns:
            num_slips: Number of slips extracted
        """
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Detect slips
        boxes, confidences, image = self.detect_slips(
            image_path, 
            min_area_ratio=min_area_ratio,
            max_area_ratio=max_area_ratio
        )
        
        if len(boxes) == 0:
            print("\n⚠ No slips detected. Try adjusting the confidence threshold or area ratios.")
            return 0
        
        # Sort boxes by position (top to bottom, left to right)
        sorted_indices = sorted(range(len(boxes)), key=lambda i: (boxes[i][1], boxes[i][0]))
        boxes = [boxes[i] for i in sorted_indices]
        confidences = [confidences[i] for i in sorted_indices]
        
        # Extract and save each slip
        print(f"\nExtracting slips to '{output_dir}'...")
        for idx, ((x, y, w, h), conf) in enumerate(zip(boxes, confidences), 1):
            # Extract slip region
            slip = image[y:y+h, x:x+w]
            
            # Save slip
            output_path = os.path.join(output_dir, f"slip_{idx}.jpg")
            cv2.imwrite(output_path, slip)
            print(f"  Saved: slip_{idx}.jpg ({w}x{h}px, conf={conf:.2f})")
        
        # Create visualization
        vis_image = image.copy()
        for idx, ((x, y, w, h), conf) in enumerate(zip(boxes, confidences), 1):
            # Draw bounding box
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 3)
            
            # Add label with slip number and confidence
            label = f"Slip {idx} ({conf:.2f})"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # Draw label background
            cv2.rectangle(vis_image, (x, y - label_size[1] - 10), 
                         (x + label_size[0], y), (0, 255, 0), -1)
            
            # Draw label text
            cv2.putText(vis_image, label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Save visualization
        vis_path = os.path.join(output_dir, "detection_visualization.jpg")
        cv2.imwrite(vis_path, vis_image)
        print(f"\n✓ Saved visualization: detection_visualization.jpg")
        
        return len(boxes)


def main():
    """
    Main function to run CNN-based slip detection.
    """
    print("=" * 60)
    print("CNN-Based Payment Slip Detector (YOLOv8)")
    print("=" * 60)
    
    # Configuration
    input_image = "input_image.jpg"
    output_dir = "output_slips_cnn"
    
    # Check if input image exists
    if not os.path.exists(input_image):
        print(f"\n❌ Error: Image file '{input_image}' not found!")
        print("Please place your image as 'input_image.jpg' in the project folder.")
        return
    
    try:
        # Initialize detector
        # Options: 'n' (fastest), 's', 'm' (balanced), 'l', 'x' (most accurate)
        detector = CNNSlipDetector(model_size='n', confidence_threshold=0.25)
        
        # Detect and extract slips
        num_slips = detector.extract_and_save_slips(
            input_image, 
            output_dir=output_dir,
            min_area_ratio=0.01,  # Minimum 1% of image
            max_area_ratio=0.4    # Maximum 40% of image
        )
        
        print("\n" + "=" * 60)
        if num_slips > 0:
            print(f"✓ SUCCESS! Extracted {num_slips} payment slips")
            print(f"✓ Slips saved in '{output_dir}' folder")
        else:
            print("⚠ No slips detected")
            print("\nTroubleshooting tips:")
            print("1. Try lowering confidence_threshold (e.g., 0.15)")
            print("2. Adjust min_area_ratio and max_area_ratio")
            print("3. Use a larger model (model_size='m' or 'l')")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
