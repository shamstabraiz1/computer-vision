import cv2
import numpy as np
import os
from pathlib import Path


def find_text_density_regions(gray, window_size=50):
    """
    Find regions with high text density by analyzing variance.
    
    Args:
        gray: Grayscale image
        window_size: Size of sliding window
        
    Returns:
        density_map: Map showing text density
    """
    # Calculate local variance (text regions have high variance)
    mean = cv2.blur(gray.astype(float), (window_size, window_size))
    mean_sq = cv2.blur((gray.astype(float) ** 2), (window_size, window_size))
    variance = mean_sq - (mean ** 2)
    
    # Normalize to 0-255
    variance = np.clip(variance, 0, 255).astype(np.uint8)
    
    return variance


def detect_slip_regions_adaptive(image):
    """
    Detect payment slip regions using adaptive methods.
    
    Args:
        image: Input BGR image
        
    Returns:
        boxes: List of bounding boxes [(x, y, w, h), ...]
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # Method 1: Text density analysis
    density = find_text_density_regions(gray, window_size=30)
    
    # Threshold to get high-density regions
    _, binary = cv2.threshold(density, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological operations to merge nearby regions
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
    morph = cv2.morphologyEx(morph, cv2.MORPH_DILATE, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Calculate area threshold
    min_area = (h * w) * 0.015  # At least 1.5% of image
    max_area = (h * w) * 0.35   # At most 35% of image
    
    boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        
        if min_area <= area <= max_area:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check aspect ratio
            aspect = float(w) / h if h > 0 else 0
            if 0.2 <= aspect <= 5.0:  # Very permissive
                boxes.append((x, y, w, h))
    
    # Sort boxes by position (top to bottom, left to right)
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    
    return boxes, morph


def split_into_grid(image, rows=2, cols=3):
    """
    Split image into a grid and extract non-empty cells.
    
    Args:
        image: Input image
        rows: Number of rows
        cols: Number of columns
        
    Returns:
        slips: List of extracted slip images
        boxes: List of bounding boxes
    """
    h, w = image.shape[:2]
    cell_h = h // rows
    cell_w = w // cols
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    slips = []
    boxes = []
    
    for i in range(rows):
        for j in range(cols):
            y1 = i * cell_h
            y2 = (i + 1) * cell_h if i < rows - 1 else h
            x1 = j * cell_w
            x2 = (j + 1) * cell_w if j < cols - 1 else w
            
            cell = image[y1:y2, x1:x2]
            cell_gray = gray[y1:y2, x1:x2]
            
            # Check if cell contains content (not mostly white/empty)
            mean_val = np.mean(cell_gray)
            std_val = np.std(cell_gray)
            
            # If there's significant variation, it likely contains a slip
            if std_val > 20 and mean_val < 240:  # Not too bright and has variation
                slips.append(cell)
                boxes.append((x1, y1, x2 - x1, y2 - y1))
    
    return slips, boxes


def split_slips(image_path, output_dir="output_slips"):
    """
    Main function to split payment slips from an image.
    
    Args:
        image_path: Path to input image
        output_dir: Directory to save extracted slips
        
    Returns:
        num_slips: Number of slips extracted
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    print(f"Loaded image: {image.shape[1]}x{image.shape[0]} pixels")
    
    # Try adaptive detection first
    print("Attempting adaptive detection...")
    boxes, debug_morph = detect_slip_regions_adaptive(image)
    
    # Save debug image
    cv2.imwrite(os.path.join(output_dir, "debug_morph.jpg"), debug_morph)
    
    print(f"Adaptive detection found {len(boxes)} regions")
    
    # If adaptive detection doesn't find enough slips, try grid method
    if len(boxes) < 3:
        print("Trying grid-based detection...")
        # Try different grid configurations
        for rows, cols in [(2, 3), (3, 2), (2, 2), (3, 3)]:
            slips_grid, boxes_grid = split_into_grid(image, rows, cols)
            print(f"  Grid {rows}x{cols}: found {len(slips_grid)} regions")
            
            if len(slips_grid) >= len(boxes):
                boxes = boxes_grid
                break
    
    print(f"Final detection: {len(boxes)} payment slips")
    
    # Extract slips from boxes
    slips = []
    for x, y, w, h in boxes:
        slip = image[y:y+h, x:x+w]
        slips.append(slip)
    
    # Save each slip
    for idx, slip in enumerate(slips, 1):
        output_path = os.path.join(output_dir, f"slip_{idx}.jpg")
        cv2.imwrite(output_path, slip)
        print(f"Saved: {output_path} ({slip.shape[1]}x{slip.shape[0]} pixels)")
    
    # Create visualization
    vis_image = image.copy()
    for x, y, w, h in boxes:
        cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 3)
        
    vis_path = os.path.join(output_dir, "detection_visualization.jpg")
    cv2.imwrite(vis_path, vis_image)
    print(f"Saved visualization: {vis_path}")
    
    return len(slips)


if __name__ == "__main__":
    # Path to the input image
    input_image = "input_image.jpg"
    
    # Check if file exists
    if not os.path.exists(input_image):
        print(f"Error: Image file '{input_image}' not found!")
        print("Please update the 'input_image' variable with the correct path.")
        exit(1)
    
    # Split the slips
    try:
        num_slips = split_slips(input_image, output_dir="output_slips")
        print(f"\n✓ Successfully extracted {num_slips} payment slips!")
        print(f"✓ Slips saved in 'output_slips' folder")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
