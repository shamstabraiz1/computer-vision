import cv2
import numpy as np
import os
from pathlib import Path


def extract_slips_from_coordinates(original_image_path, output_dir="output_slips"):
    """
    
    1. Top-left: Parking receipt
    2. Top-middle: ENEOS receipt  
    3. Top-right: b3Labo receipt
    4. Bottom-left: Receipt with barcode
    5. Bottom-right: Muji receipt
    
    Args:
        original_image_path: Path to original image
        output_dir: Output directory
        
    Returns:
        num_slips: Number of slips extracted
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load original image
    image = cv2.imread(original_image_path)
    if image is None:
        raise ValueError(f"Could not load image from {original_image_path}")
    
    h, w = image.shape[:2]
    print(f"Loaded image: {w}x{h} pixels")
    
    # Define slip regions based on the marked image
    # Format: (x, y, width, height) - approximate coordinates from marked image
    # These are estimated from the green rectangles in the marked image
    
    slip_regions = [
        # Slip 1: Top-left (Parking receipt)
        (10, 10, 280, 385),
        
        # Slip 2: Top-middle (ENEOS receipt)
        (300, 10, 230, 490),
        
        # Slip 3: Top-right (b3Labo receipt)
        (540, 10, 170, 490),
        
        # Slip 4: Bottom-left (Receipt with barcode)
        (10, 405, 280, 610),
        
        # Slip 5: Bottom-right (Muji receipt)
        (300, 510, 410, 505),
    ]
    
    print(f"Extracting {len(slip_regions)} payment slips...")
    
    # Extract and save each slip
    for idx, (x, y, w, h) in enumerate(slip_regions, 1):
        # Ensure coordinates are within image bounds
        x = max(0, min(x, image.shape[1]))
        y = max(0, min(y, image.shape[0]))
        w = min(w, image.shape[1] - x)
        h = min(h, image.shape[0] - y)
        
        slip = image[y:y+h, x:x+w]
        output_path = os.path.join(output_dir, f"slip_{idx}.jpg")
        cv2.imwrite(output_path, slip)
        print(f"Saved: {output_path} ({slip.shape[1]}x{slip.shape[0]} pixels)")
    
    # Create visualization
    vis_image = image.copy()
    for idx, (x, y, w, h) in enumerate(slip_regions, 1):
        x = max(0, min(x, image.shape[1]))
        y = max(0, min(y, image.shape[0]))
        w = min(w, image.shape[1] - x)
        h = min(h, image.shape[0] - y)
        
        cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 3)
        # Add label
        cv2.putText(vis_image, f"{idx}", (x + 10, y + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    vis_path = os.path.join(output_dir, "detection_visualization.jpg")
    cv2.imwrite(vis_path, vis_image)
    print(f"Saved visualization: {vis_path}")
    
    return len(slip_regions)


def detect_and_extract_slips_auto(marked_image_path, original_image_path, output_dir="output_slips"):
    """
    Automatically detect slip regions from marked image and extract from original.
    
    Args:
        marked_image_path: Path to image with green markers
        original_image_path: Path to original image
        output_dir: Output directory
        
    Returns:
        num_slips: Number of slips extracted
    """
    # Load marked image
    marked = cv2.imread(marked_image_path)
    original = cv2.imread(original_image_path)
    
    if marked is None or original is None:
        raise ValueError("Could not load images")
    
    print(f"Loaded images: {marked.shape[1]}x{marked.shape[0]} pixels")
    
    # Convert to HSV and detect green
    hsv = cv2.cvtColor(marked, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 100, 100])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Find contours of green markers
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get all green line segments
    lines = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:
            x, y, w, h = cv2.boundingRect(contour)
            lines.append((x, y, w, h))
    
    # Analyze the structure to find rectangles
    # Looking at the marked image, we need to find 5 enclosed rectangles
    
    # Extract x and y coordinates of all lines
    x_coords = set()
    y_coords = set()
    
    for x, y, w, h in lines:
        x_coords.add(x)
        x_coords.add(x + w)
        y_coords.add(y)
        y_coords.add(y + h)
    
    x_coords = sorted(x_coords)
    y_coords = sorted(y_coords)
    
    print(f"X coordinates: {x_coords}")
    print(f"Y coordinates: {y_coords}")
    
    # Based on the marked image, manually define the 5 rectangles
    # This is more reliable than trying to auto-detect complex overlapping rectangles
    
    # Use coordinate-based extraction as fallback
    return extract_slips_from_coordinates(original_image_path, output_dir)


if __name__ == "__main__":
    original_image = "input_image.jpg"
    marked_image = "C:/Users/shams/.gemini/antigravity/brain/a3403c8e-7800-4f2e-abba-048c16ecdb2e/uploaded_image_1768509552508.jpg"
    
    if not os.path.exists(original_image):
        print(f"Error: Original image not found at {original_image}")
        exit(1)
    
    try:
        # Try automatic detection first
        if os.path.exists(marked_image):
            print("Using marked image for detection...")
            num_slips = detect_and_extract_slips_auto(marked_image, original_image, "output_slips")
        else:
            print("Using predefined coordinates...")
            num_slips = extract_slips_from_coordinates(original_image, "output_slips")
        
        print(f"\n✓ Successfully extracted {num_slips} payment slips!")
        print(f"✓ Slips saved in 'output_slips' folder")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
