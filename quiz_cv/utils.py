"""
Utility functions for image processing and slip detection.
"""

import cv2
import numpy as np
from typing import Tuple, List


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Order points in the order: top-left, top-right, bottom-right, bottom-left.
    
    Args:
        pts: Array of 4 points
        
    Returns:
        Ordered array of points
    """
    rect = np.zeros((4, 2), dtype="float32")
    
    # Sum and difference to find corners
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    
    rect[0] = pts[np.argmin(s)]      # Top-left has smallest sum
    rect[2] = pts[np.argmax(s)]      # Bottom-right has largest sum
    rect[1] = pts[np.argmin(diff)]   # Top-right has smallest difference
    rect[3] = pts[np.argmax(diff)]   # Bottom-left has largest difference
    
    return rect


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    Apply perspective transformation to get a top-down view of the region.
    
    Args:
        image: Input image
        pts: Four corner points of the region
        
    Returns:
        Warped image with perspective correction
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    # Compute width of the new image
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    # Compute height of the new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # Destination points for the transform
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")
    
    # Compute perspective transform matrix and apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    return warped


def resize_image(image: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
    """
    Resize image while maintaining aspect ratio.
    
    Args:
        image: Input image
        max_width: Maximum width
        max_height: Maximum height
        
    Returns:
        Resized image
    """
    height, width = image.shape[:2]
    
    # Calculate scaling factor
    scale = min(max_width / width, max_height / height)
    
    # Only resize if image is larger than max dimensions
    if scale < 1:
        new_width = int(width * scale)
        new_height = int(height * scale)
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    return image


def visualize_detections(image: np.ndarray, contours: List[np.ndarray], 
                        color: Tuple[int, int, int] = (0, 255, 0), 
                        thickness: int = 2) -> np.ndarray:
    """
    Draw bounding boxes around detected contours for visualization.
    
    Args:
        image: Input image
        contours: List of contours to visualize
        color: BGR color for bounding boxes
        thickness: Line thickness
        
    Returns:
        Image with drawn bounding boxes
    """
    vis_image = image.copy()
    
    for i, contour in enumerate(contours):
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Draw rectangle
        cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, thickness)
        
        # Add label
        cv2.putText(vis_image, f"Slip {i+1}", (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return vis_image


def get_contour_area(contour: np.ndarray) -> float:
    """
    Calculate the area of a contour.
    
    Args:
        contour: Input contour
        
    Returns:
        Area of the contour
    """
    return cv2.contourArea(contour)


def get_aspect_ratio(contour: np.ndarray) -> float:
    """
    Calculate the aspect ratio (width/height) of a contour's bounding box.
    
    Args:
        contour: Input contour
        
    Returns:
        Aspect ratio
    """
    x, y, w, h = cv2.boundingRect(contour)
    return float(w) / h if h > 0 else 0


def sort_contours(contours: List[np.ndarray], method: str = "left-to-right") -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Sort contours based on their position.
    
    Args:
        contours: List of contours
        method: Sorting method ("left-to-right", "right-to-left", "top-to-bottom", "bottom-to-top")
        
    Returns:
        Tuple of (sorted contours, bounding boxes)
    """
    reverse = False
    i = 0
    
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    
    # Get bounding boxes
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    
    # Sort by x or y coordinate
    (contours, bounding_boxes) = zip(*sorted(zip(contours, bounding_boxes),
                                            key=lambda b: b[1][i], reverse=reverse))
    
    return list(contours), list(bounding_boxes)
