"""
Configuration parameters for payment slip detection and splitting.
"""

# Image preprocessing parameters
GAUSSIAN_BLUR_KERNEL = (5, 5)
CANNY_THRESHOLD_1 = 50
CANNY_THRESHOLD_2 = 150

# Contour detection parameters
MIN_SLIP_AREA = 8000  # Minimum area in pixels for a valid slip
MAX_SLIP_AREA = 600000  # Maximum area in pixels for a valid slip
MIN_ASPECT_RATIO = 0.35  # Minimum width/height ratio
MAX_ASPECT_RATIO = 3.0  # Maximum width/height ratio

# Contour approximation
APPROX_EPSILON_FACTOR = 0.02  # Factor for contour approximation

# Output parameters
OUTPUT_IMAGE_QUALITY = 95  # JPEG quality (1-100)
OUTPUT_FORMAT = "jpg"  # Output image format
PADDING = 10  # Padding around detected slips in pixels

# Visualization parameters (for debugging)
BBOX_COLOR = (0, 255, 0)  # Green color for bounding boxes
BBOX_THICKNESS = 2

# Morphological operations
MORPH_KERNEL_SIZE = (3, 3)
MORPH_ITERATIONS = 2

# Resize parameters (if input image is too large)
MAX_IMAGE_WIDTH = 2000
MAX_IMAGE_HEIGHT = 2000
