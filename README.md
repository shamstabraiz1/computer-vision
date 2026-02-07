# Dollar Bill Value Detection using HOG + SVM

## Quiz 2_CV

This project implements a **dollar bill denomination classifier** using classic Computer Vision techniques to detect whether a bill is $1, $5, $10, or $20.

---

## 📊 Results

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 100% |
| **Cross-Validation Accuracy** | 94.61% |
| **Training Samples** | 130 |
| **Test Samples** | 31 |

---

## 🗂️ Project Structure

## 🔬 Computer Vision Techniques

### 1. HOG (Histogram of Oriented Gradients)

HOG is a **feature descriptor** used in computer vision for object detection. It works by:

1. **Gradient Computation**: Calculates the gradient magnitude and direction at each pixel
2. **Cell Division**: Divides the image into small cells (8×8 pixels)
3. **Histogram Creation**: Creates a histogram of gradient orientations for each cell (9 bins, 0°-180°)
4. **Block Normalization**: Groups cells into blocks (2×2 cells) and normalizes for lighting invariance
5. **Feature Vector**: Concatenates all normalized histograms into a single feature vector

**Why HOG works for bills:**
- Captures **edges and contours** (portraits, numbers, borders)
- **Invariant to lighting changes** due to normalization
- **Compact representation** (3,780 features per image)

```python
# HOG Parameters used
win_size = (128, 64)      # Image size
block_size = (16, 16)     # Block size for normalization
block_stride = (8, 8)     # Overlap between blocks
cell_size = (8, 8)        # Cell size for histogram
nbins = 9                 # Number of orientation bins
```

### 2. SVM (Support Vector Machine)

SVM is a **supervised learning algorithm** that finds the optimal hyperplane to separate classes:

- **Kernel**: RBF (Radial Basis Function) - maps features to higher dimensions
- **C Parameter**: 10 (controls trade-off between margin and misclassification)
- **Gamma**: 'scale' (kernel coefficient)

**Grid Search**: Tried 16 combinations of C and gamma with 3-fold cross-validation to find best parameters.

---

## 📝 Code Explanation

### bill_detection.py

#### Imports (Lines 11-24)
```python
import os, shutil, random          # File operations
import numpy as np                  # Numerical operations
import cv2                          # OpenCV for image processing
from PIL import Image               # Image loading (TIF support)
import matplotlib.pyplot as plt     # Visualization
from sklearn.svm import SVC         # Support Vector Machine
from sklearn.preprocessing import StandardScaler  # Feature normalization
from sklearn.metrics import classification_report, confusion_matrix
```

#### Configuration (Lines 27-40)
```python
BASE_DIR = r"c:\University\Semester7\Computer Vision\Quiz 1\Quiz 2"
IMG_SIZE = (128, 64)    # Width x Height - optimal for HOG
TEST_SPLIT = 0.2        # 20% for testing
CLASSES = ['1', '5', '10', '20']
```

#### Key Functions

| Function | Lines | Description |
|----------|-------|-------------|
| `prepare_data()` | 43-79 | Splits dataset into 80% train, 20% test |
| `extract_hog_features()` | 82-99 | Extracts HOG descriptor from grayscale image |
| `load_and_preprocess_image()` | 102-117 | Loads image, resizes to 128×64, converts to grayscale |
| `load_dataset()` | 120-143 | Loads all images from a directory and extracts features |
| `train_svm()` | 146-175 | Trains SVM with grid search for best parameters |
| `evaluate_model()` | 178-211 | Evaluates on test set, prints metrics, saves confusion matrix |
| `show_sample_predictions()` | 251-278 | Displays 8 random test predictions |
| `main()` | 281-311 | Orchestrates the entire pipeline |

---

### Detailed Function Breakdown

#### 1. `prepare_data()` - Data Splitting

```python
def prepare_data():
    for class_name in CLASSES:  # Loop through ['1', '5', '10', '20']
        # Get all image files
        images = [f for f in os.listdir(source_dir) 
                  if f.endswith(('.tif', '.tiff', '.jpg', '.png'))]
        
        # Shuffle randomly
        random.shuffle(images)
        
        # Split: first 20% for test, rest for train
        test_count = int(len(images) * 0.2)
        test_images = images[:test_count]
        train_images = images[test_count:]
        
        # Copy files to respective folders
        for img in test_images:
            shutil.copy2(src, dst)  # Copy to Test/
        for img in train_images:
            shutil.copy2(src, dst)  # Copy to Train/
```

**Key Point**: Test images are **never** present in training data.

---

#### 2. `extract_hog_features()` - Feature Extraction

```python
def extract_hog_features(image):
    # Create HOG descriptor with specific parameters
    hog = cv2.HOGDescriptor(
        win_size,      # (128, 64)
        block_size,    # (16, 16)
        block_stride,  # (8, 8)
        cell_size,     # (8, 8)
        nbins          # 9 orientation bins
    )
    
    # Compute features - returns array of 3780 values
    features = hog.compute(image)
    return features.flatten()
```

**Feature Vector Size Calculation**:
- Blocks in width: (128 - 16) / 8 + 1 = 15
- Blocks in height: (64 - 16) / 8 + 1 = 7
- Cells per block: 2 × 2 = 4
- Bins per cell: 9
- **Total: 15 × 7 × 4 × 9 = 3,780 features**

---

#### 3. `load_and_preprocess_image()` - Image Preprocessing

```python
def load_and_preprocess_image(image_path):
    # Read image with OpenCV
    img = cv2.imread(image_path)
    
    # Fallback for TIF files
    if img is None:
        pil_img = Image.open(image_path)
        img = np.array(pil_img.convert('RGB'))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Resize to 128x64 (required for HOG)
    img_resized = cv2.resize(img, (128, 64))
    
    # Convert to grayscale (HOG works on gradient, color not needed)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    return gray, img_resized
```

---

#### 4. `train_svm()` - Training with Grid Search

```python
def train_svm(X_train, y_train):
    # Normalize features (mean=0, std=1)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Define parameter grid for search
    param_grid = {
        'C': [0.1, 1, 10, 100],          # Regularization
        'gamma': ['scale', 'auto', 0.01, 0.1]  # Kernel coefficient
    }
    
    # SVM with RBF kernel
    svm = SVC(kernel='rbf', probability=True)
    
    # Grid search with 3-fold cross-validation
    grid_search = GridSearchCV(svm, param_grid, cv=3)
    grid_search.fit(X_train_scaled, y_train)
    
    # Returns best model and scaler for test data
    return grid_search.best_estimator_, scaler
```

**Best Parameters Found**: C=10, gamma='scale'

---

#### 5. `evaluate_model()` - Testing

```python
def evaluate_model(model, scaler, X_test, y_test):
    # Scale test features using same scaler
    X_test_scaled = scaler.transform(X_test)
    
    # Get predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Print classification report (precision, recall, f1)
    print(classification_report(y_test, y_pred))
    
    # Create and save confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.savefig('confusion_matrix.png')
```

---

## 🚀 How to Run

### Prerequisites
```bash
pip install opencv-python numpy matplotlib scikit-learn seaborn pillow
```

### Run Training
```bash
cd "c:\University\Semester7\Computer Vision\Quiz 1\Quiz 2"
python bill_detection.py
```

### Expected Output
```
============================================================
   DOLLAR BILL VALUE DETECTION
   Using HOG Features + SVM Classifier
   Quiz 2 - Computer Vision
============================================================
STEP 1: Preparing Train/Test Split
$1 bill: 72 train, 18 test
$5 bill: 11 train, 2 test
$10 bill: 18 train, 4 test
$20 bill: 29 train, 7 test

STEP 2: Training SVM Classifier
Feature vector size: 3780
Best parameters: {'C': 10, 'gamma': 'scale'}
Best cross-validation accuracy: 94.61%

STEP 3: Evaluating on Test Set
TEST ACCURACY: 100.00%
```

---

## 📈 Output Files

| File | Description |
|------|-------------|
| `confusion_matrix.png` | Shows true vs predicted labels (perfect diagonal = no errors) |
| `sample_predictions.png` | 8 random test images with their predictions |

---

## 🔑 Key Takeaways

1. **HOG Features** are excellent for bill recognition because they capture edges, textures, and shapes
2. **SVM with RBF kernel** works well for image classification when features are properly extracted
3. **Data augmentation** was not needed due to the nature of HOG features (already robust to small variations)
4. **Class imbalance** ($1 has more samples) was handled implicitly by SVM
5. **100% accuracy** achieved on the test set, showing the method is effective for this task
