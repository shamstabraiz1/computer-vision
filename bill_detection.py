"""
Dollar Bill Value Detection using HOG Features + SVM
Quiz 2 - Computer Vision

This script uses classic Computer Vision techniques:
1. HOG (Histogram of Oriented Gradients) for feature extraction
2. SVM (Support Vector Machine) for classification
"""

import os
import shutil
import random
import numpy as np
import cv2
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots without display
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
import seaborn as sns

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Configuration
BASE_DIR = r"c:\University\Semester7\Computer Vision\Quiz 1\Quiz 2"
DATASET_DIR = os.path.join(BASE_DIR, "Bill_dataset_temp", "Bill_dataset")
TRAIN_DIR = os.path.join(BASE_DIR, "Train")
TEST_DIR = os.path.join(BASE_DIR, "Test")

IMG_SIZE = (128, 64)  # Width x Height - optimal for HOG
TEST_SPLIT = 0.2  # 20% for testing

CLASSES = ['1', '5', '10', '20']
CLASS_LABELS = {'1': 0, '5': 1, '10': 2, '20': 3}


def prepare_data():
    """Split dataset into train and test sets"""
    print("=" * 50)
    print("STEP 1: Preparing Train/Test Split")
    print("=" * 50)
    
    for class_name in CLASSES:
        source_dir = os.path.join(DATASET_DIR, class_name)
        train_dest = os.path.join(TRAIN_DIR, class_name)
        test_dest = os.path.join(TEST_DIR, class_name)
        
        # Create directories if they don't exist
        os.makedirs(train_dest, exist_ok=True)
        os.makedirs(test_dest, exist_ok=True)
        
        # Get all image files (exclude .DS_Store)
        images = [f for f in os.listdir(source_dir) 
                  if f.lower().endswith(('.tif', '.tiff', '.jpg', '.jpeg', '.png'))]
        
        # Shuffle and split
        random.shuffle(images)
        test_count = max(1, int(len(images) * TEST_SPLIT))
        
        test_images = images[:test_count]
        train_images = images[test_count:]
        
        # Copy to test folder
        for img in test_images:
            src = os.path.join(source_dir, img)
            dst = os.path.join(test_dest, img)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
        
        # Copy to train folder
        for img in train_images:
            src = os.path.join(source_dir, img)
            dst = os.path.join(train_dest, img)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
        
        print(f"${class_name} bill: {len(train_images)} train, {len(test_images)} test")
    
    print("\nData split completed!")
    return True


def extract_hog_features(image):
    """
    Extract HOG (Histogram of Oriented Gradients) features from an image.
    
    HOG is a classic Computer Vision feature descriptor that captures
    edge orientations and gradients, which are excellent for object recognition.
    """
    # HOG parameters
    win_size = (128, 64)
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    nbins = 9
    
    # Create HOG descriptor
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    
    # Compute HOG features
    features = hog.compute(image)
    
    return features.flatten()


def load_and_preprocess_image(image_path):
    """Load image and preprocess for HOG feature extraction"""
    # Read image
    img = cv2.imread(image_path)
    
    if img is None:
        # Try with PIL for TIF files
        pil_img = Image.open(image_path)
        img = np.array(pil_img.convert('RGB'))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Resize to standard size for HOG
    img_resized = cv2.resize(img, IMG_SIZE)
    
    # Convert to grayscale (HOG works better on grayscale)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    return gray, img_resized


def load_dataset(data_dir):
    """Load all images and extract HOG features"""
    features = []
    labels = []
    image_paths = []
    
    for class_name in CLASSES:
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            continue
            
        images = [f for f in os.listdir(class_dir) 
                  if f.lower().endswith(('.tif', '.tiff', '.jpg', '.jpeg', '.png'))]
        
        for img_name in images:
            img_path = os.path.join(class_dir, img_name)
            try:
                gray_img, _ = load_and_preprocess_image(img_path)
                hog_features = extract_hog_features(gray_img)
                
                features.append(hog_features)
                labels.append(CLASS_LABELS[class_name])
                image_paths.append(img_path)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    
    return np.array(features), np.array(labels), image_paths


def train_svm(X_train, y_train):
    """Train SVM classifier with grid search for best parameters"""
    print("\n" + "=" * 50)
    print("STEP 2: Training SVM Classifier")
    print("=" * 50)
    
    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    print(f"Feature vector size: {X_train.shape[1]}")
    print(f"Training samples: {X_train.shape[0]}")
    
    # SVM with RBF kernel (best for image classification)
    print("\nTraining SVM with RBF kernel...")
    
    # Grid search for best parameters
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.01, 0.1]
    }
    
    svm = SVC(kernel='rbf', random_state=42, probability=True)
    grid_search = GridSearchCV(svm, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X_train_scaled, y_train)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best cross-validation accuracy: {grid_search.best_score_:.2%}")
    
    return grid_search.best_estimator_, scaler


def evaluate_model(model, scaler, X_test, y_test, test_paths):
    """Evaluate model on test set"""
    print("\n" + "=" * 50)
    print("STEP 3: Evaluating on Test Set")
    print("=" * 50)
    
    # Scale test features
    X_test_scaled = scaler.transform(X_test)
    
    # Predict
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred) * 100
    print(f"\n{'='*50}")
    print(f"TEST ACCURACY: {accuracy:.2f}%")
    print(f"{'='*50}")
    
    # Classification report
    class_labels = [f'${c}' for c in CLASSES]
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_labels))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels)
    plt.title(f'Confusion Matrix (HOG + SVM)\nTest Accuracy: {accuracy:.2f}%', fontsize=14)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'confusion_matrix.png'), dpi=150)
    plt.show()
    print(f"\nConfusion matrix saved to: {os.path.join(BASE_DIR, 'confusion_matrix.png')}")
    
    return accuracy, y_pred, y_pred_proba


def visualize_hog_features(sample_image_path):
    """Visualize HOG features on a sample image"""
    print("\n" + "=" * 50)
    print("Visualizing HOG Features")
    print("=" * 50)
    
    gray_img, color_img = load_and_preprocess_image(sample_image_path)
    
    # Compute HOG with visualization
    win_size = (128, 64)
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    nbins = 9
    
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    
    # For visualization, use skimage (if available) or just show the image
    try:
        from skimage.feature import hog as skimage_hog
        from skimage import exposure
        
        _, hog_image = skimage_hog(gray_img, orientations=9, pixels_per_cell=(8, 8),
                                    cells_per_block=(2, 2), visualize=True)
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(gray_img, cmap='gray')
        axes[1].set_title('Grayscale Image')
        axes[1].axis('off')
        
        axes[2].imshow(hog_image_rescaled, cmap='gray')
        axes[2].set_title('HOG Features Visualization')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(BASE_DIR, 'hog_visualization.png'), dpi=150)
        plt.show()
        print(f"HOG visualization saved to: {os.path.join(BASE_DIR, 'hog_visualization.png')}")
    except ImportError:
        print("skimage not available for HOG visualization, skipping...")


def show_sample_predictions(model, scaler, X_test, y_test, test_paths, y_pred):
    """Display sample predictions"""
    print("\n" + "=" * 50)
    print("Sample Predictions")
    print("=" * 50)
    
    num_samples = min(8, len(test_paths))
    indices = random.sample(range(len(test_paths)), num_samples)
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    class_labels = [f'${c}' for c in CLASSES]
    
    for i, idx in enumerate(indices):
        ax = axes[i]
        
        # Load original image
        _, color_img = load_and_preprocess_image(test_paths[idx])
        ax.imshow(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))
        
        true_label = class_labels[y_test[idx]]
        pred_label = class_labels[y_pred[idx]]
        
        color = 'green' if y_test[idx] == y_pred[idx] else 'red'
        ax.set_title(f'True: {true_label}\nPred: {pred_label}', color=color, fontsize=12)
        ax.axis('off')
    
    plt.suptitle('Sample Predictions (Green=Correct, Red=Incorrect)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'sample_predictions.png'), dpi=150)
    plt.show()
    print(f"Sample predictions saved to: {os.path.join(BASE_DIR, 'sample_predictions.png')}")


def main():
    """Main function"""
    print("\n" + "=" * 60)
    print("   DOLLAR BILL VALUE DETECTION")
    print("   Using HOG Features + SVM Classifier")
    print("   Quiz 2 - Computer Vision")
    print("=" * 60)
    
    # Step 1: Prepare data
    prepare_data()
    
    # Step 2: Load and extract features
    print("\n" + "=" * 50)
    print("Extracting HOG Features...")
    print("=" * 50)
    
    print("Loading training data...")
    X_train, y_train, train_paths = load_dataset(TRAIN_DIR)
    print(f"Training set: {len(X_train)} samples")
    
    print("Loading test data...")
    X_test, y_test, test_paths = load_dataset(TEST_DIR)
    print(f"Test set: {len(X_test)} samples")
    
    # Step 3: Train SVM
    model, scaler = train_svm(X_train, y_train)
    
    # Step 4: Evaluate
    accuracy, y_pred, _ = evaluate_model(model, scaler, X_test, y_test, test_paths)
    
    # Step 5: Visualize HOG features
    if train_paths:
        visualize_hog_features(train_paths[0])
    
    # Step 6: Show sample predictions
    show_sample_predictions(model, scaler, X_test, y_test, test_paths, y_pred)
    
    print("\n" + "=" * 60)
    print(f"   TRAINING COMPLETE!")
    print(f"   Final Test Accuracy: {accuracy:.2f}%")
    print("=" * 60)
    print(f"\nSaved files:")
    print(f"  - Confusion Matrix: {os.path.join(BASE_DIR, 'confusion_matrix.png')}")
    print(f"  - HOG Visualization: {os.path.join(BASE_DIR, 'hog_visualization.png')}")
    print(f"  - Sample Predictions: {os.path.join(BASE_DIR, 'sample_predictions.png')}")


if __name__ == "__main__":
    main()