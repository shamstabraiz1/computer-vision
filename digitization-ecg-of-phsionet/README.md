# ECG Signal Digitization from PhysioNet Images

##  Competition Overview
This project is part of the **PhysioNet ECG Image Digitization** Kaggle competition.  
The objective is to convert scanned ECG images into accurate **1-D digital ECG signals** that match the original ground-truth waveforms.

---

##  Approach Summary
This notebook implements a **classical image-processing–based pipeline** (non-deep learning) to digitize ECG signals from images.  
The solution relies on traditional **computer vision and signal processing techniques** rather than neural networks.

---

##  Methodology

### 1. Image Preprocessing
- Convert ECG images to grayscale
- Apply Gaussian blur to reduce noise
- Use thresholding to isolate ECG traces

### 2. Signal Extraction
- Scan image column-wise to detect waveform pixels
- Extract vertical positions of the ECG trace
- Interpolate missing or noisy segments

### 3. Signal Smoothing
- Apply **Savitzky–Golay filter** to smooth extracted signals
- Reduce sharp spikes and artifacts

### 4. Normalization
- Normalize signal amplitude
- Resize signal length to the required output format

### 5. Submission Generation
- Convert processed signals into CSV format
- Ensure compatibility with Kaggle submission rules

---

##  Results

- **Public Score:** `-3.33`

The negative public score indicates that the extracted ECG signals do not sufficiently match the ground-truth signals under the competition evaluation metric.

### Limitations
- Classical image-processing methods are sensitive to:
  - Noise and low-contrast images
  - ECG grid line interference
  - Signal discontinuities
- No learning-based feature extraction
- Minor extraction errors significantly affect final score

---

##  Key Learnings
- ECG digitization is a complex task requiring robust signal understanding
- Rule-based pipelines struggle with diverse image conditions
- Deep learning approaches are more suitable for competitive performance

---

##  Future Improvements
- Use CNN / U-Net–based segmentation models
- Separate grid removal and waveform extraction
- Improve signal scaling and alignment
- Handle multi-lead ECG images more effectively

---

##  Technologies Used
- Python
- OpenCV
- NumPy
- SciPy
- Pandas

---

##  Files
- `digitization-ecg-of-physionet.ipynb` — Main notebook
- `submission.csv` — Generated submission file

---

##  Disclaimer
This project represents a **learning-based experimental attempt** using classical techniques.  
The low public score highlights the limitations of non-learning approaches for this task.
