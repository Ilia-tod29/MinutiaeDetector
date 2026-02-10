# Fingerprint Minutiae Detector

A complete Python implementation of fingerprint minutiae detection using the **Crossing Number (CN) method** on 1-pixel-wide skeletons.

## Features

- **Complete preprocessing pipeline**: CLAHE, noise reduction, ROI segmentation, binarization
- **Robust skeletonization**: Automatic fallback between opencv-contrib, scikit-image, and custom Zhang-Suen
- **CN-based minutiae detection**: Detects ridge endings (CN=1) and bifurcations (CN=3)
- **Advanced filtering**: Boundary exclusion, deduplication
- **Rich visualization**: Color-coded overlay with green (endings) and red (bifurcations) markers
- **Detailed logging**: Step-by-step progress and statistics

## Installation

### Instalation flow
```bash
# Navigate to project directory
cd /path-to-folder/MinutiaeDetector

# Install dependencies
pip install -r requirements.txt

# Optional: Install opencv-contrib for better thinning
pip install opencv-contrib-python

# Optional: Install scikit-image as fallback
pip install scikit-image
```

### Recommended Installation (Manual)
```bash
pip install opencv-contrib-python numpy
```

### Full Installation (All Fallbacks) (Manual)
```bash
pip install opencv-contrib-python numpy scikit-image
```

## Usage

### Basic Usage
```bash
python minutiae_cn.py --input fingerprint.tif --outdir outputs/
```

### Recommended Settings
```bash
python minutiae_cn.py --input fingerprint.png --outdir outputs/ --clahe --adaptive
```

### Advanced Settings
```bash
python minutiae_cn.py \
  --input fingerprint.bmp \
  --outdir outputs/ \
  --clahe \
  --adaptive \
  --margin 20 \
  --dedup_radius 12 \
  --roi_blocksize 16 \
  --circle_radius 5
```

### For Dark Ridge Images
```bash
python minutiae_cn.py --input fingerprint.tif --outdir outputs/ --clahe --adaptive --invert
```

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--input` | str | *required* | Path to input fingerprint image |
| `--outdir` | str | `outputs/` | Output directory for results |
| `--clahe` | flag | off | Use CLAHE contrast enhancement (recommended) |
| `--adaptive` | flag | off | Use adaptive threshold (recommended) |
| `--invert` | flag | off | Invert binary if ridges are dark |
| `--margin` | int | 16 | Border/ROI margin in pixels |
| `--dedup_radius` | int | 10 | Deduplication radius in pixels |
| `--roi_blocksize` | int | 16 | Block size for ROI segmentation |
| `--circle_radius` | int | 5 | Minutiae marker radius |

## Output Files

The script generates the following files in the output directory:

1. **01_preprocessed.png** - Contrast enhanced and denoised image
2. **02_roi_mask.png** - Segmented fingerprint region (white = fingerprint)
3. **03_binary.png** - Binarized ridge map (white = ridges)
4. **04_skeleton.png** - 1-pixel-wide skeleton
5. **05_overlay.png** - Final visualization with minutiae markers
   - 🟢 Green circles = Ridge endings
   - 🔴 Red circles = Bifurcations

## Algorithm Overview

### 1. Preprocessing
- **CLAHE**: Contrast Limited Adaptive Histogram Equalization for local contrast
- **Gaussian Blur**: Removes high-frequency noise
- **Bilateral Filter**: Edge-preserving smoothing

### 2. ROI Segmentation
- Block-wise variance computation
- Otsu thresholding
- Morphological cleanup (open/close)
- Hole filling

### 3. Binarization
- **Adaptive Threshold**: Local threshold based on Gaussian-weighted neighborhood
- **Otsu Threshold**: Global threshold (fallback)

### 4. Morphological Cleanup
- Opening: removes small noise
- Closing: fills small gaps
- ROI masking

### 5. Skeletonization
Priority-based method selection:
1. **cv2.ximgproc.thinning** (opencv-contrib) - Zhang-Suen
2. **scikit-image skeletonize** - Morphological thinning
3. **Custom Zhang-Suen** - Pure Python fallback

### 6. Crossing Number Method

For each skeleton pixel, compute CN using 8-neighborhood:

```
Neighbor layout:
  p8  p1  p2
  p7  p   p3
  p6  p5  p4

CN = 0.5 × Σ|pi - p(i+1)| for i=1..8 (circular)
```

**Classification:**
- CN = 1 → Ridge ending
- CN = 3 → Bifurcation

### 7. Filtering
- **Boundary filter**: Remove minutiae near ROI/image borders
- **Deduplication**: Merge minutiae within specified radius

## FVC2002 Dataset Usage

To use with FVC2002 fingerprint images:

1. Download dataset from [FVC2002](https://www.kaggle.com/datasets/nageshsingh/fvc2002-fingerprints)
2. Extract to a directory
3. Run on individual images:

```bash
python minutiae_cn.py \
  --input path/to/FVC2002/DB1_B/101_1.tif \
  --outdir outputs/fvc2002/ \
  --clahe --adaptive
```

## Performance Tips

- Use `--clahe` for better contrast on uneven illumination
- Use `--adaptive` for better binarization on varying lighting
- Increase `--margin` if getting false minutiae at borders
- Decrease `--dedup_radius` if losing real minutiae
- Increase `--roi_blocksize` for faster ROI segmentation (less accurate)

## Troubleshooting

### Too many false minutiae
- Increase `--margin` to 20-30
- Increase `--dedup_radius` to 12-15
- Check ROI segmentation in `02_roi_mask.png`

### Missing real minutiae
- Decrease `--margin` to 10-12
- Decrease `--dedup_radius` to 6-8
- Check skeleton quality in `04_skeleton.png`

### Inverted ridges (ridges appear as valleys)
- Add `--invert` flag

### Poor quality skeleton
- Install opencv-contrib-python for better thinning
- Check binary image quality in `03_binary.png`

## Technical Notes

### Crossing Number Theory
The Crossing Number is defined as:
$$CN(p) = \frac{1}{2} \sum_{i=1}^{8} |p_i - p_{(i \bmod 8) + 1}|$$

where $p_i$ are binary values {0,1} of the 8 neighbors in circular order.

### Zhang-Suen Thinning
- Two-pass iterative algorithm
- Preserves topology and connectivity
- Removes pixels based on:
  - Number of black neighbors (2-6)
  - Number of black-to-white transitions (must be 1)
  - Specific patterns to prevent disconnection

## Examples

### Example 1: Basic Detection
```bash
python minutiae_cn.py --input samples/fingerprint1.png --outdir out1/
```

### Example 2: High-Quality Detection
```bash
python minutiae_cn.py \
  --input samples/fingerprint2.tif \
  --outdir out2/ \
  --clahe \
  --adaptive \
  --margin 20 \
  --dedup_radius 12
```

### Example 3: Dark Ridge Image
```bash
python minutiae_cn.py \
  --input samples/dark_ridges.bmp \
  --outdir out3/ \
  --clahe \
  --adaptive \
  --invert
```
