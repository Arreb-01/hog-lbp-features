# Image Local Feature Extraction: HoG & LBP

A Python implementation for extracting and visualizing **HoG (Histogram of Oriented Gradients)** and **LBP (Local Binary Patterns)** features from images using OpenCV and scikit-image.

## Features

- **HoG Feature Extraction**: Detects edges and gradients for object detection
- **LBP Feature Extraction**: Captures local texture patterns
- **Visualization**: Generates comprehensive visual comparisons of features
- **Batch Processing**: Process multiple images automatically

## Installation

Install required dependencies:

```bash
pip install opencv-python scikit-image matplotlib numpy
```

## Usage

### Basic Usage

Run the script to process all test images:

```bash
python hog_lbp_features.py
```

### Process Custom Images

Edit the `image_files` list in the `main()` function:

```python
image_files = [
    ('your_image.jpg', 'output_prefix'),
    # Add more images as needed
]
```

## Output Files

For each input image, the script generates:

| File | Description |
|------|-------------|
| `{prefix}_hog.jpg` | Original image + HoG feature map + feature histogram |
| `{prefix}_lbp.jpg` | Original image + LBP texture map + histogram |
| `{prefix}_combined.jpg` | 2x2 comparison grid (Original, Grayscale, HoG, LBP) |

## Feature Comparison

| Feature Type | Typical Dimension | Best For |
|--------------|-------------------|----------|
| **HoG** | High (e.g., 56448) | Edge/contour detection, object detection |
| **LBP** | Low (e.g., 18) | Texture classification, face recognition |

### HoG Characteristics
- Robust to illumination changes
- Captures gradient orientation information
- Widely used in pedestrian detection

### LBP Characteristics
- Robust to illumination changes
- Captures local texture patterns
- Computationally efficient
- Rotation-invariant variants available

## Parameters

### HoG Parameters
```python
cell_size=(8, 8)      # Size of each cell
block_size=(2, 2)     # Block size in cells
orientations=9        # Number of gradient orientation bins
```

### LBP Parameters
```python
radius=2              # Radius of circle sampling
n_points=16           # Number of sampling points
method='uniform'      # LBP method: 'default', 'uniform', 'ror', etc.
```

## Example Output

```
==================================================
Processing: test_01.jpg
==================================================
Original image size: (7952, 5304, 3)
Resized image: (400, 267, 3)

==================================================
HoG Feature Extraction
==================================================
HoG Feature Dimension: (56448,)
HoG Feature Range: [0.0000, 1.0000]

==================================================
LBP Feature Extraction
==================================================
LBP Feature Dimension: (18,)
LBP Feature Range: [0, 17]
```

## Project Structure

```
.
├── hog_lbp_features.py      # Main script
├── test_01.jpg              # Sample image 1
├── test_02.jpg              # Sample image 2
├── README.md                # This file
└── *_hog.jpg, *_lbp.jpg, *_combined.jpg  # Output files
```

## Requirements

- Python 3.7+
- OpenCV (`cv2`)
- scikit-image
- matplotlib
- numpy

## References

- Dalal, N., & Triggs, B. (2005). "Histograms of oriented gradients for human detection"
- Ojala, T., Pietikäinen, M., & Mäenpää, T. (2002). "Multiresolution gray-scale and rotation invariant texture classification with local binary patterns"
