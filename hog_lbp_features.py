"""
HoG and LBP Feature Extraction and Visualization
Using OpenCV and scikit-image
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import feature
from skimage.feature import hog
from skimage import color
import os


def extract_hog_features(image, cell_size=(8, 8), block_size=(2, 2), orientations=9):
    """
    Extract HoG features

    Args:
        image: Input image
        cell_size: Cell size
        block_size: Block size (in cells)
        orientations: Number of orientation bins
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Extract HoG features
    hog_features, hog_image = hog(
        gray,
        orientations=orientations,
        pixels_per_cell=cell_size,
        cells_per_block=block_size,
        visualize=True,
        feature_vector=True
    )

    print(f"HoG Feature Dimension: {hog_features.shape}")
    print(f"HoG Feature Range: [{hog_features.min():.4f}, {hog_features.max():.4f}]")

    return gray, hog_features, hog_image


def extract_lbp_features(image, radius=3, n_points=24, method='uniform'):
    """
    Extract LBP features

    Args:
        image: Input image
        radius: LBP radius
        n_points: Number of sampling points
        method: LBP method ('default', 'ror', 'uniform', 'nri_uniform', 'var')
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Extract LBP features
    lbp = feature.local_binary_pattern(gray, n_points, radius, method=method)

    # Calculate LBP histogram
    if method == 'uniform':
        n_bins = n_points + 2
    else:
        n_bins = 256

    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)

    print(f"LBP Feature Dimension: {hist.shape}")
    print(f"LBP Feature Range: [{lbp.min():.0f}, {lbp.max():.0f}]")

    return gray, lbp, hist


def visualize_hog_features(image, hog_image, hog_features, save_path='hog_result.jpg'):
    """
    Visualize HoG features
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else image, cmap='gray')
    axes[0].set_title('Original Image', fontsize=14)
    axes[0].axis('off')

    # HoG feature map
    axes[1].imshow(hog_image, cmap='gray')
    axes[1].set_title('HoG Feature Map', fontsize=14)
    axes[1].axis('off')

    # HoG feature histogram
    axes[2].bar(range(len(hog_features)), hog_features, color='steelblue')
    axes[2].set_title(f'HoG Feature Distribution ({len(hog_features)} dims)', fontsize=14)
    axes[2].set_xlabel('Feature Dimension', fontsize=12)
    axes[2].set_ylabel('Feature Value', fontsize=12)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"HoG result saved to: {save_path}")
    plt.close()


def visualize_lbp_features(image, lbp_image, hist, save_path='lbp_result.jpg'):
    """
    Visualize LBP features
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else image, cmap='gray')
    axes[0].set_title('Original Image', fontsize=14)
    axes[0].axis('off')

    # LBP feature map
    im = axes[1].imshow(lbp_image, cmap='gray')
    axes[1].set_title('LBP Texture Map', fontsize=14)
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046)

    # LBP histogram
    axes[2].bar(range(len(hist)), hist, color='coral')
    axes[2].set_title(f'LBP Histogram ({len(hist)} bins)', fontsize=14)
    axes[2].set_xlabel('LBP Value', fontsize=12)
    axes[2].set_ylabel('Frequency', fontsize=12)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"LBP result saved to: {save_path}")
    plt.close()


def visualize_combined(image, gray, hog_image, lbp_image, save_path='combined_result.jpg'):
    """
    Combined visualization of all features
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Original image
    axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else image, cmap='gray')
    axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')

    # Grayscale
    axes[0, 1].imshow(gray, cmap='gray')
    axes[0, 1].set_title('Grayscale', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')

    # HoG feature map
    axes[1, 0].imshow(hog_image, cmap='gray')
    axes[1, 0].set_title('HoG Gradient Features', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')

    # LBP feature map
    axes[1, 1].imshow(lbp_image, cmap='jet')
    axes[1, 1].set_title('LBP Texture Features', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')

    plt.suptitle('Local Feature Extraction Comparison (HoG vs LBP)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Combined result saved to: {save_path}")
    plt.close()


def process_image(image_path, output_prefix):
    """
    Process a single image: extract HoG and LBP features and visualize

    Args:
        image_path: Path to input image
        output_prefix: Prefix for output files
    """
    # Read image
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Cannot read image {image_path}")
        return

    print(f"\n{'='*50}")
    print(f"Processing: {image_path}")
    print(f"{'='*50}")
    print(f"Original image size: {image.shape}")

    # Resize for better visualization
    max_dim = 400
    h, w = image.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        print(f"Resized image: {image.shape}")

    # Extract HoG features
    print(f"\n{'='*50}")
    print("HoG Feature Extraction")
    print(f"{'='*50}")
    gray, hog_features, hog_image = extract_hog_features(
        image,
        cell_size=(8, 8),
        block_size=(2, 2),
        orientations=9
    )

    # Extract LBP features
    print(f"\n{'='*50}")
    print("LBP Feature Extraction")
    print(f"{'='*50}")
    gray, lbp_image, lbp_hist = extract_lbp_features(
        image,
        radius=2,
        n_points=16,
        method='uniform'
    )

    # Visualize results
    print(f"\n{'='*50}")
    print("Generating Visualization")
    print(f"{'='*50}")

    visualize_hog_features(image, hog_image, hog_features, f'{output_prefix}_hog.jpg')
    visualize_lbp_features(image, lbp_image, lbp_hist, f'{output_prefix}_lbp.jpg')
    visualize_combined(image, gray, hog_image, lbp_image, f'{output_prefix}_combined.jpg')

    # Print feature comparison
    print(f"\n{'='*50}")
    print("Feature Summary")
    print(f"{'='*50}")
    print(f"{'Feature Type':<15} {'Dimension':<15} {'Purpose':<20}")
    print("-" * 50)
    print(f"{'HoG':<15} {hog_features.shape[0]:<15} {'Edge/Gradient':<20}")
    print(f"{'LBP':<15} {lbp_hist.shape[0]:<15} {'Texture':<20}")
    print("-" * 50)
    print("\nHoG: Robust to illumination, good for edge/contour detection")
    print("LBP: Robust to illumination, good for texture description")


def main():
    """Main function"""
    # List of images to process
    image_files = [
        ('test_01.jpg', 'test01'),
        ('test_02.jpg', 'test02'),
    ]

    for image_file, output_prefix in image_files:
        if os.path.exists(image_file):
            process_image(image_file, output_prefix)
        else:
            print(f"Warning: {image_file} not found, skipping...")

    print(f"\n{'='*50}")
    print("All processing completed!")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
