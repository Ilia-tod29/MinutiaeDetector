#!/usr/bin/env python3
"""
FINGERPRINT MINUTIAE DETECTOR - Crossing Number Method
=======================================================

This script detects fingerprint minutiae (ridge endings and bifurcations) using
the Crossing Number (CN) method on a 1-pixel-wide skeleton.

INSTALLATION:
    pip install opencv-python numpy argparse
    
    Optional (for better thinning):
    pip install opencv-contrib-python
    
    Optional (fallback thinning):
    pip install scikit-image

USAGE:
    python minutiae_cn.py --input fingerprint.tif --outdir outputs/
    python minutiae_cn.py --input fingerprint.png --outdir outputs/ --clahe --adaptive --margin 16 --dedup_radius 10
    python minutiae_cn.py --input fingerprint.bmp --outdir outputs/ --clahe --adaptive --invert --roi_blocksize 8

PARAMETERS:
    --input: Path to input fingerprint image (required)
    --outdir: Output directory for results (default: outputs/)
    --clahe: Use CLAHE contrast enhancement (recommended)
    --adaptive: Use adaptive threshold (recommended)
    --invert: Invert binary image if ridges are dark
    --margin: Margin from ROI/image border in pixels (default: 16)
    --dedup_radius: Radius for merging duplicate minutiae (default: 10)
    --roi_blocksize: Block size for ROI segmentation (default: 16)
    --circle_radius: Radius of minutiae markers (default: 5)
"""

import os
import sys
import argparse
import logging
from typing import Tuple, List, Dict
import numpy as np
import cv2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MinutiaeDetector:
    
    def __init__(self, config: Dict):
        """
        Initialize the detector with configuration.
        
        Args:
            config: Dictionary containing all configuration parameters
        """
        self.config = config
        self.original_image = None
        self.preprocessed = None
        self.roi_mask = None
        self.binary = None
        self.skeleton = None
        self.minutiae = {'endings': [], 'bifurcations': []}
        
    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load and convert image to grayscale.
        
        WHY: Supports multiple formats (BMP, PNG, JPG, TIF) and ensures grayscale.
        HOW: Uses cv2.imread with UNCHANGED flag, then converts to grayscale if needed.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Grayscale image as numpy array
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load image
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            if img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        
        logger.info(f"Loaded image: {img.shape}, dtype: {img.dtype}")
        self.original_image = img
        return img
    
    def enhance_contrast(self, image: np.ndarray, use_clahe: bool = True) -> np.ndarray:
        """
        Enhance image contrast using CLAHE or histogram equalization.
        
        WHY: Fingerprint images often have uneven illumination and low contrast.
             CLAHE (Contrast Limited Adaptive Histogram Equalization) enhances local
             contrast while preventing over-amplification of noise.
        
        HOW: CLAHE divides image into tiles and applies histogram equalization to each
             with a clip limit to prevent noise amplification. Falls back to global
             histogram equalization if CLAHE is disabled.
        
        Args:
            image: Input grayscale image
            use_clahe: Whether to use CLAHE (recommended) or simple histogram equalization
            
        Returns:
            Contrast-enhanced image
        """
        if use_clahe:
            # CLAHE parameters:
            # clipLimit: Threshold for contrast limiting (2.0 is typical)
            # tileGridSize: Size of grid for histogram equalization (8x8 is standard)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(image)
            logger.info("Applied CLAHE contrast enhancement")
        else:
            # Simple global histogram equalization
            enhanced = cv2.equalizeHist(image)
            logger.info("Applied histogram equalization")
        
        return enhanced
    
    def reduce_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Reduce noise while preserving ridge structures.
        
        WHY: Noise in fingerprint images can create false minutiae.
             Gaussian blur smooths the image while bilateral filter preserves edges.
        
        HOW: First applies Gaussian blur (5x5 kernel) for general smoothing,
             then bilateral filter which smooths flat regions while keeping edges sharp.
             Bilateral filter uses spatial and intensity differences.
        
        Args:
            image: Input image
            
        Returns:
            Denoised image
        """
        # Gaussian blur: removes high-frequency noise
        # (5,5) kernel size, 0 sigmaX means auto-calculate from kernel size
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        
        # Bilateral filter: edge-preserving smoothing
        # Parameters: d=9 (diameter), sigmaColor=75, sigmaSpace=75
        # sigmaColor: filter in color space (larger = more colors mixed)
        # sigmaSpace: filter in coordinate space (larger = farther pixels influence)
        denoised = cv2.bilateralFilter(blurred, d=9, sigmaColor=75, sigmaSpace=75)
        
        logger.info("Applied noise reduction (Gaussian + Bilateral)")
        return denoised
    
    def segment_roi(self, image: np.ndarray, block_size: int = 16) -> np.ndarray:
        """
        Segment fingerprint region of interest (ROI) from background.
        
        WHY: Fingerprint images often contain background regions without ridge information.
             These areas can produce false minutiae and should be excluded.
        
        HOW: 
        1. Compute local variance in blocks - fingerprint regions have high variance
           due to ridge patterns, background has low variance (uniform).
        2. Threshold variance map to separate foreground/background.
        3. Apply morphological operations to clean up the mask:
           - Close: fills small holes
           - Open: removes small noise
        4. Fill holes to get a solid ROI.
        5. Optionally erode slightly to avoid boundary artifacts.
        
        Args:
            image: Input grayscale image
            block_size: Size of blocks for variance computation
            
        Returns:
            Binary mask (255=foreground, 0=background)
        """
        h, w = image.shape
        
        # Compute block-wise variance
        # High variance = ridge patterns, Low variance = background
        variance_map = np.zeros((h // block_size, w // block_size), dtype=np.float32)
        
        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                block = image[i:i+block_size, j:j+block_size]
                variance_map[i//block_size, j//block_size] = np.var(block)
        
        # Normalize variance map
        variance_map = cv2.normalize(variance_map, None, 0, 255, cv2.NORM_MINMAX)
        variance_map = variance_map.astype(np.uint8)
        
        # Resize back to original size
        variance_map = cv2.resize(variance_map, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # Threshold to get binary mask
        # Use Otsu's method to automatically find threshold
        _, roi_mask = cv2.threshold(variance_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations to clean up mask
        # Close: fills small holes in foreground
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_CLOSE, kernel_close)
        
        # Open: removes small noise in background
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_OPEN, kernel_open)
        
        # Fill holes
        contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        roi_mask_filled = np.zeros_like(roi_mask)
        cv2.drawContours(roi_mask_filled, contours, -1, 255, thickness=cv2.FILLED)
        
        # Optional: erode slightly to avoid boundary artifacts
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        roi_mask_final = cv2.erode(roi_mask_filled, kernel_erode, iterations=1)
        
        logger.info(f"Segmented ROI with block size {block_size}")
        self.roi_mask = roi_mask_final
        return roi_mask_final
    
    def binarize(self, image: np.ndarray, adaptive: bool = True, invert: bool = False) -> np.ndarray:
        """
        Binarize image to separate ridges from valleys.
        
        WHY: Minutiae detection requires a clear binary representation where
             ridges are 1 and background is 0.
        
        HOW: 
        - Adaptive threshold: computes local threshold for each pixel based on
          neighborhood. Better for uneven illumination. Uses Gaussian-weighted mean.
        - Global threshold: Uses Otsu's method to find optimal global threshold.
        - Invert option handles datasets where ridges are dark instead of bright.
        
        Args:
            image: Input grayscale image
            adaptive: Use adaptive threshold (recommended) vs global
            invert: Invert result if ridges are dark
            
        Returns:
            Binary image (0 or 255)
        """
        if adaptive:
            # Adaptive Gaussian threshold
            # Parameters:
            # - blockSize=11: size of neighborhood for threshold calculation
            # - C=2: constant subtracted from weighted mean
            binary = cv2.adaptiveThreshold(
                image, 
                255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                blockSize=11,
                C=2
            )
            logger.info("Applied adaptive threshold (Gaussian)")
        else:
            # Global threshold with Otsu's method
            # Otsu's method automatically finds optimal threshold
            _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            logger.info("Applied Otsu's threshold")
        
        # Invert if ridges are dark
        if invert:
            binary = cv2.bitwise_not(binary)
            logger.info("Inverted binary image")
        
        return binary
    
    def morphological_cleanup(self, binary: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
        """
        Clean up binary image using morphological operations.
        
        WHY: Binary images may have small noise artifacts and broken ridges.
             Morphological operations clean these up while preserving overall structure.
        
        HOW:
        - Opening: removes small objects (erosion followed by dilation)
        - Closing: fills small holes in ridges (dilation followed by erosion)
        - Apply ROI mask to keep only fingerprint region
        
        Args:
            binary: Input binary image
            roi_mask: ROI mask to apply
            
        Returns:
            Cleaned binary image (0 or 255)
        """
        # Opening: removes small noise
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)
        
        # Closing: fills small gaps in ridges
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close)
        
        # Apply ROI mask
        cleaned = cv2.bitwise_and(cleaned, cleaned, mask=roi_mask)
        
        logger.info("Applied morphological cleanup")
        return cleaned
    
    def skeletonize(self, binary: np.ndarray) -> np.ndarray:
        """
        Thin ridges to 1-pixel width (skeletonization).
        
        WHY: Crossing Number method requires ridges to be exactly 1 pixel wide
             to accurately detect minutiae points.
        
        HOW: Tries multiple methods in order of preference:
        1. cv2.ximgproc.thinning (opencv-contrib): Fast, reliable Zhang-Suen
        2. scikit-image skeletonize: Robust, widely used
        3. Custom Zhang-Suen implementation: Pure Python fallback
        
        Zhang-Suen algorithm:
        - Iteratively removes pixels from ridge boundaries
        - Preserves connectivity and topology
        - Stops when no more pixels can be removed
        
        Args:
            binary: Input binary image (0 or 255)
            
        Returns:
            Skeleton image (0 or 1)
        """
        # Convert to binary {0, 1}
        binary_01 = (binary // 255).astype(np.uint8)
        
        # Try opencv-contrib thinning first
        try:
            import cv2
            if hasattr(cv2, 'ximgproc'):
                skeleton = cv2.ximgproc.thinning(binary, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
                skeleton = (skeleton // 255).astype(np.uint8)
                logger.info("Skeletonization: Using cv2.ximgproc.thinning (Zhang-Suen)")
                return skeleton
        except (ImportError, AttributeError):
            logger.warning("cv2.ximgproc.thinning not available, trying scikit-image skeletonize")
            try:
                from skimage.morphology import skeletonize
                skeleton = skeletonize(binary_01).astype(np.uint8)
                logger.info("Skeletonization: Using scikit-image skeletonize")
                return skeleton
            except ImportError:
                logger.error("scikit-image not available, please use 'pip install scikit-image' or 'pip install opencv-contrib-python'. To make sure you have access to cv2.ximgproc.thinning.")
                sys.exit(1)
        
        return skeleton
    
    
    def compute_crossing_number(self, skeleton: np.ndarray, x: int, y: int) -> int:
        """
        Compute Crossing Number for a pixel using 8-neighborhood.
        
        WHY: CN method is a robust way to classify minutiae points based on
             ridge topology around a point.
        
        HOW: 
        1. Get 8 neighbors in circular order: p1..p8
        2. Create closed loop by appending p1 at end
        3. Compute CN = 0.5 * sum(|pi - p(i+1)|) for i=1..8
        
        Neighbor ordering (critical for correct CN):
             p8  p1  p2
             p7  p   p3
             p6  p5  p4
        
        CN interpretation:
        - CN = 1: Ridge ending (one transition)
        - CN = 2: Normal ridge continuation
        - CN = 3: Bifurcation (three transitions)
        - CN = 4: Crossing (rare, usually filtered)
        
        Args:
            skeleton: Binary skeleton image {0, 1}
            x, y: Pixel coordinates
            
        Returns:
            Crossing number (0, 1, 2, 3, or 4)
        """
        # Get 8 neighbors in circular order (convert to int to avoid overflow)
        p1 = int(skeleton[y-1, x])      # top
        p2 = int(skeleton[y-1, x+1])    # top-right
        p3 = int(skeleton[y, x+1])      # right
        p4 = int(skeleton[y+1, x+1])    # bottom-right
        p5 = int(skeleton[y+1, x])      # bottom
        p6 = int(skeleton[y+1, x-1])    # bottom-left
        p7 = int(skeleton[y, x-1])      # left
        p8 = int(skeleton[y-1, x-1])    # top-left
        
        # Create circular list (p9 = p1)
        neighbors = [p1, p2, p3, p4, p5, p6, p7, p8, p1]
        
        # Compute CN = 0.5 * sum(|pi - p(i+1)|)
        cn = 0
        for i in range(8):
            cn += abs(neighbors[i] - neighbors[i+1])
        cn = cn // 2
        
        return cn
    
    def detect_minutiae_raw(self, skeleton: np.ndarray, margin: int) -> Tuple[List, List]:
        """
        Detect all minutiae candidates using Crossing Number method.
        
        WHY: Initial detection before filtering - finds all potential minutiae.
        
        HOW:
        1. Iterate over all skeleton pixels (value = 1)
        2. Skip pixels too close to image borders (margin)
        3. Compute CN for each skeleton pixel
        4. Classify: CN=1 => ending, CN=3 => bifurcation
        
        Args:
            skeleton: Binary skeleton {0, 1}
            margin: Border margin to skip
            
        Returns:
            Tuple of (endings list, bifurcations list), each containing (x, y) tuples
        """
        h, w = skeleton.shape
        endings = []
        bifurcations = []
        
        # Traverse skeleton pixels
        for y in range(margin, h - margin):
            for x in range(margin, w - margin):
                if skeleton[y, x] == 1:
                    cn = self.compute_crossing_number(skeleton, x, y)
                    
                    if cn == 1:
                        endings.append((x, y))
                    elif cn == 3:
                        bifurcations.append((x, y))
        
        logger.info(f"Raw minutiae: {len(endings)} endings, {len(bifurcations)} bifurcations")
        return endings, bifurcations
    
    def filter_boundary_minutiae(self, minutiae: List, roi_mask: np.ndarray, margin: int) -> List:
        """
        Filter minutiae too close to ROI boundary.
        
        WHY: Minutiae near ROI boundary are often false detections caused by
             the boundary itself rather than actual ridge features.
        
        HOW: 
        1. Erode ROI mask by margin pixels
        2. Keep only minutiae inside eroded mask
        
        Args:
            minutiae: List of (x, y) coordinates
            roi_mask: ROI mask image
            margin: Distance from boundary
            
        Returns:
            Filtered list of minutiae
        """
        # Erode mask to exclude boundary region
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (margin*2, margin*2))
        eroded_mask = cv2.erode(roi_mask, kernel, iterations=1)
        
        # Filter minutiae
        filtered = []
        for (x, y) in minutiae:
            if eroded_mask[y, x] > 0:
                filtered.append((x, y))
        
        return filtered
    
    def deduplicate_minutiae(self, minutiae: List, radius: int) -> List:
        """
        Merge duplicate minutiae within a radius.
        
        WHY: Multiple minutiae may be detected for the same physical feature
             due to noise or skeleton imperfections.
        
        HOW:
        1. Sort minutiae (for consistent results)
        2. For each minutia, check if any previous minutia is within radius
        3. If not, add to filtered list
        4. Uses Euclidean distance
        
        Args:
            minutiae: List of (x, y) coordinates
            radius: Merging radius in pixels
            
        Returns:
            Deduplicated list of minutiae
        """
        if not minutiae:
            return []
        
        # Sort for consistent results
        minutiae = sorted(minutiae)
        
        filtered = [minutiae[0]]
        
        for (x, y) in minutiae[1:]:
            # Check distance to all kept minutiae
            too_close = False
            for (fx, fy) in filtered:
                dist = np.sqrt((x - fx)**2 + (y - fy)**2)
                if dist < radius:
                    too_close = True
                    break
            
            if not too_close:
                filtered.append((x, y))
        
        return filtered
    
    def filter_minutiae(self, endings: List, bifurcations: List, 
                       roi_mask: np.ndarray, margin: int, dedup_radius: int) -> Tuple[List, List]:
        """
        Apply all filtering steps to minutiae.
        
        WHY: Combines all filtering logic to remove false minutiae.
        
        HOW:
        1. Filter boundary minutiae (near ROI edge)
        2. Deduplicate (merge close minutiae)
        
        Args:
            endings: List of ending minutiae
            bifurcations: List of bifurcation minutiae
            roi_mask: ROI mask
            margin: Boundary margin
            dedup_radius: Deduplication radius
            
        Returns:
            Tuple of (filtered endings, filtered bifurcations)
        """
        # Filter boundary minutiae
        endings_filtered = self.filter_boundary_minutiae(endings, roi_mask, margin)
        bifurcations_filtered = self.filter_boundary_minutiae(bifurcations, roi_mask, margin)
        
        logger.info(f"After boundary filter: {len(endings_filtered)} endings, {len(bifurcations_filtered)} bifurcations")
        
        # Deduplicate
        endings_final = self.deduplicate_minutiae(endings_filtered, dedup_radius)
        bifurcations_final = self.deduplicate_minutiae(bifurcations_filtered, dedup_radius)
        
        logger.info(f"After deduplication: {len(endings_final)} endings, {len(bifurcations_final)} bifurcations")
        
        return endings_final, bifurcations_final
    
    def visualize_results(self, base_image: np.ndarray, endings: List, 
                         bifurcations: List, circle_radius: int = 5) -> np.ndarray:
        """
        Create visualization overlay with minutiae marked.
        
        WHY: Visual representation helps verify detection quality and debug issues.
        
        HOW:
        1. Convert base image to RGB (if grayscale)
        2. Draw green circles for endings
        3. Draw red circles for bifurcations
        4. Use thick circles for visibility
        
        Args:
            base_image: Base image (grayscale or RGB)
            endings: List of ending minutiae (x, y)
            bifurcations: List of bifurcation minutiae (x, y)
            circle_radius: Radius of marker circles
            
        Returns:
            RGB overlay image
        """
        # Convert to RGB if grayscale
        if len(base_image.shape) == 2:
            overlay = cv2.cvtColor(base_image, cv2.COLOR_GRAY2RGB)
        else:
            overlay = base_image.copy()
        
        # Ensure 8-bit for drawing
        if overlay.dtype != np.uint8:
            overlay = cv2.normalize(overlay, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Draw bifurcations in red
        for (x, y) in bifurcations:
            cv2.circle(overlay, (x, y), circle_radius, (0, 0, 255), 2)
        
        # Draw endings in green
        for (x, y) in endings:
            cv2.circle(overlay, (x, y), circle_radius, (0, 255, 0), 2)
        
        logger.info(f"Created visualization: {len(endings)} green circles (endings), "
                   f"{len(bifurcations)} red circles (bifurcations)")
        
        return overlay
    
    def process(self, image_path: str, output_dir: str):
        """
        Main processing pipeline.
        
        WHY: Orchestrates entire detection workflow from loading to output.
        
        HOW: Executes all steps in sequence:
        1. Load image
        2. Preprocess (enhance, denoise)
        3. Segment ROI
        4. Binarize
        5. Cleanup
        6. Skeletonize
        7. Detect minutiae
        8. Filter minutiae
        9. Visualize and save results
        
        Args:
            image_path: Path to input image
            output_dir: Directory for output files
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load image
        logger.info("=" * 60)
        logger.info("STEP 1: Loading image")
        img = self.load_image(image_path)
        
        # Preprocessing
        logger.info("=" * 60)
        logger.info("STEP 2: Preprocessing")
        
        # Contrast enhancement
        if self.config['use_clahe']:
            enhanced = self.enhance_contrast(img, use_clahe=True)
        else:
            enhanced = self.enhance_contrast(img, use_clahe=False)
        
        # Noise reduction
        denoised = self.reduce_noise(enhanced)
        self.preprocessed = denoised
        
        # Save preprocessed
        cv2.imwrite(os.path.join(output_dir, '01_preprocessed.png'), self.preprocessed)
        
        # ROI Segmentation
        logger.info("=" * 60)
        logger.info("STEP 3: ROI Segmentation")
        roi_mask = self.segment_roi(self.preprocessed, self.config['roi_blocksize'])
        cv2.imwrite(os.path.join(output_dir, '02_roi_mask.png'), roi_mask)
        
        # Binarization
        logger.info("=" * 60)
        logger.info("STEP 4: Binarization")
        binary = self.binarize(self.preprocessed, 
                              adaptive=self.config['use_adaptive'],
                              invert=self.config['invert'])
        
        # Morphological cleanup
        logger.info("=" * 60)
        logger.info("STEP 5: Morphological cleanup")
        binary_clean = self.morphological_cleanup(binary, roi_mask)
        self.binary = binary_clean
        cv2.imwrite(os.path.join(output_dir, '03_binary.png'), self.binary)
        
        # Skeletonization
        logger.info("=" * 60)
        logger.info("STEP 6: Skeletonization")
        skeleton = self.skeletonize(self.binary)
        self.skeleton = skeleton
        
        # Save skeleton (convert to 0/255 for visibility)
        skeleton_vis = (skeleton * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(output_dir, '04_skeleton.png'), skeleton_vis)
        
        skeleton_pixels = np.sum(skeleton)
        logger.info(f"Total skeleton pixels: {skeleton_pixels}")
        
        # Minutiae detection
        logger.info("=" * 60)
        logger.info("STEP 7: Minutiae detection (Crossing Number)")
        endings_raw, bifurcations_raw = self.detect_minutiae_raw(
            skeleton, self.config['margin']
        )
        
        # Minutiae filtering
        logger.info("=" * 60)
        logger.info("STEP 8: Minutiae filtering")
        endings_final, bifurcations_final = self.filter_minutiae(
            endings_raw, bifurcations_raw, roi_mask,
            self.config['margin'], self.config['dedup_radius']
        )
        
        self.minutiae['endings'] = endings_final
        self.minutiae['bifurcations'] = bifurcations_final
        
        # Visualization
        logger.info("=" * 60)
        logger.info("STEP 9: Visualization")
        overlay = self.visualize_results(
            self.preprocessed, endings_final, bifurcations_final,
            circle_radius=self.config['circle_radius']
        )
        cv2.imwrite(os.path.join(output_dir, '05_overlay.png'), overlay)
        
        # Print summary
        logger.info("=" * 60)
        logger.info("DETECTION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Input image: {image_path}")
        logger.info(f"Image size: {img.shape}")
        logger.info(f"Total skeleton pixels: {skeleton_pixels}")
        logger.info(f"Raw detections: {len(endings_raw)} endings, {len(bifurcations_raw)} bifurcations")
        logger.info(f"Final detections: {len(endings_final)} endings, {len(bifurcations_final)} bifurcations")
        logger.info(f"Output directory: {output_dir}")
        logger.info("=" * 60)


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Fingerprint Minutiae Detector using Crossing Number method',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python minutiae_cn.py --input fingerprint.tif --outdir outputs/
  python minutiae_cn.py --input fp.png --clahe --adaptive --margin 20
  python minutiae_cn.py --input fp.bmp --clahe --adaptive --invert --roi_blocksize 8
        """
    )
    
    parser.add_argument('--input', type=str, required=False,
                       help='Path to input fingerprint image')
    parser.add_argument('--outdir', type=str, default='outputs',
                       help='Output directory (default: outputs/)')
    parser.add_argument('--clahe', action='store_true',
                       help='Use CLAHE contrast enhancement (recommended)')
    parser.add_argument('--adaptive', action='store_true',
                       help='Use adaptive threshold (recommended)')
    parser.add_argument('--invert', action='store_true',
                       help='Invert binary image if ridges are dark')
    parser.add_argument('--margin', type=int, default=16,
                       help='Margin from border/ROI in pixels (default: 16)')
    parser.add_argument('--dedup_radius', type=int, default=10,
                       help='Radius for deduplication in pixels (default: 10)')
    parser.add_argument('--roi_blocksize', type=int, default=16,
                       help='Block size for ROI segmentation (default: 16)')
    parser.add_argument('--circle_radius', type=int, default=5,
                       help='Radius of minutiae markers (default: 5)')
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Check if input is provided
    if not args.input:
        logger.error("Error: --input argument is required")
        logger.info("\nUsage: python minutiae_cn.py --input <image_path> [options]")
        logger.info("\nFor full help: python minutiae_cn.py --help")
        sys.exit(1)
    
    # Build configuration
    config = {
        'use_clahe': args.clahe,
        'use_adaptive': args.adaptive,
        'invert': args.invert,
        'margin': args.margin,
        'dedup_radius': args.dedup_radius,
        'roi_blocksize': args.roi_blocksize,
        'circle_radius': args.circle_radius
    }
    
    # Create detector and process
    detector = MinutiaeDetector(config)
    
    try:
        detector.process(args.input, args.outdir)
        logger.info("\n✓ Processing completed successfully!")
        logger.info(f"✓ Check {args.outdir}/ for output images")
    except Exception as e:
        logger.error(f"\n✗ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
