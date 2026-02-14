"""Image processing facade that wraps src/ modules for the Streamlit demo.

This module provides a single :class:`ImageProcessor` class whose static
methods delegate to the lower-level algorithms in ``src/``.  It handles
colour-space conversions, kernel-size validation, and list-wrapping so
the UI layer (``app.py``) never has to deal with those details.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple, Union

import cv2
import numpy as np
from skimage import exposure
from skimage.feature import hog

# ---------------------------------------------------------------------------
# Path configuration – add ``src/`` to the import search path once.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_PATH = str(_PROJECT_ROOT / "src")

if _SRC_PATH not in sys.path:
    sys.path.insert(0, _SRC_PATH)

import denoising  # noqa: E402
import edge_detection  # noqa: E402
import segmentation  # noqa: E402
import spatial_frequency_filters  # noqa: E402
import thresholding  # noqa: E402

__all__ = ["ImageProcessor"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_odd(value: int) -> int:
    """Return *value* unchanged if odd, otherwise return ``value + 1``."""
    return value if value % 2 != 0 else value + 1


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class ImageProcessor:
    """Stateless wrapper around the ``src/`` image-processing modules.

    Every public method is a :func:`staticmethod` – the class is used
    purely as a namespace so callers can write
    ``ImageProcessor.load_image(...)`` without instantiation.
    """

    # ------------------------------------------------------------------
    # Image I/O & conversion
    # ------------------------------------------------------------------

    @staticmethod
    def load_image(file_buffer) -> np.ndarray:
        """Read an uploaded file buffer and return an RGB ``uint8`` array.

        Raises
        ------
        ValueError
            If the buffer cannot be decoded as an image.
        """
        raw_bytes = np.asarray(bytearray(file_buffer.read()), dtype=np.uint8)
        bgr = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Could not decode the uploaded image.")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    @staticmethod
    def to_gray(image: np.ndarray) -> np.ndarray:
        """Convert to single-channel grayscale (no-op copy if already gray)."""
        if image.ndim == 2:
            return image.copy()
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    @staticmethod
    def calculate_histogram(
        image: np.ndarray,
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Return the intensity histogram(s) of *image*.

        * Grayscale input → single ``(256, 1)`` array.
        * Colour input    → list of three such arrays (one per channel).
        """
        if image.ndim == 2:
            return cv2.calcHist([image], [0], None, [256], [0, 256])
        return [
            cv2.calcHist([image], [ch], None, [256], [0, 256])
            for ch in range(3)
        ]

    # ------------------------------------------------------------------
    # 1. Denoising
    # ------------------------------------------------------------------

    @staticmethod
    def add_salt_pepper_noise(
        image: np.ndarray, amount: float, s_vs_p: float
    ) -> np.ndarray:
        """Apply salt-and-pepper noise (image is converted to gray)."""
        gray = ImageProcessor.to_gray(image)
        return denoising.saltAndPepperNoise([gray], amount, s_vs_p)[0]

    @staticmethod
    def add_gaussian_noise(
        image: np.ndarray, mean: float, std: float
    ) -> np.ndarray:
        """Apply additive Gaussian noise (image is converted to gray)."""
        gray = ImageProcessor.to_gray(image)
        return denoising.gaussianNoise([gray], mean, std)[0]

    @staticmethod
    def apply_median_filter(image: np.ndarray, kernel: int) -> np.ndarray:
        """Denoise with a median filter (*kernel* forced odd)."""
        return denoising.medianFilter([image], _ensure_odd(kernel))[0]

    @staticmethod
    def apply_bilateral_filter(
        image: np.ndarray,
        diameter: int,
        sigma_color: float,
        sigma_space: float,
    ) -> np.ndarray:
        """Denoise with a bilateral Gaussian filter."""
        return denoising.bilateralGaussian(
            [image], diameter, sigma_color, sigma_space
        )[0]

    # ------------------------------------------------------------------
    # 2. Edge detection
    # ------------------------------------------------------------------

    @staticmethod
    def _to_bgr(image: np.ndarray) -> np.ndarray:
        """Convert *image* to BGR as expected by the edge-detection module."""
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    @staticmethod
    def detect_edges_prewitt(image: np.ndarray) -> np.ndarray:
        """Prewitt edge detection."""
        return edge_detection.prewitt_ED([ImageProcessor._to_bgr(image)])[0]

    @staticmethod
    def detect_edges_kirsch(image: np.ndarray) -> np.ndarray:
        """Kirsch 8-direction edge detection."""
        return edge_detection.kirsch_ED([ImageProcessor._to_bgr(image)])[0]

    @staticmethod
    def detect_edges_marr_hildreth(
        image: np.ndarray, sigma: float, threshold: float
    ) -> np.ndarray:
        """Marr-Hildreth (LoG) edge detection."""
        return edge_detection.Marr_Hildreth_ED(
            [ImageProcessor._to_bgr(image)], sigma, threshold
        )[0]

    @staticmethod
    def detect_edges_canny(
        image: np.ndarray, min_thresh: int, max_thresh: int
    ) -> np.ndarray:
        """Canny edge detection."""
        return edge_detection.canny_ED(
            [ImageProcessor._to_bgr(image)], min_thresh, max_thresh
        )[0]

    # ------------------------------------------------------------------
    # 3. Segmentation
    # ------------------------------------------------------------------

    @staticmethod
    def segment_kmeans(image: np.ndarray, k: int) -> np.ndarray:
        """K-means colour segmentation with *k* clusters."""
        if image.ndim == 2:
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            rgb = image.copy()
        return segmentation.segment_colors([rgb], k)[0]

    # ------------------------------------------------------------------
    # 4. Spatial & frequency-domain filters
    # ------------------------------------------------------------------

    @staticmethod
    def apply_average_filter(image: np.ndarray, kernel: int) -> np.ndarray:
        """Spatial average (box) filter."""
        return spatial_frequency_filters.average_filter(
            ImageProcessor.to_gray(image), kernel
        )

    @staticmethod
    def apply_gaussian_filter(image: np.ndarray, kernel: int) -> np.ndarray:
        """Spatial Gaussian filter (*kernel* forced odd)."""
        return spatial_frequency_filters.gaussian_filter(
            ImageProcessor.to_gray(image), _ensure_odd(kernel)
        )

    @staticmethod
    def apply_spatial_median(image: np.ndarray, kernel: int) -> np.ndarray:
        """Spatial median filter (*kernel* forced odd)."""
        return spatial_frequency_filters.median_filter(
            ImageProcessor.to_gray(image), _ensure_odd(kernel)
        )

    @staticmethod
    def apply_sharpening_filter(image: np.ndarray) -> np.ndarray:
        """Laplacian-based sharpening filter."""
        return spatial_frequency_filters.sharpening_filter(
            ImageProcessor.to_gray(image)
        )

    @staticmethod
    def apply_sobel_filter(image: np.ndarray) -> np.ndarray:
        """Sobel gradient magnitude, normalised to ``uint8``."""
        raw = spatial_frequency_filters.sobel_filter(
            ImageProcessor.to_gray(image)
        )
        return cv2.normalize(
            raw, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )

    @staticmethod
    def apply_frequency_filters(
        image: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(low_pass, high_pass)`` images via FFT filtering."""
        lpf, hpf = spatial_frequency_filters.frequency_filters(
            ImageProcessor.to_gray(image)
        )
        lpf_view = cv2.normalize(
            np.abs(lpf), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )
        hpf_view = cv2.normalize(
            np.abs(hpf), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )
        return lpf_view, hpf_view

    # ------------------------------------------------------------------
    # 5. Thresholding
    # ------------------------------------------------------------------

    @staticmethod
    def threshold_simple(
        gray: np.ndarray, value: int, inverse: bool
    ) -> np.ndarray:
        """Simple global thresholding."""
        return thresholding.apply_threshold_simple(gray, value, inverse)

    @staticmethod
    def threshold_adaptive_mean(
        gray: np.ndarray, block_size: int, constant_c: int
    ) -> np.ndarray:
        """Adaptive mean thresholding (*block_size* forced odd)."""
        return thresholding.apply_threshold_adaptive_mean(
            gray, _ensure_odd(block_size), constant_c
        )

    @staticmethod
    def threshold_adaptive_gaussian(
        gray: np.ndarray, block_size: int, constant_c: int
    ) -> np.ndarray:
        """Adaptive Gaussian thresholding (*block_size* forced odd)."""
        return thresholding.apply_threshold_adaptive_gaussian(
            gray, _ensure_odd(block_size), constant_c
        )

    @staticmethod
    def threshold_otsu(gray: np.ndarray) -> Tuple[float, np.ndarray]:
        """Otsu binarisation. Returns ``(optimal_threshold, binary_image)``."""
        return thresholding.apply_threshold_otsu(gray)

    # ------------------------------------------------------------------
    # 6. HOG feature extraction
    # ------------------------------------------------------------------

    @staticmethod
    def compute_hog(image: np.ndarray) -> np.ndarray:
        """Compute HOG features and return a ``uint8`` visualisation image."""
        gray = ImageProcessor.to_gray(image)
        _, hog_image = hog(
            gray,
            orientations=8,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            visualize=True,
            block_norm="L2-Hys",
        )
        rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
        max_val = rescaled.max()
        if max_val == 0:
            return np.zeros_like(rescaled, dtype=np.uint8)
        return (rescaled / max_val * 255).astype(np.uint8)