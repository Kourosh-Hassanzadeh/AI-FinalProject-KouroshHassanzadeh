import numpy as np
import cv2
import sys
sys.path.append('.')
from utils.load_data import load_data
from utils.plot import plot_images

def apply_threshold_simple(
        image_gray: np.ndarray, 
        threshold_value: int, 
        inverse: bool = False
    ):
        """
        Applies simple global thresholding.

        Args:
            image_gray (np.ndarray): Grayscale input image.
            threshold_value (int): Threshold value (0-255).
            inverse (bool): If True, applies THRESH_BINARY_INV.

        Returns:
            np.ndarray: Binary image.
        """
        type_flag = cv2.THRESH_BINARY_INV if inverse else cv2.THRESH_BINARY
        _, result = cv2.threshold(image_gray, threshold_value, 255, type_flag)
        return result

@staticmethod
def apply_threshold_adaptive_mean(
    image_gray: np.ndarray, 
    block_size: int, 
    constant_c: int
):
    """
    Applies Adaptive Mean Thresholding.

    Args:
        image_gray (np.ndarray): Grayscale input image.
        block_size (int): Size of a pixel neighborhood (must be odd).
        constant_c (int): Constant subtracted from the mean.

    Returns:
        np.ndarray: Binary image.
    """
    if block_size % 2 == 0:
        block_size += 1
        
    return cv2.adaptiveThreshold(
        image_gray, 
        255, 
        cv2.ADAPTIVE_THRESH_MEAN_C, 
        cv2.THRESH_BINARY, 
        block_size, 
        constant_c
    )

@staticmethod
def apply_threshold_adaptive_gaussian(
    image_gray: np.ndarray, 
    block_size: int, 
    constant_c: int
):
    """
    Applies Adaptive Gaussian Thresholding.

    Args:
        image_gray (np.ndarray): Grayscale input image.
        block_size (int): Size of a pixel neighborhood (must be odd).
        constant_c (int): Constant subtracted from the weighted mean.

    Returns:
        np.ndarray: Binary image.
    """
    if block_size % 2 == 0:
        block_size += 1
        
    return cv2.adaptiveThreshold(
        image_gray, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 
        block_size, 
        constant_c
    )

@staticmethod
def apply_threshold_otsu(image_gray: np.ndarray):
    """
    Applies Otsu's Binarization to automatically find the optimal threshold.

    Args:
        image_gray (np.ndarray): Grayscale input image.

    Returns:
        Tuple[float, np.ndarray]: 
            - The optimal threshold value calculated by Otsu.
            - The resulting binary image.
    """
    # Otsu's thresholding requires an 8-bit input image
    image_8bit = np.uint8(image_gray)
    otsu_threshold, result = cv2.threshold(
        image_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return otsu_threshold, result


if __name__ == "__main__":

    img = load_data("data/apply_filters_&_thresholding/Lena.bmp")

    simple_thresh = apply_threshold_simple(
        img, threshold_value=127
    )

    adaptive_mean = apply_threshold_adaptive_mean(
        img, block_size=11, constant_c=2
    )

    adaptive_gaussian = apply_threshold_adaptive_gaussian(
        img, block_size=11, constant_c=2
    )

    otsu_value, otsu_thresh = apply_threshold_otsu(img)

    images = [
        img,
        simple_thresh,
        adaptive_mean,
        adaptive_gaussian,
        otsu_thresh
    ]

    titles = [
        "Original Image",
        "Simple Threshold",
        "Adaptive Mean",
        "Adaptive Gaussian",
        f"Otsu (T={otsu_value:.2f})"
    ]

    plot_images(images, titles)