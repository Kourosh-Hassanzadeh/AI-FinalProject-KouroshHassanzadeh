import cv2
import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('.')
from utils.plot import visualize_with_metrics
from utils.load_data import load_dataSet


def saltAndPepperNoise(org_data, amount=0.1, s_vs_p=0.5):
    """
    Add salt-and-pepper noise to a list of grayscale images.

    Parameters
    ----------
    org_data : list of numpy.ndarray
        List of original grayscale images.
    amount : float, optional
        Proportion of image pixels to be replaced with noise
        (default is 0.1).
    s_vs_p : float, optional
        Proportion of salt noise versus pepper noise (default is 0.5).

    Returns
    -------
    list of numpy.ndarray
        List of images corrupted with salt-and-pepper noise.
    """
    data = copy.deepcopy(org_data)

    for idx in range(len(data)):
        img = data[idx]
        h, w = img.shape

        num_salt = int(np.ceil(amount * img.size * s_vs_p))
        num_pepper = int(np.ceil(amount * img.size * (1.0 - s_vs_p)))

        # Salt
        ys = np.random.randint(0, h, num_salt)
        xs = np.random.randint(0, w, num_salt)
        img[ys, xs] = 255

        # Pepper
        ys = np.random.randint(0, h, num_pepper)
        xs = np.random.randint(0, w, num_pepper)
        img[ys, xs] = 0

    return data


def gaussianNoise(data, mean=10, std=25):
    """
    Add Gaussian noise to a list of grayscale images.

    Parameters
    ----------
    data : list of numpy.ndarray
        List of original grayscale images.
    mean : float, optional
        Mean of the Gaussian distribution (default is 10).
    std : float, optional
        Standard deviation of the Gaussian distribution (default is 25).

    Returns
    -------
    list of numpy.ndarray
        List of images corrupted with Gaussian noise.
    """
    gaussian_images = copy.deepcopy(data)

    for i in range(len(gaussian_images)):

        row, col = gaussian_images[i].shape
        mean = mean
        sigma = std
        gauss = np.random.normal(mean, sigma, (row, col))
        gauss = gauss.reshape(row, col)
        gaussian_images[i] = gaussian_images[i] + gauss
        gaussian_images[i] = (gaussian_images[i]).astype(np.uint8)

    return gaussian_images


def medianFilter(data, kernel=5):
    """
    Apply median filtering to a list of images.

    Parameters
    ----------
    data : list of numpy.ndarray
        List of noisy grayscale images.
    kernel : int, optional
        Size of the median filter kernel (must be odd).
        Default is 5.

    Returns
    -------
    list of numpy.ndarray
        List of denoised images after median filtering.
    """
    median_images = copy.deepcopy(data)

    for i in range(len(median_images)):
        median_images[i] = cv2.medianBlur(median_images[i], ksize=kernel)

    return median_images


def bilateralGaussian(data, kernel=3, sigColor=75, sigSpace=75):
    """
    Apply bilateral filtering to a list of images.

    The bilateral filter smooths images while preserving edges
    by combining spatial and intensity-based weighting.

    Parameters
    ----------
    data : list of numpy.ndarray
        List of noisy grayscale images.
    kernel : int, optional
        Diameter of each pixel neighborhood (default is 3).
    sigColor : float, optional
        Filter sigma in the color space (default is 75).
    sigSpace : float, optional
        Filter sigma in the coordinate space (default is 75).

    Returns
    -------
    list of numpy.ndarray
        List of denoised images after bilateral filtering.
    """
    bilateral_gaussian_images = copy.deepcopy(data)

    for i in range(len(bilateral_gaussian_images)):
        bilateral_gaussian_images[i] = cv2.bilateralFilter(
            bilateral_gaussian_images[i], d=kernel, sigmaColor=sigColor, sigmaSpace=sigSpace)

    return bilateral_gaussian_images


if __name__ == '__main__':

    images = load_dataSet('data/denoising')

    salt_and_pepper_noisy_images = saltAndPepperNoise(images)
    median_filtered = medianFilter(salt_and_pepper_noisy_images)
    visualize_with_metrics(images, median_filtered)

    # gassuan_noisy_images = gaussianNoise(images)
    # bilateral_filtered = bilateralGaussian(gassuan_noisy_images)
    # visualize_with_metrics(images, bilateral_filtered)
