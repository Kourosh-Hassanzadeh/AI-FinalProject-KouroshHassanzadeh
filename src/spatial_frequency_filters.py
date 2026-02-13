import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('.')
from utils.load_data import load_data
from utils.plot import plot_images


def average_filter(img, kernel_size):
    mean_filtered = cv2.blur(img, (kernel_size, kernel_size))

    return mean_filtered


def gaussian_filter(img, kernel_size):
    gaussian_filtered = cv2.GaussianBlur(
        img, (kernel_size, kernel_size), sigmaX=1.0)
    return gaussian_filtered


def median_filter(img, kernel_size):
    median_filtered = cv2.medianBlur(img, kernel_size)

    return median_filtered


def sharpening_filter(img):
    kernel_sharpen = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])
    sharpened = cv2.filter2D(img, -1, kernel_sharpen)
    return sharpened


def sobel_filter(img):
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = cv2.magnitude(sobel_x, sobel_y)

    return sobel_combined


def frequency_filters(img):
    # FFT and Shift
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude = np.log(1 + np.abs(fshift))

    # Low Pass Filter
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2

    # Gaussian LPF
    sigma = 30
    x = np.arange(cols)
    y = np.arange(rows)
    X, Y = np.meshgrid(x, y)
    gaussian_lpf = np.exp(-((X-ccol)**2 + (Y-crow)**2) / (2*sigma**2))

    f_lpf = fshift * gaussian_lpf
    img_lpf = np.abs(np.fft.ifft2(np.fft.ifftshift(f_lpf)))

    # High Pass filter
    gaussian_hpf = 1 - gaussian_lpf
    f_hpf = fshift * gaussian_hpf
    img_hpf = np.abs(np.fft.ifft2(np.fft.ifftshift(f_hpf)))

    return img_lpf, img_hpf


if __name__ == "__main__":

    # filters
    img = load_data('data/apply_filters_&_thresholding/Lena.bmp')
    average_filtered = average_filter(img, 3)
    median_filtered = median_filter(img, 3)
    gaussian_filtered = gaussian_filter(img, 3)
    sharpening_filtered = sharpening_filter(img)
    sobel_filtered = sobel_filter(img)
    low_pass_filtered, high_pass_filtered = frequency_filters(img)

    images = [
    img,
    average_filtered,
    median_filtered,
    gaussian_filtered,
    sharpening_filtered,
    sobel_filtered,
    low_pass_filtered,
    high_pass_filtered
    ]

    titles = [
        "Original Image",
        "Average Filter",
        "Median Filter",
        "Gaussian Filter",
        "Sharpening Filter",
        "Sobel Filter",
        "Low Pass (Frequency)",
        "High Pass (Frequency)"
    ]

    plot_images(images, titles)
