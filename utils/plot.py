import matplotlib.pyplot as plt
import math
import cv2
from skimage.metrics import mean_squared_error
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio

def plot_images(images, titles=None, cols=4, figsize=(18, 10)):
    """
    Plot multiple images in a grid layout.

    Parameters
    ----------
    images : list of numpy.ndarray
        List of images to display.
    titles : list of str, optional
        Titles corresponding to each image.
    cols : int, optional
        Number of columns in the grid (default is 4).
    figsize : tuple, optional
        Figure size (default is (18, 10)).
    """

    n_images = len(images)
    rows = math.ceil(n_images / cols)

    plt.figure(figsize=figsize)

    for i, img in enumerate(images):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img, cmap='gray')
        
        if titles is not None and i < len(titles):
            plt.title(titles[i])
        
        plt.axis('off')

    plt.tight_layout()
    plt.show()
    
    
def visualize_with_metrics(org_images, pictures):
    """
    Visualize processed images alongside quality metrics.

    For each image, this function computes PSNR, MSE, and SSIM
    between the original image and the corresponding processed image,
    then displays the processed images with the metrics shown in the title.

    Parameters
    ----------
    org_images : list of numpy.ndarray
        List of original reference grayscale images.

    pictures : list of numpy.ndarray
        List of processed or noisy grayscale images to compare
        against the original images.

    Returns
    -------
    None
        Displays the images in a grid layout using matplotlib.
    """
    num_images = len(pictures)
    num_rows = (num_images + 4) // 5

    fig, axes = plt.subplots(num_rows, 5, figsize=(15, num_rows * 3))

    for i, img in enumerate(pictures):
        row = i // 5
        col = i % 5
        psnr = peak_signal_noise_ratio(org_images[i], pictures[i])
        mse = mean_squared_error(org_images[i], pictures[i])
        ssim = structural_similarity(org_images[i], pictures[i])
        axes[row, col].imshow(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
        axes[row, col].set_title(f'psnr: {psnr:.4f}\nmse: {mse:.4f}\nssim: {ssim:.4f}')
        axes[row, col].axis("off")
        
    if num_images % 5 != 0:
        for j in range(num_images % 5, 5):
            axes[num_rows-1, j].remove()

    plt.tight_layout()
    plt.show()
