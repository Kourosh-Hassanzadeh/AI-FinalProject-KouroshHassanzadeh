import matplotlib.pyplot as plt
import math

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
