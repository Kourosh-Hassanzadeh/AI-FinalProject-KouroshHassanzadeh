import cv2
import numpy as np
import sys
sys.path.append('.')
from utils.load_data import load_dataSet
from utils.plot import plot_images

def segment_colors(images, num_colors):
    """
    Segments the colors in a list of images using K-means clustering.

    Args:
    - images: List of input images
    - num_colors: Number of colors for segmentation

    Returns:
    - segmented_images: List of segmented images
    """

    segmented_images = []
    
    for image in images:
        
        # Reshape image pixels to a 2D array
        pixels = image.reshape(-1, 3)
        pixels = np.float32(pixels)
        
        # Set criteria for K-means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
        
        # Apply K-means clustering
        retval, labels, centers = cv2.kmeans(pixels, num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # Convert centers to uint8
        centers = np.uint8(centers)
        
        # Retrieve segmented data based on the labels
        segmented_data = centers[labels.flatten()]
        
        # Reshape the segmented data to the original image shape
        segmented_image = segmented_data.reshape((image.shape))
        
        segmented_images.append(segmented_image)
    
    return segmented_images

if __name__ == "__main__":
    
    images = load_dataSet("data/segmentation", rgb=True)
    # Perform segmentation with different numbers of colors
    # segmented_8_colors = segment_colors(images, 8)
    # segmented_16_colors = segment_colors(images, 16)
    segmented_32_colors = segment_colors(images, 32)

    plot_images(segmented_32_colors)