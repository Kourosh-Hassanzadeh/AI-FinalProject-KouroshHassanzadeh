import cv2
from skimage import exposure
import matplotlib.pyplot as plt
from skimage.feature import hog
import sys
sys.path.append('.')
from utils.load_data import load_dataSet

def create_hog(images):
    """
    Computes Histogram of Oriented Gradients (HOG) features for a set of input images.

    Args:
    - images: List of input images in RGB format

    Returns:
    - hog_images: List of HOG feature representations of the input images
    """
    
    hog_images = []
    
    for image in images:
        
        
        # Compute HOG features and HOG image visualization
        hog_features, hog_image = hog(image, orientations=8, pixels_per_cell=(8, 8),
                              cells_per_block=(2, 2), visualize=True, block_norm='L2-Hys')


        # Display the original image and its corresponding HOG features visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
        ax1.axis('off')
        ax1.imshow(image, cmap=plt.cm.gray)
        ax1.set_title('Input Image')
        
        # Rescale intensity for better visualization of the HOG image
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
        hog_images.append(hog_image_rescaled)

        ax2.axis('off')
        ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
        ax2.set_title('HOG Features')

        plt.show()


if __name__ == "__main__":
    
    images = load_dataSet("data/histogram_analysis")
    create_hog(images)