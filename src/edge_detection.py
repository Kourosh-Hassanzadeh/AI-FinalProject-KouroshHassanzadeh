import cv2
import numpy as np
import sys
sys.path.append('.')
from utils.load_data import load_dataSet
from utils.plot import plot_images

def prewitt_ED(images):
    """
    Performs Prewitt edge detection on a list of input images.

    Args:
    - images: List of input images (in BGR format)

    Returns:
    - prewitt_images: List of Prewitt edge-detected images
    """
    
    prewitt_images = []
    
    kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    
    for image in images: 
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        grad_x = cv2.filter2D(image, -1, kernel_x)
        grad_y = cv2.filter2D(image, -1, kernel_y)
        edge_detected = cv2.addWeighted(cv2.convertScaleAbs(grad_x), 0.5, 
                                     cv2.convertScaleAbs(grad_y), 0.5, 0)
        prewitt_images.append(edge_detected)
    
    return prewitt_images

def kirsch_ED(images):
    """
    Performs Kirsch edge detection on a list of input images.

    Args:
    - images: List of input images (in BGR format)

    Returns:
    - kirsch_images: List of Kirsch edge-detected images
    """
    
    kirsch_images = []
    
    # Define Kirsch masks for edge detection in 8 directions
    KIRSCH_K1   = np.array([[ 5, -3, -3], [ 5,  0, -3], [ 5, -3, -3]], dtype=np.float32) / 15
    KIRSCH_K2   = np.array([[-3, -3,  5], [-3,  0,  5], [-3, -3,  5]], dtype=np.float32) / 15
    KIRSCH_K3   = np.array([[-3, -3, -3], [ 5,  0, -3], [ 5,  5, -3]], dtype=np.float32) / 15
    KIRSCH_K4   = np.array([[-3,  5,  5], [-3,  0,  5], [-3, -3, -3]], dtype=np.float32) / 15
    KIRSCH_K5   = np.array([[-3, -3, -3], [-3,  0, -3], [ 5,  5,  5]], dtype=np.float32) / 15
    KIRSCH_K6   = np.array([[ 5,  5,  5], [-3,  0, -3], [-3, -3, -3]], dtype=np.float32) / 15
    KIRSCH_K7   = np.array([[-3, -3, -3], [-3,  0,  5], [-3,  5,  5]], dtype=np.float32) / 15
    KIRSCH_K8   = np.array([[ 5,  5, -3], [ 5,  0, -3], [-3, -3, -3]], dtype=np.float32) / 15
    
    for image in images:
        
        # Convert the image to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Kirsch masks to detect edges in 8 directions and take the maximum response
        edges    = np.maximum(cv2.filter2D(image, cv2.CV_8U, KIRSCH_K1),
              np.maximum(cv2.filter2D(image, cv2.CV_8U, KIRSCH_K2),
              np.maximum(cv2.filter2D(image, cv2.CV_8U, KIRSCH_K3),
              np.maximum(cv2.filter2D(image, cv2.CV_8U, KIRSCH_K4),
              np.maximum(cv2.filter2D(image, cv2.CV_8U, KIRSCH_K5),
              np.maximum(cv2.filter2D(image, cv2.CV_8U, KIRSCH_K6),
              np.maximum(cv2.filter2D(image, cv2.CV_8U, KIRSCH_K7),
                            cv2.filter2D(image, cv2.CV_8U, KIRSCH_K8),
                           )))))))
        
        kirsch_images.append(edges)
    
    return kirsch_images

def Marr_Hildreth_ED(images, sigma=1.4, threshold=0.5):
    """
    Performs Marr-Hildreth edge detection on a list of input images.

    Args:
    - images: List of input images (in BGR format)
    - sigma: Standard deviation for Gaussian smoothing (default: 1.4)
    - threshold: Threshold value for edge detection (default: 0.5)

    Returns:
    - MH_images: List of Marr-Hildreth edge-detected images
    """
    
    MH_images = []
    
    for image in images:
        
        # Convert the image to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian smoothing
        smoothed_image = cv2.GaussianBlur(image, (5, 5), sigma)
        
        # Compute the Laplacian of the smoothed image
        laplacian = cv2.Laplacian(smoothed_image, cv2.CV_64F)
        
        # Find zero crossings in the Laplacian image
        edges = np.zeros_like(laplacian)
        rows, cols = laplacian.shape
        
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                neighbors = [laplacian[i-1, j], laplacian[i+1, j], laplacian[i, j-1], laplacian[i, j+1],
                            laplacian[i-1, j-1], laplacian[i-1, j+1], laplacian[i+1, j-1], laplacian[i+1, j+1]]
                if any(np.sign(neighbors[:-1]) != np.sign(neighbors[1:])):
                    edges[i, j] = 255
        
        edges = 255 - edges
                    
        # Apply thresholding to the edge image
        edges[edges >= (threshold * 255)] = 255
        edges[edges < (threshold * 255)] = 0
        
        MH_images.append(edges.astype(np.uint8))
              
    return MH_images

def canny_ED(images, min_threshold=100, max_threshold=200):
    """
    Performs Canny edge detection on a list of input images.

    Args:
    - images: List of input images (in BGR format)
    - min_threshold: Minimum threshold value (default: 100)
    - max_threshold: Maximum threshold value (default: 200)

    Returns:
    - canny_images: List of Canny edge-detected images
    """
    
    canny_images = []
    
    for image in images:
        
        # Convert the image to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Canny edge detection
        edges = cv2.Canny(image, min_threshold, max_threshold)
        
        canny_images.append(edges)
    
    return canny_images

        
if __name__ == '__main__':
    
    images = load_dataSet('data/edge_detection', rgb=True)
    
    # prewitt_images = prewitt_ED(images)
    # plot_images(prewitt_images)
    
    # kirsch_images = kirsch_ED(images)
    # plot_images(kirsch_images)
    
    # MH_images = Marr_Hildreth_ED(images)
    # plot_images(MH_images)
    
    canny_images = canny_ED(images)
    plot_images(canny_images)