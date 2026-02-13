import cv2

def load_data(path):
    """
    Load an image from the given file path in grayscale mode.

    Args:
        path (str): Path to the image file.

    Returns:
        numpy.ndarray: Grayscale image as a NumPy array.
                       Returns None if the image cannot be loaded.
    """
    gray_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    
    return gray_img