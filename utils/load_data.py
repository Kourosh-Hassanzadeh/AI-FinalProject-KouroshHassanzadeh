import cv2
import os

def load_data(path, rgb=False):
    """
    Load an image from the given file path in grayscale mode.

    Args:
        path (str): Path to the image file.

    Returns:
        numpy.ndarray: Grayscale image as a NumPy array.
                       Returns None if the image cannot be loaded.
    """
    if rgb:
        img = cv2.imread(path)
    else:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        
    return img

def load_dataSet(path, rgb=False):
    """
    Load all images from a specified directory.

    This function reads all files in the given directory,
    loads each image in grayscale format using `load_data`,
    and returns them as a list.

    Parameters
    ----------
    path : str
        Path to the directory containing image files.

    Returns
    -------
    list of numpy.ndarray
        List of loaded grayscale images.
    """
    images = []
    for file in os.listdir(path):
        img_path = os.path.join(path, file)
        img = load_data(img_path, rgb)
        images.append(img)
        
    return images
        