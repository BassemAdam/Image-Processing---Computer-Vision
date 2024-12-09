import cv2
import numpy as np
from scipy import ndimage
from skimage.filters import threshold_otsu
from skimage.morphology import disk, opening, closing
from skimage.segmentation import watershed

def apply_dilation(mask, kernel_size=5, iterations=40):
    """
    Apply dilation to expand mask regions.
    
    Args:
        mask: Input binary mask
        kernel_size: Size of square kernel
        iterations: Number of dilation iterations
    """
    # Ensure mask is uint8 numpy array
    if mask.max() <= 1.0:
        mask = (mask * 255).astype(np.uint8)
    else:
        mask = mask.astype(np.uint8)
    
    # Create kernel
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # Apply dilation
    dilated_mask = cv2.dilate(mask, kernel, iterations=iterations)
    
    return dilated_mask

def apply_erosion(mask, kernel_size=5, iterations=20):
    """
    Apply erosion to shrink mask regions.
    
    Args:
        mask: Input binary mask
        kernel_size: Size of square kernel
        iterations: Number of erosion iterations
    """
     # Ensure mask is uint8 numpy array
    if mask.max() <= 1.0:
        mask = (mask * 255).astype(np.uint8)
    else:
        mask = mask.astype(np.uint8)
        
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    eroded_mask = cv2.erode(mask, kernel, iterations=iterations)
    return eroded_mask
def apply_mask(image, mask):
    """
    Apply a binary mask to an image.
    
    Parameters:
        image (numpy.ndarray): The original image.
        mask (numpy.ndarray): The binary mask to apply (should be 0 or 255).
    
    Returns:
        numpy.ndarray: The masked image.
    """
    # Ensure mask is binary (0 or 255)
    mask = (mask > 0).astype(np.uint8) * 255

    # If image is RGB, keep it RGB
    if len(image.shape) == 3:
        masked_image = cv2.bitwise_and(image, image, mask=mask)
    else:
        # For grayscale images
        masked_image = cv2.bitwise_and(image, mask)
    
    return masked_image

from skimage.filters import threshold_otsu
def normalize_image_range(image):
    """
    Normalize image values to 0-255 range if needed.
    
    Args:
        image (numpy.ndarray): Input image
        
    Returns:
        numpy.ndarray: Image with values in 0-255 range
    """
    if image.max() <= 1.0:
        return (image * 255).astype(np.uint8)
    return image.astype(np.uint8)

def get_minimal_mask_approach1(image):
    """
    First approach using Otsu thresholding and morphological operations.
    """
    # Normalize image
    image = normalize_image_range(image)
    
    # Convert to grayscale if RGB
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    # Apply Otsu's thresholding
    thresh = threshold_otsu(gray)
    binary = (gray > thresh).astype(np.uint8) * 255
    
    # Apply morphological operations
    selem = disk(6)
    mask = closing(binary, selem)  # Close small holes
    mask = opening(mask, selem)    # Remove small objects
    
    # Fill holes
    mask = ndimage.binary_fill_holes(mask).astype(np.uint8) * 255
    
    return mask

def get_minimal_mask_approach2(image):
    """
    Second approach using watershed segmentation.
    """
    # Normalize and convert image depth
    if image.dtype == np.float64 or image.dtype == np.float32:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)
    # Convert to grayscale if RGB
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 11, 2)
    
    # Noise removal
    kernel = np.ones((3,3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Sure background
    sure_bg = cv2.dilate(binary, kernel, iterations=3)
    
    # Finding sure foreground
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.5*dist_transform.max(), 255, 0)
    
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown==255] = 0
    
    # Apply watershed
    markers = cv2.watershed(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), markers)
    mask = np.zeros_like(gray)
    mask[markers > 1] = 255
    
    return mask

def threshold_bright_regions(image, threshold=200):
    """
    Threshold image to keep only bright pixels.
    
    Args:
        image: RGB image (0-255)
        threshold: Brightness threshold value (0-255)
    
    Returns:
        Binary mask where bright pixels are 255, others 0
    """
    # Ensure image is in correct range
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
        
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # Extract value channel (brightness)
    _, _, v = cv2.split(hsv)
    
    # Threshold value channel
    _, bright_mask = cv2.threshold(v, threshold, 255, cv2.THRESH_BINARY)
    
    return bright_mask


"""
hresholding
Calculates threshold for each pixel based on neighborhood
Uses Gaussian-weighted sum of neighborhood values
Better than global thresholding for uneven lighting
11: neighborhood size
2: constant subtracted from mean
Morphological Operations
morphologyEx: Combines basic morphological operations
MORPH_OPEN: Erosion followed by dilation (removes noise)
dilate: Expands white regions based on kernel
Distance Transform
Calculates distance from each pixel to nearest zero pixel
DIST_L2: Euclidean distance
Creates gradient from object edges to centers
Connected Components
Labels connected regions in binary image
Each separate object gets unique integer label
Returns number of labels and labeled image
Watershed
Segmentation algorithm treating image as topographic surface
Markers define starting points for flooding
Creates boundaries at points where different flood regions meet
Basic Operations
Pixel-wise subtraction between images
Used here to find uncertain regions between definite background/foreground
"""