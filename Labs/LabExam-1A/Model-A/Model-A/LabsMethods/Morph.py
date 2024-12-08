import numpy as np
from scipy import ndimage

def is_rgb(image):
    """Check if image is RGB."""
    return len(image.shape) == 3 and image.shape[2] == 3

def process_rgb_channels(image, function, *args, **kwargs):
    """Apply function to each RGB channel."""
    r, g, b = image[:,:,0], image[:,:,1], image[:,:,2]
    r_processed = function(r, *args, **kwargs)
    g_processed = function(g, *args, **kwargs)
    b_processed = function(b, *args, **kwargs)
    return np.dstack((r_processed, g_processed, b_processed))

def create_kernel(shape='square', size=3):
    """Create morphological kernel/structuring element."""
    if shape == 'square':
        return np.ones((size, size), dtype=np.uint8)
    elif shape == 'cross':
        kernel = np.zeros((size, size), dtype=np.uint8)
        kernel[size//2,:] = 1
        kernel[:,size//2] = 1
        return kernel

def erode(image, kernel=None):
    """Perform morphological erosion."""
    if is_rgb(image):
        return process_rgb_channels(image, erode, kernel)
    
    if kernel is None:
        kernel = create_kernel()
    return ndimage.minimum_filter(image, footprint=kernel)

def dilate(image, kernel=None):
    """Perform morphological dilation."""
    if is_rgb(image):
        return process_rgb_channels(image, dilate, kernel)
    
    if kernel is None:
        kernel = create_kernel()
    return ndimage.maximum_filter(image, footprint=kernel)

def opening(image, kernel=None):
    """Perform morphological opening (erosion followed by dilation)."""
    if is_rgb(image):
        return process_rgb_channels(image, opening, kernel)
    
    if kernel is None:
        kernel = create_kernel()
    eroded = erode(image, kernel)
    return dilate(eroded, kernel)

def closing(image, kernel=None):
    """Perform morphological closing (dilation followed by erosion)."""
    if is_rgb(image):
        return process_rgb_channels(image, closing, kernel)
    
    if kernel is None:
        kernel = create_kernel()
    dilated = dilate(image, kernel)
    return erode(dilated, kernel)

def gradient(image, kernel=None):
    """Perform morphological gradient (dilation - erosion)."""
    if is_rgb(image):
        return process_rgb_channels(image, gradient, kernel)
    
    if kernel is None:
        kernel = create_kernel()
    return dilate(image, kernel) - erode(image, kernel)