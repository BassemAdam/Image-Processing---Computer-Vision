import numpy as np

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
    if kernel is None:
        kernel = create_kernel()
    return np.min(image[kernel > 0])

def dilate(image, kernel=None):
    """Perform morphological dilation."""
    if kernel is None:
        kernel = create_kernel()
    return np.max(image[kernel > 0])

def opening(image, kernel=None):
    """Perform morphological opening (erosion followed by dilation)."""
    if kernel is None:
        kernel = create_kernel()
    eroded = erode(image, kernel)
    return dilate(eroded, kernel)

def closing(image, kernel=None):
    """Perform morphological closing (dilation followed by erosion)."""
    if kernel is None:
        kernel = create_kernel()
    dilated = dilate(image, kernel)
    return erode(dilated, kernel)

def gradient(image, kernel=None):
    """Perform morphological gradient (dilation - erosion)."""
    if kernel is None:
        kernel = create_kernel()
    return dilate(image, kernel) - erode(image, kernel)