import cv2
import numpy as np
from matplotlib import pyplot as plt

# ---------------------------Noise reduction Methods--------------------------------
def apply_gaussian_blur_rgb(image, kernel_size=(5, 5), sigma=1.5):
    """
    Applies Gaussian blur to an RGB image by processing each channel independently.
    
    Args:
        image: RGB image (numpy array)
        kernel_size: Tuple, size of the Gaussian kernel
        sigma: Standard deviation of the Gaussian filter
    
    Returns:
        Denoised RGB image using Gaussian blur
    """
    # Split channels, apply Gaussian Blur, and merge back
    channels = cv2.split(image)
    blurred_channels = [cv2.GaussianBlur(channel, kernel_size, sigma) for channel in channels]
    return cv2.merge(blurred_channels)

def apply_median_blur_rgb(image, kernel_size=5):
    """
    Applies Median blur to an RGB image by processing each channel independently.
    Handles both float [0-1] and uint8 [0-255] ranges.
    
    Args:
        image: RGB image (numpy array)
        kernel_size: Integer, size of the kernel (must be odd)
    
    Returns:
        Denoised RGB image
    """
    # Check input range
    is_normalized = image.max() <= 1.0
    
    # Convert to uint8 if needed
    if is_normalized:
        image_uint8 = (image * 255).astype(np.uint8)
    else:
        image_uint8 = image.astype(np.uint8)
    
    # Split channels and apply median blur
    channels = cv2.split(image_uint8)
    blurred_channels = [cv2.medianBlur(channel, kernel_size) for channel in channels]
    result = cv2.merge(blurred_channels)
    
    # Return in original range
    if is_normalized:
        return result.astype(np.float32) / 255
    return result

# ---------------------------Illumination Issues Uneven Lighting--------------------------------
def apply_clahe_rgb(image, clip_limit=2.0, grid_size=(8, 8)):
    """
    Applies CLAHE to RGB image. Handles both float [0-1] and uint8 [0-255] ranges.
    
    Args:
        image: RGB image (numpy array)
        clip_limit: Threshold for contrast limiting
        grid_size: Size of grid for histogram equalization
        
    How it works ? 
    Divide Image

    Splits image into small tiles (grid_size)
    Each tile processed independently
    In Each Tile

    Calculates histogram of pixel values
    Clips histogram at clip_limit to prevent over-amplification
    Redistributes clipped pixels across histogram
    Applies histogram equalization
    Bilinear Interpolation

    Combines tile borders smoothly
    Prevents artificial boundaries
    Why LAB Color Space

    L channel: Lightness/brightness
    A,B channels: Color information
    Only enhances L channel to preserve colors
    Visual example:

    Key benefits:

    Better local contrast
    Prevents noise amplification
    Preserves edges
    Avoids artificial boundaries
    Works well for uneven lighting
    Clip Limit Effects
    Purpose
    Controls contrast enhancement level
    Prevents over-amplification of noise
    Limits histogram peaks
    """
    # Check input range
    is_normalized = image.max() <= 1.0
    
    # Convert to uint8 if needed
    if is_normalized:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)
    
    # Convert to LAB color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    
    # Split channels
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    cl_channel = clahe.apply(l_channel)
    
    # Merge channels
    merged = cv2.merge((cl_channel, a_channel, b_channel))
    result = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    
    # Return in original range
    if is_normalized:
        return result.astype(np.float32) / 255
    return result

def enhance_image(image, params=None):
    """
    Complete image enhancement pipeline including edge enhancement and color correction.
    """
    if params is None:
        params = {
            'median_kernel': 3,
            'gaussian_kernel': (3,3),
            'gaussian_sigma': 1,
            'clahe_clip': 6.0,
            'clahe_grid': (5,5),
            'nlmeans_h': 15,
            'nlmeans_template': 11,
            'nlmeans_search': 35,
            'sharpen_strength': 5,
            'white_balance_strength': 1.1
        }
    
    # Check input range
    is_normalized = image.max() <= 1.0
    if is_normalized:
        image = (image * 255).astype(np.uint8)
    
    try:
        # 1. Initial Denoising
        denoised = apply_median_blur_rgb(image, params['median_kernel'])
        smoothed = apply_gaussian_blur_rgb(denoised, 
                                         params['gaussian_kernel'],
                                         params['gaussian_sigma'])
        
        # 2. Color Cast Removal
        lab = cv2.cvtColor(smoothed, cv2.COLOR_RGB2LAB)
        avg_a = np.average(lab[:, :, 1])
        avg_b = np.average(lab[:, :, 2])
        strength = params['white_balance_strength']
        lab[:, :, 1] = lab[:, :, 1] - ((avg_a - 128) * (lab[:, :, 0] / 255.0) * strength)
        lab[:, :, 2] = lab[:, :, 2] - ((avg_b - 128) * (lab[:, :, 0] / 255.0) * strength)
        color_corrected = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # 3. Contrast Enhancement
        enhanced = apply_clahe_rgb(color_corrected,
                                 params['clahe_clip'],
                                 params['clahe_grid'])
        
        # 4. Edge Enhancement
        kernel = np.array([[0, -1, 0],
                          [-1, params['sharpen_strength'], -1],
                          [0, -1, 0]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        # 5. Final Denoising
        final = cv2.fastNlMeansDenoisingColored(sharpened, None,
                                               params['nlmeans_h'],
                                               params['nlmeans_h'],
                                               params['nlmeans_template'],
                                               params['nlmeans_search'])
        
        # Return in original range
        if is_normalized:
            return final.astype(np.float32) / 255
        return final
        
    except Exception as e:
        print(f"Error in enhancement pipeline: {str(e)}")
        return image
