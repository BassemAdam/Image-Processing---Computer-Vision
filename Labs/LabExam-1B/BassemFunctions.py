import skimage.io as io
import matplotlib.pyplot as plt
import numpy as np
from skimage.exposure import histogram
from matplotlib.pyplot import bar
from skimage.color import rgb2gray,rgb2hsv,rgba2rgb
import cv2
import numpy as np
from skimage import exposure
# Convolution:
from scipy.signal import convolve2d
from scipy import fftpack
import math
import cv2
import numpy as np
from skimage import exposure
from skimage.util import random_noise
from skimage.filters import median, gaussian
from skimage.feature import canny
from skimage.measure import label
from skimage.color import label2rgb
from skimage.morphology import disk
# Edges
from skimage.filters import sobel_h, sobel, sobel_v,roberts, prewitt

def preprocess_image(image_path, output_format='rgb'):
    """
    Preprocess image to desired format (rgb or grayscale)
    
    Args:
        image_path (str): Path to input image
        output_format (str): Desired output format ('rgb' or 'grayscale')
    
    Returns:
        numpy.ndarray: Processed image
    """
    # Validate input format
    if output_format.lower() not in ['rgb', 'grayscale']:
        raise ValueError("output_format must be 'rgb' or 'grayscale'")
    
    # Load image
    img = io.imread(image_path)
    
    # Check image dimensions
    if len(img.shape) < 2:
        raise ValueError("Invalid image format: Image must be 2D or 3D")
    
    # Convert based on input/output requirements
    if len(img.shape) == 2:  # Already grayscale
        if output_format.lower() == 'rgb':
            # Convert grayscale to RGB
            img = np.stack((img,)*3, axis=-1)
    else:  # 3D image
        if img.shape[2] == 4:  # RGBA
            img = rgba2rgb(img)
        
        if output_format.lower() == 'grayscale':
            img = rgb2gray(img)
    
    return img

# Example usage:
# rgb_image = preprocess_image('image.png', 'rgb')
# gray_image = preprocess_image('image.png', 'grayscale')

def print_image_details(image):
    """
    Print details of the image including dimensions, data type, min/max values, and a sample of pixel values.
    
    Args:
        image (str or numpy.ndarray): Path to input image or image array
    """
    # Check if input is a file path or an image array
    if isinstance(image, str):
        img = io.imread(image)
    elif isinstance(image, np.ndarray):
        img = image
    else:
        raise ValueError("Input must be a file path or an image array")
    
    # Print dimensions and data type
    print(f"Image dimensions: {img.shape}")
    print(f"Data type: {img.dtype}")
    
    # Print min and max values
    min_val = np.min(img)
    max_val = np.max(img)
    print(f"Min value: {min_val}")
    print(f"Max value: {max_val}")
    
    # Determine if image is too large
    if img.size > 10000:  # Arbitrary threshold for large image
        print("Image is too large to display all values. Showing a small sample:")
        sample = img[:5, :5] if len(img.shape) == 2 else img[:5, :5, :]
        print(sample)
    else:
        print("Image values:")
        print(img)

# Example usage:
# print_image_details('image.png')
# print_image_details(img_rgb)

def remove_gaussian_noise(image, sigma):
    """
    Remove Gaussian noise from an image using a Gaussian filter.
    
    Args:
        image (numpy.ndarray): Input image
        sigma (float): Standard deviation for Gaussian kernel
    
    Returns:
        numpy.ndarray: Denoised image
    """
    # Apply Gaussian filter to remove noise
    denoised_image = gaussian(image, sigma=sigma, channel_axis=-1 if image.ndim == 3 else None)
    
    return denoised_image

def remove_salt_and_pepper_noise(image):
    """
    Remove salt and pepper noise from an image using a median filter.
    
    Args:
        image (numpy.ndarray): Input image
    
    Returns:
        numpy.ndarray: Denoised image
    """
    # Apply median filter to remove noise
    denoised_image = median(image, disk(1), channel_axis=-1 if image.ndim == 3 else None)
    
    return denoised_image

def remove_noise(image, noise_type, sigma=1):
    """
    Remove specified type of noise from an image.
    
    Args:
        image (numpy.ndarray): Input image
        noise_type (str): Type of noise ('gaussian' or 'salt_and_pepper')
        sigma (float, optional): Standard deviation for Gaussian kernel (default is 1)
    
    Returns:
        numpy.ndarray: Denoised image
    """
    if noise_type == 'gaussian':
        return remove_gaussian_noise(image, sigma)
    elif noise_type == 'salt_and_pepper':
        return remove_salt_and_pepper_noise(image)
    else:
        raise ValueError("noise_type must be 'gaussian' or 'salt_and_pepper'")

# Example usage:
# img_rgb = preprocess_image('image.png', 'rgb')
# img_rgb = remove_noise(img_rgb, 'gaussian', sigma=3)
# img_rgb = remove_noise(img_rgb, 'salt_and_pepper')
# print_image_details(img_rgb)

def enhance_contrast(image, method='clahe', **kwargs):
    """
    Enhance the contrast of an image using the specified method.
    
    Args:
        image (numpy.ndarray): Input image
        method (str): Contrast enhancement method ('clahe', 'hist_eq', 'linear')
        **kwargs: Additional parameters for the specific enhancement method
    
    Returns:
        numpy.ndarray: Contrast-enhanced image
    """
    # Ensure the image is in the correct format
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = (image * 255).astype(np.uint8)
    
    if method == 'clahe':
        clip_limit = kwargs.get('clip_limit', 2.0)
        tile_grid_size = kwargs.get('tile_grid_size', (8, 8))
        
        if len(image.shape) == 2:  # Grayscale image
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            enhanced_image = clahe.apply(image)
        else:  # RGB or RGBA image
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to the L channel
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            l_clahe = clahe.apply(l)
            
            # Merge the CLAHE enhanced L channel back with A and B channels
            lab_clahe = cv2.merge((l_clahe, a, b))
            
            # Convert back to RGB color space
            enhanced_image = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
    
    elif method == 'hist_eq':
        if len(image.shape) == 2:  # Grayscale image
            enhanced_image = cv2.equalizeHist(image)
        else:  # RGB or RGBA image
            # Convert to YUV color space
            yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            y, u, v = cv2.split(yuv)
            
            # Apply histogram equalization to the Y channel
            y_eq = cv2.equalizeHist(y)
            
            # Merge the equalized Y channel back with U and V channels
            yuv_eq = cv2.merge((y_eq, u, v))
            
            # Convert back to RGB color space
            enhanced_image = cv2.cvtColor(yuv_eq, cv2.COLOR_YUV2RGB)
    
    elif method == 'linear':
        p2 = kwargs.get('p2', 2)
        p98 = kwargs.get('p98', 98)
        
        # Apply linear contrast stretching
        p2, p98 = np.percentile(image, (p2, p98))
        enhanced_image = exposure.rescale_intensity(image, in_range=(p2, p98))
    
    else:
        raise ValueError("Unsupported contrast enhancement method. Choose 'clahe', 'hist_eq', or 'linear'.")
    
    return enhanced_image

# Example usage in Jupyter Notebook:
# img_rgb_c2 = enhance_contrast(img_rgb_c1, method='clahe', clip_limit=3.0, tile_grid_size=(8, 8))
# img_rgb_c2 = enhance_contrast(img_rgb_c1, method='hist_eq')
# img_rgb_c2 = enhance_contrast(img_rgb_c1, method='linear', p2=5, p98=95)

def correct_color_cast(image, method='white_balance', **kwargs):
    """
    Correct color cast issues.
    
    Args:
        image: Input image
        method: 'white_balance' or 'gray_world'
        **kwargs:
            - percentile (1-99, for white balance)
    """
    if method == 'white_balance':
        percentile = kwargs.get('percentile', 95)
        
        # Calculate white point for each channel
        r, g, b = cv2.split(image)
        r_white = np.percentile(r, percentile)
        g_white = np.percentile(g, percentile)
        b_white = np.percentile(b, percentile)
        
        # Scale channels
        scale = max(r_white, g_white, b_white) / np.array([r_white, g_white, b_white])
        return cv2.merge([
            np.clip(r * scale[0], 0, 255).astype(np.uint8),
            np.clip(g * scale[1], 0, 255).astype(np.uint8),
            np.clip(b * scale[2], 0, 255).astype(np.uint8)
        ])
    
    elif method == 'gray_world':
        return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

def adjust_exposure(image, method='clahe', **kwargs):
    """
    Adjust image exposure and brightness.
    
    Args:
        image: Input image
        method: 'clahe', 'hist_eq', or 'brightness'
        **kwargs: 
            - clahe: clip_limit (1-5), tile_grid_size (2-16)
            - brightness: alpha (contrast, 0.5-3), beta (brightness, -100 to 100)
    """
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = (image * 255).astype(np.uint8)
        
    if method == 'brightness':
        alpha = kwargs.get('alpha', 1.0)
        beta = kwargs.get('beta', 0)
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    return enhance_contrast(image, method, **kwargs)

def correct_color_cast(image, method='white_balance', **kwargs):
    """
    Correct color cast issues.
    
    Args:
        image: Input image
        method: 'white_balance' or 'gray_world'
        **kwargs:
            - percentile (1-99, for white balance)
    """
    if method == 'white_balance':
        percentile = kwargs.get('percentile', 95)
        
        # Calculate white point for each channel
        r, g, b = cv2.split(image)
        r_white = np.percentile(r, percentile)
        g_white = np.percentile(g, percentile)
        b_white = np.percentile(b, percentile)
        
        # Scale channels
        scale = max(r_white, g_white, b_white) / np.array([r_white, g_white, b_white])
        return cv2.merge([
            np.clip(r * scale[0], 0, 255).astype(np.uint8),
            np.clip(g * scale[1], 0, 255).astype(np.uint8),
            np.clip(b * scale[2], 0, 255).astype(np.uint8)
        ])
    
    elif method == 'gray_world':
        return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)


def reduce_noise(image, method='nlmeans', **kwargs):
    """
    Reduce image noise.
    
    Args:
        image: Input image
        method: 'nlmeans' or 'gaussian'
        **kwargs:
            - h (strength, 1-20 for nlmeans)
            - sigma (0.1-5.0 for gaussian)
    """
    if method == 'nlmeans':
        h = kwargs.get('h', 10)
        return cv2.fastNlMeansDenoisingColored(image, None, h, h, 7, 21)
    
    elif method == 'gaussian':
        return remove_gaussian_noise(image, kwargs.get('sigma', 1.0))

# Example usage:
# img_fixed = adjust_exposure(img, method='clahe', clip_limit=3.0)
# img_fixed = correct_perspective(img, [[x1,y1], [x2,y2], [x3,y3], [x4,y4]])
# img_fixed = correct_color_cast(img, method='white_balance', percentile=95)
# img_fixed = reduce_noise(img, method='nlmeans', h=10)