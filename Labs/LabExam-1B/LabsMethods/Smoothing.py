import numpy as np
from scipy import ndimage
import math

def gaussian_kernel(size, sigma):
    """Generate 2D Gaussian kernel."""
    kernel = np.zeros((size, size))
    center = size // 2
    
    for i in range(size):
        for j in range(size):
            x, y = i - center, j - center
            kernel[i, j] = (1/(2*np.pi*sigma**2)) * np.exp(-(x**2 + y**2)/(2*sigma**2))
    
    return kernel / kernel.sum()

def mean_filter(image, kernel_size=3):
    """Apply mean/average filtering."""
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
    return ndimage.convolve(image, kernel)

def gaussian_filter(image, kernel_size=3, sigma=1.0):
    """Apply Gaussian smoothing."""
    kernel = gaussian_kernel(kernel_size, sigma)
    return ndimage.convolve(image, kernel)

def median_filter(image, kernel_size=3):
    """Apply median filtering."""
    return ndimage.median_filter(image, size=kernel_size)

def bilateral_filter(image, kernel_size=5, sigma_spatial=1.0, sigma_intensity=50.0):
    """Apply bilateral filtering."""
    height, width = image.shape
    padding = kernel_size // 2
    padded = np.pad(image, padding, mode='reflect')
    output = np.zeros_like(image)
    
    for i in range(height):
        for j in range(width):
            # Extract local region
            region = padded[i:i+kernel_size, j:j+kernel_size]
            center = region[kernel_size//2, kernel_size//2]
            
            # Calculate gaussian spatial weights
            spatial_weight = gaussian_kernel(kernel_size, sigma_spatial)
            
            # Calculate intensity weights
            intensity_diff = region - center
            intensity_weight = np.exp(-(intensity_diff**2)/(2*sigma_intensity**2))
            
            # Combine weights
            weights = spatial_weight * intensity_weight
            weights = weights / np.sum(weights)
            
            # Apply weights
            output[i, j] = np.sum(region * weights)
    
    return output

def adaptive_gaussian(image, kernel_size=3, max_sigma=2.0):
    """Apply adaptive Gaussian smoothing based on local variance."""
    local_variance = ndimage.generic_filter(image, np.var, size=kernel_size)
    max_variance = np.max(local_variance)
    
    # Adjust sigma based on local variance
    sigma_map = (1 - local_variance/max_variance) * max_sigma
    
    output = np.zeros_like(image)
    padding = kernel_size // 2
    padded = np.pad(image, padding, mode='reflect')
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            kernel = gaussian_kernel(kernel_size, sigma_map[i, j])
            region = padded[i:i+kernel_size, j:j+kernel_size]
            output[i, j] = np.sum(region * kernel)
    
    return output

def kuwahara_filter(image, kernel_size=5):
    """Apply Kuwahara filter for edge-preserving smoothing."""
    padding = kernel_size // 2
    padded = np.pad(image, padding, mode='reflect')
    output = np.zeros_like(image)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Define four regions (top-left, top-right, bottom-left, bottom-right)
            regions = [
                padded[i:i+padding+1, j:j+padding+1],
                padded[i:i+padding+1, j+padding:j+kernel_size],
                padded[i+padding:i+kernel_size, j:j+padding+1],
                padded[i+padding:i+kernel_size, j+padding:j+kernel_size]
            ]
            
            # Calculate mean and variance for each region
            means = [np.mean(region) for region in regions]
            variances = [np.var(region) for region in regions]
            
            # Select value from region with minimum variance
            min_var_idx = np.argmin(variances)
            output[i, j] = means[min_var_idx]
    
    return output