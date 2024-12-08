import numpy as np
from scipy import ndimage
import math

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
    if is_rgb(image):
        return process_rgb_channels(image, mean_filter, kernel_size)
    
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
    return ndimage.convolve(image, kernel)

def gaussian_filter(image, kernel_size=3, sigma=1.0):
    """Apply Gaussian smoothing."""
    if is_rgb(image):
        return process_rgb_channels(image, gaussian_filter, kernel_size, sigma)
    
    kernel = gaussian_kernel(kernel_size, sigma)
    return ndimage.convolve(image, kernel)

def median_filter(image, kernel_size=3):
    """Apply median filtering."""
    if is_rgb(image):
        return process_rgb_channels(image, median_filter, kernel_size)
    
    return ndimage.median_filter(image, size=kernel_size)

def bilateral_filter(image, kernel_size=5, sigma_spatial=1.0, sigma_intensity=50.0):
    """Apply bilateral filtering."""
    if is_rgb(image):
        return process_rgb_channels(image, bilateral_filter, 
                                  kernel_size, sigma_spatial, sigma_intensity)
    
    height, width = image.shape
    padding = kernel_size // 2
    padded = np.pad(image, padding, mode='reflect')
    output = np.zeros_like(image)
    
    for i in range(height):
        for j in range(width):
            region = padded[i:i+kernel_size, j:j+kernel_size]
            center = region[kernel_size//2, kernel_size//2]
            spatial_weight = gaussian_kernel(kernel_size, sigma_spatial)
            intensity_diff = region - center
            intensity_weight = np.exp(-(intensity_diff**2)/(2*sigma_intensity**2))
            weights = spatial_weight * intensity_weight
            weights = weights / np.sum(weights)
            output[i, j] = np.sum(region * weights)
    
    return output

def adaptive_gaussian(image, kernel_size=3, max_sigma=2.0):
    """Apply adaptive Gaussian smoothing based on local variance."""
    if is_rgb(image):
        return process_rgb_channels(image, adaptive_gaussian, kernel_size, max_sigma)
    
    local_variance = ndimage.generic_filter(image, np.var, size=kernel_size)
    max_variance = np.max(local_variance)
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
    if is_rgb(image):
        return process_rgb_channels(image, kuwahara_filter, kernel_size)
    
    padding = kernel_size // 2
    padded = np.pad(image, padding, mode='reflect')
    output = np.zeros_like(image)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            regions = [
                padded[i:i+padding+1, j:j+padding+1],
                padded[i:i+padding+1, j+padding:j+kernel_size],
                padded[i+padding:i+kernel_size, j:j+padding+1],
                padded[i+padding:i+kernel_size, j+padding:j+kernel_size]
            ]
            means = [np.mean(region) for region in regions]
            variances = [np.var(region) for region in regions]
            min_var_idx = np.argmin(variances)
            output[i, j] = means[min_var_idx]
    
    return output

# def non_local_means(image, patch_size=7, search_size=21, h=10):
    """Apply Non-Local Means denoising with proper edge handling.
    
    Args:
        image: Input image (grayscale or RGB)
        patch_size: Size of patches to compare
        search_size: Size of search window
        h: Filter strength (higher h = more smoothing)
    """
    if is_rgb(image):
        return process_rgb_channels(image, non_local_means, patch_size, search_size, h)
    
    # Pad image to handle edges
    pad_size = max(search_size, patch_size)
    padded = np.pad(image, pad_size, mode='reflect')
    
    # Initialize output
    height, width = image.shape
    output = np.zeros_like(image, dtype=np.float32)
    weights_sum = np.zeros_like(image, dtype=np.float32)
    
    # Gaussian kernel for patch comparison
    patch_gaussian = gaussian_kernel(patch_size, patch_size/3)
    
    # Search window radius
    search_radius = search_size // 2
    patch_radius = patch_size // 2
    
    # Process each pixel
    for i in range(height):
        for j in range(width):
            # Convert image coordinates to padded coordinates
            pi = i + pad_size
            pj = j + pad_size
            
            # Extract reference patch
            ref_patch = padded[pi-patch_radius:pi+patch_radius+1, 
                             pj-patch_radius:pj+patch_radius+1]
            
            # Define search region with bounds checking
            search_start_i = max(pad_size, pi - search_radius)
            search_start_j = max(pad_size, pj - search_radius)
            search_end_i = min(pad_size + height, pi + search_radius + 1)
            search_end_j = min(pad_size + width, pj + search_radius + 1)
            
            # Compare patches in search window
            for si in range(search_start_i, search_end_i):
                for sj in range(search_start_j, search_end_j):
                    # Extract comparison patch
                    comp_patch = padded[si-patch_radius:si+patch_radius+1,
                                      sj-patch_radius:sj+patch_radius+1]
                    
                    # Calculate patch distance
                    diff = ((ref_patch - comp_patch) ** 2) * patch_gaussian
                    dist = np.sum(diff) / (patch_size ** 2)
                    
                    # Calculate weight
                    weight = np.exp(-dist / (h ** 2))
                    
                    # Accumulate weighted value
                    output[i, j] += weight * padded[si, sj]
                    weights_sum[i, j] += weight
    
    # Normalize
    output = output / (weights_sum + 1e-6)  # Add small epsilon to avoid division by zero
    
    return np.clip(output, 0, 255).astype(np.uint8)