import numpy as np

def compute_histogram(image, bins=256):
    """Compute histogram of image."""
    return np.histogram(image, bins=bins, range=(0, 255))[0]

def cumulative_distribution(hist):
    """Compute cumulative distribution function."""
    return np.cumsum(hist) / float(np.sum(hist))

def histogram_equalization(image):
    """Perform histogram equalization."""
    hist = compute_histogram(image)
    cdf = cumulative_distribution(hist)
    
    # Create lookup table
    lookup_table = np.uint8(255 * cdf)
    
    # Map original pixels to new values
    return lookup_table[image]

def contrast_stretching(image, low_percentile=2, high_percentile=98):
    """Perform contrast stretching using percentiles."""
    low = np.percentile(image, low_percentile)
    high = np.percentile(image, high_percentile)
    
    # Scale image to full range
    scaled = ((image - low) / (high - low) * 255)
    return np.clip(scaled, 0, 255).astype(np.uint8)

def gamma_correction(image, gamma=1.0):
    """Apply gamma correction."""
    # Normalize image to 0-1 range
    normalized = image / 255.0
    
    # Apply gamma correction
    corrected = np.power(normalized, gamma)
    
    # Scale back to 0-255 range
    return np.uint8(corrected * 255)

def clahe(image, clip_limit=2.0, grid_size=(8, 8)):
    """Contrast Limited Adaptive Histogram Equalization."""
    # Get image dimensions
    height, width = image.shape
    
    # Calculate tile size
    tile_height = height // grid_size[0]
    tile_width = width // grid_size[1]
    
    # Initialize output image
    result = np.zeros_like(image)
    
    # Process each tile
    for i in range(0, height, tile_height):
        for j in range(0, width, tile_width):
            tile = image[i:i+tile_height, j:j+tile_width]
            
            # Apply histogram equalization with clipping
            hist = compute_histogram(tile)
            hist = np.clip(hist, 0, clip_limit)
            
            # Normalize histogram
            hist = hist / float(np.sum(hist))
            
            # Create mapping function
            cdf = np.cumsum(hist)
            mapping = np.uint8(255 * cdf)
            
            # Apply mapping to tile
            result[i:i+tile_height, j:j+tile_width] = mapping[tile]
    
    return result