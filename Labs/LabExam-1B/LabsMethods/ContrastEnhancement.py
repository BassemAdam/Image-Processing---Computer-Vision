import numpy as np
from skimage.color import rgb2hsv, hsv2rgb

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

def compute_histogram(image, bins=256):
    """Compute histogram of image."""
    return np.histogram(image, bins=bins, range=(0, 255))[0]

def cumulative_distribution(hist):
    """Compute cumulative distribution function."""
    return np.cumsum(hist) / float(np.sum(hist))

def histogram_equalization(image):
    """Perform histogram equalization on grayscale or RGB image."""
    if is_rgb(image):
        return process_rgb_channels(image, histogram_equalization)
    
    hist = compute_histogram(image)
    cdf = cumulative_distribution(hist)
    lookup_table = np.uint8(255 * cdf)
    return lookup_table[image.astype(np.uint8)]

def contrast_stretching(image, low_percentile=2, high_percentile=98):
    """Perform contrast stretching using percentiles."""
    if is_rgb(image):
        return process_rgb_channels(image, contrast_stretching, 
                                  low_percentile, high_percentile)
    
    low = np.percentile(image, low_percentile)
    high = np.percentile(image, high_percentile)
    scaled = ((image - low) / (high - low) * 255)
    return np.clip(scaled, 0, 255).astype(np.uint8)

def gamma_correction(image, gamma=1.0):
    """Apply gamma correction."""
    if is_rgb(image):
        return process_rgb_channels(image, gamma_correction, gamma)
    
    normalized = image / 255.0
    corrected = np.power(normalized, gamma)
    return np.uint8(corrected * 255)

def clahe(image, clip_limit=2.0, grid_size=(8, 8)):
    """Contrast Limited Adaptive Histogram Equalization."""
    if is_rgb(image):
        return process_rgb_channels(image, clahe, clip_limit, grid_size)
    
    height, width = image.shape
    tile_height = height // grid_size[0]
    tile_width = width // grid_size[1]
    result = np.zeros_like(image)
    
    for i in range(0, height, tile_height):
        for j in range(0, width, tile_width):
            tile = image[i:i+tile_height, j:j+tile_width]
            hist = compute_histogram(tile)
            hist = np.clip(hist, 0, clip_limit)
            hist = hist / float(np.sum(hist))
            cdf = np.cumsum(hist)
            mapping = np.uint8(255 * cdf)
            result[i:i+tile_height, j:j+tile_width] = mapping[tile.astype(np.uint8)]
    
    return result

def adjust_hsv(image, hue_shift=0, sat_scale=1.0, val_scale=1.0):
    """
    Adjust image using HSV color space.
    
    Args:
        image: RGB image (0-255)
        hue_shift: Hue adjustment (-180 to 180)
        sat_scale: Saturation scaling factor (0 to 2)
        val_scale: Value/brightness scaling factor (0 to 2)
    """
    # Convert to float32 for processing
    image = image.astype(np.float32) / 255.0
    
    # Convert RGB to HSV
    hsv = rgb2hsv(image)
    
    # Adjust Hue (0-1 range)
    hsv[:,:,0] = (hsv[:,:,0] + hue_shift/360.0) % 1.0
    
    # Adjust Saturation
    hsv[:,:,1] = np.clip(hsv[:,:,1] * sat_scale, 0, 1)
    
    # Adjust Value/Brightness
    hsv[:,:,2] = np.clip(hsv[:,:,2] * val_scale, 0, 1)
    
    # Convert back to RGB
    adjusted = hsv2rgb(hsv)
    
    # Convert to uint8
    return (adjusted * 255).astype(np.uint8)

# # Example usage:
# # Remove blue color cast
# img_color_fixed = adjust_hsv(img_rgb, hue_shift=-10, sat_scale=0.9)

# # Increase brightness
# img_brightness_fixed = adjust_hsv(img_rgb, val_scale=1.3)

# # Fix both
# img_final = adjust_hsv(img_rgb, hue_shift=-10, sat_scale=0.9, val_scale=1.3)

def getImageWithHist(image):
    Number_of_Pixels = image.shape[0]*image.shape[1]
    H = np.zeros(255)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            H[image[i][j]-1] += 1 
    P = H  / Number_of_Pixels       
    
    CDF = np.zeros(255)
    CDF[0] = P[0]
    for i in range(1,255):
        CDF[i] = CDF[i-1] + P[i]
    T = np.round(CDF * 255) 

    # convert to new levels
    new_image = np.zeros(image.shape)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            new_image[i][j] = T[image[i][j]-1]

    return new_image
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


def getImageWithHist_color(image, color='yellow'):
    """
    Apply histogram equalization to specific color channels.
    
    Args:
        image: RGB image array
        color: Color to adjust ('red', 'green', 'blue', 'yellow', 'cyan', 'magenta')
    """
    normalize_image_range(image)
    if not is_rgb(image):
        raise ValueError("Input must be RGB image")
    
    # Ensure input image is uint8
    image = image.astype(np.uint8)
    output = image.copy()
    
    # Define channels to process based on color
    channels = {
        'red': [0],
        'green': [1],
        'blue': [2],
        'yellow': [0, 1],  # Red + Green
        'cyan': [1, 2],    # Green + Blue
        'magenta': [0, 2]  # Red + Blue
    }
    
    if color.lower() not in channels:
        raise ValueError(f"Unsupported color: {color}")
    
    # Process selected channels
    for channel in channels[color.lower()]:
        # Get channel data
        ch_data = image[:,:,channel]
        
        # Calculate histogram (ensure integer indices)
        hist = np.zeros(256, dtype=np.int32)
        for i in range(256):
            hist[i] = np.sum(ch_data == i)
        
        # Calculate CDF
        cdf = hist.cumsum()
        
        # Normalize CDF
        cdf = ((cdf - cdf.min()) * 255) / (cdf.max() - cdf.min())
        
        # Create lookup table (ensure integer indices)
        lookup_table = np.uint8(cdf)
        
        # Apply transformation using integer indexing
        output[:,:,channel] = lookup_table[ch_data]
    
    return output

# Usage:
# img_fixed = getImageWithHist_color(img_after_median_filter, color='yellow')