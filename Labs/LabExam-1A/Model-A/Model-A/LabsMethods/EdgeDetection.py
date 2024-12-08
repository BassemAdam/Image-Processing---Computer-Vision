import numpy as np
from scipy import ndimage

def pad_image(image, pad_width):
    """Add zero padding to image."""
    return np.pad(image, pad_width, mode='constant', constant_values=0)

def sobel_operator():
    """Return Sobel operators for x and y directions."""
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
    
    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])
    return sobel_x, sobel_y

def prewitt_operator():
    """Return Prewitt operators for x and y directions."""
    prewitt_x = np.array([[-1, 0, 1],
                          [-1, 0, 1],
                          [-1, 0, 1]])
    
    prewitt_y = np.array([[-1, -1, -1],
                          [0, 0, 0],
                          [1, 1, 1]])
    return prewitt_x, prewitt_y

def is_rgb(image):
    """Check if image is RGB."""
    return len(image.shape) == 3 and image.shape[2] == 3

def process_rgb_channels(image, function, *args, **kwargs):
    """Apply function to each RGB channel."""
    # Convert to float for processing
    r, g, b = image[:,:,0], image[:,:,1], image[:,:,2]
    r_processed = function(r, *args, **kwargs)
    g_processed = function(g, *args, **kwargs)
    b_processed = function(b, *args, **kwargs)
    # Combine channels
    return np.dstack((r_processed, g_processed, b_processed))

def detect_edges_sobel(image):
    """Detect edges using Sobel operator."""
    if is_rgb(image):
        return process_rgb_channels(image, detect_edges_sobel)
    
    sobel_x, sobel_y = sobel_operator()
    
    # Apply convolution
    gradient_x = ndimage.convolve(image, sobel_x)
    gradient_y = ndimage.convolve(image, sobel_y)
    
    # Calculate magnitude
    magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    
    # Normalize to 0-255 range
    magnitude = (magnitude / magnitude.max() * 255).astype(np.uint8)
    
    return magnitude

def detect_edges_prewitt(image):
    """Detect edges using Prewitt operator."""
    if is_rgb(image):
        return process_rgb_channels(image, detect_edges_prewitt)
    
    prewitt_x, prewitt_y = prewitt_operator()
    
    gradient_x = ndimage.convolve(image, prewitt_x)
    gradient_y = ndimage.convolve(image, prewitt_y)
    
    magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    magnitude = (magnitude / magnitude.max() * 255).astype(np.uint8)
    
    return magnitude

def laplacian_operator():
    """Return Laplacian operator."""
    return np.array([[0, 1, 0],
                     [1, -4, 1],
                     [0, 1, 0]])

def detect_edges_laplacian(image):
    """Detect edges using Laplacian operator."""
    if is_rgb(image):
        return process_rgb_channels(image, detect_edges_laplacian)
    
    laplacian = laplacian_operator()
    edges = ndimage.convolve(image, laplacian)
    return np.uint8(np.absolute(edges))

def canny_edge_detection(image, low_threshold=50, high_threshold=150, sigma=1):
    """Implement Canny edge detection."""
    if is_rgb(image):
        # For RGB, process intensity channel only
        # Convert to grayscale using luminosity method
        gray = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
        return canny_edge_detection(gray, low_threshold, high_threshold, sigma)
    
    # 1. Gaussian smoothing
    smoothed = ndimage.gaussian_filter(image, sigma)
    
    # 2. Compute gradients
    sobel_x, sobel_y = sobel_operator()
    gradient_x = ndimage.convolve(smoothed, sobel_x)
    gradient_y = ndimage.convolve(smoothed, sobel_y)
    
    # Calculate gradient magnitude and direction
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient_direction = np.arctan2(gradient_y, gradient_x)
    
    # 3. Non-maximum suppression
    height, width = image.shape
    suppressed = np.zeros_like(gradient_magnitude)
    
    for i in range(1, height-1):
        for j in range(1, width-1):
            angle = gradient_direction[i, j] * 180 / np.pi
            angle = angle % 180
            
            if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                neighbors = [gradient_magnitude[i, j+1], gradient_magnitude[i, j-1]]
            elif 22.5 <= angle < 67.5:
                neighbors = [gradient_magnitude[i+1, j-1], gradient_magnitude[i-1, j+1]]
            elif 67.5 <= angle < 112.5:
                neighbors = [gradient_magnitude[i+1, j], gradient_magnitude[i-1, j]]
            else:
                neighbors = [gradient_magnitude[i-1, j-1], gradient_magnitude[i+1, j+1]]
            
            if gradient_magnitude[i, j] >= max(neighbors):
                suppressed[i, j] = gradient_magnitude[i, j]
    
    # 4. Double thresholding
    strong_edges = (suppressed > high_threshold)
    weak_edges = (suppressed >= low_threshold) & (suppressed <= high_threshold)
    
    # 5. Edge tracking by hysteresis
    final_edges = np.copy(strong_edges)
    
    for i in range(1, height-1):
        for j in range(1, width-1):
            if weak_edges[i, j]:
                if np.any(strong_edges[i-1:i+2, j-1:j+2]):
                    final_edges[i, j] = 1
    
    return np.uint8(final_edges * 255)


def unsharp_mask(image, radius=1, amount=1.0, threshold=0):
    """
    Advanced unsharp masking with edge-awareness.
    Handles both [0,1] and [0,255] ranges.
    
    Args:
        image: Input image (either 0-1 or 0-255 range)
        radius: Gaussian blur radius
        amount: Sharpening strength (1.0-2.0 typical)
        threshold: Minimum brightness difference to sharpen
    """
    if is_rgb(image):
        return process_rgb_channels(image, unsharp_mask, radius, amount, threshold)
    
    # Detect input range
    is_normalized = image.max() <= 1.0
    
    # Convert to float for processing
    if is_normalized:
        image = image.astype(np.float32)
        threshold = threshold / 255.0  # Scale threshold to 0-1 range
    else:
        image = image.astype(np.float32)
    
    # Create blurred version
    blurred = ndimage.gaussian_filter(image, radius)
    
    # Calculate high-pass (detail) component
    highpass = image - blurred
    
    # Calculate local contrast for edge-awareness
    local_contrast = ndimage.gaussian_filter(np.abs(highpass), radius * 2)
    
    # Create edge-aware mask
    edge_mask = 1.0 / (1.0 + np.exp(-local_contrast + threshold))
    
    # Apply sharpening with edge-awareness
    sharpened = image + (highpass * amount * edge_mask)
    
    # Clip values and return in original range
    if is_normalized:
        return np.clip(sharpened, 0, 1).astype(np.float32)
    return np.clip(sharpened, 0, 255).astype(np.uint8)

def adaptive_sharpen(image, radius=1, amount=1.0, threshold=0):
    """Adaptive sharpening based on local image properties.
    
    Args:
        image: Input image
        radius: Base radius for Gaussian operations
        amount: Base sharpening amount
        threshold: Minimum difference for sharpening
    """
    if is_rgb(image):
        return process_rgb_channels(image, adaptive_sharpen, radius, amount, threshold)
    
    # Convert to float
    image = image.astype(np.float32)
    
    # Calculate local variance for adaptivity
    local_var = ndimage.gaussian_filter((image - ndimage.gaussian_filter(image, radius))**2, radius*2)
    
    # Create adaptive radius and amount maps
    radius_map = radius * (1.0 + np.exp(-local_var/128.0))
    amount_map = amount * (1.0 + local_var/128.0)
    
    # Apply unsharp masking with varying parameters
    height, width = image.shape
    output = np.zeros_like(image)
    
    for i in range(height):
        for j in range(width):
            r = radius_map[i,j]
            patch = image[max(0,i-2):min(height,i+3), max(0,j-2):min(width,j+3)]
            blurred = ndimage.gaussian_filter(patch, r)
            highpass = patch - blurred
            output[i,j] = image[i,j] + (highpass[2,2] * amount_map[i,j])
    
    return np.clip(output, 0, 255).astype(np.uint8)

def edge_aware_sharpen(image, radius=1, amount=1.0, threshold=10):
    """
    Edge-aware sharpening using gradient information.
    Handles both [0,1] and [0,255] ranges.
    
    Args:
        image: Input image (either 0-1 or 0-255 range)
        radius: Gaussian blur radius
        amount: Sharpening strength
        threshold: Edge detection threshold (will be scaled if image is 0-1)
    """
    if is_rgb(image):
        return process_rgb_channels(image, edge_aware_sharpen, radius, amount, threshold)
    
    # Detect input range
    is_normalized = image.max() <= 1.0
    
    if is_normalized:
        # Scale threshold to 0-1 range
        threshold = threshold / 255.0
        image_255 = image
    else:
        image_255 = image / 255.0  # Normalize for gradient computation
    
    # Get edge information using Sobel
    sobel_x, sobel_y = sobel_operator()
    grad_x = ndimage.convolve(image_255, sobel_x)
    grad_y = ndimage.convolve(image_255, sobel_y)
    gradient = np.sqrt(grad_x**2 + grad_y**2)
    
    # Create edge-aware mask
    edge_mask = gradient > threshold
    
    # Apply unsharp masking only to non-edge regions
    sharpened = unsharp_mask(image, radius, amount)
    
    # Blend based on edge mask
    result = np.where(edge_mask, image, sharpened)
    
    # Return in original range
    if is_normalized:
        return np.clip(result, 0, 1).astype(np.float32)
    return np.clip(result, 0, 255).astype(np.uint8)