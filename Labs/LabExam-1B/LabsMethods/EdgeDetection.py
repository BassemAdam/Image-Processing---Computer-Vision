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

def detect_edges_sobel(image):
    """Detect edges using Sobel operator."""
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
    laplacian = laplacian_operator()
    edges = ndimage.convolve(image, laplacian)
    return np.uint8(np.absolute(edges))

def canny_edge_detection(image, low_threshold=50, high_threshold=150, sigma=1):
    """Implement Canny edge detection."""
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