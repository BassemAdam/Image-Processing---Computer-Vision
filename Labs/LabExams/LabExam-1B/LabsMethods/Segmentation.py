import numpy as np
from scipy import ndimage
from sklearn.cluster import KMeans

def is_rgb(image):
    """Check if image is RGB."""
    return len(image.shape) == 3 and image.shape[2] == 3

def rgb_to_gray(image):
    """Convert RGB to grayscale using luminosity method."""
    return np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])

def otsu_threshold(image):
    """Implement Otsu's thresholding method."""
    if is_rgb(image):
        image = rgb_to_gray(image)
        
    hist = np.histogram(image, bins=256, range=(0, 256))[0]
    total_pixels = np.sum(hist)
    
    max_variance = 0
    optimal_threshold = 0
    current_sum = 0
    total_sum = sum(i * hist[i] for i in range(256))
    
    for threshold in range(256):
        current_sum += threshold * hist[threshold]
        w_background = sum(hist[:threshold])
        w_foreground = total_pixels - w_background
        
        if w_background == 0 or w_foreground == 0:
            continue
            
        mean_background = current_sum / w_background
        mean_foreground = (total_sum - current_sum) / w_foreground
        variance = w_background * w_foreground * (mean_background - mean_foreground) ** 2
        
        if variance > max_variance:
            max_variance = variance
            optimal_threshold = threshold
    
    return image > optimal_threshold

def region_growing(image, seed_point, threshold=10):
    """Implement region growing segmentation."""
    if is_rgb(image):
        image = rgb_to_gray(image)
        
    height, width = image.shape
    segmented = np.zeros_like(image, dtype=np.bool_)
    seed_value = image[seed_point]
    stack = [seed_point]
    segmented[seed_point] = True
    
    while stack:
        current_point = stack.pop()
        x, y = current_point
        
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            new_x, new_y = x + dx, y + dy
            if (0 <= new_x < height and 0 <= new_y < width and
                not segmented[new_x, new_y] and
                abs(int(image[new_x, new_y]) - int(seed_value)) <= threshold):
                segmented[new_x, new_y] = True
                stack.append((new_x, new_y))
    
    return segmented

def watershed_segmentation(image):
    """Implement watershed segmentation."""
    if is_rgb(image):
        image = rgb_to_gray(image)
        
    gradient_x = ndimage.sobel(image, axis=0)
    gradient_y = ndimage.sobel(image, axis=1)
    gradient = np.sqrt(gradient_x**2 + gradient_y**2)
    
    local_min = ndimage.minimum_filter(gradient, size=3) == gradient
    markers, num_features = ndimage.label(local_min)
    watershed = ndimage.watershed_ift(gradient.astype(np.uint8), markers)
    
    return watershed

def kmeans_segmentation(image, n_clusters=3):
    """Implement k-means clustering segmentation."""
    if is_rgb(image):
        # For RGB, reshape to (#pixels, 3) for clustering
        pixels = image.reshape(-1, 3)
    else:
        pixels = image.reshape((-1, 1))
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(pixels)
    return labels.reshape(image.shape[:-1] if is_rgb(image) else image.shape)

def adaptive_threshold(image, block_size=11, c=2):
    """Implement adaptive thresholding."""
    if is_rgb(image):
        image = rgb_to_gray(image)
        
    mean = ndimage.uniform_filter(image, size=block_size)
    return image > (mean - c)

def multi_otsu_threshold(image, n_classes=3):
    """Implement multi-level Otsu thresholding."""
    if is_rgb(image):
        image = rgb_to_gray(image)
        
    hist = np.histogram(image, bins=256, range=(0, 256))[0]
    thresholds = np.zeros(n_classes - 1, dtype=np.int32)
    max_variance = 0
    
    for t1 in range(0, 256):
        for t2 in range(t1 + 1, 256):
            w0 = np.sum(hist[:t1])
            w1 = np.sum(hist[t1:t2])
            w2 = np.sum(hist[t2:])
            
            if w0 == 0 or w1 == 0 or w2 == 0:
                continue
                
            m0 = np.sum(np.arange(0, t1) * hist[:t1]) / w0
            m1 = np.sum(np.arange(t1, t2) * hist[t1:t2]) / w1
            m2 = np.sum(np.arange(t2, 256) * hist[t2:]) / w2
            
            variance = (w0 * m0**2 + w1 * m1**2 + w2 * m2**2)
            
            if variance > max_variance:
                max_variance = variance
                thresholds = [t1, t2]
    
    return thresholds