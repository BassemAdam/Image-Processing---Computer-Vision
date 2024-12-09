import cv2
import numpy as np
from skimage.feature import canny
from skimage.transform import probabilistic_hough_line
import matplotlib.pyplot as plt
import numpy as np
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

def detect_and_highlight_lines_opencv(
    image, angle_range=(40, 60), canny_threshold=(50, 150), hough_threshold=150
):
    """
    Detect and highlight lines within a specified angle range using OpenCV.

    Parameters:
    - image: Input image as a numpy array (loaded externally).
    - angle_range: Tuple (min_angle, max_angle) in degrees for the slope angles to filter lines.
    - canny_threshold: Tuple (low_threshold, high_threshold) for Canny edge detection.
    - hough_threshold: Integer threshold for the Hough Line Transform.

    Returns:
    - highlighted_image: Image with the detected lines highlighted.
    """
    normalize_image_range(image)
    # Convert the image to grayscale if it's not already
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    # Edge detection
    edges = cv2.Canny(
        gray_image, threshold1=canny_threshold[0], threshold2=canny_threshold[1]
    )

    # Hough Line detection
    lines = cv2.HoughLines(edges, rho=1, theta=np.pi / 180, threshold=hough_threshold)

    # Highlight lines within the specified angle range
    highlighted_image = image.copy()
    if lines is not None:
        for rho, theta in lines[:, 0]:
            angle = np.degrees(theta)
            if angle_range[0] <= angle <= angle_range[1]:
                # Draw the line
                a, b = np.cos(theta), np.sin(theta)
                x0, y0 = a * rho, b * rho
                x1, y1 = int(x0 + 1000 * (-b)), int(y0 + 1000 * a)
                x2, y2 = int(x0 - 1000 * (-b)), int(y0 - 1000 * a)
                cv2.line(highlighted_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return highlighted_image

def detect_and_highlight_lines_skimage(image, angle_range=(40, 60), canny_sigma=2.0, 
                                     line_length=50, line_gap=3, color=(255, 0, 0)):
    """
    Detect and highlight lines using skimage with direct image drawing.
    
    Parameters:
        image: RGB/BGR image array
        angle_range: (min_angle, max_angle) in degrees
        canny_sigma: Gaussian sigma for edge detection
        line_length: Minimum line length
        line_gap: Maximum line gap
        color: Line color in BGR format
    """
    # Convert to grayscale while preserving original
    if len(image.shape) == 3:
        from skimage.color import rgb2gray
        gray_image = rgb2gray(image)
    else:
        gray_image = image.copy()
    
    # Create output image
    highlighted_image = image.copy()
    
    # Edge detection
    edges = canny(gray_image, sigma=canny_sigma)
    
    # Detect lines
    lines = probabilistic_hough_line(edges, 
                                   threshold=10,
                                   line_length=line_length,
                                   line_gap=line_gap)
    
    # Draw lines within angle range
    for line in lines:
        p0, p1 = line
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        angle = np.degrees(np.arctan2(dy, dx))
        
        if angle_range[0] <= abs(angle) <= angle_range[1]:
            # Convert points to integer coordinates
            start_point = (int(p0[0]), int(p0[1]))
            end_point = (int(p1[0]), int(p1[1]))
            
            # Draw line on image
            cv2.line(highlighted_image, 
                    start_point, 
                    end_point, 
                    color, 
                    2)  # thickness=2
    
    return highlighted_image
