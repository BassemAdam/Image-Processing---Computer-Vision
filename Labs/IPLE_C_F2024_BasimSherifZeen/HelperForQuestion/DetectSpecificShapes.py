import cv2
import numpy as np

def detect_points(image):
    """
    Detect intersection points in an image.
    
    Args:
        image: Input image (grayscale or RGB, values in [0,1] or [0,255])
        
    Returns:
        List of intersection points [(x,y),...]
    """
    # Input validation and normalization
    if len(image.shape) == 3:  # RGB image
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Handle [0,1] range
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(image, (3,3), 3)
    
    # Step 1: Edge detection using Canny
    # First argument: input image
    # Second/Third: lower/upper thresholds for edge detection
    # apertureSize: size of Sobel kernel for gradients
    edges = cv2.Canny(blurred, 70, 210, apertureSize=7, L2gradient=True)
    
    # Step 2: Probabilistic Hough Line Transform
    # rho: distance resolution (pixels)
    # theta: angle resolution (radians)
    # threshold: minimum number of intersections to detect line
    # minLineLength: minimum length of line
    # maxLineGap: maximum gap between line segments
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, 
                           threshold=100, minLineLength=50, maxLineGap=10)
    
    if lines is None:
        return []
    
    def line_intersection(line1, line2):
        """
        Find intersection point of two lines.
        
        Args:
            line1, line2: Lines in format [x1,y1,x2,y2]
            
        Returns:
            Intersection point (x,y) or None if parallel
        """
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        
        # Calculate line coefficients (Ax + By = C)
        A1 = y2 - y1  # First line
        B1 = x1 - x2
        C1 = A1 * x1 + B1 * y1
        
        A2 = y4 - y3  # Second line
        B2 = x3 - x4
        C2 = A2 * x3 + B2 * y3
        
        # Calculate determinant to check if lines are parallel
        determinant = A1 * B2 - A2 * B1
        if determinant == 0:
            return None
        
        # Calculate intersection point using Cramer's rule
        x = (B2 * C1 - B1 * C2) / determinant
        y = (A1 * C2 - A2 * C1) / determinant
        
        return int(x), int(y)
    
    # Step 3: Find all valid intersections
    intersections = []
    for i, line1 in enumerate(lines):
        for j, line2 in enumerate(lines):
            if i < j:  # Avoid duplicate comparisons
                intersect = line_intersection(line1[0], line2[0])
                if intersect is not None:
                    intersections.append(intersect)
    
    # Step 4: Filter out invalid intersections
    height, width = image.shape
    valid_points = [point for point in intersections 
                   if 0 <= point[0] < width and 0 <= point[1] < height]
    
    
     # Cluster nearby points
    clustered_points = cluster_points(valid_points, distance_threshold=10)
    
    return clustered_points

# Example usage:
# img = preprocess_image('Q2.png', 'rgb')  # Load image
# points = detect_points(img)  # Detect points
# print("Detected Points:", points)

def draw_detected_points(image, points, radius=5, thickness=2):
    """
    Draw detected points with different colors for identification.
    """
    # Input validation and normalization
    is_normalized = image.max() <= 1.0
    if is_normalized:
        image_uint8 = (image * 255).astype(np.uint8)
    else:
        image_uint8 = image.astype(np.uint8)
    
    # Convert grayscale to RGB if needed
    if len(image_uint8.shape) == 2:
        marked_image = cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2RGB)
    else:
        marked_image = image_uint8.copy()
    
    # Define distinct colors for points (BGR format)
    colors = [
        (255, 0, 0),    # Blue
        (0, 255, 0),    # Green
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 0, 0),    # Dark Blue
        (0, 128, 0),    # Dark Green
        (0, 0, 128),    # Dark Red
    ]
    
    # Create color mapping dictionary
    color_mapping = {}
    
    # Draw each point with different color and label
    for i, (x, y) in enumerate(points):
        color = colors[i % len(colors)]
        # Draw circle
        cv2.circle(marked_image, (x, y), radius, color, thickness)
        
        # Add point label
        label = f"P{i+1}"
        # Position text above the circle
        text_x = x - 10
        text_y = y - radius - 5
        
        # Draw text with white background for visibility
        (text_width, text_height), _ = cv2.getTextSize(label, 
                                                      cv2.FONT_HERSHEY_SIMPLEX,
                                                      0.5, 1)
        cv2.rectangle(marked_image, 
                     (text_x, text_y - text_height),
                     (text_x + text_width, text_y + 5),
                     (255, 255, 255),
                     -1)
        
        # Draw text
        cv2.putText(marked_image, label,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 1)
        
        color_mapping[f"Point {i+1}"] = {
            'coordinates': (x, y),
            'color': color
        }
    
    # Return in original range
    if is_normalized:
        marked_image = marked_image.astype(np.float32) / 255
    
    return marked_image, color_mapping

# Usage:
"""
img = preprocess_image('Q2.png', 'rgb')
points = detect_points(img)
marked_img, mapping = draw_detected_points(img, points)

# Print point mapping
for point, info in mapping.items():
    print(f"{point}: {info['coordinates']}")
"""

def cluster_points(points, distance_threshold=10):
    """
    Cluster nearby points and return single representative point per cluster.
    
    Args:
        points: List of (x,y) points
        distance_threshold: Maximum distance between points in same cluster
    """
    if not points:
        return []
    
    # Initialize clusters
    clusters = []
    used_points = set()
    
    for i, point1 in enumerate(points):
        if i in used_points:
            continue
            
        # Start new cluster
        current_cluster = [point1]
        used_points.add(i)
        
        # Find all points close to this one
        for j, point2 in enumerate(points):
            if j in used_points:
                continue
                
            # Calculate distance between points
            dist = np.sqrt((point1[0] - point2[0])**2 + 
                            (point1[1] - point2[1])**2)
            
            if dist <= distance_threshold:
                current_cluster.append(point2)
                used_points.add(j)
        
        # Add cluster to list
        clusters.append(current_cluster)
    
    # Calculate center point for each cluster
    final_points = []
    for cluster in clusters:
        x_mean = int(np.mean([p[0] for p in cluster]))
        y_mean = int(np.mean([p[1] for p in cluster]))
        final_points.append((x_mean, y_mean))
    
    return final_points
    
   