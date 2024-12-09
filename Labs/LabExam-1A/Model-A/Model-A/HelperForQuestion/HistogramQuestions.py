
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr


def create_histogram_from_equation(equation_str, width=256, height=256):
    """
    Creates histogram values from a mathematical equation.
    
    Args:
        equation_str: String of mathematical equation (e.g., 'x + 1', '2*x**2')
        width: Number of x values (default 256 for pixel intensities)
        height: Maximum height of histogram (default 256)
        
    Returns:
        Array of y-values representing histogram heights
    """
    # 1. Create symbolic variable for equation parsing
    x = sp.Symbol('x')
    # Example: x becomes symbolic variable that can be used in equations
    
    # 2. Parse string equation into symbolic expression
    expr = parse_expr(equation_str)
    # Example: '2*x + 1' becomes mathematical expression 2x + 1
    
    # 3. Generate evenly spaced x values from 0 to 255
    x_vals = np.linspace(0, 255, width)
    # Example: array([0, 1, 2, ..., 254, 255])
    
    # 4. Convert symbolic expression to numpy function
    equation_func = sp.lambdify(x, expr, 'numpy')
    # Example: for '2*x + 1' creates function f(x) = 2x + 1
    
    # 5. Evaluate function for all x values
    y_vals = equation_func(x_vals)
    # Example for '2*x + 1':
    # array([1, 3, 5, ..., 509, 511])
    
 
    
    
    return y_vals

# Example usage with different equations:
"""
# Linear equation: y = 2x + 1
hist1 = create_histogram_from_equation('2*x + 1')
# Sample output: [1, 3, 5, ..., 509, 511] -> normalized to [0, 1, 2, ..., 254, 255]

# Quadratic equation: y = x²
hist2 = create_histogram_from_equation('x**2')
# Sample output: [0, 1, 4, ..., 64516, 65025] -> normalized to [0, 0, 0, ..., 254, 255]

# Sinusoidal equation: y = sin(x/10)*100 + 100
hist3 = create_histogram_from_equation('sin(x/10)*100 + 100')
# Sample output: [100, 110, 119, ..., 91, 82] -> normalized to [128, 140, 152, ..., 116, 105]
"""


def generate_image_from_histogram(hist_values, width=256, height=256):
    """
    Generate grayscale image from histogram values.
    Each column x is filled with intensity value x up to histogram height.
    
    Args:
        hist_values: Array of histogram values for each intensity level
        width: Width of output image (default 256 for intensity levels)
        height: Height of output image (default 256)
        
    Returns:
        grayscale image as numpy array (height x width)
    """
    # Create empty black image
    image = np.zeros((height, width), dtype=np.uint8)
    
    # Process each column (x position/intensity level)
    for x, count in enumerate(hist_values):
        if count > 0:
                # Convert count to integer for array slicing
            count = int(count)
            # Fill column with intensity value x up to count height
            # Start from bottom of image (height-count) up to height
            start_row = height - count
            image[start_row:height, x] = x
            
    return image

def display_histogram(hist_values, equation_str=None, y_max=None):
    """
    Display histogram of pixel intensity distribution.
    
    Args:
        hist_values: Array of histogram values
        equation_str: Equation string for title (optional)
        y_max: Maximum y-axis value (optional)
    """
    plt.figure(figsize=(8, 4))
    
    # Plot histogram bars
    plt.bar(np.arange(256), hist_values, width=1, color='gray', alpha=0.8)
    
    # Add title and labels
    title = "Pixel Intensity Distribution"
    if equation_str:
        title += f" for {equation_str}"
    plt.title(title)
    plt.xlabel("Pixel Intensity (x)")
    plt.ylabel("Frequency")
    
    # Add grid and set y-axis limit
    plt.grid(True, alpha=0.2)
    if y_max is not None:
        plt.ylim(0, y_max)
    
    plt.tight_layout()
    plt.show()

