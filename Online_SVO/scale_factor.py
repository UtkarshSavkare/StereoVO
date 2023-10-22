import numpy as np

# Known dimensions of the checkerboard pattern (in millimeters)
checkerboard_width_mm = 180
checkerboard_height_mm = 140

# Camera calibration parameters
focal_length_mm = 1.930 # Example focal length in pixels
image_width_mm = 0.003 # Example image width in pixels

# Calculate the scale factor
scale_factor = checkerboard_width_mm / (2 * focal_length_mm / image_width_mm)
print(scale_factor)