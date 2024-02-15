import cv2
import numpy as np

from PIL import Image

def process_image(image_path):
    # Load the image in grayscale
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Threshold the image to obtain a binary image
    _, binary_image = cv2.threshold(original_image, 128, 255, cv2.THRESH_BINARY)

    # Apply morphological operations to remove noise and small objects
    kernel = np.ones((5,5), np.uint8)
    cleaned_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=1)

    # Use a larger kernel to close gaps
    large_kernel = np.ones((10,10), np.uint8)
    cleaned_image = cv2.morphologyEx(cleaned_image, cv2.MORPH_CLOSE, large_kernel, iterations=1)

    # Invert image if necessary so that the background is white and the lines are black
    if np.mean(original_image) < 128:
        cleaned_image = cv2.bitwise_not(cleaned_image)

    # Thinning to get single pixel width lines
    # This requires a specific function not available in OpenCV, can be done with skimage or custom implementation

    return cleaned_image

# Process the image
image_path = 'map.png'  # The provided image path
processed_image = process_image(image_path)

# Save the processed image
output_path = 'processed_map.png'
cv2.imwrite(output_path, processed_image)

edges = cv2.Canny(processed_image, 100, 200)
cv2.imwrite('edges.png', edges)


image = cv2.imread('processed_map.png', cv2.IMREAD_GRAYSCALE)
contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
import IPython
IPython.embed()
mask=np.zeros_like(inverted_edges)
cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)
inverted_mask = cv2.bitwise_not(mask)

cv2.imwrite('out.png',inverted_mask)