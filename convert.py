import numpy as np
import cv2

pgm_file = 'my_map.pgm'
image = cv2.imread(pgm_file, cv2.IMREAD_GRAYSCALE)


# center crop image
height, width = image.shape
new_height = height - 1600
new_width = width - 1600
start_row = int((height - new_height) / 2)
start_col = int((width - new_width) / 2)
end_row = start_row + new_height
end_col = start_col + new_width
image = image[start_row:end_row, start_col:end_col]
cv2.imwrite('my_map.png', image)

_, white = cv2.threshold(image, 220, 255, cv2.THRESH_BINARY)
_, dark = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)
dark = 255 - dark

inner_region = cv2.bitwise_or(white, dark)

# # perform open operation
kernel = np.ones((5,5),np.uint8)
new_img = cv2.morphologyEx(inner_region, cv2.MORPH_CLOSE, kernel)
new_img = cv2.morphologyEx(new_img, cv2.MORPH_OPEN, kernel)

# save the binary image
cv2.imwrite('new_image.png', new_img)