import numpy as np
import cv2

pgm_file = 'rtabmap_ece165_new.pgm'
image = cv2.imread(pgm_file, cv2.IMREAD_GRAYSCALE)

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
