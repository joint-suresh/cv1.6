import cv2
import matplotlib.pyplot as plt
import numpy as np
IMG_PATH = 'image1.png'
image = cv2.imread(IMG_PATH)
scaled = cv2.resize(image, (0, 0), fx=0.5, fy=0.5) 
rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
h, w = image.shape[:2]
M_shear = np.array([
    [1, 0.3, 0],
    [0, 1, 0]
], dtype = np.float32)
sheared = cv2.warpAffine(image, M_shear, (int(w + 0.3*h), h))
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
scaled_rgb = cv2.cvtColor(scaled, cv2.COLOR_BGR2RGB)
rotated_rgb = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)
sheared_rgb = cv2.cvtColor(sheared, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(20,5))

plt.subplot(1, 4, 1)
plt.title("Original Image")
plt.imshow(image_rgb)
plt.axis('off')

plt.subplot(1, 4, 2)
plt.title("Scaled")
plt.imshow(scaled_rgb)
plt.axis('off')

plt.subplot(1, 4, 3)
plt.title("Rotated")
plt.imshow(rotated_rgb)
plt.axis('off')

plt.subplot(1, 4, 4)
plt.title("Sheared")
plt.imshow(sheared_rgb)
plt.axis('off')

plt.show()