# Exercise 1: Image Processing - Contrast Stretching and Linear Filtering
import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- Step 1: Load the image ---
img = cv2.imread('image1.png', cv2.IMREAD_GRAYSCALE)  # Replace with your image file
if img is None:
    print("Error: Image not found.")
    exit()

# --- Step 2: Contrast Stretching ---
# Formula: new_pixel = (pixel - min) * (255 / (max - min))
min_val = np.min(img)
max_val = np.max(img)
contrast_stretched = ((img - min_val) * (255.0 / (max_val - min_val))).astype(np.uint8)

# --- Step 3: Linear Filtering (Smoothing) ---
# Using a simple 5x5 averaging filter
kernel = np.ones((5, 5), np.float32) / 25
filtered = cv2.filter2D(contrast_stretched, -1, kernel)

# --- Step 4: Plot results ---
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.title('Original Image')
plt.imshow(img, cmap='gray')

plt.subplot(2, 3, 2)
plt.title('Contrast Stretched')
plt.imshow(contrast_stretched, cmap='gray')

plt.subplot(2, 3, 3)
plt.title('Linear Filtered')
plt.imshow(filtered, cmap='gray')

plt.subplot(2, 3, 4)
plt.title('Original Histogram')
plt.hist(img.ravel(), 256, [0, 256])

plt.subplot(2, 3, 5)
plt.title('Contrast Stretched Histogram')
plt.hist(contrast_stretched.ravel(), 256, [0, 256])

plt.subplot(2, 3, 6)
plt.title('Filtered Histogram')
plt.hist(filtered.ravel(), 256, [0, 256])

plt.tight_layout()
plt.show()
