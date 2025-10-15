# Experiment 1: Canny Edge Detection
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('sanji2.png', cv2.IMREAD_GRAYSCALE)
blur = cv2.GaussianBlur(img, (5,5), 1.4)
edges = cv2.Canny(blur, 100, 200)

plt.figure(figsize=(8,6))
plt.imshow(edges, cmap='gray')
plt.title('Canny Edge Detection')
plt.axis('off')
plt.show()

