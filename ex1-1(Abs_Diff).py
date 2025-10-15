import cv2
import numpy as np

# --- Step 1: Load the two input images ---
# Make sure both images have the same size and number of channels
M1 = cv2.imread("image1.png")  # Replace with your actual image path
M2 = cv2.imread("image2.png")  # Replace with your actual image path

# Check if images loaded successfully
if M1 is None or M2 is None:
    print("Error: One or both images could not be loaded. Check file paths.")
    exit()

# --- Step 2: Resize images to match dimensions (optional safeguard) ---
if M1.shape != M2.shape:
    print("Warning: Images have different sizes — resizing second image to match the first.")
    M2 = cv2.resize(M2, (M1.shape[1], M1.shape[0]))

# --- Step 3: Compute the pixel-wise absolute difference ---
# Formula: Out(x, y) = abs(M1(x, y) – M2(x, y))
Out = cv2.absdiff(M1, M2)

# --- Step 4: Display the output image ---
cv2.imshow("Input Image 1", M1)
cv2.imshow("Input Image 2", M2)
cv2.imshow("Absolute Difference", Out)

# Wait until a key is pressed, then close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()

# --- Optional: Save the output image to disk ---
cv2.imwrite("output_difference.jpg", Out)
