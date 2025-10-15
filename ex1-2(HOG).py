import cv2
import numpy as np

# --- Step 1: Preprocess the Data (Resize to 64 x 128) ---
# Load image (grayscale for HOG)
image = cv2.imread('image1.png', cv2.IMREAD_GRAYSCALE)  # Replace with your file path
if image is None:
    print("Error: Image not found.")
    exit()

# Resize to 64x128 (standard HOG window size)
image = cv2.resize(image, (64, 128))

# --- Step 2: Calculate Gradients (direction x and y) ---
# Use Sobel operator to find gradients
gx = cv2.Sobel(np.float32(image), cv2.CV_32F, 1, 0, ksize=1)
gy = cv2.Sobel(np.float32(image), cv2.CV_32F, 0, 1, ksize=1)

# --- Step 3: Calculate Magnitude and Orientation ---
magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

# --- Step 4: Calculate Histogram of Gradients in 8×8 cells ---
cell_size = (8, 8)
num_bins = 9  # 0-180 degrees divided into 9 bins

# Compute number of cells
cell_x = image.shape[1] // cell_size[1]
cell_y = image.shape[0] // cell_size[0]

# Create an array to store histogram for each cell
hist = np.zeros((cell_y, cell_x, num_bins))

# Compute histogram per cell
for i in range(cell_y):
    for j in range(cell_x):
        # Extract cell
        mag_cell = magnitude[i*8:(i+1)*8, j*8:(j+1)*8]
        ang_cell = angle[i*8:(i+1)*8, j*8:(j+1)*8]
        
        # Create histogram for the cell
        bin_indices = np.int32(ang_cell / 20) % num_bins  # Each bin covers 20 degrees
        for b in range(num_bins):
            hist[i, j, b] = np.sum(mag_cell[bin_indices == b])

# --- Step 5: Normalize gradients in 16×16 blocks ---
# Each block = 2x2 cells
block_size = (2, 2)
eps = 1e-5  # small value to avoid division by zero
hog_features = []

for i in range(cell_y - 1):
    for j in range(cell_x - 1):
        block = hist[i:i+2, j:j+2, :].ravel()
        norm = np.sqrt(np.sum(block**2) + eps**2)
        block = block / norm
        hog_features.append(block)

hog_features = np.concatenate(hog_features)

# --- Step 6: Features for the Complete Image ---
print("HOG feature vector length:", len(hog_features))
print("Feature vector (first 20 values):", hog_features[:20])

# Optional visualization (show original and gradient magnitude)
cv2.imshow('Original Image', image)
cv2.imshow('Gradient Magnitude', cv2.convertScaleAbs(magnitude))
cv2.waitKey(0)
cv2.destroyAllWindows()
