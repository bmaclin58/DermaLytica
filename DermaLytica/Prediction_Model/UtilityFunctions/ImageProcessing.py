import cv2
import numpy as np

from DermaLytica.Prediction_Model.GlobalVariables import IMAGE_SIZE


def create_mask_otsu(image):
	"""
	Create an enhanced binary mask using an improved preprocessing pipeline:
	1. Convert to grayscale.
	2. Enhance contrast using CLAHE.
	3. Denoise with a bilateral filter.
	4. Sharpen using an unsharp mask filter.
	5. Smooth with a Gaussian blur.
	6. Apply Otsu's thresholding.
	7. Clean up with morphological operations.
	"""
	# Convert image to grayscale
	gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

	# Enhance local contrast using CLAHE
	clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8, 8))
	enhanced = clahe.apply(gray)

	# Use a bilateral filter to reduce noise while preserving edges
	denoised = cv2.bilateralFilter(enhanced, d = 9, sigmaColor = 75, sigmaSpace = 75)

	# Sharpen the image using an unsharp masking kernel
	sharpening_kernel = np.array(
			[[-1, -1, -1],
			 [-1, 9, -1],
			 [-1, -1, -1]]
			)
	sharpened = cv2.filter2D(denoised, -1, sharpening_kernel)

	# Apply Gaussian Blur to reduce any high-frequency artifacts
	blurred = cv2.GaussianBlur(sharpened, (5, 5), 0)

	# Apply Otsu's thresholding to create the binary mask
	_, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

	# Use morphological opening to remove small noise artifacts from the mask
	kernel_morph = np.ones((3, 3), np.uint8)
	mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_morph, iterations = 1)

	# Convert to a 3-channel image for the model
	mask_3channel = cv2.cvtColor(mask_clean, cv2.COLOR_GRAY2RGB)
	#print(f'Mask shape')

	return mask_3channel


def preprocess_image(image_data):
	"""
	Preprocess an image for model input
	"""
	# Convert to RGB if needed
	
	if len(image_data.shape) == 2:  # If grayscale
		image_data = cv2.cvtColor(image_data, cv2.COLOR_GRAY2RGB)
	elif image_data.shape[2] == 4:  # If RGBA
		image_data = cv2.cvtColor(image_data, cv2.COLOR_RGBA2RGB)

	# Resize to model input size
	image_data = cv2.resize(image_data, IMAGE_SIZE)

	# Normalize pixel values
	image_data = image_data.astype(np.float32) / 255.0

	return image_data
