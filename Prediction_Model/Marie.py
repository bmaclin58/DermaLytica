from django.conf import settings
import os
import numpy as np
import cv2
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import tensorflow as tf
import json
from PIL import Image
import io
import base64

# Define constants
MODEL_PATH = os.path.join(settings.BASE_DIR, 'Prediction_Model', 'models', 'derma_model.h5')
AGE_MEAN = 57.70533017
AGE_STD = 14.11323567
IMAGE_SIZE = (224, 224)
OPTIMAL_THRESHOLD = 0.2494


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


def prepare_metadata(age, gender, location):
	"""
	Prepare the metadata array for model input

	Format: [Age, Gender_female, Gender_male, Location_Back, Location_Front Torso, Location_Head & Neck,
				 Location_Legs, Location_Mouth & Groin, Location_Palms & Soles, Location_Shoulders & Arms,
				 Location_Side Torso (Ribs)]
	"""
	# Standardize age
	age_standardized = (float(age) - AGE_MEAN) / AGE_STD

	# Initialize metadata array
	metadata = np.zeros(11, dtype = np.float32)
	metadata[0] = age_standardized

	# Set gender (one-hot encoding)
	if gender.lower() == 'female':
		metadata[1] = 1.0
	else:
		metadata[2] = 1.0

	# Set location (one-hot encoding)
	location_mapping = {
			'Back':              3,
			'Front Torso':       4,
			'Head & Neck':       5,
			'Legs':              6,
			'Mouth & Groin':     7,
			'Palms & Soles':     8,
			'Shoulders & Arms':  9,
			'Side Torso (Ribs)': 10
			}

	if location in location_mapping:
		metadata[location_mapping[location]] = 1.0

	return metadata


# Load the model
model = None
try:
	model = tf.keras.models.load_model(MODEL_PATH)
	print("Model loaded successfully")
except Exception as e:
	print(f"Error loading model: {e}")


@csrf_exempt
@require_http_methods(["POST"])
def predict_lesion(request):
	"""
	API endpoint to predict if a skin lesion is benign or malignant
	"""
	try:
		# Parse incoming JSON data
		data = json.loads(request.body)

		# Extract parameters
		image_base64 = data.get('image', '')
		age = data.get('age', 0)
		gender = data.get('gender', 'male')
		location = data.get('location', 'Back')

		# Validate inputs
		if not image_base64:
			return JsonResponse({'error': 'Image is required'}, status = 400)

		try:
			age = float(age)
			if age <= 0 or age > 120:
				return JsonResponse({'error': 'Invalid age'}, status = 400)
		except (ValueError, TypeError):
			return JsonResponse({'error': 'Age must be a number'}, status = 400)

		if gender.lower() not in ['male', 'female']:
			return JsonResponse({'error': 'Gender must be male or female'}, status = 400)

		valid_locations = [
				'Back', 'Front Torso', 'Head & Neck', 'Legs',
				'Mouth & Groin', 'Palms & Soles', 'Shoulders & Arms', 'Side Torso (Ribs)'
				]
		if location not in valid_locations:
			return JsonResponse({'error': f'Location must be one of: {", ".join(valid_locations)}'}, status = 400)

		# Decode the base64 image
		try:
			# Remove the base64 prefix if present
			if ',' in image_base64:
				image_base64 = image_base64.split(',')[1]

			image_bytes = base64.b64decode(image_base64)
			image = np.array(Image.open(io.BytesIO(image_bytes)))
		except Exception as e:
			return JsonResponse({'error': f'Invalid image format: {str(e)}'}, status = 400)

		# Preprocess image and create mask
		try:
			mask = create_mask_otsu(image)  # Create the mask and then resize
			preprocessed_image = preprocess_image(image)
			mask = preprocess_image(mask)  # Resize and normalize mask
		except Exception as e:
			return JsonResponse({'error': f'Error preprocessing image: {str(e)}'}, status = 500)

		# Prepare metadata
		metadata = prepare_metadata(age, gender, location)

		# Make prediction
		if model is None:
			return JsonResponse({'error': 'Model not loaded'}, status = 500)

		try:
			# Prepare inputs in the format expected by the model
			image_input = np.expand_dims(preprocessed_image, axis = 0)
			mask_input = np.expand_dims(mask, axis = 0)
			metadata_input = np.expand_dims(metadata, axis = 0)

			# Make prediction
			prediction = model.predict(
					{
							'image_input':    image_input,
							'mask_input':     mask_input,
							'metadata_input': metadata_input
							}
					)

			# Extract prediction value
			prediction_value = float(prediction[0][0])

			# Apply optimal threshold. More accurate and leans towards avoiding false negatives
			is_malignant = prediction_value >= OPTIMAL_THRESHOLD

			# Prepare response
			response = {
					'prediction':     int(is_malignant),  # 0 for benign, 1 for malignant
					'probability':    prediction_value,
					'threshold_used': OPTIMAL_THRESHOLD,
					'classification': 'Malignant' if is_malignant else 'Benign',
					'confidence':     prediction_value if is_malignant else 1 - prediction_value
					}

			return JsonResponse(response)

		except Exception as e:
			return JsonResponse({'error': f'Error making prediction: {str(e)}'}, status = 500)

	except Exception as e:
		return JsonResponse({'error': f'Unexpected error: {str(e)}'}, status = 500)
