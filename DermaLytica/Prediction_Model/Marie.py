import base64
import io
import os

import numpy as np
import tensorflow as tf
from PIL import Image
from django.conf import settings
from django.http import JsonResponse

from DermaLytica.Prediction_Model.UtilityFunctions.ImageProcessing import create_mask_otsu, preprocess_image
from DermaLytica.Prediction_Model.UtilityFunctions.PrepMetadata import prepare_metadata

# Define constants
MODEL_PATH = os.path.join(settings.BASE_DIR, 'Prediction_Model', 'models', 'derma_model.h5')
AGE_MEAN = 57.70533017
AGE_STD = 14.11323567
IMAGE_SIZE = (224, 224)
OPTIMAL_THRESHOLD = 0.2494

# Load the model
model = None
try:
	model = tf.keras.models.load_model(MODEL_PATH)
	print("Model loaded successfully")
except Exception as e:
	print(f"Error loading model: {e}")

def predict_lesion(image, age, gender, location) -> JsonResponse:
	"""
	API endpoint to predict if a skin lesion is benign or malignant
	"""
	try:
		# Decode the base64 image
		try:
			# Remove the base64 prefix if present
			if ',' in image:
				image = image.split(',')[1]

			image_bytes = base64.b64decode(image)
			image = np.array(Image.open(io.BytesIO(image_bytes)))

		except Exception as e:
			return JsonResponse({'error': f'Invalid image format: {str(e)}'}, status=400)

		# Preprocess image and create mask
		try:
			mask = create_mask_otsu(image)  # Create the mask and then resize
			preprocessed_image = preprocess_image(image)
			mask = preprocess_image(mask)  # Resize and normalize mask

		except Exception as e:
			return JsonResponse({'error': f'Error preprocessing image: {str(e)}'}, status=500)

		# Prepare metadata
		metadata = prepare_metadata(age, gender, location)

		# Make prediction
		if model is None:
			return JsonResponse({'error': 'Model not loaded'}, status=500)

		try:
			# Prepare inputs in the format expected by the model
			image_input = np.expand_dims(preprocessed_image, axis=0)
			mask_input = np.expand_dims(mask, axis=0)
			metadata_input = np.expand_dims(metadata, axis=0)

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
			return JsonResponse({'error': f'Error making prediction: {str(e)}'}, status=500)

	except Exception as e:
		return JsonResponse({'error': f'Unexpected error: {str(e)}'}, status=500)
