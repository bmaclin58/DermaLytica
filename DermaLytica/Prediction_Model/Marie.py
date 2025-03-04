import base64
import io

import numpy as np
import tensorflow as tf
from PIL import Image
from django.http import JsonResponse
from pyjsparser.parser import false

from DermaLytica.GPS import dermatologistLookup
from DermaLytica.Prediction_Model.GlobalVariables import MODEL_PATH, OPTIMAL_THRESHOLD
from DermaLytica.Prediction_Model.UtilityFunctions.ImageProcessing import create_mask_otsu, preprocess_image
from DermaLytica.Prediction_Model.UtilityFunctions.PrepMetadata import prepare_metadata

model = None

def get_model():
	global model
	if model is None:
		try:
			model = tf.keras.models.load_model(
					MODEL_PATH,
					compile=false)
			print("Model loaded successfully")

		except Exception as e:
			print(f"Error loading model: {e}")
	return model


def predict_lesion(image, age, gender, location, zipCode):
	"""
	API endpoint to predict if a skin lesion is benign or malignant
	"""
	model = get_model()
	if model is None:
		return JsonResponse({'error': 'Model not loaded'}, status=500)

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

			dermatology_Lists = None

			if is_malignant and zipCode:
				dermatology_Lists = dermatologistLookup(zipCode)

			# Prepare response
			response = {
					'prediction':        int(is_malignant),  # 0 for benign, 1 for malignant
					'probability':       prediction_value,
					'threshold_used':    OPTIMAL_THRESHOLD,
					'classification':    'Malignant' if is_malignant else 'Benign',
					'confidence':        prediction_value if is_malignant else 1 - prediction_value,
					'dermatology_Lists': dermatology_Lists
			}

			return response

		except Exception as e:
			return JsonResponse({'error': f'Error making prediction: {str(e)}'}, status=500)

	except Exception as e:
		return JsonResponse({'error': f'Unexpected error: {str(e)}'}, status=500)
