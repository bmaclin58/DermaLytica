import base64
import io

import numpy as np
import tensorflow as tf
from PIL import Image
from django.http import JsonResponse

from DermaLytica.GPS import dermatologistLookup
from DermaLytica.Prediction_Model.GlobalVariables import MODEL_PATH, OPTIMAL_THRESHOLD
from DermaLytica.Prediction_Model.UtilityFunctions.ImageProcessing import create_mask_otsu, preprocess_image
from DermaLytica.Prediction_Model.UtilityFunctions.PrepMetadata import prepare_metadata

# Global interpreter
model = None


def get_model():
	global model
	if model is None:
		try:
			model = tf.lite.Interpreter(MODEL_PATH)
			model.allocate_tensors()
			print("TFLite Model loaded successfully")
		except Exception as e:
			print(f"Error loading TFLite model: {e}")
	return model


def get_io_details(model):
	input_details = model.get_input_details()
	output_details = model.get_output_details()
	return input_details, output_details


def predict_lesion(image, age, gender, location, zipCode):
	"""
	API endpoint to predict if a skin lesion is benign or malignant
	"""
	model = get_model()
	if model is None:
		return JsonResponse({'error': 'Model not loaded'}, status=500)

	try:
		# Decode and process the image
		try:
			if ',' in image:
				image = image.split(',')[1]

			image_bytes = base64.b64decode(image)
			image = np.array(Image.open(io.BytesIO(image_bytes)))

		except Exception as e:
			return JsonResponse({'error': f'Invalid image format: {str(e)}'}, status=400)

		# Preprocess image and create mask
		try:
			mask = create_mask_otsu(image)
			preprocessed_image = preprocess_image(image)
			mask = preprocess_image(mask)

		except Exception as e:
			return JsonResponse({'error': f'Error preprocessing image: {str(e)}'}, status=500)

		# Prepare metadata
		metadata = prepare_metadata(age, gender, location)

		# Get input/output details from the model
		input_details, output_details = get_io_details(model)

		# Prepare inputs in expected format
		image_input = np.expand_dims(preprocessed_image, axis=0).astype(np.float32)
		mask_input = np.expand_dims(mask, axis=0).astype(np.float32)
		metadata_input = np.expand_dims(metadata, axis=0).astype(np.float32)

		# Set the model's inputs
		model.set_tensor(input_details[0]['index'], image_input)
		model.set_tensor(input_details[1]['index'], mask_input)
		model.set_tensor(input_details[2]['index'], metadata_input)

		# Run inference
		model.invoke()

		# Get the output
		prediction = model.get_tensor(output_details[0]['index'])[0][0]

		# Apply thresholding
		is_malignant = prediction >= OPTIMAL_THRESHOLD
		dermatology_Lists = None

		if is_malignant and zipCode:
			dermatology_Lists = dermatologistLookup(zipCode)

		# Prepare response
		response = {
				'prediction':        int(is_malignant),
				'probability':       float(prediction),
				'threshold_used':    OPTIMAL_THRESHOLD,
				'classification':    'Malignant' if is_malignant else 'Benign',
				'confidence':        prediction if is_malignant else 1 - prediction,
				'dermatology_Lists': dermatology_Lists
		}
		for key, value in response.items():
			print(f'{key}: {value}')

		return response

	except Exception as e:
		return JsonResponse({'error': f'Error making prediction: {str(e)}'}, status=500)
