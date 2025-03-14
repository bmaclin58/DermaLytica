import base64
import io

import numpy as np
from PIL import Image

from DermaLytica.GPS import dermatologistLookup
from DermaLytica.Prediction_Model.GlobalVariables import OPTIMAL_THRESHOLD, downloadModel
from DermaLytica.Prediction_Model.UtilityFunctions.ImageProcessing import create_mask_otsu, preprocess_image
from DermaLytica.Prediction_Model.UtilityFunctions.PrepMetadata import prepare_metadata

# Global variable for model instance
_model = None

def get_model():
	"""Lazy-load the model only when needed"""
	global _model

	if _model is None:
		try:
			MODEL_PATH = downloadModel()
			import tensorflow as tf
			tf.config.set_visible_devices([], 'GPU')
			_model = tf.lite.Interpreter(model_path = MODEL_PATH)
			_model.allocate_tensors()
			print("TFLite Model loaded successfully")
		except Exception as e:
			print(f"Error loading TFLite model: {e}")
	return _model


def get_io_details(model):
	print("Getting input/output details")
	input_details = model.get_input_details()
	print(input_details)
	output_details = model.get_output_details()
	print(output_details)
	return input_details, output_details


def predict_lesion(image, age, gender, location, zipCode):
	"""
	API endpoint to predict if a skin lesion is benign or malignant
	"""
	model = get_model()
	if model is None:
		print('Model not loaded')
		return {"classification": "Error", "confidence": 0, "dermatology_Lists": []}

	try:
		# Handle different image input types
		try:
			# If image is a Django ImageFieldFile
			if hasattr(image, 'read'):
				image_data = image.read()
				image = np.array(Image.open(io.BytesIO(image_data)))
			# If image is a base64 string
			elif isinstance(image, str):
				if ',' in image:
					image = image.split(',')[1]
				image_bytes = base64.b64decode(image)
				image = np.array(Image.open(io.BytesIO(image_bytes)))
			# If image is already a numpy array
			elif isinstance(image, np.ndarray):
				pass
			else:
				raise ValueError(f"Unsupported image type: {type(image)}")

		except Exception as e:
			print(f'Invalid image format: {str(e)}')
			return {"classification": "Error", "confidence": 0, "dermatology_Lists": []}

		# Preprocess image and create mask
		try:
			print('Preprocessing image and creating mask')
			mask = create_mask_otsu(image)
			print("Mask shape")
			preprocessed_image = preprocess_image(image)
			print(f"Image shape: {preprocessed_image.shape}")
			mask = preprocess_image(mask)
			print(f"Mask shape: {mask.shape}")

		except Exception as e:
			print(f'Error preprocessing image: {str(e)}')
			return {"classification": "Error", "confidence": 0, "dermatology_Lists": []}

		# Prepare metadata
		metadata = prepare_metadata(age, gender, location)

		# Get input/output details from the model
		input_details, output_details = get_io_details(model)

		# Prepare inputs in expected format
		image_input = np.expand_dims(preprocessed_image, axis = 0).astype(np.float32)
		mask_input = np.expand_dims(mask, axis = 0).astype(np.float32)
		metadata_input = np.expand_dims(metadata, axis = 0).astype(np.float32)

		# Set the model's inputs
		model.set_tensor(input_details[0]['index'], mask_input)  # mask input
		model.set_tensor(input_details[1]['index'], metadata_input)  # metadata input
		model.set_tensor(input_details[2]['index'], image_input)  # image input

		print('Inputs set, Invoke model')
		# Run inference
		model.invoke()
		print('Model invoked, Get output')

		# Get the output
		prediction = model.get_tensor(output_details[0]['index'])[0][0]

		# Apply thresholding
		is_malignant = prediction >= OPTIMAL_THRESHOLD
		print(f'Prediction: {prediction}, Threshold: {OPTIMAL_THRESHOLD}, Malignant: {is_malignant}')
		dermatology_Lists = None
		print(f'Zip Code: {zipCode}')
		if is_malignant and zipCode:
			dermatology_Lists = dermatologistLookup(zipCode)

		# Prepare response
		response = {
				'prediction':        int(is_malignant),
				'probability':       float(prediction),
				'threshold_used':    OPTIMAL_THRESHOLD,
				'classification':    'Malignant' if is_malignant else 'Benign',
				'confidence':        float(prediction if is_malignant else 1 - prediction) * 100,
				'dermatology_Lists': dermatology_Lists
				}
		for key, value in response.items():
			print(f'{key}: {value}')

		return response

	except Exception as e:
		print(f'Error making prediction: {str(e)}')
		return {"classification": "Error", "confidence": 0, "dermatology_Lists": []}
