import os

from django.conf import settings

import os
import requests

def downloadModel ():
	MODEL_URL = "https://storage.googleapis.com/dermalyticsdrive/models/KERAS_model.tflite"
	MODEL_PATH = os.path.join(settings.BASE_DIR, 'DermaLytica', 'Prediction_Model', 'AI_Models', 'KERAS_model.tflite')

	# Ensure the directory exists
	os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

	# Download the model if not already present
	if not os.path.exists(MODEL_PATH):
	    print(f"Downloading model from {MODEL_URL}...")
	    response = requests.get(MODEL_URL)
	    with open(MODEL_PATH, "wb") as f:
	        f.write(response.content)
	    print("Download complete.")
	return MODEL_PATH

AGE_MEAN = 57.70533017
AGE_STD = 14.11323567
IMAGE_SIZE = (224, 224)
OPTIMAL_THRESHOLD = 0.2494
