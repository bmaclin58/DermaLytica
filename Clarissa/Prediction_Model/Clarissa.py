import os

import requests

from settings import get_secret

CLARISSA_HF_ENDPOINT_URL = (
		get_secret("CLARISSA_HF_ENDPOINT_URL")
		or os.getenv("CLARISSA_HF_ENDPOINT_URL")
)


def clarissa_Prediction(image):
	if hasattr(image, "read"):  # Django ImageFieldFile
		image_bytes = image.read()
	elif isinstance(image, (bytes, bytearray)):
		image_bytes = image
	else:
		with open(image, "rb") as f:
			image_bytes = f.read()

	headers = {
			"Accept": "application/json",
			"Content-Type": "image/png"
			}

	response = requests.post(
			CLARISSA_HF_ENDPOINT_URL,
			headers = headers,
			data = image_bytes,
			timeout = 25,
			)

	return response
