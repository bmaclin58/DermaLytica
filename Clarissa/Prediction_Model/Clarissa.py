import os

import requests

from settings import get_secret

CLARISSA_HF_ENDPOINT_URL = (
		get_secret("CLARISSA_HF_ENDPOINT_URL")
		or os.getenv("CLARISSA_HF_ENDPOINT_URL")
)

def clarissa_Prediction(image):
	"""
	image: file path (str) OR bytes
	"""

	if isinstance(image, (str, os.PathLike)):
		with open(image, "rb") as f:
			image_bytes = f.read()
	else:
		# assume bytes-like
		image_bytes = image

	headers = {
			"Accept"      : "image/png",
			"Content-Type": "application/json",
			}

	response = requests.post(
			url = CLARISSA_HF_ENDPOINT_URL,
			headers = headers,
			data = image_bytes,
			timeout = 25,  # IMPORTANT on Cloud Run
			)

	return response
