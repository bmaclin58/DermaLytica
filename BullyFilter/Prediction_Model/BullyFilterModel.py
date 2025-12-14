import os

import requests

from settings import get_secret

BULLYFILTER_HF_ENDPOINT_URL = (
		get_secret("BULLYFILTER_HF_ENDPOINT_URL")
		or os.getenv("BULLYFILTER_HF_ENDPOINT_URL")
)


def BullyFilter_Prediction(payload):
	headers = {
			"Accept"      : "application/json",
			"Content-Type": "application/json",
			}
	text = {
			"inputs"    : payload,
			"parameters": {
					"max_new_tokens": 100,
					},
			}

	response = requests.post(
			url = BULLYFILTER_HF_ENDPOINT_URL,
			headers = headers,
			json = text,
			)
	return response.json()
