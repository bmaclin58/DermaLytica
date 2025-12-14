import os

import requests
from django.conf import settings
from django.core.exceptions import ImproperlyConfigured


def _get_endpoint_url() -> str | None:
	# Prefer env var first (simple + reliable in containers)
	url = os.getenv("BULLYFILTER_HF_ENDPOINT_URL")
	if url:
		return url

	# Fall back to your settings.get_secret if it exists
	get_secret = getattr(settings, "get_secret", None)
	if callable(get_secret):
		return get_secret("BULLYFILTER_HF_ENDPOINT_URL")

	return None


BULLYFILTER_HF_ENDPOINT_URL = _get_endpoint_url()


def bullyfilter_request(payload: str, *, max_new_tokens: int = 100, timeout: int = 45) -> requests.Response:
	if not BULLYFILTER_HF_ENDPOINT_URL:
		raise ImproperlyConfigured(
				"BULLYFILTER_HF_ENDPOINT_URL is not set (env var or Secret Manager).",
				)

	headers = {
			"Accept"      : "application/json",
			"Content-Type": "application/json",
			}
	body = {
			"inputs"    : payload,
			"parameters": { "max_new_tokens": max_new_tokens },
			}

	resp = requests.post(
			url = BULLYFILTER_HF_ENDPOINT_URL,
			headers = headers,
			json = body,
			timeout = timeout,
			)
	# Let callers decide how to handle non-2xx
	return resp


def BullyFilter_Prediction(payload: str, *, max_new_tokens: int = 100) -> object:
	resp = bullyfilter_request(payload, max_new_tokens = max_new_tokens)
	resp.raise_for_status()
	return resp.json()
