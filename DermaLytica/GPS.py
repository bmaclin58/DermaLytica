import json

import requests

from settings import get_secret


def dermatologistLookup(zipCode) -> list:
	"""
	   Finds dermatologists near a given zip code using Google Places API Text Search.

	   Args:
		   zipCode: The zip code (string or integer) to search within.

	   Returns:
		   A list of dictionaries, where each dictionary contains information
		   about a dermatologist found (name, address, rating). Returns an empty
		   list if no results are found or an error occurs.
	   """
	MAPS_API_KEY = get_secret("MAPS_API_KEY")

	if not MAPS_API_KEY:
		print("API Key not configured. Cannot perform search.")
		return []

	# New Places API Text Search endpoint (v1)
	endpoint_url = "https://places.googleapis.com/v1/places:searchText"

	# Headers required for the new API
	headers = {
			'Content-Type':     'application/json',
			'X-Goog-Api-Key':   MAPS_API_KEY,
			'X-Goog-FieldMask': 'places.displayName,places.formattedAddress,places.rating,places.userRatingCount,places.nationalPhoneNumber,places.websiteUri'
	}

	# Request body
	data = {
			'textQuery':    f'Dermatologists in {zipCode}',
			'includedType': 'doctor'
	}
	# --- Make API Request ---
	results = []
	try:
		response = requests.post(endpoint_url, json=data, headers=headers, timeout=25)
		response.raise_for_status()  # Raise an exception for bad status codes

		if response.status_code != 200:
			try:
				error_content = response.json()
				print(f"Error response body: {json.dumps(error_content, indent=2)}")
			except:
				print(f"Raw error response: {response.text[:500]}...")
		if response.status_code == 403:
			error_message = "Access denied (403 Forbidden). Common causes include:"
			error_details = [
					"API key restrictions (IP, referrer, or app restrictions)",
					"API not enabled in Google Cloud Console",
					"Billing not enabled for your Google Cloud project",
					"Places API quota exceeded",
					"Invalid authentication scope or credentials"
			]
			print(error_message)
			for detail in error_details:
				print(f"- {detail}")

			# Try to extract error details from response
			try:
				error_data = response.json()
				if 'error' in error_data:
					print(
							f"Google API error details: {error_data['error'].get('message', 'No details provided')}")
					print(f"Error code: {error_data['error'].get('code', 'Unknown')}")
					if 'details' in error_data['error']:
						for detail in error_data['error']['details']:
							print(f"Detail: {detail}")
			except:
				pass

			return []

		data = response.json()

		# Check if we have places in the response
		if 'places' in data:
			# Extract information for each place found
			for place in data.get('places', []):
				formattedRatings = f" {place.get('rating', 0.0):.1f} ({place.get('userRatingCount', 0)} reviews)"
				business_info = {
						'name':            place.get('displayName', {}).get('text', 'N/A'),
						'display_address': place.get('formattedAddress', 'N/A'),
						'rating':          formattedRatings,
						'display_phone':   place.get('nationalPhoneNumber', 'N/A'),
						'url':             place.get('websiteUri', '#'),
				}
				results.append(business_info)
		else:
			print(f"No dermatologists found for zip code {zipCode} or API returned unexpected format.")

		return results


	except requests.exceptions.RequestException as e:

		print(f"Network error during Google Places API request: {e}")

		# Enhanced request exception handling
		if isinstance(e, requests.exceptions.SSLError):
			print("SSL Error: This could indicate certificate issues or network security configuration problems")

		elif isinstance(e, requests.exceptions.ConnectionError):
			print("Connection Error: Check network connectivity or possible outage of the Google API service")

		elif isinstance(e, requests.exceptions.Timeout):
			print("Timeout Error: The request took too long to complete")

		return []

	except json.JSONDecodeError as json_err:
		print(f"Error decoding JSON response from Google Places API: {json_err}")
		print(f"Response content that couldn't be parsed: {response.text[:200]}...")

		return []

	except Exception as e:
		# Catch any other unexpected errors during processing
		print(f"An unexpected error occurred: {e}")

		import traceback
		print(f"Traceback: {traceback.format_exc()}")

		return []
