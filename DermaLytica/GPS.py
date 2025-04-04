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
						'url':             place.get('websiteUri', 'N/A'),
				}
				results.append(business_info)
		else:
			print(f"No dermatologists found for zip code {zipCode} or API returned unexpected format.")

		return results

	except requests.exceptions.RequestException as e:
		print(f"Network error during Google Places API request: {e}")
		return []
	except json.JSONDecodeError:
		print("Error decoding JSON response from Google Places API")
		return []
	except Exception as e:
		# Catch any other unexpected errors during processing
		print(f"An unexpected error occurred: {e}")
		return []
