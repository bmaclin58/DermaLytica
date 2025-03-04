import requests
from django.conf import settings


def dermatologistLookup (zipCode):

	url = f"https://api.yelp.com/v3/businesses/search?location={zipCode}&term=dermatologist&radius=20000&sort_by=best_match&limit=20"

	headers = {
	    "accept": "application/json",
	    "authorization": settings.YELP_API_KEY
	}

	response = requests.get(url, headers=headers)

	return response.text
