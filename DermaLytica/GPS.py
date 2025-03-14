import requests
import json
from django.conf import settings

from settings import YELP_API_KEY, get_secret


def dermatologistLookup(zipCode) -> list:
    url = f"https://api.yelp.com/v3/businesses/search?location={zipCode}&radius=20000&categories=Dermatologists&categories=dermatology&sort_by=best_match&limit=8"
    print(url)

    YELP_API_KEY = get_secret("YELP_API_KEY")

    headers = {
        "accept": "application/json",
        "authorization": f"Bearer {YELP_API_KEY}"
        }
    print(settings.YELP_API_KEY)
    print(headers["authorization"])

    response = requests.get(url, headers=headers)

    try:
        # Get the text content of the response and then parse it
        data = response.json()

        # List to store info
        results = []

        if 'businesses' not in data:
            print("No businesses found in the response")
            return results

        for business in data['businesses']:

            business_info = {
                'name': business.get('name', ''),
                'url': business.get('url', ''),
                'rating': business.get('rating', 0),
                'display_address': ', '.join(business.get('location', {}).get('display_address', [])),
                'display_phone': business.get('display_phone', ''),
                'distance': round(business.get('distance', 0) / 1609.34, 2)  # Convert meters to miles
            }

            results.append(business_info)

        return results

    except json.JSONDecodeError:
        print("Error decoding JSON response")
        return []
    except Exception as e:
        print(f"Error extracting data: {str(e)}")
        return []
