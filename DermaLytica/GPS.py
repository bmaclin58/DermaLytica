import requests
import json
from django.conf import settings


def dermatologistLookup(zipCode) -> list:
    url = f"https://api.yelp.com/v3/businesses/search?location={zipCode}&radius=20000&categories=Dermatologists&categories=dermatology&sort_by=best_match&limit=10"

    headers = {
        "accept": "application/json",
        "authorization": settings.YELP_API_KEY
        }

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
'''
# Example usage
try:
    dermatologists = dermatologistLookup("23325")
    for i, dermatologist in enumerate(dermatologists, 1):
        print(f"Dermatologist {i}:")
        print(f"Name: {dermatologist['name']}")
        print(f"URL: {dermatologist['url']}")
        print(f"Rating: {dermatologist['rating']}")
        print(f"Address: {dermatologist['display_address']}")
        print(f"Phone: {dermatologist['display_phone']}")
        print(f"Distance: {dermatologist['distance']} miles")
        print("-" * 50)
except Exception as e:
    print(f"Error running the example: {str(e)}")
'''
