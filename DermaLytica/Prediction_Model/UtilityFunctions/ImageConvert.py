from PIL import Image
import os
from io import BytesIO
from django.core.files.base import ContentFile


def convert_to_jpg(image_field):
	"""
	Convert any image to JPG format

	Args:
		image_field: A Django ImageField containing the uploaded image

	Returns:
		A ContentFile with the image converted to JPG
	"""
	# Open the uploaded image
	img = Image.open(image_field)

	# Convert to RGB mode (removing alpha channel if present)
	if img.mode != 'RGB':
		img = img.convert('RGB')

	# Create a BytesIO object to store the converted image
	output = BytesIO()

	# Save the image as JPEG to the BytesIO object
	img.save(output, format='JPEG', quality=95)
	output.seek(0)

	# Generate a new filename with .jpg extension
	original_name = os.path.splitext(image_field.name)[0]
	new_name = f"{original_name}.jpg"

	# Return a ContentFile that can be saved to a model field
	return ContentFile(output.getvalue(), name=new_name)
