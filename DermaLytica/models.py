from django.db import models

IMAGE_LOCATION = {
		'Back':              'Back',
		'Front Torso':       'Front Torso',
		'Head & Neck':       'Head & Neck',
		'Legs':              'Legs',
		'Mouth & Groin':     'Mouth & Groin',
		'Palms & Soles':     'Palms & Soles',
		'Shoulders & Arms':  'Shoulders & Arms',
		'Side Torso (Ribs)': 'Side Torso (Ribs)'
		}

class userData (models.Model):
	image = models.ImageField()
	age = models.IntegerField()
	gender = models.CharField(
			choices=[('Male','Male'),('Female','Female')],
			max_length=6
	)
	location = models.CharField(choices=IMAGE_LOCATION, max_length=20)
	zipCode = models.CharField(
			max_length=5,
			blank=True,)

	class Meta:
		app_label = 'DermaLytica'
