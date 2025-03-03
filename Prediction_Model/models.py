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
	image = models.ImageField(upload_to='images/')
	age = models.IntegerField()
	gender = models.CharField(choices=[('Male','Male'),('Female','Female')])
	location = models.CharField(choices=IMAGE_LOCATION)
