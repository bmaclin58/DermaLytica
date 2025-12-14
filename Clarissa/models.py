from django.db import models

class mriData (models.Model):
	image = models.ImageField()

	class Meta:
		app_label = 'Clarissa'
