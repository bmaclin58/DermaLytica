from django.apps import AppConfig

from BullyFilter.Prediction_Model.WarmUp import warm_hf_endpoint_once


class BullyFilterConfig(AppConfig):
	default_auto_field = 'django.db.models.BigAutoField'
	name = 'BullyFilter'

	def ready(self):
		warm_hf_endpoint_once()
