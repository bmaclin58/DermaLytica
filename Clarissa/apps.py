from django.apps import AppConfig

from Clarissa.Prediction_Model.WarmUp import warm_hf_endpoint_once


class ClarissaConfig(AppConfig):
	default_auto_field = 'django.db.models.BigAutoField'
	name = 'Clarissa'
