from django.apps import AppConfig

from DermaLytica.Prediction_Model.Marie import get_model


class DermaLyticaConfig(AppConfig):
	default_auto_field = 'django.db.models.BigAutoField'
	name = 'DermaLytica'
	
	def ready(self):
		model = get_model()  # Preload the prediction model
