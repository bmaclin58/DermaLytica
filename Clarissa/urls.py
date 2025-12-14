from Clarissa.views import Clarissa_HomeView, Clarissa_PredictionView
from django.urls import path

urlpatterns = [
	path('', Clarissa_HomeView.as_view(), name = 'Clarissa-home'),
	path('Prediction/', Clarissa_PredictionView.as_view(), name = 'Clarissa-prediction')
		]
