from django.shortcuts import render
from django.views.generic import FormView, TemplateView

from DermaLytica.EntryForm import InputForm
from DermaLytica.Prediction_Model.Marie import predict_lesion
from DermaLytica.Prediction_Model.UtilityFunctions.ImageConvert import convert_to_jpg


class DermaLitica_HomeView(FormView):
	template_name = 'HomePage.html'
	form_class = InputForm

	def form_valid(self, form):
		# self.object = form.save()
		instance = form.save(commit=False)

		image = instance.image
		age = instance.age
		gender = instance.gender
		location = instance.location
		zipCode = instance.zipCode

		# Translate and standardize the image only if it's not a .jpg or .jpeg
		if not image.name.lower().endswith(('.jpg', '.jpeg')):
			image = convert_to_jpg(image)
			instance.image = image

		# Process Metadata
		prediction_results = predict_lesion(image, age, gender, location, zipCode)
		context = {
				'prediction':        prediction_results.get('classification'),
				'confidence':        prediction_results.get('confidence') ,  # Convert to percentage
				'dermatology_Lists': prediction_results.get('dermatology_Lists', None),
		}

		return render(self.request, 'Prediction Page.html', context)


class PredictionView(TemplateView):
	"""
	View for displaying prediction results directly if needed
	(e.g., for navigating back to results without reprocessing).
	"""
	template_name = 'Prediction Page.html'

	def get_context_data(self, **kwargs):
		context = super().get_context_data(**kwargs)
		return context
