from django.shortcuts import redirect, render
from django.views.generic import FormView, TemplateView

from DermaLytica.EntryForm import InputForm
from DermaLytica.Prediction_Model.Marie import predict_lesion
from DermaLytica.Prediction_Model.UtilityFunctions.ImageConvert import convert_to_jpg


class DermaLytica_HomeView(FormView):
	template_name = 'HomePage.html'
	form_class = InputForm

	def form_valid(self, form):
		instance = form.save(commit=False)

		image = instance.image
		age = instance.age
		gender = instance.gender
		location = instance.location
		zipCode = instance.zipCode

		if not image.name.lower().endswith(('.jpg', '.jpeg')):
			image = convert_to_jpg(image)
			instance.image = image

		prediction_results = predict_lesion(image, age, gender, location, zipCode)

		# Store the results in session
		self.request.session['prediction_results'] = {
				'prediction':        prediction_results.get('classification'),
				'confidence':        prediction_results.get('confidence'),
				'dermatology_Lists': prediction_results.get('dermatology_Lists', None),
		}

		# Redirect to the PredictionView
		return redirect('prediction')


class PredictionView(TemplateView):
	template_name = 'Prediction Page.html'

	def get_context_data(self, **kwargs):
		context = super().get_context_data(**kwargs)

		# Get the prediction results from session
		prediction_results = self.request.session.get('prediction_results', {})
		context.update(prediction_results)

		# Clear the session data after retrieving it
		self.request.session.pop('prediction_results', None)

		return context
