from django.shortcuts import render
from django.urls import reverse
from django.views.generic import FormView, TemplateView

from DermaLytica.EntryForm import InputForm
from DermaLytica.Prediction_Model.Marie import predict_lesion
from DermaLytica.Prediction_Model.UtilityFunctions.ImageConvert import convert_to_jpg


class DermaLytica_HomeView(FormView):
	template_name = 'DermaHomePage.html'
	form_class = InputForm

	def get_form_action(self):
		""" Change the form action to submit to 'prediction' view instead of itself """
		return reverse('prediction')


class PredictionView(TemplateView):
	template_name = 'Prediction Page.html'

	def post(self, request, *args, **kwargs):
		form = InputForm(request.POST, request.FILES)  # Get form data

		if form.is_valid():
			instance = form.save(commit = False)

			image = instance.image
			age = instance.age
			gender = instance.gender
			location = instance.location
			zipCode = instance.zipCode

			if not image.name.lower().endswith(('.jpg', '.jpeg')):
				image = convert_to_jpg(image)
				instance.image = image

			# Make prediction
			prediction_results = predict_lesion(image, age, gender, location, zipCode)

			# Pass the results to the template
			context = {
					"prediction":        prediction_results.get("classification"),
					"confidence":        prediction_results.get("confidence"),
					"dermatology_Lists": prediction_results.get("dermatology_Lists", []),
					}
			return self.render_to_response(context)

		# If the form is invalid, return to the home page with errors
		return render(request, "DermaHomePage.html", {"form": form})
