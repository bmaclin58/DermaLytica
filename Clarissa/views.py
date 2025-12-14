import threading

from django.shortcuts import render
from django.urls import reverse
from django.views.generic import FormView, TemplateView

from Clarissa.EntryForm import MRIInputForm
from Clarissa.Prediction_Model.Clarissa import clarissa_Prediction


class Clarissa_HomeView(FormView):
	template_name = 'Clarissa_AI/ClarissaHomePage.html'
	form_class = MRIInputForm

	def dispatch(self, request, *args, **kwargs):
		from Clarissa.Prediction_Model.WarmUp import warm_hf_endpoint_once
		# Fire-and-forget warmup (non-blocking)
		threading.Thread(
				target = warm_hf_endpoint_once,
				daemon = True,
				).start()

		return super().dispatch(request, *args, **kwargs)

	def get_form_action(self):
		""" Change the form action to submit to 'prediction' view instead of itself """
		return reverse('Clarissa-prediction')


class Clarissa_PredictionView(TemplateView):
	template_name = 'Clarissa_AI/Clarissa_Prediction Page.html'

	def post(self, request, *args, **kwargs):
		form = MRIInputForm(request.POST, request.FILES)  # Get form data

		if form.is_valid():
			instance = form.save(commit = False)

			image = instance.image

			# Make prediction
			prediction_results = clarissa_Prediction(image)

			prediction = prediction_results.json() [0]

			predictionImage = prediction ['image_base64']
			predictionLabel = prediction ['label']
			predictionConfidence = prediction ['confidence']

			# Pass the results to the template
			context = {
					"prediction": predictionLabel,
					"confidence": predictionConfidence,
					"Image64"   : predictionImage,
					}
			return self.render_to_response(context)

		# If the form is invalid, return to the home page with errors
		return render(request, "Clarissa_AI/Clarissa_Prediction Page.html", { "form": form })
