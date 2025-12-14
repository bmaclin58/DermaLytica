import threading

from django.shortcuts import render
from django.urls import reverse
from django.views.generic import FormView, TemplateView

from Clarissa.EntryForm import MRIInputForm
from Clarissa.Prediction_Model.Clarissa import clarissa_Prediction
from Clarissa.Prediction_Model.WarmUp import warm_hf_endpoint_once

class Clarissa_HomeView(FormView):
	template_name = 'Clarissa_AI/ClarissaHomePage.html'
	form_class = MRIInputForm

	def dispatch(self, request, *args, **kwargs):

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
		form = MRIInputForm(request.POST, request.FILES)

		if not form.is_valid():
			return render(request, "Clarissa_AI/Clarissa_Prediction Page.html", { "form": form })

		image_file = form.cleaned_data ["image"]  # InMemoryUploadedFile

		# Forward to HF
		resp = clarissa_Prediction(image_file)
		resp.raise_for_status()

		pred = resp.json() [0]

		confidence = pred ["confidence"]
		confidence = round(confidence * 100, 2)

		context = {
				"prediction": pred ["label"],
				"confidence": confidence,
				"Image64"   : pred ["image_base64"],
				}
		return self.render_to_response(context)
