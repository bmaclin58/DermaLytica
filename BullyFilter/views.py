import threading

from django.urls import reverse_lazy
from django.views.generic import FormView

from BullyFilter.EntryForm import BullyInputForm
from BullyFilter.Prediction_Model.BullyFilterModel import BullyFilter_Prediction


class BullyFilter_HomeView(FormView):
	template_name = "BullyFilter/BullyFilterHomePage.html"
	form_class = BullyInputForm
	success_url = reverse_lazy("Bully-home")

	def dispatch(self, request, *args, **kwargs):
		try:
			from BullyFilter.Prediction_Model.WarmUp import warm_hf_endpoint_once
			threading.Thread(target = warm_hf_endpoint_once, daemon = True).start()
		except Exception as e:
			print(f"BullyFilter warmup failed: {e}")
		return super().dispatch(request, *args, **kwargs)

	def form_valid(self, form):
		text = form.cleaned_data ["text"]

		text_result = None
		error = None

		try:
			prediction_results = BullyFilter_Prediction(text)

			# HF typical success: list[{"generated_text": "..."}]
			if isinstance(prediction_results, list) and prediction_results:
				text_result = prediction_results [0].get("generated_text")
			# HF typical error: {"error": "..."}
			elif isinstance(prediction_results, dict):
				error = prediction_results.get("error") or str(prediction_results)
			else:
				error = f"Unexpected response format: {type(prediction_results).__name__}"

		except Exception as e:
			error = str(e)

		context = self.get_context_data(
				form = form,
				textResult = text_result,
				originalText = text,
				error = error,
				)
		return self.render_to_response(context)
