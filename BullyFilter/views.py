import threading

from django.views.generic import FormView
from django.urls import reverse_lazy

from BullyFilter.EntryForm import BullyInputForm
from BullyFilter.Prediction_Model.BullyFilterModel import BullyFilter_Prediction


class BullyFilter_HomeView(FormView):
	template_name = 'BullyFilter/BullyFilterHomePage.html'
	form_class = BullyInputForm
	success_url = reverse_lazy('Bully-home')  # same page

	def dispatch(self, request, *args, **kwargs):
		# Warm up the Hugging Face endpoint (fire-and-forget)
		try:
			from BullyFilter.Prediction_Model.WarmUp import warm_hf_endpoint_once
			threading.Thread(
				target=warm_hf_endpoint_once,
				daemon=True,
			).start()
		except Exception:
			# Don't 500 the page just because warmup failed
			print("BullyFilter warmup failed")
		return super().dispatch(request, *args, **kwargs)


	def form_valid(self, form):
		"""
		Handle POST:
		- Run the HF model
		- Re-render the SAME template with results
		"""
		text = form.cleaned_data['text']

		prediction_results = BullyFilter_Prediction(text)
		deToxed = prediction_results[0]['generated_text']

		context = self.get_context_data(
			form=form,
			textResult=deToxed,
			originalText=text,
		)
		return self.render_to_response(context)
