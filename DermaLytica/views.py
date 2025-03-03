from django.shortcuts import render
from django.views.generic import FormView, TemplateView

from DermaLytica.EntryForm import InputForm
from DermaLytica.Prediction_Model.Marie import predict_lesion
from DermaLytica.Prediction_Model.UtilityFunctions.ImageConvert import convert_to_jpg
from DermaLytica.Prediction_Model.UtilityFunctions.PrepMetadata import prepare_metadata


class homeView (FormView):
	template_name = 'HomePage.html'
	form_class = InputForm

	def form_valid(self, form):
		# self.object = form.save()
		form_data = form.cleaned_data

		image = form_data.get('image')
		age = form_data.get('age')
		gender = form_data.get('gender')
		location = form_data.get('location')
		zipCode = form_data.get('zipCode')

		# Translate and standardize the image only if it's not a .jpg or .jpeg
		if not image.name.lower().endswith(('.jpg', '.jpeg')):
			image = convert_to_jpg(image)

		# Process Metadata
		prediction = predict_lesion(image,age,gender,location)

		return render(self.request, 'Prediction Page.html', context)

class evaluationView (TemplateView):
	template_name = 'Prediction Page.html'
