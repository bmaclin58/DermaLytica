from django.views.generic import FormView, TemplateView

from DermaLytica.EntryForm import InputForm


class homeView (FormView):
	template_name = 'HomePage.html'
	form_class = InputForm

class evaluationView (TemplateView):
	template_name = 'Prediction Page.html'
