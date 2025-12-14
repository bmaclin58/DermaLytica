from django import forms
from crispy_forms.helper import FormHelper
from crispy_forms.layout import Layout, Div, Column, Submit
from crispy_forms.bootstrap import FormActions


class BullyInputForm(forms.Form):
	text = forms.CharField(
		label="Enter text to analyze",
		required=True,
		max_length=1000,
		widget=forms.Textarea(
			attrs={
				'rows': 6,
				'placeholder': 'Enter text to analyze for bullying or toxicity...',
				'class': 'form-control',
			}
		),
		help_text="Text will be processed by the Bully Filter model.",
	)

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

		self.helper = FormHelper()
		self.helper.form_method = 'post'
		self.helper.form_action = '.'
		self.helper.form_tag = False  # form tag handled in template

		self.helper.layout = Layout(
			Div(
				Column('text', css_class='form-group col-6'),
				css_class='row m-3 justify-content-center',
			),
			FormActions(
				Submit('submit', 'Submit', css_class='btn btn-primary'),
				css_class='text-center',
			),
		)
