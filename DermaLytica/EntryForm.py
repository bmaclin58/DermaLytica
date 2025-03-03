from crispy_forms.bootstrap import FormActions
from crispy_forms.helper import FormHelper
from crispy_forms.layout import Column, Div, Layout, Submit
from django import forms

from DermaLytica.models import userData


class InputForm(forms.ModelForm):
	class Meta:
		model = userData
		fields = ('image', 'age', 'gender', 'location', 'zipCode')

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

		# FormHelper
		self.helper = FormHelper()
		self.helper.form_method = 'post'
		self.helper.form_action = '.'

		self.fields['zipCode'].label = "Zip Code (Optional)"
		self.fields['location'].label = "Body Location of Image"

		self.helper.form_tag = True
		self.helper.layout = Layout(
				Div(
						Column('gender', css_class='form-group col-6'),
						Column('age', css_class='form-group col-6'),
						css_class='row m-3 justify-content-center'
				),
				Div(
						Column('location', css_class='form-group col-6'),
						Column('image', css_class='form-group col-6'),
						css_class='row m-3 justify-content-center'
				),
				Div(
						Column('zipCode', css_class='form-group col-6'),
						css_class='row m-3 justify-content-center'
				),

		)
