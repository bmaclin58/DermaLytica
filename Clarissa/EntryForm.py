from crispy_forms.bootstrap import FormActions
from crispy_forms.helper import FormHelper
from crispy_forms.layout import Column, Div, Layout, Submit
from django import forms

from Clarissa.models import mriData


class MRIInputForm(forms.ModelForm):
	class Meta:
		model = mriData
		fields = 'image'

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

		# FormHelper
		self.helper = FormHelper()
		self.helper.form_method = 'post'
		self.helper.form_action = '.'

		self.helper.form_tag = False
		# self.helper.disable_csrf = True
		self.helper.layout = Layout(
				Div(
						Column('image', css_class = 'form-group col-6'),
						css_class = 'row m-3 justify-content-center',
						),
				FormActions(
						Submit('submit', 'Submit', css_class = 'btn btn-primary'),
						css_class = 'text-center',
						),

				)
