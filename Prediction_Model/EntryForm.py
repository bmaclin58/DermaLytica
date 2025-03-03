from django import forms

from Prediction_Model.models import IMAGE_LOCATION, userData


class EntryForm(forms.ModelForm):
	image = forms.ImageField(
			required=True,

			)
	age = forms.IntegerField()
	location= forms.ChoiceField(
			choices=IMAGE_LOCATION,
			required=True,
			widget=forms.Select(attrs={'class': 'form-control'}),
			)

	gender = forms.ChoiceField(
			choices=[('Male','Male'),('Female','Female')],
			required=True,)
