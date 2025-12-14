from django.urls import path

from BullyFilter.views import BullyFilter_HomeView

urlpatterns = [
	path('', BullyFilter_HomeView.as_view(), name = 'Bully-home'),
		]
