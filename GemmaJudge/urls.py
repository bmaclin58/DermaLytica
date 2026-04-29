from django.urls import path

from . import views


urlpatterns = [
    path("", views.home, name="gemmajudge-home"),
    path("api/search/cards/", views.search_cards_api, name="gemmajudge-search-cards"),
    path("api/search/rules/", views.search_rules_api, name="gemmajudge-search-rules"),
    path("api/ask-rules/", views.ask_rules_api, name="gemmajudge-ask-rules"),
    path("api/health/", views.health_api, name="gemmajudge-health"),
]
