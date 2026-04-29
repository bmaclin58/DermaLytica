import json
from typing import Any

from django.http import HttpRequest, JsonResponse, StreamingHttpResponse
from django.shortcuts import render
from django.views.decorators.http import require_GET, require_POST

from .services.events import encode_stream_event
from .services.oracle import stream_adjudication as stream_adjudication_service
from .services.retrieval import search_cards as search_cards_service, search_rules as search_rules_service
from .services.runtime import build_health_report


def home(request: HttpRequest):
	return render(request, "GemmaJudge/GemmaJudgeHomePage.html")


def search_cards(query: str) -> list [dict [str, Any]]:
	return search_cards_service(query)


def search_rules(query: str) -> list [dict [str, Any]]:
	return search_rules_service(query)


def stream_adjudication(
		question: str,
		format_name: str | None,
		messages: list [dict [str, Any]] | None,
		mode: str,
		):
	return stream_adjudication_service(
			question = question,
			format_name = format_name,
			messages = messages,
			mode = mode,
			)


@require_GET
def search_cards_api(request: HttpRequest) -> JsonResponse:
	query = request.GET.get("q", "")
	return JsonResponse(search_cards(query), safe = False)


@require_GET
def search_rules_api(request: HttpRequest) -> JsonResponse:
	query = request.GET.get("q", "")
	return JsonResponse(search_rules(query), safe = False)


@require_POST
def ask_rules_api(request: HttpRequest) -> StreamingHttpResponse | JsonResponse:
	payload = json.loads(request.body or "{}")
	question = str(payload.get("question", "")).strip()
	if not question:
		return JsonResponse({ "error": "question is required" }, status = 400)

	format_name = payload.get("format")
	messages = payload.get("messages") or []
	mode = str(payload.get("mode") or ("thinking" if payload.get("thinking") else "quick")).lower()

	def response_stream():
		for event in stream_adjudication(
				question = question,
				format_name = format_name,
				messages = messages,
				mode = mode,
				):
			yield encode_stream_event(event)

	return StreamingHttpResponse(
			response_stream(),
			content_type = "application/x-ndjson",
			)


@require_GET
def health_api(request: HttpRequest) -> JsonResponse:
	return JsonResponse(build_health_report())
