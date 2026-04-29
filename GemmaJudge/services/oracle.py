from __future__ import annotations

from typing import Any, Iterable

from .events import build_stream_event
from .inference import stream_inference_events
from .retrieval import retrieve_rules_and_glossary_context


def _build_retrieval_trace(retrieval_result: dict[str, Any]) -> list[str]:
    trace_lines: list[str] = []
    cards = retrieval_result.get("cards_for_ui", [])
    rules = retrieval_result.get("rules_for_ui", [])
    if cards:
        trace_lines.append("Detected cards: " + ", ".join(card["name"] for card in cards if card.get("name")))
    else:
        trace_lines.append("Detected cards: none")

    if rules:
        trace_lines.append(
            "Retrieved rules: " + ", ".join(rule["rule_id"] for rule in rules[:6] if rule.get("rule_id"))
        )
    else:
        trace_lines.append("Retrieved rules: none")

    for warning in retrieval_result.get("warnings", []):
        trace_lines.append(warning)

    return trace_lines


def stream_adjudication(
    question: str,
    format_name: str | None = None,
    messages: list[dict[str, Any]] | None = None,
    mode: str = "quick",
) -> Iterable[dict[str, Any]]:
    active_mode = "thinking" if str(mode).lower() == "thinking" else "quick"
    yield build_stream_event("status", {"message": "Starting adjudication.", "mode": active_mode})

    try:
        retrieval_result = retrieve_rules_and_glossary_context(question=question, format_name=format_name)
        yield build_stream_event("cards", {"cards": retrieval_result.get("cards_for_ui", [])})
        yield build_stream_event("rules", {"rules": retrieval_result.get("rules_for_ui", [])})

        for line in _build_retrieval_trace(retrieval_result):
            yield build_stream_event("thinking", {"chunk": line, "source": "retrieval"})

        yield build_stream_event("status", {"message": "Querying Gemma endpoint.", "mode": active_mode})

        for event in stream_inference_events(
            question=question,
            format_name=format_name,
            messages=messages,
            retrieval_result=retrieval_result,
            mode=active_mode,
        ):
            yield event
    except Exception as exc:
        yield build_stream_event("error", {"message": str(exc)})
    finally:
        yield build_stream_event("done", {"ok": True})

