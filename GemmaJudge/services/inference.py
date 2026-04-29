from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from django.conf import settings
from openai import OpenAI

from .events import build_stream_event


SYSTEM_PROMPT = (
    "You are a Magic: The Gathering rules adjudicator. "
    "Use the provided context unless the answer follows directly from ordinary reasoning. "
    "When relevant, cite the specific card text, card ruling, glossary term, or comprehensive rule used. "
    "If the answer depends on missing game-state values, give a conditional answer instead of guessing. "
    "If the context is insufficient, say what is missing instead of inventing facts."
)


THINKING_RESPONSE_INSTRUCTIONS = (
    "Respond using exactly two XML-style sections: "
    "<analysis>your visible reasoning</analysis><final>your final ruling</final>."
)

ANALYSIS_OPEN_TAG = "<analysis>"
ANALYSIS_CLOSE_TAG = "</analysis>"
FINAL_OPEN_TAG = "<final>"
FINAL_CLOSE_TAG = "</final>"


def create_openai_client() -> OpenAI:
    return OpenAI(
        base_url="https://api.novita.ai/openai",
        api_key=settings.GEMMA_JUDGE_ENDPOINT,
    )


def build_model_prompt(
    question: str,
    format_name: str | None,
    messages: list[dict[str, Any]] | None,
    retrieval_result: dict[str, Any],
) -> str:
    """Build the final prompt passed to the fixed Gemma-family endpoint."""
    history_lines = []
    for message in messages or []:
        role = str(message.get("role", "user")).upper()
        text = str(message.get("text") or message.get("content") or "").strip()
        if text:
            history_lines.append(f"{role}: {text}")

    history_block = "\n".join(history_lines) if history_lines else "No prior conversation."
    active_format = format_name or retrieval_result.get("format_name") or "Unspecified"

    return f"""
QUESTION
{question}

ACTIVE FORMAT
{active_format}

CHAT HISTORY
{history_block}

GAME CONTEXT
{retrieval_result.get("game_context_block", "Format / ruleset: unspecified")}

RETRIEVED CONTEXT
{retrieval_result.get("combined_context", "")}

ANSWER INSTRUCTIONS
- Apply the ACTIVE FORMAT when deciding which rules matter.
- If the answer depends on missing values, give a conditional answer instead of assuming them.
- Distinguish combat damage, commander damage, regular damage, and life loss when relevant.
- Cite the relevant card text, ruling, glossary term, or comprehensive rule when available.
""".strip()


def parse_thinking_response(response: str) -> dict[str, str]:
    """Split a tagged visible-reasoning response into analysis and final answer."""
    text = str(response or "").strip()
    match = re.search(
        r"<analysis>\s*(?P<analysis>.*?)\s*</analysis>\s*<final>\s*(?P<final>.*?)\s*</final>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return {"thinking": "", "answer": text}

    return {
        "thinking": match.group("analysis").strip(),
        "answer": match.group("final").strip(),
    }


def _chunk_text(text: str, chunk_size: int = 700) -> Iterable[str]:
    remaining = str(text or "")
    while remaining:
        yield remaining[:chunk_size]
        remaining = remaining[chunk_size:]


def _extract_stream_content(chunk: Any) -> str | None:
    choices = getattr(chunk, "choices", None) or []
    if not choices:
        return None

    delta = getattr(choices[0], "delta", None)
    return getattr(delta, "content", None)


def _find_tag(text: str, tag: str) -> int:
    return text.lower().find(tag.lower())


def _split_for_partial_tag(text: str, tag: str) -> tuple[str, str]:
    lowered_text = text.lower()
    lowered_tag = tag.lower()
    max_suffix = min(len(text), len(tag) - 1)
    for size in range(max_suffix, 0, -1):
        if lowered_text.endswith(lowered_tag[:size]):
            return text[:-size], text[-size:]
    return text, ""


def _strip_partial_tag_suffix(text: str, tag: str) -> str:
    safe_text, _ = _split_for_partial_tag(text, tag)
    return safe_text


def _strip_visible_reasoning_tags(text: str) -> str:
    cleaned = re.sub(r"</?(analysis|final)>", "", str(text or ""), flags=re.IGNORECASE)
    return cleaned.strip()


class _ThinkingStreamParser:
    def __init__(self) -> None:
        self.state = "before_analysis"
        self.buffer = ""
        self.raw_parts: list[str] = []
        self.between_sections: list[str] = []
        self.answer_emitted = False
        self.thinking_emitted = False

    def feed(self, text: str) -> list[tuple[str, str]]:
        self.raw_parts.append(text)
        self.buffer += text
        emitted: list[tuple[str, str]] = []

        while True:
            if self.state == "before_analysis":
                tag_index = _find_tag(self.buffer, ANALYSIS_OPEN_TAG)
                if tag_index != -1:
                    self.buffer = self.buffer[tag_index + len(ANALYSIS_OPEN_TAG) :]
                    self.state = "in_analysis"
                    continue

                _, self.buffer = _split_for_partial_tag(self.buffer, ANALYSIS_OPEN_TAG)
                break

            if self.state == "in_analysis":
                tag_index = _find_tag(self.buffer, ANALYSIS_CLOSE_TAG)
                if tag_index != -1:
                    thinking_text = self.buffer[:tag_index]
                    emitted.extend(self._emit_thinking(thinking_text))
                    self.buffer = self.buffer[tag_index + len(ANALYSIS_CLOSE_TAG) :]
                    self.state = "after_analysis"
                    continue

                thinking_text, self.buffer = _split_for_partial_tag(self.buffer, ANALYSIS_CLOSE_TAG)
                emitted.extend(self._emit_thinking(thinking_text))
                break

            if self.state == "after_analysis":
                tag_index = _find_tag(self.buffer, FINAL_OPEN_TAG)
                if tag_index != -1:
                    between_text = self.buffer[:tag_index]
                    if between_text:
                        self.between_sections.append(between_text)
                    self.buffer = self.buffer[tag_index + len(FINAL_OPEN_TAG) :]
                    self.state = "in_final"
                    continue

                between_text, self.buffer = _split_for_partial_tag(self.buffer, FINAL_OPEN_TAG)
                if between_text:
                    self.between_sections.append(between_text)
                break

            if self.state == "in_final":
                tag_index = _find_tag(self.buffer, FINAL_CLOSE_TAG)
                if tag_index != -1:
                    answer_text = self.buffer[:tag_index]
                    emitted.extend(self._emit_answer(answer_text))
                    self.buffer = self.buffer[tag_index + len(FINAL_CLOSE_TAG) :]
                    self.state = "done"
                    continue

                answer_text, self.buffer = _split_for_partial_tag(self.buffer, FINAL_CLOSE_TAG)
                emitted.extend(self._emit_answer(answer_text))
                break

            self.buffer = ""
            break

        return emitted

    def finish(self) -> list[tuple[str, str]]:
        emitted: list[tuple[str, str]] = []
        full_text = "".join(self.raw_parts)

        if self.state == "in_analysis":
            remainder = _strip_partial_tag_suffix(self.buffer, ANALYSIS_CLOSE_TAG)
            emitted.extend(self._emit_thinking(remainder))
            self.buffer = ""
        elif self.state == "in_final":
            remainder = _strip_partial_tag_suffix(self.buffer, FINAL_CLOSE_TAG)
            emitted.extend(self._emit_answer(remainder))
            self.buffer = ""

        if not self.answer_emitted:
            if self.state == "after_analysis":
                fallback_source = "".join(self.between_sections) + _strip_partial_tag_suffix(self.buffer, FINAL_OPEN_TAG)
                fallback_text = _strip_visible_reasoning_tags(fallback_source)
                if not fallback_text:
                    fallback_text = _strip_visible_reasoning_tags(full_text)
            else:
                fallback_text = _strip_visible_reasoning_tags(full_text)

            emitted.extend(self._emit_answer(fallback_text))

        self.state = "done"
        self.buffer = ""
        return emitted

    def _emit_thinking(self, text: str) -> list[tuple[str, str]]:
        if not text:
            return []

        self.thinking_emitted = True
        return [("thinking", text)]

    def _emit_answer(self, text: str) -> list[tuple[str, str]]:
        if not text:
            return []

        self.answer_emitted = True
        return [("answer", chunk) for chunk in _chunk_text(text)]


def stream_inference_events(
    question: str,
    format_name: str | None,
    messages: list[dict[str, Any]] | None,
    retrieval_result: dict[str, Any],
    mode: str = "quick",
) -> Iterable[dict[str, Any]]:
    """Stream model output as answer or thinking/answer event pairs."""
    prompt = build_model_prompt(
        question=question,
        format_name=format_name,
        messages=messages,
        retrieval_result=retrieval_result,
    )

    user_content = prompt
    if mode == "thinking":
        user_content = f"{prompt}\n\nVISIBLE REASONING MODE\n{THINKING_RESPONSE_INSTRUCTIONS}"

    stream = create_openai_client().chat.completions.create(
        model=settings.GEMMA_JUDGE_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        stream=True,
        max_tokens=(
            settings.GEMMA_JUDGE_MAX_TOKENS_THINKING
            if mode == "thinking"
            else settings.GEMMA_JUDGE_MAX_TOKENS_QUICK
        ),
        temperature=0.2,
        top_p=1,
    )

    if mode != "thinking":
        for chunk in stream:
            content = _extract_stream_content(chunk)
            if content:
                yield build_stream_event("answer", {"chunk": content})
        return

    parser = _ThinkingStreamParser()
    for chunk in stream:
        content = _extract_stream_content(chunk)
        if content:
            for event_type, event_chunk in parser.feed(content):
                if event_type == "thinking":
                    yield build_stream_event("thinking", {"chunk": event_chunk, "source": "model"})
                else:
                    yield build_stream_event("answer", {"chunk": event_chunk})

    for event_type, event_chunk in parser.finish():
        if event_type == "thinking":
            yield build_stream_event("thinking", {"chunk": event_chunk, "source": "model"})
        else:
            yield build_stream_event("answer", {"chunk": event_chunk})
