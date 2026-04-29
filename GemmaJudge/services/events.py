from __future__ import annotations

import json
from typing import Any


def build_stream_event(event_type: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build the canonical event payload emitted by the stream endpoint."""
    return {
        "type": event_type,
        "payload": payload or {},
    }


def encode_stream_event(event: dict[str, Any]) -> bytes:
    """Encode a single stream event as newline-delimited JSON."""
    return (json.dumps(event, ensure_ascii=True) + "\n").encode("utf-8")

