from __future__ import annotations

import re
from pathlib import Path
from typing import Mapping

from django.core.exceptions import ImproperlyConfigured


REQUIRED_SETTING_NAMES = (
    "GEMMA_JUDGE_ENDPOINT",
    "CLD_USER",
    "HMAC_K",
    "DUCKDB_PATH",
    "CHROMA_DB_PATH",
    "LOCAL_EMBEDDING_MODEL_PATH",
)

WINDOWS_ABSOLUTE_PATH_RE = re.compile(r"^[A-Za-z]:[\\/]")


def _looks_like_local_path(value: str) -> bool:
    return value.startswith(("/", "\\", "./", ".\\", "../", "..\\", "~")) or bool(
        WINDOWS_ABSOLUTE_PATH_RE.match(value)
    )


def normalize_storage_path(value: str | Path | None) -> str:
    """Preserve local paths and normalize GCS values into explicit gs:// URIs."""
    if not value:
        return ""

    cleaned = str(value).strip()
    if not cleaned:
        return ""

    if cleaned.startswith(("gs://", "gcs://")):
        return cleaned.replace("gcs://", "gs://", 1)

    if _looks_like_local_path(cleaned):
        return cleaned

    return f"gs://{cleaned.lstrip('/')}"


def normalize_gcs_uri(value: str | None) -> str:
    """Backward-compatible wrapper for storage path normalization."""
    return normalize_storage_path(value)


def is_gcs_uri(value: str | Path | None) -> bool:
    return normalize_storage_path(value).startswith("gs://")


def validate_required_settings(
    settings_map: Mapping[str, str | Path | None],
    required_keys: tuple[str, ...] = REQUIRED_SETTING_NAMES,
) -> dict[str, str]:
    """Validate the runtime settings used by the service layer."""
    missing = [
        key
        for key in required_keys
        if not str(settings_map.get(key, "") or "").strip()
    ]

    if missing:
        joined = ", ".join(sorted(missing))
        raise ImproperlyConfigured(f"Missing required Gemma Judge settings: {joined}")

    return {
        key: str(settings_map[key]).strip()
        for key in required_keys
    }
