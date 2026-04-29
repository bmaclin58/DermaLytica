from __future__ import annotations

import logging
import sys
from pathlib import Path

import chromadb
from django.conf import settings
from openai import OpenAI

from .duckdb_gateway import DuckDBGateway
from .storage_sync import RuntimeAssetSpec, RuntimeAssetSyncService


logger = logging.getLogger(__name__)
_BOOTSTRAPPED = False


def _skip_startup_sync() -> bool:
    skip_commands = {"test", "makemigrations", "migrate", "collectstatic"}
    return any(arg in skip_commands for arg in sys.argv[1:])


def get_duckdb_gateway() -> DuckDBGateway:
    return DuckDBGateway(
        duckdb_path=settings.GEMMA_JUDGE_DUCKDB_PATH,
        key_id=settings.CLD_USER,
        secret=settings.HMAC_K,
    )


def get_asset_sync_service() -> RuntimeAssetSyncService:
    specs = [
        RuntimeAssetSpec(
            name="chroma",
            remote_uri=settings.GEMMA_JUDGE_CHROMA_DB_PATH,
            mounted_dir=Path(settings.GEMMA_JUDGE_MOUNTED_CHROMA_DIR) if settings.GEMMA_JUDGE_MOUNTED_CHROMA_DIR else None,
            local_dir=Path(settings.GEMMA_JUDGE_LOCAL_CHROMA_DIR),
            sentinel_relative_path="chroma.sqlite3",
        ),
        RuntimeAssetSpec(
            name="embedding_model",
            remote_uri=settings.GEMMA_JUDGE_LOCAL_EMBEDDING_MODEL_PATH,
            mounted_dir=Path(settings.GEMMA_JUDGE_MOUNTED_EMBEDDING_MODEL_DIR)
            if settings.GEMMA_JUDGE_MOUNTED_EMBEDDING_MODEL_DIR
            else None,
            local_dir=Path(settings.GEMMA_JUDGE_LOCAL_EMBEDDING_MODEL_DIR),
            sentinel_relative_path="modules.json",
        ),
    ]
    return RuntimeAssetSyncService(asset_specs=specs, duckdb_gateway=get_duckdb_gateway())


def bootstrap_runtime(force: bool = False) -> None:
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED and not force:
        return
    if _skip_startup_sync() and not force:
        return
    if not settings.GEMMA_JUDGE_SYNC_ON_STARTUP and not force:
        return
    asset_service = get_asset_sync_service()
    for spec in asset_service.asset_specs:
        source_details = asset_service.describe_asset_source(spec)
        logger.info(
            "Gemma Judge %s source resolved to %s (%s).",
            spec.name,
            source_details["source"],
            source_details["location"],
        )
    asset_service.ensure_assets(force=force)
    _BOOTSTRAPPED = True


def build_health_report() -> dict[str, dict[str, str]]:
    report: dict[str, dict[str, str]] = {}
    duckdb_gateway = get_duckdb_gateway()
    asset_service = get_asset_sync_service()
    asset_sources = {
        spec.name: asset_service.describe_asset_source(spec)
        for spec in asset_service.asset_specs
    }

    try:
        count = duckdb_gateway.query_scalar("SELECT COUNT(*) FROM mtg_db.cardInfo")
        report["duckdb"] = {
            "status": "healthy",
            "detail": f"cardInfo rows visible: {count}",
            "source": duckdb_gateway.describe_source(),
        }
    except Exception as exc:
        report["duckdb"] = {
            "status": "unhealthy",
            "detail": str(exc),
            "source": duckdb_gateway.describe_source(),
        }

    try:
        bootstrap_runtime(force=False)
        client = chromadb.PersistentClient(path=str(settings.GEMMA_JUDGE_LOCAL_CHROMA_DIR))
        client.get_collection("mtg_rules_Chroma")
        report["chroma"] = {
            "status": "healthy",
            "detail": "Local Chroma collections opened successfully.",
            "source": asset_sources["chroma"]["source"],
            "embedding_source": asset_sources["embedding_model"]["source"],
        }
    except Exception as exc:
        report["chroma"] = {
            "status": "unhealthy",
            "detail": str(exc),
            "source": asset_sources["chroma"]["source"],
            "embedding_source": asset_sources["embedding_model"]["source"],
        }

    try:
        OpenAI(base_url="https://api.novita.ai/openai", api_key=settings.GEMMA_JUDGE_ENDPOINT)
        report["novita"] = {"status": "healthy", "detail": "Novita client constructed successfully."}
    except Exception as exc:
        report["novita"] = {"status": "unhealthy", "detail": str(exc)}

    overall = "healthy" if all(item["status"] == "healthy" for item in report.values()) else "degraded"
    report["overall"] = {"status": overall, "detail": "Gemma Oracle runtime dependency check."}
    return report
