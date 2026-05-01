from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

from .duckdb_gateway import DuckDBGateway


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RuntimeAssetSpec:
    name: str
    remote_uri: str
    local_dir: Path
    sentinel_relative_path: str
    mounted_dir: Path | None = None


class RuntimeAssetSyncService:
    """Sync Chroma and embedding-model assets into a local runtime cache."""

    def __init__(self, asset_specs: list[RuntimeAssetSpec], duckdb_gateway: DuckDBGateway):
        self.asset_specs = asset_specs
        self.duckdb_gateway = duckdb_gateway

    def is_asset_ready(self, spec: RuntimeAssetSpec) -> bool:
        return (spec.local_dir / spec.sentinel_relative_path).exists()

    def is_mounted_asset_ready(self, spec: RuntimeAssetSpec) -> bool:
        if spec.mounted_dir is None:
            return False
        return (spec.mounted_dir / spec.sentinel_relative_path).exists()

    def describe_asset_source(self, spec: RuntimeAssetSpec) -> dict[str, str]:
        if self.is_mounted_asset_ready(spec):
            return {
                "source": "mounted",
                "location": str(spec.mounted_dir),
            }
        return {
            "source": "remote",
            "location": spec.remote_uri,
        }

    def ensure_assets(self, force: bool = False) -> list[Path]:
        synced: list[Path] = []
        for spec in self.asset_specs:
            if force or not self.is_asset_ready(spec):
                source = self.sync_asset(spec, force=force)
                logger.info("Prepared Gemma Judge %s cache from %s source.", spec.name, source)
                synced.append(spec.local_dir)
        return synced

    def sync_asset(self, spec: RuntimeAssetSpec, force: bool = False) -> str:
        if force and spec.local_dir.exists():
            shutil.rmtree(spec.local_dir)

        spec.local_dir.mkdir(parents=True, exist_ok=True)

        source_details = self.describe_asset_source(spec)
        if source_details["source"] == "mounted":
            shutil.copytree(spec.mounted_dir, spec.local_dir, dirs_exist_ok=True)
        else:
            self._sync_remote_asset(spec)

        if not self.is_asset_ready(spec):
            raise RuntimeError(
                f"Runtime asset sync did not produce expected sentinel file: {spec.local_dir / spec.sentinel_relative_path}"
            )

        return source_details["source"]

    def _sync_remote_asset(self, spec: RuntimeAssetSpec) -> None:
        for remote_file in self.duckdb_gateway.glob_remote_files(spec.remote_uri):
            if remote_file.endswith("/"):
                continue

            relative_path = remote_file.replace(spec.remote_uri.rstrip("/") + "/", "", 1)
            target_path = spec.local_dir / relative_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_bytes(self.duckdb_gateway.read_remote_blob(remote_file))
