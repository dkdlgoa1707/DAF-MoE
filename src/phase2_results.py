"""Hash-gated Phase 2 result persistence and resume checks."""

import json
from pathlib import Path

from src.data.provenance import stable_hash
from src.phase2_protocol import PROTOCOL_VERSION


def build_execution_manifest(
    data_manifest,
    model_name,
    resolved_config,
    search_space_hash,
    seed,
):
    manifest = dict(data_manifest)
    manifest.update(
        {
            "protocol_version": PROTOCOL_VERSION,
            "model_name": model_name,
            "resolved_config": resolved_config,
            "resolved_config_hash": stable_hash(resolved_config),
            "search_space_hash": search_space_hash,
            "random_seed": int(seed),
        }
    )
    manifest.pop("manifest_hash", None)
    manifest["manifest_hash"] = stable_hash(manifest)
    return manifest


def reusable_result(path, expected_manifest):
    path = Path(path)
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    manifest = payload.get("resume_manifest", payload.get("manifest", {}))
    return (
        manifest.get("protocol_version") == PROTOCOL_VERSION
        and manifest.get("manifest_hash") == expected_manifest.get("manifest_hash")
        and manifest.get("resolved_config_hash")
        == expected_manifest.get("resolved_config_hash")
    )


def write_result(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path
