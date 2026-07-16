"""Deterministic Phase 2 run provenance and manifest hashing."""

import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np


PROTOCOL_VERSION = "phase2-v1"


def canonicalize(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): canonicalize(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [canonicalize(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def stable_hash(value) -> str:
    payload = json.dumps(
        canonicalize(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def current_git_sha(repository_root=None) -> str:
    root = Path(repository_root or Path(__file__).resolve().parents[2])
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def build_run_manifest(
    *,
    dataset_name,
    schema_version,
    schema_hash,
    split_hash,
    adapter,
    seed,
    subsample_size=None,
    missing_counts=None,
    unseen_category_counts=None,
    git_sha=None,
):
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "git_sha": git_sha or current_git_sha(),
        "dataset_name": dataset_name,
        "dataset_schema_version": str(schema_version),
        "dataset_schema_hash": schema_hash,
        "split_index_hash": split_hash,
        "preprocessing_class": adapter.__class__.__name__,
        "preprocessing_version": adapter.version,
        "fitted_state_hash": adapter.state_hash,
        "missing_counts": missing_counts or {},
        "unseen_category_counts": unseen_category_counts or {},
        "random_seed": int(seed),
        "subsample_size": subsample_size,
    }
    manifest["manifest_hash"] = stable_hash(manifest)
    return manifest
