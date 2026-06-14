from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .constants import DEFAULT_ARTIFACT_DIR, DEFAULT_METRICS_PATH


ARTIFACT_FILES = {
    "baseline_subset": "baseline_subset.parquet",
    "scale_subset": "scale_subset.parquet",
    "baseline_shingles": "baseline_shingles.parquet",
    "scale_shingles": "scale_shingles.parquet",
    "baseline_pairs": "baseline_pairs.parquet",
    "candidates": "candidates.parquet",
    "verified_pairs": "verified_pairs.parquet",
    "clusters": "clusters.parquet",
    "search_index": "search_index.pkl",
}


def ensure_artifact_dir(artifact_dir: Path | None = None) -> Path:
    path = Path(artifact_dir or DEFAULT_ARTIFACT_DIR)
    path.mkdir(parents=True, exist_ok=True)
    return path


def artifact_path(name: str, artifact_dir: Path | None = None) -> Path:
    base = ensure_artifact_dir(artifact_dir)
    if name not in ARTIFACT_FILES:
        raise KeyError(f"Unknown artifact name: {name}")
    return base / ARTIFACT_FILES[name]


def write_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def read_dataframe(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def read_metrics(path: Path | None = None) -> dict[str, Any]:
    metrics_path = Path(path or DEFAULT_METRICS_PATH)
    if not metrics_path.exists():
        return {}
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def _merge_dicts(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def merge_metrics(updates: dict[str, Any], path: Path | None = None) -> dict[str, Any]:
    metrics_path = Path(path or DEFAULT_METRICS_PATH)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    merged = _merge_dicts(read_metrics(metrics_path), updates)
    metrics_path.write_text(json.dumps(merged, indent=2, sort_keys=True), encoding="utf-8")
    return merged
