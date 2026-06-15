from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .artifacts import artifact_path
from .broadcast_cache import (
    broadcast_cache_id,
    image_cache_summary,
    load_metadata,
    metadata_path,
    source_signature,
)


def _bytes_to_gib(value: int | float) -> float:
    return round(float(value) / (1024**3), 2)


def _meminfo() -> dict[str, Any]:
    values: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, raw = line.split(":", 1)
            number = int(raw.strip().split()[0]) * 1024
            values[key] = number
    except (OSError, ValueError):
        return {"available_bytes": None, "total_bytes": None, "swap_free_bytes": None}
    return {
        "available_bytes": values.get("MemAvailable"),
        "available_gib": _bytes_to_gib(values.get("MemAvailable", 0)),
        "total_bytes": values.get("MemTotal"),
        "total_gib": _bytes_to_gib(values.get("MemTotal", 0)),
        "swap_free_bytes": values.get("SwapFree"),
        "swap_free_gib": _bytes_to_gib(values.get("SwapFree", 0)),
    }


def _diskinfo(path: Path) -> dict[str, Any]:
    usage = shutil.disk_usage(path)
    return {
        "path": str(path),
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
        "free_gib": _bytes_to_gib(usage.free),
        "used_percent": round(usage.used / usage.total * 100, 1) if usage.total else None,
    }


def build_run_preflight(
    artifact_dir: Path | str,
    top_n: int = 5,
    samples_per_cluster: int = 12,
    use_llm: bool = True,
    images: bool = False,
) -> dict[str, Any]:
    """Inspect local data/resources before preparing the broadcast."""
    artifact_path_dir = Path(artifact_dir)
    cache_id = broadcast_cache_id(artifact_path_dir, top_n, samples_per_cluster, use_llm)
    meta = load_metadata(artifact_path_dir, top_n, samples_per_cluster, use_llm)
    source = source_signature(artifact_path_dir)
    mem = _meminfo()
    disk = _diskinfo(artifact_path_dir if artifact_path_dir.exists() else artifact_path_dir.parent)

    required = {
        "clusters": artifact_path("clusters", artifact_path_dir),
        "scale_shingles": artifact_path("scale_shingles", artifact_path_dir),
    }
    missing = [name for name, path in required.items() if not path.exists()]

    scale = source.get("scale_shingles") or {}
    clusters = source.get("clusters") or {}
    scale_size_gib = _bytes_to_gib(scale.get("size_bytes", 0))
    cluster_rows = int(clusters.get("rows") or 0)
    scale_rows = int(scale.get("rows") or 0)
    cache_hit = meta is not None
    image_summary = meta.get("images") if meta else {"total": top_n, "available": 0, "missing": [], "complete": False}

    warnings: list[str] = []
    recommendations: list[str] = []
    status = "ok"

    if missing:
        status = "danger"
        warnings.append(f"Missing required artifact(s): {', '.join(missing)}.")
        recommendations.append("Run the LSH pipeline first, or point --artifact-dir to lsh_combined/lsh_full that already exists.")

    available_bytes = mem.get("available_bytes")
    if available_bytes is not None:
        available_gib = float(mem.get("available_gib") or 0)
        if available_gib < 2:
            status = "danger"
            warnings.append(f"Only {available_gib} GiB RAM is available.")
            recommendations.append("Close heavy apps or reboot before preparing a broadcast.")
        elif available_gib < 6 and scale_size_gib > 1 and not cache_hit:
            status = "warning" if status == "ok" else status
            warnings.append(
                f"{artifact_path_dir.name} is large ({scale_size_gib} GiB scale_shingles) and cache is empty."
            )
            recommendations.append("Use jupyter/output/lsh_combined for a quick demo, or pregenerate cache once while the machine is idle.")

    if disk["free_gib"] < 2:
        status = "danger"
        warnings.append(f"Only {disk['free_gib']} GiB disk space is free.")
        recommendations.append("Free disk space before generating images or new pipeline artifacts.")
    elif images and disk["free_gib"] < 5:
        status = "warning" if status == "ok" else status
        warnings.append("Disk space is low for image generation.")
        recommendations.append("Skip image generation now, or delete old generated media first.")

    if scale_size_gib > 1 and not cache_hit:
        status = "warning" if status == "ok" else status
        recommendations.append(
            "First run will scan parquet metadata/data in streaming mode; later runs should use the saved metadata cache."
        )

    if images and cache_hit and not image_summary.get("complete"):
        status = "warning" if status == "ok" else status
        recommendations.append("Only missing images will be generated; existing cached images will be reused.")

    if cache_hit:
        recommendations.append("Cache is valid, so script generation will be skipped.")
    elif status != "danger":
        recommendations.append("No valid metadata cache yet; this run will build and save one.")

    return {
        "status": status,
        "warnings": warnings,
        "recommendations": recommendations,
        "artifact_dir": str(artifact_path_dir.resolve()),
        "artifact_rows": {"clusters": cluster_rows, "scale_shingles": scale_rows},
        "artifact_sizes_gib": {
            "clusters": _bytes_to_gib(clusters.get("size_bytes", 0)),
            "scale_shingles": scale_size_gib,
        },
        "cache": {
            "id": cache_id,
            "hit": cache_hit,
            "metadata_path": str(metadata_path(cache_id)),
            "images": image_summary,
        },
        "memory": mem,
        "disk": disk,
    }
