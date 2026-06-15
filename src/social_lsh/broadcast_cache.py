from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .artifacts import artifact_path
from .constants import REPO_ROOT


CACHE_VERSION = 1
NEWS_DIR = REPO_ROOT / "jupyter" / "output" / "news"
IMAGES_DIR = NEWS_DIR / "images"
METADATA_DIR = NEWS_DIR / "metadata"


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return slug or "artifact"


def artifact_cache_id(artifact_dir: Path | str) -> str:
    path = Path(artifact_dir).resolve()
    digest = hashlib.blake2b(str(path).encode("utf-8"), digest_size=6).hexdigest()
    return f"{_slug(path.name)}_{digest}"


def broadcast_cache_id(
    artifact_dir: Path | str,
    top_n: int,
    samples_per_cluster: int,
    use_llm: bool,
) -> str:
    mode = "llm" if use_llm else "tmpl"
    return f"{artifact_cache_id(artifact_dir)}_top{top_n}_s{samples_per_cluster}_{mode}"


def metadata_path(cache_id: str) -> Path:
    return METADATA_DIR / f"{cache_id}.json"


def image_file(cache_id: str, cluster_id: int | str) -> Path:
    return IMAGES_DIR / cache_id / f"cluster_{cluster_id}.png"


def legacy_image_file(cluster_id: int | str) -> Path:
    return IMAGES_DIR / f"cluster_{cluster_id}.png"


def public_image_path(cache_id: str, cluster_id: int | str) -> str:
    return f"/news-image/{cache_id}/{cluster_id}"


def _parquet_info(path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {}
    try:
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(path)
        info.update(
            {
                "rows": int(pf.metadata.num_rows),
                "row_groups": int(pf.metadata.num_row_groups),
                "columns": list(pf.schema_arrow.names),
            }
        )
    except Exception as exc:
        info["metadata_error"] = str(exc)
    return info


def artifact_file_signature(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    data: dict[str, Any] = {
        "path": str(path),
        "exists": True,
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if path.suffix == ".parquet":
        data.update(_parquet_info(path))
    return data


def source_signature(artifact_dir: Path | str) -> dict[str, Any]:
    artifact_path_dir = Path(artifact_dir)
    return {
        "artifact_dir": str(artifact_path_dir.resolve()),
        "clusters": artifact_file_signature(artifact_path("clusters", artifact_path_dir)),
        "scale_shingles": artifact_file_signature(artifact_path("scale_shingles", artifact_path_dir)),
    }


def _signature_matches(saved: dict[str, Any], current: dict[str, Any]) -> bool:
    for name in ("clusters", "scale_shingles"):
        left = saved.get(name) or {}
        right = current.get(name) or {}
        if not left.get("exists") or not right.get("exists"):
            return False
        if left.get("size_bytes") != right.get("size_bytes"):
            return False
        if left.get("mtime_ns") != right.get("mtime_ns"):
            return False
    return True


def attach_cached_images(items: list[dict[str, Any]], cache_id: str) -> list[dict[str, Any]]:
    attached: list[dict[str, Any]] = []
    for item in items:
        record = dict(item)
        cluster_id = record.get("cluster_id")
        if cluster_id is None:
            attached.append(record)
            continue

        cached = image_file(cache_id, cluster_id)
        legacy = legacy_image_file(cluster_id)
        if cached.exists():
            record["image_path"] = public_image_path(cache_id, cluster_id)
            record["image_file"] = str(cached)
            record["image_exists"] = True
        elif legacy.exists():
            record["image_path"] = f"/news-image/{cluster_id}"
            record["image_file"] = str(legacy)
            record["image_exists"] = True
        else:
            record.pop("image_path", None)
            record.pop("image_file", None)
            record["image_exists"] = False
        attached.append(record)
    return attached


def image_cache_summary(items: list[dict[str, Any]], cache_id: str) -> dict[str, Any]:
    cluster_ids = [item.get("cluster_id") for item in items if item.get("cluster_id") is not None]
    missing = [
        int(cluster_id)
        for cluster_id in cluster_ids
        if not image_file(cache_id, cluster_id).exists() and not legacy_image_file(cluster_id).exists()
    ]
    return {
        "total": len(cluster_ids),
        "available": len(cluster_ids) - len(missing),
        "missing": missing,
        "complete": not missing,
    }


def _segments_with_cached_images(segments: list[dict[str, Any]], items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    image_by_cluster = {
        item.get("cluster_id"): item.get("image_path")
        for item in items
        if item.get("cluster_id") is not None and item.get("image_path")
    }
    updated: list[dict[str, Any]] = []
    for segment in segments:
        record = dict(segment)
        cluster_id = record.get("cluster_id")
        if cluster_id in image_by_cluster:
            record["image_path"] = image_by_cluster[cluster_id]
        updated.append(record)
    return updated


def build_metadata(
    artifact_dir: Path | str,
    top_n: int,
    samples_per_cluster: int,
    use_llm: bool,
    items: list[dict[str, Any]],
    segments: list[dict[str, Any]],
) -> dict[str, Any]:
    cache_id = broadcast_cache_id(artifact_dir, top_n, samples_per_cluster, use_llm)
    items_with_images = attach_cached_images(items, cache_id)
    segments_with_images = _segments_with_cached_images(segments, items_with_images)
    return {
        "version": CACHE_VERSION,
        "cache_id": cache_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "params": {
            "top_n": int(top_n),
            "samples_per_cluster": int(samples_per_cluster),
            "use_llm": bool(use_llm),
        },
        "source_artifacts": source_signature(artifact_dir),
        "items": items_with_images,
        "segments": segments_with_images,
        "transcript": "\n".join(str(segment.get("text", "")) for segment in segments_with_images),
        "images": image_cache_summary(items_with_images, cache_id),
    }


def save_metadata(
    artifact_dir: Path | str,
    top_n: int,
    samples_per_cluster: int,
    use_llm: bool,
    items: list[dict[str, Any]],
    segments: list[dict[str, Any]],
) -> dict[str, Any]:
    meta = build_metadata(artifact_dir, top_n, samples_per_cluster, use_llm, items, segments)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    metadata_path(meta["cache_id"]).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def load_metadata(
    artifact_dir: Path | str,
    top_n: int,
    samples_per_cluster: int,
    use_llm: bool,
) -> dict[str, Any] | None:
    cache_id = broadcast_cache_id(artifact_dir, top_n, samples_per_cluster, use_llm)
    path = metadata_path(cache_id)
    if not path.exists():
        return None
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None

    params = meta.get("params") or {}
    if meta.get("version") != CACHE_VERSION:
        return None
    if int(params.get("top_n", -1)) != int(top_n):
        return None
    if int(params.get("samples_per_cluster", -1)) != int(samples_per_cluster):
        return None
    if bool(params.get("use_llm")) != bool(use_llm):
        return None
    if not _signature_matches(meta.get("source_artifacts") or {}, source_signature(artifact_dir)):
        return None

    items = attach_cached_images([dict(item) for item in meta.get("items", [])], cache_id)
    meta["items"] = items
    meta["images"] = image_cache_summary(items, cache_id)
    meta["segments"] = _segments_with_cached_images([dict(segment) for segment in meta.get("segments", [])], items)
    meta["transcript"] = "\n".join(str(segment.get("text", "")) for segment in meta["segments"])
    return meta
