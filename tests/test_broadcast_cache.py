from __future__ import annotations

import pandas as pd

import social_lsh.broadcast_cache as broadcast_cache
from social_lsh.broadcast_cache import broadcast_cache_id, image_file, load_metadata, save_metadata
from social_lsh.runtime_check import build_run_preflight


def _write_artifacts(artifact_dir):
    artifact_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"tweet_id": 1, "cluster_id": 10, "cluster_size": 3}]
    ).to_parquet(artifact_dir / "clusters.parquet", index=False)
    pd.DataFrame(
        [{"tweet_id": 1, "text": "story text", "topic_label": "unknown", "shingle_count": 4}]
    ).to_parquet(artifact_dir / "scale_shingles.parquet", index=False)


def test_broadcast_metadata_attaches_cached_images(tmp_path, monkeypatch) -> None:
    artifact_dir = tmp_path / "lsh_test"
    _write_artifacts(artifact_dir)
    monkeypatch.setattr(broadcast_cache, "METADATA_DIR", tmp_path / "news" / "metadata")
    monkeypatch.setattr(broadcast_cache, "IMAGES_DIR", tmp_path / "news" / "images")

    items = [{"cluster_id": 10, "cluster_size": 3, "headline": "H", "summary": "S", "topic": "unknown"}]
    segments = [{"kind": "story", "text": "S", "cluster_id": 10, "image_path": None}]
    save_metadata(artifact_dir, 1, 2, False, items, segments)

    cache_id = broadcast_cache_id(artifact_dir, 1, 2, False)
    image_path = image_file(cache_id, 10)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"png")

    loaded = load_metadata(artifact_dir, 1, 2, False)
    assert loaded is not None
    assert loaded["items"][0]["image_path"] == f"/news-image/{cache_id}/10"
    assert loaded["segments"][0]["image_path"] == f"/news-image/{cache_id}/10"
    assert loaded["images"]["complete"] is True


def test_broadcast_metadata_invalidates_when_artifact_changes(tmp_path, monkeypatch) -> None:
    artifact_dir = tmp_path / "lsh_test"
    _write_artifacts(artifact_dir)
    monkeypatch.setattr(broadcast_cache, "METADATA_DIR", tmp_path / "news" / "metadata")
    monkeypatch.setattr(broadcast_cache, "IMAGES_DIR", tmp_path / "news" / "images")

    save_metadata(artifact_dir, 1, 2, False, [], [])
    assert load_metadata(artifact_dir, 1, 2, False) is not None

    pd.DataFrame(
        [
            {"tweet_id": 1, "cluster_id": 10, "cluster_size": 3},
            {"tweet_id": 2, "cluster_id": 20, "cluster_size": 2},
        ]
    ).to_parquet(artifact_dir / "clusters.parquet", index=False)
    assert load_metadata(artifact_dir, 1, 2, False) is None


def test_preflight_reports_cache_hit(tmp_path, monkeypatch) -> None:
    artifact_dir = tmp_path / "lsh_test"
    _write_artifacts(artifact_dir)
    monkeypatch.setattr(broadcast_cache, "METADATA_DIR", tmp_path / "news" / "metadata")
    monkeypatch.setattr(broadcast_cache, "IMAGES_DIR", tmp_path / "news" / "images")

    save_metadata(artifact_dir, 1, 2, False, [], [])
    check = build_run_preflight(artifact_dir, top_n=1, samples_per_cluster=2, use_llm=False)
    assert check["status"] == "ok"
    assert check["cache"]["hit"] is True
