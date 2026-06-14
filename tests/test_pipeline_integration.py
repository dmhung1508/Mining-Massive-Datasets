from __future__ import annotations

from pathlib import Path

import pandas as pd

from social_lsh.artifacts import artifact_path, read_dataframe, write_dataframe
from social_lsh.pipeline import build_shingles_artifacts, run_baseline, run_lsh, verify_and_cluster
from social_lsh.search import prepare_search_index, search_similar_tweets


def _synthetic_frame() -> pd.DataFrame:
    rows = [
        {"tweet_id": 1, "user_id": 100, "text": "Breaking news from ukraine front line report today", "timestamp": "2024-01-01 00:00:00", "date": "2024-01-01"},
        {"tweet_id": 2, "user_id": 101, "text": "Breaking news from ukraine front line report today https://t.co/abc", "timestamp": "2024-01-01 00:01:00", "date": "2024-01-01"},
        {"tweet_id": 3, "user_id": 102, "text": "breaking news from ukraine front line report today", "timestamp": "2024-01-01 00:02:00", "date": "2024-01-01"},
        {"tweet_id": 4, "user_id": 103, "text": "Markets closed lower as oil prices moved sharply", "timestamp": "2024-01-01 00:03:00", "date": "2024-01-01"},
        {"tweet_id": 5, "user_id": 104, "text": "Markets closed lower as oil prices moved sharply today", "timestamp": "2024-01-01 00:04:00", "date": "2024-01-01"},
        {"tweet_id": 6, "user_id": 105, "text": "Completely unrelated sports update from another match", "timestamp": "2024-01-01 00:05:00", "date": "2024-01-01"},
    ]
    return pd.DataFrame(rows)


def test_pipeline_end_to_end_on_synthetic_data(tmp_path: Path) -> None:
    baseline_subset = _synthetic_frame()
    scale_subset = _synthetic_frame()

    write_dataframe(baseline_subset, artifact_path("baseline_subset", tmp_path))
    write_dataframe(scale_subset, artifact_path("scale_subset", tmp_path))

    build_shingles_artifacts(artifact_dir=tmp_path, shingle_size=3)
    baseline_path, _ = run_baseline(artifact_dir=tmp_path, threshold=0.8)
    candidates_path, lsh_metrics = run_lsh(artifact_dir=tmp_path, seed=42)
    outputs = verify_and_cluster(artifact_dir=tmp_path, threshold=0.8)

    baseline_pairs = read_dataframe(baseline_path)
    candidates = read_dataframe(candidates_path)
    verified_pairs = read_dataframe(outputs["verified_pairs"])
    clusters = read_dataframe(outputs["clusters"])

    assert not baseline_pairs.empty
    assert not candidates.empty
    assert not verified_pairs.empty
    assert lsh_metrics["selected_config"]["config_name"].startswith("k")
    assert clusters["cluster_id"].nunique() >= 2
    assert int(clusters.loc[clusters["tweet_id"] == 1, "cluster_size"].iloc[0]) >= 2


def test_search_similar_tweets_returns_expected_matches(tmp_path: Path) -> None:
    baseline_subset = _synthetic_frame()
    scale_subset = _synthetic_frame()

    write_dataframe(baseline_subset, artifact_path("baseline_subset", tmp_path))
    write_dataframe(scale_subset, artifact_path("scale_subset", tmp_path))

    build_shingles_artifacts(artifact_dir=tmp_path, shingle_size=3)
    run_baseline(artifact_dir=tmp_path, threshold=0.8)
    run_lsh(artifact_dir=tmp_path, seed=42)
    verify_and_cluster(artifact_dir=tmp_path, threshold=0.8)
    prepare_search_index(artifact_dir=tmp_path, seed=42)

    results, metadata = search_similar_tweets(
        "breaking news from ukraine front line report today",
        artifact_dir=tmp_path,
        top_k=3,
        min_jaccard=0.5,
        seed=42,
    )

    assert metadata["retrieval_mode"] == "lsh_candidates"
    assert results["tweet_id"].tolist()[0] == 1
    assert set(results["tweet_id"].tolist()[:3]) == {1, 2, 3}
    assert results["jaccard"].iloc[0] == 1.0
