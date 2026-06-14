from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds
import pytest

from social_lsh.artifacts import artifact_path, read_dataframe, write_dataframe
from social_lsh.constants import DEFAULT_INPUT_PARQUET
from social_lsh.pipeline import build_shingles_artifacts, run_baseline, run_lsh, verify_and_cluster


@pytest.mark.skipif(not DEFAULT_INPUT_PARQUET.exists(), reason="tweets_final.parquet is not available")
def test_pipeline_smoke_on_real_parquet_sample(tmp_path: Path) -> None:
    dataset = ds.dataset(str(DEFAULT_INPUT_PARQUET), format="parquet", partitioning="hive")
    sample = dataset.to_table(columns=["tweet_id", "user_id", "text", "timestamp", "date"]).slice(0, 48).to_pandas()
    sample = sample.dropna(subset=["tweet_id", "user_id", "text"]).reset_index(drop=True)

    write_dataframe(sample.head(24), artifact_path("baseline_subset", tmp_path))
    write_dataframe(sample, artifact_path("scale_subset", tmp_path))

    build_shingles_artifacts(artifact_dir=tmp_path, shingle_size=3)
    run_baseline(artifact_dir=tmp_path, threshold=0.8)
    candidates_path, _ = run_lsh(artifact_dir=tmp_path, seed=42)
    outputs = verify_and_cluster(artifact_dir=tmp_path, threshold=0.8)

    candidates = read_dataframe(candidates_path)
    verified_pairs = read_dataframe(outputs["verified_pairs"])
    clusters = read_dataframe(outputs["clusters"])

    assert isinstance(candidates, pd.DataFrame)
    assert isinstance(verified_pairs, pd.DataFrame)
    assert isinstance(clusters, pd.DataFrame)
    assert {"tweet_id", "cluster_id", "cluster_size"} <= set(clusters.columns)
