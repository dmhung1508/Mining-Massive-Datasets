from __future__ import annotations

import hashlib
import heapq
import time
from pathlib import Path

import pandas as pd

from .artifacts import artifact_path, ensure_artifact_dir, merge_metrics, read_dataframe, write_dataframe
from .clustering import connected_components
from .constants import (
    BASELINE_SIZE,
    CONFIG_GRID,
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_INPUT_PARQUET,
    DEFAULT_SEED,
    DEFAULT_SHINGLE_SIZE,
    DEFAULT_VERIFY_THRESHOLD,
    LSHConfig,
    SCALE_SIZE,
)
from .minhash import evaluate_candidates, rank_configs, run_config
from .preprocessing import build_shingles, deserialize_nested_columns, serialize_nested_columns
from .similarity import exact_jaccard_pairs, verify_candidate_pairs


def _required_columns() -> list[str]:
    return ["tweet_id", "user_id", "text", "timestamp", "date"]


def _metadata_columns() -> list[str]:
    return [
        "source",
        "source_item_id",
        "source_user_id",
        "source_channel_id",
        "media_type",
        "forward_from_user_id",
        "forward_from_username",
        "topic_label",
        "topic_confidence",
        "topic_reason",
    ]


def _available_columns(column_names: list[str]) -> list[str]:
    available = set(column_names)
    return _required_columns() + [
        column for column in _metadata_columns() if column in available
    ]


def _load_with_spark(input_path: Path, sample_size: int, seed: int) -> pd.DataFrame:
    from pyspark.sql import SparkSession, functions as F

    spark = (
        SparkSession.builder.appName("uk-russia-lsh-extract")
        .master("local[*]")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    try:
        source_frame = spark.read.parquet(str(input_path))
        columns = _available_columns(source_frame.columns)
        frame = (
            source_frame.select(*columns)
            .filter(F.col("tweet_id").isNotNull())
            .filter(F.col("user_id").isNotNull())
            .filter(F.col("text").isNotNull())
        )
        subset = frame.orderBy(F.rand(seed)).limit(sample_size).toPandas()
        return subset.reset_index(drop=True)
    finally:
        spark.stop()


def _sampling_score(tweet_id: int, seed: int) -> int:
    payload = f"{seed}:{tweet_id}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), byteorder="big", signed=False)


def _stream_sample_with_pyarrow(
    input_path: Path,
    sample_size: int,
    seed: int,
) -> pd.DataFrame:
    import pyarrow.dataset as ds

    dataset = ds.dataset(str(input_path), format="parquet", partitioning="hive")
    heap: list[tuple[int, int, tuple[object, ...]]] = []
    columns = _available_columns(dataset.schema.names)

    for batch in dataset.to_batches(columns=columns, batch_size=8_192):
        batch_frame = batch.to_pandas()
        batch_frame = batch_frame.dropna(subset=["tweet_id", "user_id", "text"])
        for row in batch_frame.itertuples(index=False):
            tweet_id = int(row.tweet_id)
            score = _sampling_score(tweet_id, seed)
            payload = tuple(getattr(row, column) for column in columns)
            entry = (-score, -tweet_id, payload)

            if len(heap) < sample_size:
                heapq.heappush(heap, entry)
                continue

            if entry > heap[0]:
                heapq.heapreplace(heap, entry)

    sampled_rows = [entry[2] for entry in heap]
    sampled_rows.sort(key=lambda item: (_sampling_score(int(item[0]), seed), int(item[0])))
    return pd.DataFrame(sampled_rows, columns=columns).reset_index(drop=True)


def _load_subset(input_path: Path, sample_size: int, seed: int) -> tuple[pd.DataFrame, str]:
    try:
        return _load_with_spark(input_path, sample_size, seed), "spark"
    except Exception:
        return _stream_sample_with_pyarrow(input_path, sample_size, seed), "pyarrow"


def extract_subsets(
    input_path: Path | None = None,
    artifact_dir: Path | None = None,
    seed: int = DEFAULT_SEED,
    baseline_size: int = BASELINE_SIZE,
    scale_size: int = SCALE_SIZE,
) -> dict[str, Path]:
    artifact_root = ensure_artifact_dir(artifact_dir)
    source = Path(input_path or DEFAULT_INPUT_PARQUET)

    baseline_df, baseline_backend = _load_subset(source, baseline_size, seed)
    scale_df, scale_backend = _load_subset(source, scale_size, seed + 1)

    baseline_path = artifact_path("baseline_subset", artifact_root)
    scale_path = artifact_path("scale_subset", artifact_root)
    write_dataframe(baseline_df, baseline_path)
    write_dataframe(scale_df, scale_path)

    merge_metrics(
        {
            "extract_subsets": {
            "input_path": str(source),
            "baseline_rows": int(len(baseline_df)),
            "scale_rows": int(len(scale_df)),
            "seed": seed,
            "baseline_backend": baseline_backend,
            "scale_backend": scale_backend,
        }
        },
        artifact_root / "metrics.json",
    )
    return {"baseline_subset": baseline_path, "scale_subset": scale_path}


def build_shingles_artifacts(
    artifact_dir: Path | None = None,
    shingle_size: int = DEFAULT_SHINGLE_SIZE,
) -> dict[str, Path]:
    artifact_root = ensure_artifact_dir(artifact_dir)
    subset_specs = {
        "baseline_subset": "baseline_shingles",
        "scale_subset": "scale_shingles",
    }
    metrics: dict[str, dict[str, int]] = {}
    outputs: dict[str, Path] = {}

    for input_name, output_name in subset_specs.items():
        input_path = artifact_path(input_name, artifact_root)
        subset = read_dataframe(input_path)
        processed = build_shingles(subset, shingle_size=shingle_size)
        output_path = artifact_path(output_name, artifact_root)
        serialised = serialize_nested_columns(processed, ["tokens", "shingles"])
        write_dataframe(serialised, output_path)
        outputs[output_name] = output_path
        metrics[output_name] = {
            "source_rows": int(len(subset)),
            "kept_rows": int(len(processed)),
            "dropped_short_text_rows": int(len(subset) - len(processed)),
            "shingle_size": shingle_size,
        }

    merge_metrics({"build_shingles": metrics}, artifact_root / "metrics.json")
    return outputs


def _load_shingled_dataframe(path: Path) -> pd.DataFrame:
    return deserialize_nested_columns(read_dataframe(path), ["tokens", "shingles"])


def run_baseline(
    artifact_dir: Path | None = None,
    threshold: float = DEFAULT_VERIFY_THRESHOLD,
) -> tuple[Path, dict[str, float | int]]:
    artifact_root = ensure_artifact_dir(artifact_dir)
    baseline_path = artifact_path("baseline_shingles", artifact_root)
    baseline_df = _load_shingled_dataframe(baseline_path)
    pairs_df, metrics = exact_jaccard_pairs(baseline_df, threshold=threshold)
    output_path = artifact_path("baseline_pairs", artifact_root)
    write_dataframe(pairs_df, output_path)

    merge_metrics({"baseline": {"threshold": threshold, **metrics}}, artifact_root / "metrics.json")
    return output_path, metrics


def run_lsh(
    artifact_dir: Path | None = None,
    seed: int = DEFAULT_SEED,
    configs: tuple[LSHConfig, ...] = CONFIG_GRID,
) -> tuple[Path, dict[str, object]]:
    artifact_root = ensure_artifact_dir(artifact_dir)
    baseline_df = _load_shingled_dataframe(artifact_path("baseline_shingles", artifact_root))
    scale_df = _load_shingled_dataframe(artifact_path("scale_shingles", artifact_root))
    ground_truth = read_dataframe(artifact_path("baseline_pairs", artifact_root))

    config_results: list[dict[str, float | int | str]] = []
    for config in configs:
        baseline_candidates, run_metrics = run_config(baseline_df, config=config, seed=seed)
        eval_metrics = evaluate_candidates(baseline_candidates, ground_truth)
        config_results.append(
            {
                "config_name": config.name,
                **config.as_dict(),
                **run_metrics,
                **eval_metrics,
            }
        )

    ranked = rank_configs(config_results)
    winner_name = ranked[0]["config_name"]
    winner = next(config for config in configs if config.name == winner_name)

    started_at = time.perf_counter()
    scale_candidates, scale_metrics = run_config(scale_df, config=winner, seed=seed)
    runtime_seconds = round(time.perf_counter() - started_at, 6)
    candidates_path = artifact_path("candidates", artifact_root)
    write_dataframe(scale_candidates, candidates_path)

    metrics_payload = {
        "run_lsh": {
            "config_results": ranked,
            "selected_config": winner.as_dict() | {"config_name": winner.name},
            "scale_run": {
                **scale_metrics,
                "runtime_seconds": runtime_seconds,
                "candidate_pairs": int(len(scale_candidates)),
            },
        }
    }
    merge_metrics(metrics_payload, artifact_root / "metrics.json")
    return candidates_path, metrics_payload["run_lsh"]


def verify_and_cluster(
    artifact_dir: Path | None = None,
    threshold: float = DEFAULT_VERIFY_THRESHOLD,
) -> dict[str, Path]:
    artifact_root = ensure_artifact_dir(artifact_dir)
    scale_df = _load_shingled_dataframe(artifact_path("scale_shingles", artifact_root))
    candidates = read_dataframe(artifact_path("candidates", artifact_root))

    shingles_lookup = {
        int(row.tweet_id): set(row.shingles)
        for row in scale_df[["tweet_id", "shingles"]].itertuples(index=False)
    }
    verified_pairs, verify_metrics = verify_candidate_pairs(
        shingles_lookup,
        candidates,
        threshold=threshold,
    )
    verified_path = artifact_path("verified_pairs", artifact_root)
    write_dataframe(verified_pairs, verified_path)

    clusters = connected_components(scale_df["tweet_id"].astype(int).tolist(), verified_pairs)
    clusters_path = artifact_path("clusters", artifact_root)
    write_dataframe(clusters, clusters_path)

    merge_metrics(
        {
            "verify_and_cluster": {
                "threshold": threshold,
                **verify_metrics,
                "clusters": int(clusters["cluster_id"].nunique()),
                "largest_cluster_size": int(clusters["cluster_size"].max()) if not clusters.empty else 0,
            }
        },
        artifact_root / "metrics.json",
    )
    return {"verified_pairs": verified_path, "clusters": clusters_path}
