from __future__ import annotations

import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from .artifacts import artifact_path, ensure_artifact_dir, merge_metrics, read_dataframe
from .constants import DEFAULT_ARTIFACT_DIR, DEFAULT_SEED, LSHConfig
from .minhash import hashed_shingles, signature_matrix
from .preprocessing import deserialize_nested_columns, make_word_shingles, normalize_text, tokenize_text
from .similarity import jaccard_similarity


def _load_scale_shingles(artifact_dir: Path) -> pd.DataFrame:
    scale_path = artifact_path("scale_shingles", artifact_dir)
    scale_df = read_dataframe(scale_path)
    return deserialize_nested_columns(scale_df, ["tokens", "shingles"])


def _load_selected_config(artifact_dir: Path) -> LSHConfig:
    metrics_path = ensure_artifact_dir(artifact_dir) / "metrics.json"
    metrics = {}
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    config_payload = metrics.get("run_lsh", {}).get("selected_config")
    if not config_payload:
        raise FileNotFoundError(
            "Selected LSH config not found in metrics.json. Run scripts/run_lsh.py before searching."
        )

    return LSHConfig(
        shingle_size=int(config_payload["shingle_size"]),
        num_perm=int(config_payload["num_perm"]),
        bands=int(config_payload["bands"]),
        rows=int(config_payload["rows"]),
    )


def _build_bucket_index(
    signatures,
    tweet_ids: list[int],
    bands: int,
    rows: int,
) -> dict[tuple[int, bytes], list[int]]:
    buckets: dict[tuple[int, bytes], list[int]] = defaultdict(list)
    for doc_idx in range(signatures.shape[0]):
        tweet_id = tweet_ids[doc_idx]
        for band_index in range(bands):
            start = band_index * rows
            end = start + rows
            key = (band_index, signatures[doc_idx, start:end].tobytes())
            buckets[key].append(tweet_id)
    return dict(buckets)


def build_search_index(
    scale_df: pd.DataFrame,
    config: LSHConfig,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    hashes = [hashed_shingles(shingles) for shingles in scale_df["shingles"]]
    signatures, metadata = signature_matrix(hashes, config.num_perm, seed)
    tweet_ids = scale_df["tweet_id"].astype(int).tolist()
    bucket_index = _build_bucket_index(signatures, tweet_ids, bands=config.bands, rows=config.rows)
    return {
        "seed": seed,
        "config": config.as_dict() | {"config_name": config.name},
        "num_documents": int(len(tweet_ids)),
        "signature_runtime_seconds": metadata["runtime_seconds"],
        "bucket_count": int(len(bucket_index)),
        "bucket_index": bucket_index,
    }


def _write_search_index(index: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(index, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _read_search_index(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return pickle.load(handle)


def prepare_search_index(
    artifact_dir: Path | None = None,
    seed: int = DEFAULT_SEED,
    force_rebuild: bool = False,
) -> Path:
    artifact_root = ensure_artifact_dir(artifact_dir or DEFAULT_ARTIFACT_DIR)
    index_path = artifact_path("search_index", artifact_root)

    if index_path.exists() and not force_rebuild:
        return index_path

    config = _load_selected_config(artifact_root)
    scale_df = _load_scale_shingles(artifact_root)
    index = build_search_index(scale_df, config=config, seed=seed)
    _write_search_index(index, index_path)

    merge_metrics(
        {
            "search_index": {
                "config_name": index["config"]["config_name"],
                "num_documents": index["num_documents"],
                "bucket_count": index["bucket_count"],
                "seed": seed,
                "signature_runtime_seconds": index["signature_runtime_seconds"],
            }
        },
        artifact_root / "metrics.json",
    )
    return index_path


def _query_candidates(
    query_shingles: list[str],
    index: dict[str, Any],
) -> tuple[set[int], dict[str, Any]]:
    config = LSHConfig(
        shingle_size=int(index["config"]["shingle_size"]),
        num_perm=int(index["config"]["num_perm"]),
        bands=int(index["config"]["bands"]),
        rows=int(index["config"]["rows"]),
    )
    query_hashes = hashed_shingles(query_shingles)
    signatures, _ = signature_matrix([query_hashes], config.num_perm, int(index["seed"]))
    query_signature = signatures[0]

    candidate_ids: set[int] = set()
    for band_index in range(config.bands):
        start = band_index * config.rows
        end = start + config.rows
        key = (band_index, query_signature[start:end].tobytes())
        candidate_ids.update(index["bucket_index"].get(key, []))

    return candidate_ids, {
        "config_name": config.name,
        "bands": config.bands,
        "rows": config.rows,
        "num_perm": config.num_perm,
        "candidate_count": len(candidate_ids),
    }


def search_similar_tweets(
    query_text: str,
    artifact_dir: Path | None = None,
    top_k: int = 5,
    min_jaccard: float = 0.0,
    seed: int = DEFAULT_SEED,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    artifact_root = ensure_artifact_dir(artifact_dir or DEFAULT_ARTIFACT_DIR)
    index_path = prepare_search_index(artifact_root, seed=seed)
    index = _read_search_index(index_path)
    scale_df = _load_scale_shingles(artifact_root)
    clusters = read_dataframe(artifact_path("clusters", artifact_root))

    config = LSHConfig(
        shingle_size=int(index["config"]["shingle_size"]),
        num_perm=int(index["config"]["num_perm"]),
        bands=int(index["config"]["bands"]),
        rows=int(index["config"]["rows"]),
    )

    normalized = normalize_text(query_text)
    tokens = tokenize_text(normalized)
    shingles = make_word_shingles(tokens, config.shingle_size)
    if not shingles:
        result_columns = [
            "rank",
            "tweet_id",
            "user_id",
            "source",
            "source_item_id",
            "source_user_id",
            "source_channel_id",
            "topic_label",
            "timestamp",
            "cluster_id",
            "cluster_size",
            "jaccard",
            "text",
        ]
        metadata = {
            "retrieval_mode": "query_too_short",
            "config_name": config.name,
            "query_token_count": len(tokens),
            "query_shingle_count": 0,
            "candidate_count": 0,
        }
        return pd.DataFrame(columns=result_columns), metadata

    candidate_ids, candidate_meta = _query_candidates(shingles, index)
    retrieval_mode = "lsh_candidates"
    candidate_frame = scale_df[scale_df["tweet_id"].isin(candidate_ids)].copy()
    if candidate_frame.empty:
        retrieval_mode = "bruteforce_fallback"
        candidate_frame = scale_df.copy()

    query_shingle_set = set(shingles)
    candidate_frame["jaccard"] = candidate_frame["shingles"].map(
        lambda current: jaccard_similarity(query_shingle_set, set(current))
    )
    candidate_frame = candidate_frame.loc[candidate_frame["jaccard"] >= min_jaccard].copy()
    candidate_frame = candidate_frame.sort_values(
        ["jaccard", "shingle_count", "tweet_id"],
        ascending=[False, False, True],
    ).head(top_k)

    candidate_frame = candidate_frame.merge(clusters, on="tweet_id", how="left")
    candidate_frame = candidate_frame.assign(rank=range(1, len(candidate_frame) + 1))
    result_columns = [
        "rank",
        "tweet_id",
        "user_id",
        "source",
        "source_item_id",
        "source_user_id",
        "source_channel_id",
        "topic_label",
        "timestamp",
        "cluster_id",
        "cluster_size",
        "jaccard",
        "text",
    ]
    result = candidate_frame[
        [column for column in result_columns if column in candidate_frame.columns]
    ].reset_index(drop=True)

    metadata = {
        "retrieval_mode": retrieval_mode,
        "config_name": config.name,
        "query_text": query_text,
        "normalized_query": normalized,
        "query_token_count": len(tokens),
        "query_shingle_count": len(shingles),
        **candidate_meta,
        "returned_results": int(len(result)),
    }
    return result, metadata
