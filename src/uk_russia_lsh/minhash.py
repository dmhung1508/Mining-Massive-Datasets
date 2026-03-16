from __future__ import annotations

import hashlib
import time
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd

from .constants import LSHConfig


MINHASH_PRIME = np.int64(2_147_483_647)


def stable_hash64(text: str) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def hashed_shingles(shingles: list[str]) -> list[int]:
    return sorted({stable_hash64(shingle) % int(MINHASH_PRIME) for shingle in shingles})


def minhash_parameters(num_perm: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    a = rng.integers(1, int(MINHASH_PRIME), size=num_perm, dtype=np.int64)
    b = rng.integers(0, int(MINHASH_PRIME), size=num_perm, dtype=np.int64)
    return a, b


def signature_matrix(
    hash_lists: list[list[int]],
    num_perm: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, float | int]]:
    started_at = time.perf_counter()
    a, b = minhash_parameters(num_perm, seed)
    signatures = np.full((len(hash_lists), num_perm), int(MINHASH_PRIME), dtype=np.int64)

    for idx, hashes in enumerate(hash_lists):
        if not hashes:
            continue
        values = np.asarray(hashes, dtype=np.int64)
        transformed = ((a[:, None] * values[None, :]) + b[:, None]) % MINHASH_PRIME
        signatures[idx] = transformed.min(axis=1)

    metadata = {
        "num_documents": len(hash_lists),
        "num_perm": num_perm,
        "runtime_seconds": round(time.perf_counter() - started_at, 6),
    }
    return signatures, metadata


def generate_candidate_pairs(
    signatures: np.ndarray,
    tweet_ids: list[int],
    bands: int,
    rows: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    if signatures.shape[1] != bands * rows:
        raise ValueError("bands * rows must equal the signature width")

    candidate_pairs: set[tuple[int, int]] = set()
    buckets_seen = 0

    for band_index in range(bands):
        start = band_index * rows
        end = start + rows
        buckets: dict[tuple[int, bytes], list[int]] = defaultdict(list)

        for doc_idx in range(signatures.shape[0]):
            key = (band_index, signatures[doc_idx, start:end].tobytes())
            buckets[key].append(doc_idx)

        buckets_seen += len(buckets)
        for doc_indices in buckets.values():
            if len(doc_indices) < 2:
                continue
            for left_idx, right_idx in combinations(doc_indices, 2):
                left_id = tweet_ids[left_idx]
                right_id = tweet_ids[right_idx]
                if left_id == right_id:
                    continue
                candidate_pairs.add((min(left_id, right_id), max(left_id, right_id)))

    frame = pd.DataFrame(candidate_pairs, columns=["tweet_id_left", "tweet_id_right"])
    if not frame.empty:
        frame = frame.sort_values(["tweet_id_left", "tweet_id_right"]).reset_index(drop=True)

    return frame, {"candidate_pairs": int(len(frame)), "bucket_count": buckets_seen}


def evaluate_candidates(
    candidate_pairs: pd.DataFrame,
    ground_truth_pairs: pd.DataFrame,
) -> dict[str, float | int]:
    candidate_set = {
        (int(row.tweet_id_left), int(row.tweet_id_right))
        for row in candidate_pairs[["tweet_id_left", "tweet_id_right"]].itertuples(index=False)
    }
    ground_truth_set = {
        (int(row.tweet_id_left), int(row.tweet_id_right))
        for row in ground_truth_pairs[["tweet_id_left", "tweet_id_right"]].itertuples(index=False)
    }
    true_positives = len(candidate_set & ground_truth_set)

    precision = true_positives / len(candidate_set) if candidate_set else 0.0
    recall = true_positives / len(ground_truth_set) if ground_truth_set else 0.0

    return {
        "candidate_pairs": len(candidate_set),
        "ground_truth_pairs": len(ground_truth_set),
        "true_positives": true_positives,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
    }


def rank_configs(results: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    return sorted(
        results,
        key=lambda item: (
            -float(item["recall"]),
            int(item["candidate_pairs"]),
            -float(item["precision"]),
            item["config_name"],
        ),
    )


def run_config(
    df: pd.DataFrame,
    config: LSHConfig,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    hashes = [hashed_shingles(shingles) for shingles in df["shingles"]]
    signatures, metadata = signature_matrix(hashes, config.num_perm, seed)
    candidates, candidate_meta = generate_candidate_pairs(
        signatures=signatures,
        tweet_ids=df["tweet_id"].astype(int).tolist(),
        bands=config.bands,
        rows=config.rows,
    )
    return candidates, {
        "signature_runtime_seconds": metadata["runtime_seconds"],
        "num_documents": metadata["num_documents"],
        "num_perm": metadata["num_perm"],
        **candidate_meta,
    }
