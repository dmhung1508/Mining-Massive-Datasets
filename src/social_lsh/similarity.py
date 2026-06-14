from __future__ import annotations

import time
from itertools import combinations

import pandas as pd


def jaccard_similarity(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    intersection = len(left & right)
    union = len(left) + len(right) - intersection
    return intersection / union if union else 0.0


def _pair_upper_bound(left_size: int, right_size: int) -> float:
    larger = max(left_size, right_size)
    smaller = min(left_size, right_size)
    return smaller / larger if larger else 0.0


def exact_jaccard_pairs(
    df: pd.DataFrame,
    threshold: float,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    started_at = time.perf_counter()
    rows: list[dict[str, float | int]] = []

    records = []
    for row in df[["tweet_id", "shingles"]].itertuples(index=False):
        shingle_set = set(row.shingles)
        records.append((int(row.tweet_id), shingle_set, len(shingle_set)))

    pairs_considered = 0
    for (left_id, left_set, left_size), (right_id, right_set, right_size) in combinations(records, 2):
        pairs_considered += 1
        if _pair_upper_bound(left_size, right_size) < threshold:
            continue

        score = jaccard_similarity(left_set, right_set)
        if score >= threshold:
            rows.append(
                {
                    "tweet_id_left": min(left_id, right_id),
                    "tweet_id_right": max(left_id, right_id),
                    "jaccard": score,
                }
            )

    frame = pd.DataFrame(rows, columns=["tweet_id_left", "tweet_id_right", "jaccard"])
    if not frame.empty:
        frame = frame.sort_values(["tweet_id_left", "tweet_id_right"]).reset_index(drop=True)

    metrics = {
        "pairs_considered": pairs_considered,
        "positive_pairs": int(len(frame)),
        "runtime_seconds": round(time.perf_counter() - started_at, 6),
    }
    return frame, metrics


def verify_candidate_pairs(
    shingles_lookup: dict[int, set[str]],
    candidate_pairs: pd.DataFrame,
    threshold: float,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    started_at = time.perf_counter()
    rows: list[dict[str, float | int]] = []

    for row in candidate_pairs[["tweet_id_left", "tweet_id_right"]].itertuples(index=False):
        left_id = int(row.tweet_id_left)
        right_id = int(row.tweet_id_right)
        score = jaccard_similarity(shingles_lookup[left_id], shingles_lookup[right_id])
        if score >= threshold:
            rows.append(
                {
                    "tweet_id_left": left_id,
                    "tweet_id_right": right_id,
                    "jaccard": score,
                }
            )

    frame = pd.DataFrame(rows, columns=["tweet_id_left", "tweet_id_right", "jaccard"])
    if not frame.empty:
        frame = frame.sort_values(["tweet_id_left", "tweet_id_right"]).reset_index(drop=True)

    metrics = {
        "candidate_pairs_checked": int(len(candidate_pairs)),
        "verified_pairs": int(len(frame)),
        "runtime_seconds": round(time.perf_counter() - started_at, 6),
    }
    return frame, metrics
