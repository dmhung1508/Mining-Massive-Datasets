from __future__ import annotations

import pandas as pd

from social_lsh.constants import LSHConfig
from social_lsh.minhash import evaluate_candidates, run_config
from social_lsh.similarity import exact_jaccard_pairs, jaccard_similarity


def test_jaccard_similarity_matches_expected_value() -> None:
    left = {"a", "b", "c"}
    right = {"b", "c", "d"}
    assert jaccard_similarity(left, right) == 0.5


def test_exact_jaccard_pairs_finds_only_high_similarity_pairs() -> None:
    frame = pd.DataFrame(
        {
            "tweet_id": [1, 2, 3],
            "shingles": [
                ["alpha beta gamma", "beta gamma delta"],
                ["alpha beta gamma", "beta gamma delta"],
                ["different tokens here"],
            ],
        }
    )

    pairs, metrics = exact_jaccard_pairs(frame, threshold=0.8)

    assert metrics["positive_pairs"] == 1
    assert pairs[["tweet_id_left", "tweet_id_right"]].values.tolist() == [[1, 2]]


def test_lsh_candidate_generation_has_no_duplicate_or_self_pairs() -> None:
    frame = pd.DataFrame(
        {
            "tweet_id": [10, 20, 30],
            "shingles": [
                ["alpha beta gamma", "beta gamma delta"],
                ["alpha beta gamma", "beta gamma delta"],
                ["other shingles only"],
            ],
        }
    )

    candidates, _ = run_config(
        frame,
        config=LSHConfig(shingle_size=3, num_perm=128, bands=32, rows=4),
        seed=42,
    )

    pairs = candidates[["tweet_id_left", "tweet_id_right"]].values.tolist()
    assert pairs == [[10, 20]]
    assert evaluate_candidates(candidates, pd.DataFrame({"tweet_id_left": [10], "tweet_id_right": [20]}))["recall"] == 1.0
