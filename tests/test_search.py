from __future__ import annotations

import pandas as pd

from social_lsh.constants import LSHConfig
from social_lsh.preprocessing import make_word_shingles, normalize_text, tokenize_text
from social_lsh.search import build_search_index


def test_build_search_index_tracks_bucket_count() -> None:
    frame = pd.DataFrame(
        {
            "tweet_id": [1, 2],
            "shingles": [
                ["alpha beta gamma", "beta gamma delta"],
                ["alpha beta gamma", "beta gamma delta"],
            ],
        }
    )
    index = build_search_index(
        frame,
        config=LSHConfig(shingle_size=3, num_perm=128, bands=16, rows=8),
        seed=42,
    )

    assert index["num_documents"] == 2
    assert index["bucket_count"] > 0


def test_query_preprocessing_for_search_produces_shingles() -> None:
    normalized = normalize_text("Breaking news from Ukraine front line report today https://t.co/abc")
    tokens = tokenize_text(normalized)
    shingles = make_word_shingles(tokens, 3)

    assert normalized == "breaking news from ukraine front line report today"
    assert tokens[:4] == ["breaking", "news", "from", "ukraine"]
    assert shingles[0] == "breaking news from"
