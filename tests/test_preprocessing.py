from __future__ import annotations

import pandas as pd

from social_lsh.preprocessing import build_shingles, normalize_text, tokenize_text


def test_normalize_text_removes_urls_and_whitespace() -> None:
    raw = "Hello   WORLD\nhttps://example.com/test  "
    assert normalize_text(raw) == "hello world"


def test_tokenize_text_keeps_words_and_numbers() -> None:
    assert tokenize_text("ukraine war 2023!") == ["ukraine", "war", "2023"]


def test_build_shingles_filters_short_rows() -> None:
    frame = pd.DataFrame(
        [
            {"tweet_id": 1, "user_id": 10, "text": "alpha beta gamma delta", "timestamp": "2024-01-01", "date": "2024-01-01"},
            {"tweet_id": 2, "user_id": 11, "text": "tiny text", "timestamp": "2024-01-01", "date": "2024-01-01"},
        ]
    )

    result = build_shingles(frame, shingle_size=3)

    assert result["tweet_id"].tolist() == [1]
    assert result.iloc[0]["tokens"] == ["alpha", "beta", "gamma", "delta"]
    assert result.iloc[0]["shingles"] == ["alpha beta gamma", "beta gamma delta"]
