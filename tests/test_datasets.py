from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from uk_russia_lsh.datasets import build_combined_dataset, export_telegram_dataset


def _write_twitter_parquet(path: Path) -> None:
    frame = pd.DataFrame(
        [
            {
                "tweet_id": 1,
                "user_id": 100,
                "text": "Breaking news from Twitter",
                "timestamp": "2024-01-01T00:00:00Z",
                "date": "2024-01-01",
            },
            {
                "tweet_id": 2,
                "user_id": 101,
                "text": "Second tweet sample",
                "timestamp": "2024-01-02T00:00:00Z",
                "date": "2024-01-02",
            },
        ]
    )
    frame.to_parquet(path, index=False)


def _write_telegram_jsonl(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "message_id": 11,
            "channel_id": 12345,
            "text": "Telegram update from a channel",
            "timestamp": "2024-01-03T05:00:00+00:00",
            "forward_from": {"user_id": 77, "username": "origin_user"},
            "media": {"type": "photo"},
        },
        {
            "message_id": 12,
            "channel_id": 12345,
            "text": "   ",
            "timestamp": "2024-01-03T06:00:00+00:00",
            "forward_from": None,
            "media": None,
        },
    ]
    with (directory / "channel_12345_demo.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_export_telegram_dataset_from_jsonl(tmp_path: Path) -> None:
    local_data_dir = tmp_path / "telegram_raw"
    output_path = tmp_path / "telegram_messages.parquet"
    _write_telegram_jsonl(local_data_dir)

    metrics = export_telegram_dataset(
        output_path=output_path,
        source="jsonl",
        local_data_dir=local_data_dir,
        overwrite=True,
    )

    result = pd.read_parquet(output_path)

    assert metrics["telegram_rows"] == 1
    assert result["source"].tolist() == ["telegram"]
    assert result["text"].tolist() == ["Telegram update from a channel"]
    assert result["source_channel_id"].tolist() == ["12345"]
    assert result["media_type"].tolist() == ["photo"]


def test_build_combined_dataset_merges_twitter_and_telegram(tmp_path: Path) -> None:
    twitter_input = tmp_path / "tweets.parquet"
    local_data_dir = tmp_path / "telegram_raw"
    output_path = tmp_path / "combined_social.parquet"

    _write_twitter_parquet(twitter_input)
    _write_telegram_jsonl(local_data_dir)

    metrics = build_combined_dataset(
        output_path=output_path,
        twitter_input_path=twitter_input,
        telegram_source="jsonl",
        telegram_local_data_dir=local_data_dir,
        batch_size=1,
        overwrite=True,
    )

    result = pd.read_parquet(output_path)

    assert metrics["twitter_rows"] == 2
    assert metrics["telegram_rows"] == 1
    assert metrics["total_rows"] == 3
    assert set(result["source"].tolist()) == {"twitter", "telegram"}
    assert {"tweet_id", "user_id", "text", "timestamp", "date"} <= set(result.columns)
