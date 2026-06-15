from __future__ import annotations

import pandas as pd

from social_lsh.datasets import CANONICAL_COLUMNS, normalise_x_frame


def test_normalise_x_frame_real_schema() -> None:
    # Matches the news_monitoring schema: statusId, postedAt, account, monitorTopic.
    frame = pd.DataFrame(
        [
            {
                "statusId": "2065841595648283062",
                "text": "Iran United States war escalation in the Gulf",
                "postedAt": "2026-06-13T16:59:52.000+00:00",
                "account": "LarryBoorstein",
                "monitorTopic": "us_iran",
                "source": "x",
            }
        ]
    )
    out = normalise_x_frame(frame)
    assert list(out.columns) == CANONICAL_COLUMNS
    assert out.iloc[0]["source"] == "x"
    assert out.iloc[0]["source_item_id"] == "2065841595648283062"
    assert out.iloc[0]["source_user_id"] == "LarryBoorstein"
    assert out.iloc[0]["date"] == "2026-06-13"
    # monitorTopic drives the canonical topic label.
    assert out.iloc[0]["topic_label"] == "us_iran_war"


def test_normalise_x_frame_monitor_topic_russia() -> None:
    frame = pd.DataFrame(
        [
            {
                "statusId": "111",
                "text": "Russia Ukraine front line update",
                "postedAt": "2026-06-13T10:00:00+00:00",
                "account": "userB",
                "monitorTopic": "russia_ukraine",
            }
        ]
    )
    out = normalise_x_frame(frame)
    assert out.iloc[0]["topic_label"] == "russia_ukraine_war"


def test_normalise_x_frame_standard_twitter_schema() -> None:
    frame = pd.DataFrame(
        [
            {
                "tweetid": "1632277881693822977",
                "text": "Russia Ukraine war update near Bakhmut",
                "tweetcreatedts": "2023-03-05 07:12:27",
                "username": "reporterA",
            }
        ]
    )
    out = normalise_x_frame(frame, topic_label="russia_ukraine_war")
    assert list(out.columns) == CANONICAL_COLUMNS
    assert out.iloc[0]["source"] == "x"
    assert out.iloc[0]["text"] == "Russia Ukraine war update near Bakhmut"
    assert out.iloc[0]["date"] == "2023-03-05"
    assert out.iloc[0]["topic_label"] == "russia_ukraine_war"
    assert out.iloc[0]["source_item_id"] == "1632277881693822977"


def test_normalise_x_frame_alternate_schema() -> None:
    # Different field names (full_text/created_at/id_str/author_id).
    frame = pd.DataFrame(
        [
            {
                "id_str": "abc123",
                "full_text": "US Iran tensions rise in the Gulf",
                "created_at": "2024-01-10T05:00:00Z",
                "author_id": "999",
            }
        ]
    )
    out = normalise_x_frame(frame, topic_label="us_iran_war")
    assert out.iloc[0]["text"] == "US Iran tensions rise in the Gulf"
    assert out.iloc[0]["topic_label"] == "us_iran_war"
    assert out.iloc[0]["source_item_id"] == "abc123"


def test_normalise_x_frame_skips_empty_text_and_bad_timestamp() -> None:
    frame = pd.DataFrame(
        [
            {"tweetid": "1", "text": "   ", "created_at": "2024-01-01"},
            {"tweetid": "2", "text": "valid post", "created_at": "not-a-date"},
            {"tweetid": "3", "text": "kept post", "created_at": "2024-01-02T00:00:00Z"},
        ]
    )
    out = normalise_x_frame(frame)
    assert len(out) == 1
    assert out.iloc[0]["source_item_id"] == "3"


def test_normalise_x_frame_stable_ids_are_positive_int64() -> None:
    frame = pd.DataFrame(
        [{"tweetid": "1", "text": "post one", "created_at": "2024-01-02T00:00:00Z"}]
    )
    out = normalise_x_frame(frame)
    tid = int(out.iloc[0]["tweet_id"])
    uid = int(out.iloc[0]["user_id"])
    assert 0 < tid < 2**63
    assert 0 < uid < 2**63


def test_normalise_x_frame_empty() -> None:
    assert normalise_x_frame(pd.DataFrame()).empty
