from __future__ import annotations

from social_lsh.news import NewsObject, build_broadcast_script, build_broadcast_segments
from social_lsh.tts import TTSClient, TTSError


def _news() -> list[NewsObject]:
    return [
        NewsObject(
            cluster_id=1,
            cluster_size=10,
            headline="Giao tranh tại Bakhmut",
            summary="Pháo kích dữ dội quanh Bakhmut.",
            topic="russia_ukraine_war",
            entities=["Bakhmut"],
            scene="war town",
            mood="tense",
        ),
        NewsObject(
            cluster_id=2,
            cluster_size=5,
            headline="Căng thẳng vùng Vịnh",
            summary="Diễn biến mới quanh eo biển Hormuz.",
            topic="us_iran_war",
            entities=["Hormuz"],
            scene="city skyline",
            mood="uneasy",
        ),
    ]


def test_broadcast_segments_have_intro_stories_and_outro() -> None:
    segments = build_broadcast_segments(_news())
    kinds = [s["kind"] for s in segments]
    assert kinds[0] == "intro"
    assert kinds[-1] == "outro"
    assert kinds.count("story") == 2
    # Story segments carry cluster ids in order.
    story_ids = [s["cluster_id"] for s in segments if s["kind"] == "story"]
    assert story_ids == [1, 2]


def test_broadcast_segment_text_mentions_topic_in_vietnamese() -> None:
    segments = build_broadcast_segments(_news())
    story = next(s for s in segments if s["kind"] == "story")
    assert "Nga" in story["text"] or "Ukraine" in story["text"]
    assert "Tin thứ 1" in story["text"]


def test_broadcast_script_is_multiline_transcript() -> None:
    script = build_broadcast_script(_news())
    assert "Xin chào" in script
    assert script.count("\n") >= 3  # intro + 2 stories + outro


def test_broadcast_accepts_dicts_with_image_path() -> None:
    items = [
        {"cluster_id": 7, "cluster_size": 3, "headline": "H", "summary": "S", "topic": "unknown", "image_path": "/news-image/7"},
    ]
    segments = build_broadcast_segments(items)
    story = next(s for s in segments if s["kind"] == "story")
    assert story["image_path"] == "/news-image/7"


def test_tts_client_rejects_empty_text() -> None:
    client = TTSClient()
    try:
        client.synthesize("   ")
        assert False, "expected ValueError"
    except ValueError:
        pass
