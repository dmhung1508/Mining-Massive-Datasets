from __future__ import annotations

from social_lsh.news import (
    NewsObject,
    build_image_prompt,
    build_news_object,
    build_video_prompt,
    build_broadcast_segments,
    clean_tweet_text,
    extract_hashtags,
    pick_representative_text,
)
from social_lsh.veo import (
    GrokVideoConfig,
    ImageConfig,
    VeoConfig,
    extract_media_url,
    extract_video_url,
)


def test_clean_tweet_text_strips_urls_mentions_and_hashtag_symbols() -> None:
    raw = "Big strike near Kyiv @reporter #Russia #Ukraine https://t.co/abc"
    assert clean_tweet_text(raw) == "Big strike near Kyiv Russia Ukraine"


def test_extract_hashtags_dedupes_case_insensitively() -> None:
    texts = ["#Russia #Ukraine", "#russia attack", "#Bakhmut"]
    assert extract_hashtags(texts) == ["Russia", "Ukraine", "Bakhmut"]


def test_pick_representative_text_returns_central_post() -> None:
    # The central post shares the most vocabulary with the others.
    texts = [
        "Russia launched missile strikes on Kyiv overnight causing damage",
        "Missile strikes hit Kyiv overnight, Russia attack causes damage",
        "Completely unrelated post about cooking pasta at home",
    ]
    rep = pick_representative_text(texts)
    assert "kyiv" in rep.lower()
    assert "pasta" not in rep.lower()


def test_pick_representative_text_handles_empty() -> None:
    assert pick_representative_text([]) == ""
    assert pick_representative_text(["   ", "ab"]) == ""


def test_broadcast_uses_llm_analysis_when_present() -> None:
    items = [
        {
            "cluster_id": 5,
            "cluster_size": 8,
            "headline": "Giao tranh tại Bakhmut",
            "summary": "Pháo kích quanh Bakhmut.",
            "analysis": "Đây là diễn biến leo thang đáng chú ý tại miền đông Ukraine, cho thấy giao tranh chưa hạ nhiệt.",
            "topic": "russia_ukraine_war",
            "entities": ["Bakhmut"],
        }
    ]
    segs = build_broadcast_segments(items)
    story = next(s for s in segs if s["kind"] == "story")
    assert "leo thang" in story["text"]  # the analysis is spoken


def test_template_news_object_sets_war_scene() -> None:
    news = build_news_object(
        cluster_id=7,
        cluster_size=12,
        sample_texts=["Heavy shelling reported near Bakhmut #Ukraine #Russia"],
        topic_label="russia_ukraine_war",
        use_llm=False,
    )
    assert news.cluster_id == 7
    assert news.cluster_size == 12
    assert "Eastern European" in news.scene
    assert news.mood


def test_build_video_prompt_is_english_and_safe() -> None:
    news = NewsObject(
        cluster_id=1,
        cluster_size=3,
        headline="Shelling near Bakhmut",
        summary="Heavy artillery shelling was reported around Bakhmut overnight.",
        topic="russia_ukraine_war",
        entities=["Bakhmut", "Ukraine"],
        scene="war-affected town, smoke, overcast sky",
        mood="tense, somber",
    )
    prompt = build_video_prompt(news)
    assert "documentary" in prompt
    assert "no identifiable real faces" in prompt
    assert "Bakhmut" in prompt
    # The prompt must carry the actual story, not just a generic scene.
    assert "artillery shelling" in prompt


def test_prompts_drop_misleading_entities() -> None:
    news = NewsObject(
        cluster_id=9,
        cluster_size=4,
        headline="Reported war crimes",
        summary="Posts describe attacks on civilians in Ukraine.",
        topic="russia_ukraine_war",
        entities=["Anonymous", "Breaking", "Ukraine"],
        scene="war-affected town",
        mood="somber",
    )
    prompt = build_image_prompt(news)
    # 'Anonymous'/'Breaking' would push the model toward hacker/news-logo imagery.
    assert "Anonymous" not in prompt
    assert "Breaking" not in prompt
    assert "Ukraine" in prompt


def test_grok_video_config_validates_choices() -> None:
    assert GrokVideoConfig().as_dict() == {"duration": 6, "aspect_ratio": "16:9"}
    try:
        GrokVideoConfig(duration=7).as_dict()
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_prompt_softens_graphic_language() -> None:
    news = NewsObject(
        cluster_id=12,
        cluster_size=3,
        headline="Civilian harm reported",
        summary="Reports describe soldiers harming civilians in the region.",
        topic="russia_ukraine_war",
        entities=["Ukraine"],
        scene="war-affected town",
        mood="somber",
    )
    prompt = build_image_prompt(news)
    assert "killing" not in prompt.lower()
    assert "raping" not in prompt.lower()


def test_prompt_drops_literal_subject_for_minor_content() -> None:
    news = NewsObject(
        cluster_id=13,
        cluster_size=4,
        headline="War crimes against children reported",
        summary="Posts allege crimes against women and children.",
        topic="russia_ukraine_war",
        entities=["Ukraine"],
        scene="war-affected Eastern European town, distant smoke",
        mood="somber",
    )
    prompt = build_image_prompt(news)
    # The literal sensitive subject must not be depicted.
    assert "children" not in prompt.lower()
    assert "depicting:" not in prompt.lower()
    # But the scene context still drives the image.
    assert "war-affected" in prompt
    assert "Ukraine" in prompt


def test_veo_config_validates_choices() -> None:
    assert VeoConfig().as_dict() == {"size": "720p", "aspect_ratio": "16:9", "enhance_prompt": True}
    try:
        VeoConfig(size="999p").as_dict()
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_image_config_validates_choices() -> None:
    assert ImageConfig().as_dict() == {"background": "opaque", "quality": "low", "size": "1024x1024"}
    try:
        ImageConfig(quality="ultra").as_dict()
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_build_image_prompt_is_safe_still() -> None:
    news = NewsObject(
        cluster_id=2,
        cluster_size=5,
        headline="x",
        summary="y",
        topic="russia_ukraine_war",
        entities=["Kyiv"],
        scene="city street after a strike, smoke, dusk",
        mood="somber",
    )
    prompt = build_image_prompt(news)
    assert "still image" in prompt
    assert "no identifiable real faces" in prompt
    assert "Kyiv" in prompt


def test_extract_media_url_finds_png() -> None:
    payload = {"task_result": {"url": "https://cdn.yescale.vip/x.png"}}
    assert extract_media_url(payload) == "https://cdn.yescale.vip/x.png"
    assert extract_media_url({"task_result": {}}) is None


def test_extract_video_url_finds_mp4_anywhere() -> None:
    payload = {"task_result": {"data": {"videos": [{"url": "https://cdn.example.com/x/out.mp4"}]}}}
    assert extract_video_url(payload) == "https://cdn.example.com/x/out.mp4"
    assert extract_video_url({"task_result": {}}) is None
