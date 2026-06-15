"""Turn LSH narrative clusters into structured news objects and Veo prompts.

Pipeline position (between clustering and video generation):

    clusters + representative tweets
        -> news object {headline, summary, scene, mood, entities}   (LLM or template)
        -> Veo video prompt (English, cinematic, safe)

The LLM step uses the xAI Grok endpoint configured in .env (API_KEY, MODEL,
XAI_BASE_URL). If those are missing or the call fails, a deterministic template
fallback keeps the pipeline working offline.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

import requests

URL_RE = re.compile(r"https?://\S+")
HASHTAG_RE = re.compile(r"#(\w+)")
MENTION_RE = re.compile(r"@\w+")


@dataclass
class NewsObject:
    cluster_id: int
    cluster_size: int
    headline: str
    summary: str
    topic: str
    entities: list[str] = field(default_factory=list)
    scene: str = ""
    mood: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "cluster_size": self.cluster_size,
            "headline": self.headline,
            "summary": self.summary,
            "topic": self.topic,
            "entities": self.entities,
            "scene": self.scene,
            "mood": self.mood,
        }


def clean_tweet_text(text: str) -> str:
    """Strip URLs, mentions, and hashtag symbols so the LLM sees readable text."""
    cleaned = URL_RE.sub(" ", str(text))
    cleaned = MENTION_RE.sub(" ", cleaned)
    cleaned = HASHTAG_RE.sub(r"\1", cleaned)
    return " ".join(cleaned.split())


def extract_hashtags(texts: list[str]) -> list[str]:
    seen: list[str] = []
    for text in texts:
        for tag in HASHTAG_RE.findall(str(text)):
            tag_lower = tag.lower()
            if tag_lower not in {item.lower() for item in seen}:
                seen.append(tag)
    return seen


# Hashtags/words that mislead the image model away from the real story.
MISLEADING_ENTITIES = {
    "anonymous", "breaking", "news", "update", "live", "video", "watch",
    "retweet", "follow", "thread", "viral", "trending",
}


def _useful_entities(entities: list[str]) -> list[str]:
    return [e for e in entities if e.strip() and e.strip().lower() not in MISLEADING_ENTITIES]


def _subject_from_news(news: "NewsObject") -> str:
    """The actual story text the image/video should depict, softened for safety.

    If the story is too graphic to depict literally (e.g. crimes against
    minors), we drop the literal subject and let the scene/entities carry the
    context instead, so the media generator is not asked to render the abuse.
    """
    subject = (news.summary or news.headline or "").strip()
    if _is_too_sensitive(subject):
        return ""
    subject = _soften_sensitive_text(subject)
    if len(subject) > 280:
        subject = subject[:277].rstrip() + "..."
    return subject


# If any of these appear, do not depict the story literally.
_BLOCK_LITERAL = re.compile(
    r"\b(child|children|kid|minor|infant|baby|rape|raping|raped|sexual)\b",
    re.IGNORECASE,
)


def _is_too_sensitive(text: str) -> bool:
    return bool(_BLOCK_LITERAL.search(text))


# Graphic phrases that trip image content filters. Replace with neutral wording
# so the scene still reflects the conflict without explicit violence.
_SENSITIVE_REPLACEMENTS = [
    (re.compile(r"\braping\b", re.IGNORECASE), "harming"),
    (re.compile(r"\brape\b", re.IGNORECASE), "abuse"),
    (re.compile(r"\bkilling\b", re.IGNORECASE), "affecting"),
    (re.compile(r"\bkilled\b", re.IGNORECASE), "affected"),
    (re.compile(r"\bmassacre\b", re.IGNORECASE), "attack on"),
    (re.compile(r"\btorture\b", re.IGNORECASE), "mistreatment"),
    (re.compile(r"\bgore\b", re.IGNORECASE), ""),
    (re.compile(r"\bdead bodies\b", re.IGNORECASE), "destruction"),
    (re.compile(r"\bcorpses?\b", re.IGNORECASE), "destruction"),
    (re.compile(r"\bblood(y|shed)?\b", re.IGNORECASE), ""),
]


def _soften_sensitive_text(text: str) -> str:
    softened = text
    for pattern, replacement in _SENSITIVE_REPLACEMENTS:
        softened = pattern.sub(replacement, softened)
    return " ".join(softened.split())


# --- Prompt building -------------------------------------------------------

VIDEO_STYLE = (
    "photojournalistic news b-roll, cinematic documentary style, realistic lighting, "
    "shallow depth of field"
)
VIDEO_NEGATIVE = (
    "no on-screen text, no captions, no watermark, no logo, no readable signs, "
    "no identifiable real faces, no graphic gore, not cartoon, not hacker imagery, "
    "not computer screens unless the story is about hacking"
)


def build_video_prompt(news: "NewsObject") -> str:
    """Build an English video prompt that depicts the actual news story."""
    parts: list[str] = []
    subject = _subject_from_news(news)
    if subject:
        parts.append(f"News scene depicting: {subject}")
    if news.scene:
        parts.append(f"visual setting: {news.scene}")
    entities = _useful_entities(news.entities)
    if entities:
        parts.append("location/context: " + ", ".join(entities[:5]))
    if news.mood:
        parts.append(f"mood: {news.mood}")
    parts.append(VIDEO_STYLE)
    parts.append(VIDEO_NEGATIVE)
    return ". ".join(parts)


IMAGE_STYLE = (
    "realistic cinematic news editorial still image, photojournalistic, natural lighting, "
    "high detail"
)
IMAGE_NEGATIVE = (
    "no on-screen text, no captions, no watermark, no logo, no readable signs, "
    "no identifiable real faces, no graphic gore, not cartoon, not illustration, "
    "not hacker imagery, not computer screens unless the story is about hacking"
)


def build_image_prompt(news: "NewsObject") -> str:
    """Build an English image prompt that depicts the actual news story."""
    parts: list[str] = []
    subject = _subject_from_news(news)
    if subject:
        parts.append(f"News photo depicting: {subject}")
    if news.scene:
        parts.append(f"visual setting: {news.scene}")
    entities = _useful_entities(news.entities)
    if entities:
        parts.append("location/context: " + ", ".join(entities[:5]))
    if news.mood:
        parts.append(f"mood: {news.mood}")
    parts.append(IMAGE_STYLE)
    parts.append(IMAGE_NEGATIVE)
    return ". ".join(parts)


# --- LLM (Grok) summarisation ---------------------------------------------

SYSTEM_PROMPT = (
    "You are a news editor. You receive several near-duplicate social media posts "
    "that form one news cluster. Summarise them into a single neutral news item. "
    "Respond ONLY with compact JSON having keys: headline, summary, topic, entities, "
    "scene, mood. 'scene' must be an English VISUAL description for a video generator "
    "(places, objects, weather, time of day) and must NOT include real named people's "
    "faces. 'mood' is 2-3 English adjectives. 'entities' is a list of place/org names."
)


def _grok_settings() -> tuple[str, str, str] | None:
    api_key = (os.getenv("API_KEY") or "").strip()
    base_url = (os.getenv("XAI_BASE_URL") or "").strip()
    model = (os.getenv("MODEL") or "").strip()
    if not api_key or not base_url or not model:
        return None
    return api_key, base_url, model


def _summarise_with_grok(
    cluster_id: int,
    cluster_size: int,
    sample_texts: list[str],
    language: str,
) -> NewsObject | None:
    settings = _grok_settings()
    if settings is None:
        return None
    api_key, base_url, model = settings

    cleaned = [clean_tweet_text(text) for text in sample_texts if str(text).strip()]
    if not cleaned:
        return None

    user_prompt = (
        f"Language for headline and summary: {language}.\n"
        f"Posts in this cluster:\n- " + "\n- ".join(cleaned[:6])
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.4,
    }
    try:
        response = requests.post(
            f"{base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        parsed = _parse_json_object(content)
        if parsed is None:
            return None
        return NewsObject(
            cluster_id=cluster_id,
            cluster_size=cluster_size,
            headline=str(parsed.get("headline", "")).strip(),
            summary=str(parsed.get("summary", "")).strip(),
            topic=str(parsed.get("topic", "")).strip() or "unknown",
            entities=[str(item).strip() for item in parsed.get("entities", []) if str(item).strip()],
            scene=str(parsed.get("scene", "")).strip(),
            mood=str(parsed.get("mood", "")).strip(),
        )
    except (requests.RequestException, KeyError, ValueError):
        return None


def _parse_json_object(content: str) -> dict[str, Any] | None:
    content = content.strip()
    # Strip ```json fences if present.
    if content.startswith("```"):
        content = re.sub(r"^```[a-zA-Z]*\n?", "", content)
        content = re.sub(r"\n?```$", "", content)
    try:
        return json.loads(content)
    except ValueError:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except ValueError:
                return None
    return None


def _template_news_object(
    cluster_id: int,
    cluster_size: int,
    sample_texts: list[str],
    topic_label: str | None,
) -> NewsObject:
    cleaned = [clean_tweet_text(text) for text in sample_texts if str(text).strip()]
    lead = cleaned[0] if cleaned else "Developing story"
    headline = lead[:80].rstrip()
    summary = " ".join(cleaned[:2])[:280]
    entities = extract_hashtags(sample_texts)[:5]

    blob = " ".join(cleaned).lower()
    topic = (topic_label or "").strip().lower()

    # Infer topic from text when the label is missing.
    if not topic:
        if any(k in blob for k in ("ukraine", "russia", "putin", "kyiv", "bakhmut", "kremlin", "zelensky")):
            topic = "russia_ukraine_war"
        elif any(k in blob for k in ("iran", "tehran", "iranian", "us-iran", "hormuz")):
            topic = "us_iran_war"
        else:
            topic = "unknown"

    if "russia" in topic or "ukraine" in topic:
        scene = "war-affected Eastern European town, distant smoke, overcast sky, empty streets, military vehicles"
        mood = "tense, somber"
    elif "iran" in topic or "us_iran" in topic:
        scene = "middle eastern city skyline at dusk, news helicopter view, hazy light"
        mood = "uneasy, urgent"
    else:
        scene = "outdoor news scene related to the story, neutral daylight"
        mood = "neutral, informative"

    return NewsObject(
        cluster_id=cluster_id,
        cluster_size=cluster_size,
        headline=headline,
        summary=summary,
        topic=topic,
        entities=entities,
        scene=scene,
        mood=mood,
    )


def build_news_object(
    cluster_id: int,
    cluster_size: int,
    sample_texts: list[str],
    topic_label: str | None = None,
    language: str = "Vietnamese",
    use_llm: bool = True,
) -> NewsObject:
    """Build one news object for a cluster, preferring Grok and falling back to a template."""
    if use_llm:
        news = _summarise_with_grok(cluster_id, cluster_size, sample_texts, language)
        if news is not None and news.scene:
            return news
    return _template_news_object(cluster_id, cluster_size, sample_texts, topic_label)


# --- Broadcast script ------------------------------------------------------

_TOPIC_VI = {
    "russia_ukraine_war": "xung đột Nga - Ukraine",
    "us_iran_war": "căng thẳng Mỹ - Iran",
    "unknown": "tin tổng hợp",
}

# Vietnamese diacritic range — used to tell if text is already Vietnamese.
_VIETNAMESE_RE = re.compile(
    r"[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]",
    re.IGNORECASE,
)


def _vi_topic(topic: str) -> str:
    return _TOPIC_VI.get((topic or "").strip().lower(), "tin tổng hợp")


def _is_vietnamese(text: str) -> bool:
    return bool(_VIETNAMESE_RE.search(text or ""))


def _vi_story_text(index: int, item: dict[str, Any]) -> str:
    """Build a fully Vietnamese spoken paragraph for one cluster.

    If the headline/summary is already Vietnamese (e.g. Grok wrote it), read it.
    Otherwise describe the cluster in Vietnamese from topic + key entities, so
    the anchor never reads raw English tweets aloud.
    """
    headline = str(item.get("headline", "")).strip()
    summary = str(item.get("summary", "")).strip()
    topic_vi = _vi_topic(str(item.get("topic", "")))
    size = item.get("cluster_size")
    entities = [e for e in (item.get("entities") or []) if str(e).strip()]

    lead = f"Tin thứ {index}, liên quan đến {topic_vi}."

    if _is_vietnamese(headline) or _is_vietnamese(summary):
        body = headline or summary
        detail = summary if summary and summary != headline else ""
        parts = [lead, body.rstrip(".") + ".", detail]
    else:
        # English-only source: describe in Vietnamese using keywords.
        if entities:
            kw = ", ".join(entities[:4])
            body = f"Nhiều bài đăng đang lan truyền về chủ đề này, với các từ khóa nổi bật như {kw}."
        else:
            body = "Nhiều bài đăng tương tự đang lan truyền trên mạng xã hội về chủ đề này."
        parts = [lead, body]

    spoken = " ".join(p for p in parts if p).strip()
    if size:
        spoken += f" Hệ thống ghi nhận khoảng {int(size)} bài đăng gần như trùng lặp trong cụm này."
    return spoken


def build_broadcast_segments(
    news_items: list[dict[str, Any] | NewsObject],
    intro: str | None = None,
    outro: str | None = None,
) -> list[dict[str, Any]]:
    """Build an ordered list of spoken segments for a news broadcast.

    Each segment is {kind, text, cluster_id, image_path}. The UI reads one
    segment at a time: synthesise `text` with TTS, play it on the avatar, and
    show `image_path` if present.
    """
    items = [item.as_dict() if isinstance(item, NewsObject) else dict(item) for item in news_items]

    if intro is None:
        intro = (
            "Xin chào quý vị và các bạn. Đây là bản tin tổng hợp tự động "
            f"từ mạng xã hội, với {len(items)} câu chuyện nổi bật trong hôm nay."
        )
    if outro is None:
        outro = "Bản tin tổng hợp đến đây là kết thúc. Xin cảm ơn quý vị và các bạn đã theo dõi."

    segments: list[dict[str, Any]] = [{"kind": "intro", "text": intro, "cluster_id": None, "image_path": None}]

    for index, item in enumerate(items, start=1):
        segments.append(
            {
                "kind": "story",
                "text": _vi_story_text(index, item),
                "cluster_id": item.get("cluster_id"),
                "image_path": item.get("image_path") or item.get("video_path"),
            }
        )

    segments.append({"kind": "outro", "text": outro, "cluster_id": None, "image_path": None})
    return segments


def build_broadcast_script(news_items: list[dict[str, Any] | NewsObject]) -> str:
    """Flatten broadcast segments into a single spoken transcript."""
    return "\n".join(segment["text"] for segment in build_broadcast_segments(news_items))
