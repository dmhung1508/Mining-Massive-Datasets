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

import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

URL_RE = re.compile(r"https?://\S+")
HASHTAG_RE = re.compile(r"#(\w+)")
MENTION_RE = re.compile(r"@\w+")

# Where to log every LLM request/response. Override with SOCIAL_LSH_LOG.
LLM_LOG_PATH = os.getenv("SOCIAL_LSH_LOG", str(Path(__file__).resolve().parents[2] / "log.txt"))


def _log_llm(section: str, content: str) -> None:
    """Append an LLM interaction to the log file (best-effort, never raises)."""
    try:
        with open(LLM_LOG_PATH, "a", encoding="utf-8") as handle:
            stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            handle.write(f"\n===== {stamp} | {section} =====\n")
            handle.write(content.rstrip() + "\n")
    except OSError:
        pass


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
    representative_text: str = ""
    analysis: str = ""

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
            "representative_text": self.representative_text,
            "analysis": self.analysis,
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


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def pick_representative_text(sample_texts: list[str]) -> str:
    """Pick the central post of a cluster (the medoid by word overlap).

    The representative is the post that shares the most vocabulary with the
    others, i.e. the one that best captures what the whole cluster is about.
    Falls back to the longest cleaned post when there are too few samples.
    """
    cleaned = [clean_tweet_text(t) for t in sample_texts if str(t).strip()]
    cleaned = [c for c in cleaned if len(c) >= 10]
    if not cleaned:
        return ""
    if len(cleaned) <= 2:
        return max(cleaned, key=len)

    token_sets = [_tokens(c) for c in cleaned]
    best_idx, best_score = 0, -1.0
    for i, ti in enumerate(token_sets):
        if not ti:
            continue
        score = 0.0
        for j, tj in enumerate(token_sets):
            if i == j or not tj:
                continue
            inter = len(ti & tj)
            union = len(ti | tj)
            score += inter / union if union else 0.0
        if score > best_score:
            best_idx, best_score = i, score
    return cleaned[best_idx]


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


# --- LLM (Grok) analysis: RAG-style, plain-text output ---------------------

SYSTEM_PROMPT = (
    "Bạn là phát thanh viên thời sự. Dựa trên các bài đăng được cung cấp (cùng nói về "
    "một sự việc), hãy viết MỘT đoạn dẫn bản tin bằng tiếng Việt, văn phong trang trọng, "
    "trung lập, mạch lạc, khoảng 4 đến 6 câu. Nêu diễn biến chính, bối cảnh, các bên liên "
    "quan và vì sao đáng chú ý. Chỉ trả lời bằng đoạn văn thuần, KHÔNG dùng JSON, KHÔNG "
    "markdown, KHÔNG tiêu đề, KHÔNG gạch đầu dòng."
)


def _grok_settings() -> tuple[str, str, str] | None:
    """Read chat LLM settings, supporting OpenAI-style and legacy xAI vars."""
    api_key = (os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY") or "").strip()
    base_url = (os.getenv("base_url") or os.getenv("OPENAI_BASE_URL") or os.getenv("XAI_BASE_URL") or "").strip()
    model = (os.getenv("model") or os.getenv("MODEL") or "").strip()
    if not api_key or not base_url or not model:
        return None
    return api_key, base_url, model


def _infer_topic(text: str) -> str:
    blob = text.lower()
    if any(k in blob for k in ("ukrain", "russia", "putin", "kyiv", "kiev", "bakhmut", "kremlin", "zelensky", "moscow")):
        return "russia_ukraine_war"
    if any(k in blob for k in ("iran", "tehran", "hormuz", "us-iran", "iranian")):
        return "us_iran_war"
    return "unknown"


def _scene_for_topic(topic: str) -> tuple[str, str]:
    if "russia" in topic or "ukraine" in topic:
        return (
            "war-affected Eastern European town, distant smoke, overcast sky, empty streets, military vehicles",
            "tense, somber",
        )
    if "iran" in topic or "us_iran" in topic:
        return ("middle eastern city skyline at dusk, news helicopter view, hazy light", "uneasy, urgent")
    return ("outdoor news scene related to the story, neutral daylight", "neutral, informative")


def _summarise_with_grok(
    cluster_id: int,
    cluster_size: int,
    sample_texts: list[str],
    language: str,
    representative_text: str = "",
    max_posts: int = 12,
) -> NewsObject | None:
    settings = _grok_settings()
    if settings is None:
        return None
    api_key, base_url, model = settings

    cleaned = [clean_tweet_text(text) for text in sample_texts if str(text).strip()]
    central = representative_text or (pick_representative_text(sample_texts))
    if not central:
        return None

    # RAG-style: provide the related posts as context, ask for a plain paragraph.
    context_posts = [central] + [c for c in cleaned if c != central][: max(0, max_posts - 1)]
    context_block = "\n".join(f"- {p}" for p in context_posts)
    user_prompt = (
        "Các bài đăng cùng nói về một sự việc:\n"
        f"{context_block}\n\n"
        "Hãy viết đoạn dẫn bản tin tiếng Việt cho sự việc trên."
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    try:
        _log_llm(
            f"INPUT cluster_id={cluster_id} size={cluster_size} model={model} posts={len(context_posts)}",
            f"[SYSTEM]\n{SYSTEM_PROMPT}\n\n[USER]\n{user_prompt}",
        )
        analysis = _chat_completion(api_key, base_url, model, messages, max_tokens=600)
        _log_llm(f"OUTPUT cluster_id={cluster_id}", analysis)
        if not analysis:
            return None

        # Derive the rest locally so we don't pay the LLM for extra fields.
        topic = _infer_topic(central + " " + " ".join(cleaned[:5]))
        scene, mood = _scene_for_topic(topic)
        headline = analysis.split(".")[0].strip()[:90]
        return NewsObject(
            cluster_id=cluster_id,
            cluster_size=cluster_size,
            headline=headline,
            summary=analysis,
            topic=topic,
            entities=extract_hashtags(sample_texts)[:5],
            scene=scene,
            mood=mood,
            representative_text=central,
            analysis=analysis,
        )
    except (requests.RequestException, KeyError, ValueError) as exc:
        _log_llm(f"ERROR cluster_id={cluster_id}", f"{type(exc).__name__}: {exc}")
        return None


def _chat_completion(api_key: str, base_url: str, model: str, messages: list[dict], max_tokens: int = 600) -> str:
    """Call an OpenAI-compatible chat endpoint, tolerant to parameter differences.

    Newer OpenAI models (gpt-5.x) require `max_completion_tokens` and only accept
    the default temperature, while older/3rd-party endpoints use `max_tokens`.
    We try the modern form first, then fall back.
    """
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    base_payload = {"model": model, "messages": messages}

    attempts = [
        {**base_payload, "max_completion_tokens": max_tokens},
        {**base_payload, "max_tokens": max_tokens, "temperature": 0.5},
        {**base_payload},
    ]
    last_error = ""
    for payload in attempts:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        if resp.status_code == 200:
            return str(resp.json()["choices"][0]["message"]["content"]).strip()
        last_error = f"{resp.status_code}: {resp.text[:160]}"
    raise requests.RequestException(last_error)


def _template_news_object(
    cluster_id: int,
    cluster_size: int,
    sample_texts: list[str],
    topic_label: str | None,
) -> NewsObject:
    cleaned = [clean_tweet_text(text) for text in sample_texts if str(text).strip()]
    central = pick_representative_text(sample_texts)
    lead = central or (cleaned[0] if cleaned else "Developing story")
    headline = lead[:80].rstrip()
    summary = lead[:280]
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
        representative_text=central,
        analysis="",  # no offline analysis without an LLM
    )


def build_news_object(
    cluster_id: int,
    cluster_size: int,
    sample_texts: list[str],
    topic_label: str | None = None,
    language: str = "Vietnamese",
    use_llm: bool = True,
    max_posts: int = 12,
) -> NewsObject:
    """Build one news object for a cluster, preferring Grok and falling back to a template.

    The cluster's central (representative) post is selected first, then the LLM
    analyses it together with up to `max_posts` related posts from the cluster
    into a richer Vietnamese broadcast segment.
    """
    representative = pick_representative_text(sample_texts)
    if use_llm:
        news = _summarise_with_grok(
            cluster_id,
            cluster_size,
            sample_texts,
            language,
            representative_text=representative,
            max_posts=max_posts,
        )
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

    Reads only the news content (no technical stats like cluster size).

    Priority:
    1. LLM `analysis` of the central post (the real news content).
    2. Vietnamese headline/summary if available.
    3. A short Vietnamese line from topic + keywords as a last resort.
    """
    headline = str(item.get("headline", "")).strip()
    summary = str(item.get("summary", "")).strip()
    analysis = str(item.get("analysis", "")).strip()
    topic_vi = _vi_topic(str(item.get("topic", "")))
    entities = [e for e in (item.get("entities") or []) if str(e).strip()]

    # Vary the opener so consecutive stories don't sound identical.
    openers = ["Tiếp theo", "Đáng chú ý", "Trong một diễn biến khác", "Bên cạnh đó", "Một tin khác"]
    lead = "Tin nổi bật đầu tiên." if index == 1 else f"{openers[(index - 1) % len(openers)]}."

    if analysis and _is_vietnamese(analysis):
        head = headline if _is_vietnamese(headline) else ""
        parts = [lead, (head.rstrip(".") + "." if head else ""), analysis]
    elif _is_vietnamese(headline) or _is_vietnamese(summary):
        body = headline or summary
        detail = summary if summary and summary != headline else ""
        parts = [lead, body.rstrip(".") + ".", detail]
    else:
        # English-only source with no LLM: a concise Vietnamese line about the topic.
        if entities:
            kw = ", ".join(entities[:4])
            body = f"Mạng xã hội đang chú ý đến diễn biến {topic_vi}, nổi bật quanh {kw}."
        else:
            body = f"Mạng xã hội đang chú ý đến một diễn biến mới liên quan đến {topic_vi}."
        parts = [lead, body]

    return " ".join(p for p in parts if p).strip()


def _clip_for_speech(text: str, limit: int = 360) -> str:
    text = clean_tweet_text(text)
    text = _soften_sensitive_text(text)
    text = re.sub(r"\b(read more|breaking|update)\b[:：]?", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"[^\w\s.,;:'\"!?()/-]", " ", text)
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    clipped = text[:limit].rsplit(" ", 1)[0].rstrip(" ,.;:")
    return clipped + "..."


def _source_story_text(item: dict[str, Any]) -> str:
    """Return concrete story content from the cluster, not a generic topic line."""
    candidates = [
        str(item.get("summary", "")).strip(),
        str(item.get("headline", "")).strip(),
        str(item.get("representative_text", "")).strip(),
    ]
    cleaned: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        value = _clip_for_speech(candidate)
        key = value.lower()
        if any(key.startswith(existing.lower()) or existing.lower().startswith(key) for existing in cleaned):
            continue
        if value and key not in seen:
            cleaned.append(value)
            seen.add(key)
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    return f"{cleaned[0]} Chi tiết thêm: {cleaned[1]}"


def _vi_story_text(index: int, item: dict[str, Any]) -> str:
    """Build a news-style spoken paragraph that keeps the real story content."""
    headline = str(item.get("headline", "")).strip()
    summary = str(item.get("summary", "")).strip()
    analysis = str(item.get("analysis", "")).strip()
    topic_vi = _vi_topic(str(item.get("topic", "")))
    entities = [e for e in (item.get("entities") or []) if str(e).strip()]

    openers = [
        "Tin thứ nhất" if index == 1 else f"Tin thứ {index}",
        "Đáng chú ý",
        "Cập nhật mới nhất",
        "Trong một diễn biến khác",
        "Bên cạnh đó",
    ]
    lead = openers[min(index - 1, len(openers) - 1)] + "."

    if analysis and _is_vietnamese(analysis):
        head = headline if _is_vietnamese(headline) else ""
        parts = [lead, (head.rstrip(".") + "." if head else ""), analysis]
    elif _is_vietnamese(headline) or _is_vietnamese(summary):
        body = headline or summary
        detail = summary if summary and summary != headline else ""
        parts = [lead, body.rstrip(".") + ".", detail]
    else:
        story = _source_story_text(item)
        if story:
            parts = [lead, f"Nội dung chính về {topic_vi}: {story}"]
        else:
            kw = ", ".join(entities[:4])
            tail = f", với các từ khóa {kw}" if kw else ""
            parts = [lead, f"Có một diễn biến mới về {topic_vi}{tail}."]

    return " ".join(p for p in parts if p).strip()


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
            "Xin chào quý vị và các bạn. Sau đây là những diễn biến chính "
            "đang được quan tâm nhất hôm nay trên mạng xã hội."
        )
    if outro is None:
        outro = "Trên đây là những tin chính. Xin cảm ơn quý vị và các bạn đã theo dõi."

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
