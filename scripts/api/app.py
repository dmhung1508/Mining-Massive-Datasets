"""FastAPI backend for the Social LSH demo artifacts.

Endpoints:
- GET  /health
- GET  /metrics
- GET  /clusters/top?limit=10
- GET  /search?text=...&top_k=5&min_jaccard=0.8
- GET  /broadcast?top_n=5         -> news broadcast segments (script + images)
- POST /tts                       -> {"text": "..."} returns WAV audio
- GET  /news-image/{cluster_id}   -> generated illustration for a cluster
"""
from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from social_lsh.artifacts import artifact_path, ensure_artifact_dir, read_dataframe, read_metrics
from social_lsh.broadcast_cache import (
    broadcast_cache_id,
    image_file,
    legacy_image_file,
    load_metadata,
    public_image_path,
    save_metadata,
    source_signature,
)
from social_lsh.constants import DEFAULT_ARTIFACT_DIR, DEFAULT_SEED, OUTPUT_ROOT, REPO_ROOT
from social_lsh.news import build_broadcast_segments, build_news_object, build_image_prompt
from social_lsh.runtime_check import build_run_preflight
from social_lsh.search import prepare_search_index, search_similar_tweets
from social_lsh.tts import TTSClient, TTSError
from social_lsh.images import ImageError, OpenAIImageClient

# Load .env so API_KEY (Grok), API_VEO (media), TTS_URL, etc. are available.
load_dotenv(REPO_ROOT / ".env")


def _resolve_artifact_dir() -> Path:
    raw = os.getenv("SOCIAL_LSH_ARTIFACT_DIR", str(DEFAULT_ARTIFACT_DIR))
    return ensure_artifact_dir(Path(raw))


def _resolve_seed() -> int:
    try:
        return int(os.getenv("SOCIAL_LSH_SEED", str(DEFAULT_SEED)))
    except ValueError:
        return DEFAULT_SEED


ARTIFACT_DIR = _resolve_artifact_dir()
LATEST_ARTIFACT_DIR = OUTPUT_ROOT / "lsh_latest"
SEED = _resolve_seed()
NEWS_DIR = REPO_ROOT / "jupyter" / "output" / "news"
IMAGES_DIR = NEWS_DIR / "images"
AUDIO_DIR = NEWS_DIR / "audio"

app = FastAPI(
    title="Social LSH API",
    description="Near-duplicate detection, similar-post search, and AI news broadcast.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_tts_client = TTSClient.from_env()


@app.on_event("startup")
def _warm_search_index() -> None:
    # Build the LSH bucket index once at startup so the first /search is fast.
    try:
        prepare_search_index(artifact_dir=ARTIFACT_DIR, seed=SEED)
    except FileNotFoundError:
        # Pipeline artifacts not generated yet; /search will report the issue.
        pass


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "artifact_dir": str(ARTIFACT_DIR)}


@app.get("/metrics")
def metrics() -> dict[str, Any]:
    return read_metrics(ARTIFACT_DIR / "metrics.json")


@app.get("/clusters/top")
def clusters_top(limit: int = Query(10, ge=1, le=200)) -> list[dict[str, Any]]:
    try:
        clusters = read_dataframe(artifact_path("clusters", ARTIFACT_DIR))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="clusters artifact not found. Run the pipeline first.") from exc

    top = (
        clusters[["cluster_id", "cluster_size"]]
        .drop_duplicates()
        .sort_values(["cluster_size", "cluster_id"], ascending=[False, True])
        .head(limit)
    )
    return top.to_dict(orient="records")


@app.get("/search")
def search(
    text: str = Query(..., min_length=1, description="Query text to find similar posts."),
    top_k: int = Query(5, ge=1, le=50),
    min_jaccard: float = Query(0.0, ge=0.0, le=1.0),
) -> dict[str, Any]:
    try:
        results, metadata = search_similar_tweets(
            query_text=text,
            artifact_dir=ARTIFACT_DIR,
            top_k=top_k,
            min_jaccard=min_jaccard,
            seed=SEED,
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail="Search artifacts not found. Run the pipeline (run_lsh + verify_and_cluster) first.",
        ) from exc
    return {"metadata": metadata, "results": results.to_dict(orient="records")}


class TTSRequest(BaseModel):
    text: str


@app.post("/tts")
def tts(request: TTSRequest) -> Response:
    """Synthesise speech for a piece of text and return WAV audio."""
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="text must be non-empty")
    try:
        audio = _tts_client.synthesize(request.text)
    except (TTSError, ValueError) as exc:
        raise HTTPException(status_code=502, detail=f"TTS failed: {exc}") from exc
    return Response(content=audio, media_type="audio/wav")


def _audio_file_for_text(text: str) -> Path:
    digest = hashlib.blake2b(text.strip().encode("utf-8"), digest_size=12).hexdigest()
    return AUDIO_DIR / f"segment_{digest}.wav"


def _attach_cached_audio(segments: list[dict[str, Any]], force: bool = False) -> int:
    """Synthesize every spoken segment once and attach /news-audio paths."""
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    created = 0
    for segment in segments:
        text = str(segment.get("text") or "").strip()
        if not text:
            continue
        audio_file = _audio_file_for_text(text)
        if force or not audio_file.exists():
            try:
                audio_file.write_bytes(_tts_client.synthesize(text))
                created += 1
            except (TTSError, ValueError) as exc:
                print(f"  [warn] audio generation failed for segment {segment.get('kind')}: {exc}")
                continue
        segment["audio_path"] = f"/news-audio/{audio_file.name}"
    return created


def _artifact_has_broadcast_inputs(artifact_dir: Path) -> bool:
    return artifact_path("clusters", artifact_dir).exists() and artifact_path("scale_shingles", artifact_dir).exists()


def _broadcast_artifact_dir(source: str = "auto") -> Path:
    if source not in {"auto", "latest", "full"}:
        raise HTTPException(status_code=400, detail="source must be one of: auto, latest, full")
    if source == "full":
        return ARTIFACT_DIR
    if source == "latest":
        if not _artifact_has_broadcast_inputs(LATEST_ARTIFACT_DIR):
            raise HTTPException(
                status_code=404,
                detail="Latest artifacts not found. Run scripts/pipeline/refresh_latest_clusters.py first.",
            )
        return LATEST_ARTIFACT_DIR
    return LATEST_ARTIFACT_DIR if _artifact_has_broadcast_inputs(LATEST_ARTIFACT_DIR) else ARTIFACT_DIR


def _load_scale_for_news(needed_ids: set[int] | None = None, artifact_dir: Path | None = None) -> pd.DataFrame:
    """Load only the rows/columns the broadcast needs.

    To stay light on memory against the multi-GB lsh_full artifacts, we read the
    parquet row-group by row-group and keep only the columns we need (text,
    topic_label) and only the rows whose tweet_id is in `needed_ids`. The big
    shingle/token columns are never read.
    """
    import pyarrow.parquet as pq

    root = artifact_dir or ARTIFACT_DIR
    path = str(artifact_path("scale_shingles", root))
    pf = pq.ParquetFile(path)
    available = pf.schema_arrow.names
    wanted = [
        c
        for c in ["tweet_id", "text", "topic_label", "timestamp", "source", "shingle_count"]
        if c in available
    ]

    if needed_ids is None:
        return pf.read(columns=wanted).to_pandas()

    parts: list[pd.DataFrame] = []
    for batch in pf.iter_batches(batch_size=50_000, columns=wanted):
        chunk = batch.to_pandas()
        if "tweet_id" in chunk.columns:
            chunk = chunk[chunk["tweet_id"].isin(needed_ids)]
        if not chunk.empty:
            parts.append(chunk)
    if not parts:
        return pd.DataFrame(columns=wanted)
    return pd.concat(parts, ignore_index=True)


_TOP_CLUSTERS_CACHE: dict[str, Any] = {}
_BROADCAST_CACHE: dict[tuple, list[dict[str, Any]]] = {}


def _source_cache_token(artifact_dir: Path | None = None) -> tuple[tuple[Any, Any], tuple[Any, Any]]:
    root = artifact_dir or ARTIFACT_DIR
    sig = source_signature(root)
    clusters = sig.get("clusters") or {}
    scale = sig.get("scale_shingles") or {}
    return (
        (clusters.get("size_bytes"), clusters.get("mtime_ns")),
        (scale.get("size_bytes"), scale.get("mtime_ns")),
    )


def _top_clusters_frame(top_n: int, artifact_dir: Path | None = None):
    """Read and rank the top clusters once, then cache (clusters.parquet is large)."""
    root = artifact_dir or ARTIFACT_DIR
    token = _source_cache_token(root)
    key = ("ranked", str(root), token)
    members_key = ("members", str(root), token)
    cached = _TOP_CLUSTERS_CACHE.get(key)
    if cached is None:
        clusters = read_dataframe(artifact_path("clusters", root))
        cached = clusters[["cluster_id", "cluster_size"]].drop_duplicates()
        if root.name == "lsh_latest":
            scale_meta = _load_scale_for_news(artifact_dir=root)
            if "timestamp" in scale_meta.columns:
                timeline = clusters[["tweet_id", "cluster_id"]].merge(
                    scale_meta[["tweet_id", "timestamp"]],
                    on="tweet_id",
                    how="left",
                )
                timeline["timestamp"] = pd.to_datetime(timeline["timestamp"], errors="coerce")
                recency = (
                    timeline.groupby("cluster_id", as_index=False)["timestamp"]
                    .max()
                    .rename(columns={"timestamp": "latest_timestamp"})
                )
                cached = cached.merge(recency, on="cluster_id", how="left")
                cached = cached.sort_values(
                    ["latest_timestamp", "cluster_size", "cluster_id"],
                    ascending=[False, False, True],
                )
            else:
                cached = cached.sort_values(["cluster_size", "cluster_id"], ascending=[False, True])
        else:
            cached = cached.sort_values(["cluster_size", "cluster_id"], ascending=[False, True])
        cached = cached.reset_index(drop=True)
        _TOP_CLUSTERS_CACHE[key] = cached
        _TOP_CLUSTERS_CACHE[members_key] = clusters[["tweet_id", "cluster_id"]]
    return cached.head(top_n), _TOP_CLUSTERS_CACHE[members_key]


def _balanced_latest_clusters(ranked: pd.DataFrame, merged: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Interleave topics for a news bulletin instead of showing one topic only."""
    if ranked.empty or "topic_label" not in merged.columns:
        return ranked.head(top_n)

    topic_by_cluster = (
        merged.dropna(subset=["topic_label"])
        .groupby("cluster_id")["topic_label"]
        .agg(lambda values: values.astype(str).mode().iloc[0] if not values.astype(str).mode().empty else None)
        .reset_index()
    )
    enriched = ranked.merge(topic_by_cluster, on="cluster_id", how="left")
    if "latest_timestamp" in enriched.columns:
        enriched = enriched.sort_values(
            ["latest_timestamp", "cluster_size", "cluster_id"],
            ascending=[False, False, True],
        )

    topic_order = [
        topic
        for topic in enriched["topic_label"].dropna().astype(str).drop_duplicates().tolist()
        if topic and topic.lower() != "unknown"
    ]
    if not topic_order:
        return ranked.head(top_n)

    buckets = {
        topic: enriched.loc[enriched["topic_label"].eq(topic)].reset_index(drop=True)
        for topic in topic_order
    }
    cursors = {topic: 0 for topic in topic_order}
    selected_rows: list[pd.Series] = []
    selected_ids: set[int] = set()

    while len(selected_rows) < top_n:
        progressed = False
        for topic in topic_order:
            bucket = buckets[topic]
            cursor = cursors[topic]
            while cursor < len(bucket) and int(bucket.iloc[cursor]["cluster_id"]) in selected_ids:
                cursor += 1
            cursors[topic] = cursor
            if cursor >= len(bucket):
                continue
            row = bucket.iloc[cursor]
            selected_rows.append(row)
            selected_ids.add(int(row["cluster_id"]))
            cursors[topic] += 1
            progressed = True
            if len(selected_rows) >= top_n:
                break
        if not progressed:
            break

    if len(selected_rows) < top_n:
        for row in enriched.itertuples(index=False):
            cluster_id = int(getattr(row, "cluster_id"))
            if cluster_id in selected_ids:
                continue
            selected_rows.append(pd.Series(row._asdict()))
            selected_ids.add(cluster_id)
            if len(selected_rows) >= top_n:
                break

    selected = pd.DataFrame(selected_rows)
    return selected[[column for column in ranked.columns if column in selected.columns]].reset_index(drop=True)


def _build_news_items(
    top_n: int,
    samples_per_cluster: int,
    use_llm: bool,
    force: bool = False,
    source: str = "auto",
) -> tuple[list[dict[str, Any]], Path]:
    """Build news objects from the top clusters, attaching cached images if present.

    Reuse order: in-memory cache -> metadata manifest -> generate fresh.
    Set `force=True` to regenerate and overwrite the saved script.
    """
    artifact_dir = _broadcast_artifact_dir(source)
    cache_id = broadcast_cache_id(artifact_dir, top_n, samples_per_cluster, use_llm)
    cache_key = (cache_id, top_n, samples_per_cluster, use_llm, _source_cache_token(artifact_dir))
    if not force and cache_key in _BROADCAST_CACHE:
        return _BROADCAST_CACHE[cache_key], artifact_dir

    if not force:
        meta = load_metadata(artifact_dir, top_n, samples_per_cluster, use_llm)
        if meta:
            items = meta.get("items") or []
            _BROADCAST_CACHE[cache_key] = items
            return items, artifact_dir

    ranked_pool, all_members = _top_clusters_frame(max(top_n * 8, top_n), artifact_dir=artifact_dir)
    pool_cluster_ids = set(ranked_pool["cluster_id"].tolist())
    pool_members = all_members[all_members["cluster_id"].isin(pool_cluster_ids)]
    needed_ids = set(pool_members["tweet_id"].astype(int).tolist())

    scale = _load_scale_for_news(needed_ids=needed_ids, artifact_dir=artifact_dir)
    pool_merged = pool_members.merge(scale, on="tweet_id", how="left")
    ranked = (
        _balanced_latest_clusters(ranked_pool, pool_merged, top_n)
        if artifact_dir.name == "lsh_latest"
        else ranked_pool.head(top_n)
    )
    top_cluster_ids = set(ranked["cluster_id"].tolist())
    merged = pool_merged[pool_merged["cluster_id"].isin(top_cluster_ids)]

    # Prepare per-cluster inputs first.
    cluster_inputs = []
    for cluster in ranked.itertuples(index=False):
        members = merged.loc[merged["cluster_id"].eq(cluster.cluster_id)]
        if "shingle_count" in members.columns:
            members = members.sort_values(["shingle_count", "tweet_id"], ascending=[False, True])
        members = members.head(samples_per_cluster)
        sample_texts = members["text"].astype(str).tolist()
        topic_label = None
        if "topic_label" in members.columns:
            labels = members["topic_label"].dropna().astype(str)
            topic_label = labels.iloc[0] if not labels.empty else None
        cluster_inputs.append((int(cluster.cluster_id), int(cluster.cluster_size), sample_texts, topic_label))

    def _make(args):
        cid, csize, texts, tlabel = args
        news = build_news_object(
            cluster_id=cid,
            cluster_size=csize,
            sample_texts=texts,
            topic_label=tlabel,
            use_llm=use_llm,
            max_posts=samples_per_cluster,
        )
        record = news.as_dict()
        record["image_prompt"] = build_image_prompt(news)
        image_file = IMAGES_DIR / f"cluster_{cid}.png"
        if image_file.exists():
            record["image_path"] = f"/news-image/{cid}"
        return record

    if use_llm and len(cluster_inputs) > 1:
        # The LLM calls are independent network requests — run them concurrently.
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=min(6, len(cluster_inputs))) as pool:
            items = list(pool.map(_make, cluster_inputs))
    else:
        items = [_make(args) for args in cluster_inputs]

    segments = build_broadcast_segments(items)
    meta = save_metadata(artifact_dir, top_n, samples_per_cluster, use_llm, items, segments)
    items = meta["items"]
    _BROADCAST_CACHE[cache_key] = items
    return items, artifact_dir


@app.get("/broadcast")
def broadcast(
    top_n: int = Query(5, ge=1, le=20),
    samples_per_cluster: int = Query(12, ge=1, le=40),
    use_llm: bool = Query(False, description="Use Grok to write the script (needs API_KEY)."),
    source: str = Query("auto", description="auto uses latest-window clusters when available; full uses historical clusters."),
) -> dict[str, Any]:
    """Return ordered broadcast segments for the avatar to read aloud."""
    try:
        items, artifact_dir = _build_news_items(top_n, samples_per_cluster, use_llm, source=source)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail="Pipeline artifacts not found. Run the LSH pipeline first.",
        ) from exc
    segments = build_broadcast_segments(items)
    _attach_cached_audio(segments, force=False)
    return {"news": items, "segments": segments, "artifact_dir": str(artifact_dir), "source": artifact_dir.name}


def _generate_one_image(client, item: dict[str, Any], cache_id: str) -> bool:
    """Generate one cluster image with OpenAI. Returns True if a new image was created."""
    cluster_id = item.get("cluster_id")
    target = image_file(cache_id, cluster_id)
    if target.exists():
        item["image_path"] = public_image_path(cache_id, cluster_id)
        return False

    legacy = legacy_image_file(cluster_id)
    if legacy.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(legacy, target)
        item["image_path"] = public_image_path(cache_id, cluster_id)
        return False

    prompt = item.get("image_prompt") or item.get("scene")
    if not prompt:
        return False
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        client.generate_to_file(prompt, target, size="1024x1024", timeout=45.0)
        item["image_path"] = public_image_path(cache_id, cluster_id)
        return True
    except Exception as exc:
        print(f"  [warn] image generation failed for cluster {cluster_id}: {type(exc).__name__}: {exc}")
        return _generate_fallback_image(item, target, cache_id)


def _wrap_text(text: str, max_chars: int) -> list[str]:
    words = str(text).split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = " ".join(current + [word])
        if len(candidate) > max_chars and current:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return lines


def _generate_fallback_image(item: dict[str, Any], target: Path, cache_id: str) -> bool:
    """Create a local news-card PNG when the remote image API is slow/unavailable."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return False

    cluster_id = item.get("cluster_id")
    topic = str(item.get("topic") or "news").replace("_", " ").upper()
    headline = str(item.get("headline") or item.get("summary") or "Latest update")
    summary = str(item.get("summary") or item.get("representative_text") or "")

    width = height = 1024
    image = Image.new("RGB", (width, height), "#08111f")
    draw = ImageDraw.Draw(image)
    for y in range(height):
        blue = int(34 + 70 * (y / height))
        red = int(8 + 18 * (1 - y / height))
        draw.line([(0, y), (width, y)], fill=(red, 24, blue))
    for x in range(-200, width, 120):
        draw.line([(x, 0), (x + 420, height)], fill=(30, 95, 150), width=3)

    try:
        title_font = ImageFont.truetype("arialbd.ttf", 58)
        body_font = ImageFont.truetype("arial.ttf", 34)
        tag_font = ImageFont.truetype("arialbd.ttf", 30)
    except OSError:
        title_font = body_font = tag_font = ImageFont.load_default()

    draw.rounded_rectangle((64, 66, 960, 158), radius=24, fill=(190, 26, 45))
    draw.text((94, 92), "AI NEWS 24", fill="white", font=tag_font)
    draw.text((690, 92), topic[:22], fill=(255, 232, 170), font=tag_font)

    y = 250
    for line in _wrap_text(headline, 28)[:4]:
        draw.text((86, y), line, fill="white", font=title_font)
        y += 70

    y += 24
    for line in _wrap_text(summary, 48)[:5]:
        draw.text((90, y), line, fill=(210, 226, 244), font=body_font)
        y += 46

    draw.rounded_rectangle((70, 870, 954, 944), radius=20, outline=(130, 205, 255), width=3)
    draw.text((96, 890), "Generated fallback visual - remote image API timed out", fill=(180, 225, 255), font=tag_font)

    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target, format="PNG")
    item["image_path"] = public_image_path(cache_id, cluster_id)
    return True


def _generate_images_for(items: list[dict[str, Any]], cache_id: str, quality: str = "low") -> int:
    """Generate any missing cluster illustrations in parallel (OpenAI Images API)."""
    try:
        client = OpenAIImageClient.from_env()
    except ImageError:
        return 0  # No API key for images; skip silently.

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=min(2, max(1, len(items)))) as pool:
        results = list(pool.map(lambda it: _generate_one_image(client, it, cache_id), items))
    return sum(1 for r in results if r)


@app.get("/preflight")
def preflight(
    top_n: int = Query(5, ge=1, le=20),
    samples_per_cluster: int = Query(12, ge=1, le=40),
    use_llm: bool = Query(True),
    images: bool = Query(False),
    source: str = Query("auto"),
) -> dict[str, Any]:
    """Check local data/cache/resources before preparing a broadcast."""
    artifact_dir = _broadcast_artifact_dir(source)
    return build_run_preflight(artifact_dir, top_n, samples_per_cluster, use_llm, images)


@app.get("/prepare")
def prepare(
    top_n: int = Query(5, ge=1, le=20),
    samples_per_cluster: int = Query(12, ge=1, le=40),
    use_llm: bool = Query(True, description="Use Grok to write the script (needs API_KEY)."),
    images: bool = Query(False, description="Generate missing images now (slow). Prefer pre-generating offline."),
    audio: bool = Query(True, description="Pre-generate TTS audio so playback can start instantly."),
    force: bool = Query(False, description="Regenerate the script even if a saved one exists."),
    allow_unsafe: bool = Query(False, description="Continue even if preflight says danger."),
    source: str = Query("auto", description="auto uses latest-window clusters when available; full uses historical clusters."),
) -> dict[str, Any]:
    """Do the heavy work BEFORE the anchor speaks.

    Builds the news script (Grok, logged to log.txt), saves a metadata manifest
    under jupyter/output/news/metadata/ so later runs reuse it, and attaches any
    pre-generated images. By default it does NOT generate images on the fly.
    """
    artifact_dir = _broadcast_artifact_dir(source)
    check = build_run_preflight(artifact_dir, top_n, samples_per_cluster, use_llm, images)
    if check["status"] == "danger" and not allow_unsafe:
        raise HTTPException(
            status_code=409,
            detail={
                "message": "Preflight check failed. Fix the data/resource issue or retry with allow_unsafe=true.",
                "preflight": check,
            },
        )

    cache_id = broadcast_cache_id(artifact_dir, top_n, samples_per_cluster, use_llm)
    try:
        items, artifact_dir = _build_news_items(
            top_n,
            samples_per_cluster,
            use_llm,
            force=force,
            source=source,
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail="Pipeline artifacts not found. Run the LSH pipeline first.",
        ) from exc

    images_created = 0
    if images:
        images_created = _generate_images_for(items, cache_id)

    segments = build_broadcast_segments(items)
    audio_created = _attach_cached_audio(segments, force=force) if audio else 0
    meta = save_metadata(artifact_dir, top_n, samples_per_cluster, use_llm, items, segments)
    items = meta["items"]
    segments = meta["segments"]
    return {
        "news": items,
        "segments": segments,
        "images_created": images_created,
        "audio_created": audio_created,
        "used_llm": use_llm,
        "cache_hit": check["cache"]["hit"] and not force,
        "cache_id": cache_id,
        "metadata_path": check["cache"]["metadata_path"],
        "preflight": check,
        "artifact_dir": str(artifact_dir),
        "source": artifact_dir.name,
    }


@app.get("/news-image/{cache_id}/{cluster_id}")
def cached_news_image(cache_id: str, cluster_id: int) -> FileResponse:
    cached = image_file(cache_id, cluster_id)
    if not cached.exists():
        raise HTTPException(status_code=404, detail="image not generated for this cache")
    return FileResponse(str(cached), media_type="image/png")


@app.get("/news-image/{cluster_id}")
def news_image(cluster_id: int) -> FileResponse:
    cached = legacy_image_file(cluster_id)
    if not cached.exists():
        raise HTTPException(status_code=404, detail="image not generated for this cluster")
    return FileResponse(str(cached), media_type="image/png")


@app.get("/news-audio/{filename}")
def news_audio(filename: str) -> FileResponse:
    audio_file = AUDIO_DIR / filename
    if not audio_file.exists() or audio_file.suffix.lower() != ".wav":
        raise HTTPException(status_code=404, detail="audio not generated for this segment")
    return FileResponse(str(audio_file), media_type="audio/wav")


# --- Static mounts: Live2D avatar assets + the anchor web UI ---------------
# Mounted last so they don't shadow the API routes above.
_AVATAR_DIR = REPO_ROOT / "Ami"
_ANCHOR_DIR = REPO_ROOT / "dashboard" / "anchor"

if _AVATAR_DIR.exists():
    app.mount("/avatar", StaticFiles(directory=str(_AVATAR_DIR)), name="avatar")
if _ANCHOR_DIR.exists():
    app.mount("/", StaticFiles(directory=str(_ANCHOR_DIR), html=True), name="anchor")
