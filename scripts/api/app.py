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

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from social_lsh.artifacts import artifact_path, ensure_artifact_dir, read_dataframe, read_metrics
from social_lsh.constants import DEFAULT_ARTIFACT_DIR, DEFAULT_SEED, REPO_ROOT
from social_lsh.news import build_broadcast_segments, build_news_object, build_image_prompt
from social_lsh.preprocessing import deserialize_nested_columns
from social_lsh.search import prepare_search_index, search_similar_tweets
from social_lsh.tts import TTSClient, TTSError


def _resolve_artifact_dir() -> Path:
    raw = os.getenv("SOCIAL_LSH_ARTIFACT_DIR", str(DEFAULT_ARTIFACT_DIR))
    return ensure_artifact_dir(Path(raw))


def _resolve_seed() -> int:
    try:
        return int(os.getenv("SOCIAL_LSH_SEED", str(DEFAULT_SEED)))
    except ValueError:
        return DEFAULT_SEED


ARTIFACT_DIR = _resolve_artifact_dir()
SEED = _resolve_seed()
NEWS_DIR = REPO_ROOT / "jupyter" / "output" / "news"
IMAGES_DIR = NEWS_DIR / "images"

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


def _load_scale_for_news() -> pd.DataFrame:
    frame = read_dataframe(artifact_path("scale_shingles", ARTIFACT_DIR))
    frame = deserialize_nested_columns(frame, ["tokens", "shingles"])
    columns = [c for c in ["tweet_id", "text", "topic_label", "shingle_count"] if c in frame.columns]
    return frame[columns].copy()


def _build_news_items(top_n: int, samples_per_cluster: int, use_llm: bool) -> list[dict[str, Any]]:
    """Build news objects from the top clusters, attaching cached images if present."""
    scale = _load_scale_for_news()
    clusters = read_dataframe(artifact_path("clusters", ARTIFACT_DIR))
    merged = clusters.merge(scale, on="tweet_id", how="left")
    ranked = (
        clusters[["cluster_id", "cluster_size"]]
        .drop_duplicates()
        .sort_values(["cluster_size", "cluster_id"], ascending=[False, True])
        .head(top_n)
    )

    items: list[dict[str, Any]] = []
    for cluster in ranked.itertuples(index=False):
        members = (
            merged.loc[merged["cluster_id"].eq(cluster.cluster_id)]
            .sort_values(["shingle_count", "tweet_id"], ascending=[False, True])
            .head(samples_per_cluster)
        )
        sample_texts = members["text"].astype(str).tolist()
        topic_label = None
        if "topic_label" in members.columns:
            labels = members["topic_label"].dropna().astype(str)
            topic_label = labels.iloc[0] if not labels.empty else None

        news = build_news_object(
            cluster_id=int(cluster.cluster_id),
            cluster_size=int(cluster.cluster_size),
            sample_texts=sample_texts,
            topic_label=topic_label,
            use_llm=use_llm,
        )
        record = news.as_dict()
        record["image_prompt"] = build_image_prompt(news)
        image_file = IMAGES_DIR / f"cluster_{int(cluster.cluster_id)}.png"
        if image_file.exists():
            record["image_path"] = f"/news-image/{int(cluster.cluster_id)}"
        items.append(record)
    return items


@app.get("/broadcast")
def broadcast(
    top_n: int = Query(5, ge=1, le=20),
    samples_per_cluster: int = Query(4, ge=1, le=10),
    use_llm: bool = Query(False, description="Use Grok to write the script (needs API_KEY)."),
) -> dict[str, Any]:
    """Return ordered broadcast segments for the avatar to read aloud."""
    try:
        items = _build_news_items(top_n, samples_per_cluster, use_llm)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail="Pipeline artifacts not found. Run the LSH pipeline first.",
        ) from exc
    segments = build_broadcast_segments(items)
    return {"news": items, "segments": segments}


@app.get("/news-image/{cluster_id}")
def news_image(cluster_id: int) -> FileResponse:
    image_file = IMAGES_DIR / f"cluster_{cluster_id}.png"
    if not image_file.exists():
        raise HTTPException(status_code=404, detail="image not generated for this cluster")
    return FileResponse(str(image_file), media_type="image/png")


# --- Static mounts: Live2D avatar assets + the anchor web UI ---------------
# Mounted last so they don't shadow the API routes above.
_AVATAR_DIR = REPO_ROOT / "Ami"
_ANCHOR_DIR = REPO_ROOT / "dashboard" / "anchor"

if _AVATAR_DIR.exists():
    app.mount("/avatar", StaticFiles(directory=str(_AVATAR_DIR)), name="avatar")
if _ANCHOR_DIR.exists():
    app.mount("/", StaticFiles(directory=str(_ANCHOR_DIR), html=True), name="anchor")
