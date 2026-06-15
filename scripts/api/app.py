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
from dotenv import load_dotenv
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


def _load_scale_for_news(needed_ids: set[int] | None = None) -> pd.DataFrame:
    """Load only the rows/columns the broadcast needs.

    To stay light on memory against the multi-GB lsh_full artifacts, we read the
    parquet row-group by row-group and keep only the columns we need (text,
    topic_label) and only the rows whose tweet_id is in `needed_ids`. The big
    shingle/token columns are never read.
    """
    import pyarrow.parquet as pq

    path = str(artifact_path("scale_shingles", ARTIFACT_DIR))
    pf = pq.ParquetFile(path)
    available = pf.schema_arrow.names
    wanted = [c for c in ["tweet_id", "text", "topic_label", "shingle_count"] if c in available]

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

# Persisted scripts so we never re-generate after a restart.
SCRIPTS_DIR = REPO_ROOT / "jupyter" / "output" / "news" / "scripts"


def _script_meta_path(top_n: int, samples_per_cluster: int, use_llm: bool) -> Path:
    tag = f"{ARTIFACT_DIR.name}_top{top_n}_s{samples_per_cluster}_{'llm' if use_llm else 'tmpl'}"
    return SCRIPTS_DIR / f"{tag}.json"


def _load_persisted_script(top_n: int, samples_per_cluster: int, use_llm: bool):
    path = _script_meta_path(top_n, samples_per_cluster, use_llm)
    if not path.exists():
        return None
    try:
        items = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    # Re-attach image paths in case images were generated after the script.
    for item in items:
        cid = item.get("cluster_id")
        if (IMAGES_DIR / f"cluster_{cid}.png").exists():
            item["image_path"] = f"/news-image/{cid}"
    return items


def _save_persisted_script(top_n: int, samples_per_cluster: int, use_llm: bool, items: list[dict[str, Any]]) -> None:
    try:
        SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
        _script_meta_path(top_n, samples_per_cluster, use_llm).write_text(
            json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except OSError:
        pass


def _top_clusters_frame(top_n: int):
    """Read and rank the top clusters once, then cache (clusters.parquet is large)."""
    key = "ranked"
    cached = _TOP_CLUSTERS_CACHE.get(key)
    if cached is None:
        clusters = read_dataframe(artifact_path("clusters", ARTIFACT_DIR))
        cached = (
            clusters[["cluster_id", "cluster_size"]]
            .drop_duplicates()
            .sort_values(["cluster_size", "cluster_id"], ascending=[False, True])
            .reset_index(drop=True)
        )
        _TOP_CLUSTERS_CACHE[key] = cached
        _TOP_CLUSTERS_CACHE["members"] = clusters[["tweet_id", "cluster_id"]]
    return cached.head(top_n), _TOP_CLUSTERS_CACHE["members"]


def _build_news_items(top_n: int, samples_per_cluster: int, use_llm: bool, force: bool = False) -> list[dict[str, Any]]:
    """Build news objects from the top clusters, attaching cached images if present.

    Reuse order: in-memory cache -> persisted script file -> generate fresh.
    Set `force=True` to regenerate and overwrite the saved script.
    """
    cache_key = (top_n, samples_per_cluster, use_llm)
    if not force and cache_key in _BROADCAST_CACHE:
        return _BROADCAST_CACHE[cache_key]

    if not force:
        persisted = _load_persisted_script(top_n, samples_per_cluster, use_llm)
        if persisted:
            _BROADCAST_CACHE[cache_key] = persisted
            return persisted

    ranked, all_members = _top_clusters_frame(top_n)
    top_cluster_ids = set(ranked["cluster_id"].tolist())
    top_members = all_members[all_members["cluster_id"].isin(top_cluster_ids)]
    needed_ids = set(top_members["tweet_id"].astype(int).tolist())

    scale = _load_scale_for_news(needed_ids=needed_ids)
    merged = top_members.merge(scale, on="tweet_id", how="left")

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

    _BROADCAST_CACHE[cache_key] = items
    _save_persisted_script(top_n, samples_per_cluster, use_llm, items)
    return items


@app.get("/broadcast")
def broadcast(
    top_n: int = Query(5, ge=1, le=20),
    samples_per_cluster: int = Query(12, ge=1, le=40),
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


def _generate_one_image(client, item: dict[str, Any]) -> bool:
    """Generate one cluster image with OpenAI. Returns True if a new image was created."""
    cluster_id = item.get("cluster_id")
    image_file = IMAGES_DIR / f"cluster_{cluster_id}.png"
    if image_file.exists():
        item["image_path"] = f"/news-image/{cluster_id}"
        return False
    prompt = item.get("image_prompt") or item.get("scene")
    if not prompt:
        return False
    try:
        client.generate_to_file(prompt, image_file, size="1024x1024")
        item["image_path"] = f"/news-image/{cluster_id}"
        return True
    except (ImageError, ValueError):
        return False


def _generate_images_for(items: list[dict[str, Any]], quality: str = "low") -> int:
    """Generate any missing cluster illustrations in parallel (OpenAI Images API)."""
    try:
        client = OpenAIImageClient.from_env()
    except ImageError:
        return 0  # No API key for images; skip silently.

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=min(6, max(1, len(items)))) as pool:
        results = list(pool.map(lambda it: _generate_one_image(client, it), items))
    return sum(1 for r in results if r)


@app.get("/prepare")
def prepare(
    top_n: int = Query(5, ge=1, le=20),
    samples_per_cluster: int = Query(12, ge=1, le=40),
    use_llm: bool = Query(True, description="Use Grok to write the script (needs API_KEY)."),
    images: bool = Query(False, description="Generate missing images now (slow). Prefer pre-generating offline."),
    force: bool = Query(False, description="Regenerate the script even if a saved one exists."),
) -> dict[str, Any]:
    """Do the heavy work BEFORE the anchor speaks.

    Builds the news script (Grok, logged to log.txt), saves it to
    jupyter/output/news/scripts/ so later runs reuse it, and attaches any
    pre-generated images. By default it does NOT generate images on the fly.
    """
    try:
        items = _build_news_items(top_n, samples_per_cluster, use_llm, force=force)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail="Pipeline artifacts not found. Run the LSH pipeline first.",
        ) from exc

    images_created = 0
    if images:
        images_created = _generate_images_for(items)

    segments = build_broadcast_segments(items)
    return {
        "news": items,
        "segments": segments,
        "images_created": images_created,
        "used_llm": use_llm,
    }


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
