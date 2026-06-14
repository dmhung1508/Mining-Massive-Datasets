"""FastAPI backend for the Social LSH demo artifacts.

Endpoints (match the project pipeline diagram, step 9):
- GET /health
- GET /metrics
- GET /clusters/top?limit=10
- GET /search?text=...&top_k=5&min_jaccard=0.8
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query

from social_lsh.artifacts import artifact_path, ensure_artifact_dir, read_dataframe, read_metrics
from social_lsh.constants import DEFAULT_ARTIFACT_DIR, DEFAULT_SEED
from social_lsh.search import prepare_search_index, search_similar_tweets


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

app = FastAPI(
    title="Social LSH API",
    description="Near-duplicate detection and similar-post search over war-related social media.",
    version="0.1.0",
)


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
