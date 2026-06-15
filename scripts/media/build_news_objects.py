"""Turn top LSH clusters into structured news objects (JSON).

Reads the pipeline artifacts (clusters + scale subset), picks the largest
clusters, summarises each into a news object with a video scene description,
and writes them to a JSON file ready for video generation.

Example:
    python scripts/media/build_news_objects.py --top-n 5 --language Vietnamese
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from social_lsh.artifacts import artifact_path, ensure_artifact_dir, read_dataframe
from social_lsh.broadcast_cache import save_metadata
from social_lsh.constants import DEFAULT_ARTIFACT_DIR, REPO_ROOT
from social_lsh.news import build_broadcast_segments, build_news_object, build_image_prompt, build_video_prompt
from social_lsh.runtime_check import build_run_preflight


def _load_scale(artifact_dir: Path, needed_ids: set[int]) -> pd.DataFrame:
    import pyarrow.parquet as pq

    path = str(artifact_path("scale_shingles", artifact_dir))
    pf = pq.ParquetFile(path)
    available = pf.schema_arrow.names
    wanted = [c for c in ["tweet_id", "timestamp", "text", "topic_label", "shingle_count"] if c in available]
    parts: list[pd.DataFrame] = []
    for batch in pf.iter_batches(batch_size=50_000, columns=wanted):
        chunk = batch.to_pandas()
        chunk = chunk[chunk["tweet_id"].isin(needed_ids)]
        if not chunk.empty:
            parts.append(chunk)
    if not parts:
        return pd.DataFrame(columns=wanted)
    return pd.concat(parts, ignore_index=True)


def _top_clusters(artifact_dir: Path, clusters: pd.DataFrame, top_n: int, samples_per_cluster: int):
    ranked = (
        clusters[["cluster_id", "cluster_size"]]
        .drop_duplicates()
        .sort_values(["cluster_size", "cluster_id"], ascending=[False, True])
        .head(top_n)
    )
    top_ids = set(ranked["cluster_id"].tolist())
    members_all = clusters[clusters["cluster_id"].isin(top_ids)]
    needed = set(members_all["tweet_id"].astype(int).tolist())
    scale = _load_scale(artifact_dir, needed)
    merged = members_all.merge(scale, on="tweet_id", how="left")

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
        yield int(cluster.cluster_id), int(cluster.cluster_size), sample_texts, topic_label


def main() -> None:
    parser = argparse.ArgumentParser(description="Build news objects from top LSH clusters.")
    parser.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--output", default=str(REPO_ROOT / "jupyter" / "output" / "news" / "news_objects.json"))
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--samples-per-cluster", type=int, default=4)
    parser.add_argument("--language", default="Vietnamese", help="Language for headline/summary.")
    parser.add_argument("--no-llm", action="store_true", help="Skip Grok and use the template fallback.")
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")

    artifact_dir = ensure_artifact_dir(Path(args.artifact_dir))
    check = build_run_preflight(
        artifact_dir,
        top_n=args.top_n,
        samples_per_cluster=args.samples_per_cluster,
        use_llm=not args.no_llm,
        images=False,
    )
    print("Preflight:", check["status"])
    for warning in check["warnings"]:
        print(f"  warning: {warning}")
    for recommendation in check["recommendations"]:
        print(f"  plan: {recommendation}")
    if check["status"] == "danger":
        raise SystemExit("Preflight failed. Fix the data/resource issue before building news objects.")

    clusters = read_dataframe(artifact_path("clusters", artifact_dir))

    items = []
    for cluster_id, cluster_size, sample_texts, topic_label in _top_clusters(
        artifact_dir, clusters, args.top_n, args.samples_per_cluster
    ):
        news = build_news_object(
            cluster_id=cluster_id,
            cluster_size=cluster_size,
            sample_texts=sample_texts,
            topic_label=topic_label,
            language=args.language,
            use_llm=not args.no_llm,
        )
        record = news.as_dict()
        record["video_prompt"] = build_video_prompt(news)
        record["image_prompt"] = build_image_prompt(news)
        items.append(record)
        print(f"cluster {cluster_id} (size {cluster_size}): {news.headline[:60]}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {len(items)} news objects -> {output}")

    segments = build_broadcast_segments(items)
    meta = save_metadata(artifact_dir, args.top_n, args.samples_per_cluster, not args.no_llm, items, segments)
    print(f"Wrote metadata -> {meta['cache_id']}")


if __name__ == "__main__":
    main()
