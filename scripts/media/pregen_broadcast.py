"""Pre-generate everything a broadcast needs, BEFORE serving (Cách B).

Run this once (offline). It builds the news objects (Grok writes the script and
logs to log.txt) and generates the cluster illustrations into
jupyter/output/news/images/. When the API later serves /prepare, the images
already exist so it never waits on slow image generation.

Example:
    python scripts/media/pregen_broadcast.py --artifact-dir jupyter/output/lsh_combined --top-n 5
"""
from __future__ import annotations

import argparse
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from social_lsh.artifacts import artifact_path, ensure_artifact_dir, read_dataframe
from social_lsh.broadcast_cache import (
    broadcast_cache_id,
    image_file,
    legacy_image_file,
    load_metadata,
    public_image_path,
    save_metadata,
)
from social_lsh.constants import DEFAULT_ARTIFACT_DIR, REPO_ROOT
from social_lsh.news import build_broadcast_segments, build_image_prompt, build_news_object
from social_lsh.images import ImageError, OpenAIImageClient
from social_lsh.runtime_check import build_run_preflight

IMAGES_DIR = REPO_ROOT / "jupyter" / "output" / "news" / "images"
NEWS_JSON = REPO_ROOT / "jupyter" / "output" / "news" / "news_objects.json"


def _load_scale(artifact_dir: Path, needed_ids: set[int]) -> pd.DataFrame:
    import pyarrow.parquet as pq

    path = str(artifact_path("scale_shingles", artifact_dir))
    pf = pq.ParquetFile(path)
    available = pf.schema_arrow.names
    wanted = [c for c in ["tweet_id", "text", "topic_label", "shingle_count"] if c in available]
    parts: list[pd.DataFrame] = []
    for batch in pf.iter_batches(batch_size=50_000, columns=wanted):
        chunk = batch.to_pandas()
        chunk = chunk[chunk["tweet_id"].isin(needed_ids)]
        if not chunk.empty:
            parts.append(chunk)
    if not parts:
        return pd.DataFrame(columns=wanted)
    return pd.concat(parts, ignore_index=True)


def _top_inputs(artifact_dir: Path, top_n: int, samples_per_cluster: int):
    clusters = read_dataframe(artifact_path("clusters", artifact_dir))
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
    scale = scale[scale["tweet_id"].isin(needed)]
    merged = members_all.merge(scale, on="tweet_id", how="left")

    for cluster in ranked.itertuples(index=False):
        members = merged.loc[merged["cluster_id"].eq(cluster.cluster_id)]
        if "shingle_count" in members.columns:
            members = members.sort_values(["shingle_count", "tweet_id"], ascending=[False, True])
        members = members.head(samples_per_cluster)
        texts = members["text"].astype(str).tolist()
        topic_label = None
        if "topic_label" in members.columns:
            labels = members["topic_label"].dropna().astype(str)
            topic_label = labels.iloc[0] if not labels.empty else None
        yield int(cluster.cluster_id), int(cluster.cluster_size), texts, topic_label


def _gen_image(client, item, cache_id: str) -> bool:
    cluster_id = item["cluster_id"]
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
        client.generate_to_file(prompt, target, size="1024x1024")
        item["image_path"] = public_image_path(cache_id, cluster_id)
        print(f"  image saved: {target}")
        return True
    except (ImageError, ValueError) as exc:
        print(f"  image FAILED cluster {cluster_id}: {exc}")
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-generate broadcast script + images (offline).")
    parser.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--samples-per-cluster", type=int, default=12)
    parser.add_argument("--quality", default="low", choices=["low", "medium", "high"])
    parser.add_argument("--no-llm", action="store_true", help="Skip Grok, use template.")
    parser.add_argument("--no-images", action="store_true", help="Only build the script, skip images.")
    parser.add_argument("--force", action="store_true", help="Regenerate the script even if metadata cache exists.")
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    artifact_dir = ensure_artifact_dir(Path(args.artifact_dir))
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    use_llm = not args.no_llm
    cache_id = broadcast_cache_id(artifact_dir, args.top_n, args.samples_per_cluster, use_llm)

    check = build_run_preflight(
        artifact_dir,
        top_n=args.top_n,
        samples_per_cluster=args.samples_per_cluster,
        use_llm=use_llm,
        images=not args.no_images,
    )
    print("==> Preflight")
    print(f"  status: {check['status']}")
    print(f"  artifact rows: {check['artifact_rows']}")
    print(f"  scale size: {check['artifact_sizes_gib']['scale_shingles']} GiB")
    print(f"  RAM available: {check['memory'].get('available_gib')} GiB")
    for warning in check["warnings"]:
        print(f"  warning: {warning}")
    for recommendation in check["recommendations"]:
        print(f"  plan: {recommendation}")
    if check["status"] == "danger":
        raise SystemExit("Preflight failed. Fix the data/resource issue before pregenerating.")

    meta = None if args.force else load_metadata(artifact_dir, args.top_n, args.samples_per_cluster, use_llm)
    if meta:
        print(f"==> Using cached script metadata: {check['cache']['metadata_path']}")
        items = meta["items"]
    else:
        print("==> Building news objects (Grok writes the script, logs to log.txt)...")
        inputs = list(_top_inputs(artifact_dir, args.top_n, args.samples_per_cluster))

        def _make(args_tuple):
            cid, csize, texts, tlabel = args_tuple
            news = build_news_object(
                cluster_id=cid,
                cluster_size=csize,
                sample_texts=texts,
                topic_label=tlabel,
                use_llm=use_llm,
                max_posts=args.samples_per_cluster,
            )
            record = news.as_dict()
            record["image_prompt"] = build_image_prompt(news)
            return record

        with ThreadPoolExecutor(max_workers=min(6, max(1, len(inputs)))) as pool:
            items = list(pool.map(_make, inputs))
        for it in items:
            print(f"  cluster {it['cluster_id']}: {it['headline'][:60]}")

    NEWS_JSON.parent.mkdir(parents=True, exist_ok=True)
    NEWS_JSON.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    segments = build_broadcast_segments(items)
    meta = save_metadata(artifact_dir, args.top_n, args.samples_per_cluster, use_llm, items, segments)
    print(f"   legacy script -> {NEWS_JSON}")
    print(f"   metadata -> {check['cache']['metadata_path']}")

    if not args.no_images:
        print("==> Generating images in parallel (OpenAI, done once)...")
        client = OpenAIImageClient.from_env()
        with ThreadPoolExecutor(max_workers=min(6, max(1, len(items)))) as pool:
            created = sum(pool.map(lambda it: _gen_image(client, it, cache_id), items))
        segments = build_broadcast_segments(items)
        meta = save_metadata(artifact_dir, args.top_n, args.samples_per_cluster, use_llm, items, segments)
        print(f"   {created} new image(s) in {IMAGES_DIR / cache_id}")
        print(f"   metadata updated -> {check['cache']['metadata_path']}")

    print("\nDone. Now serve with images ready:")
    print(f"  python scripts/api/serve_api.py --artifact-dir {artifact_dir} --port 8765")


if __name__ == "__main__":
    main()
