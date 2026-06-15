"""Generate Veo videos from news objects.

Reads news_objects.json (produced by build_news_objects.py), submits each
video_prompt to the Veo API, polls until completion, downloads the videos,
and writes a manifest mapping clusters to video files.

Example:
    python scripts/media/generate_videos.py --size 720p --aspect-ratio 16:9
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from social_lsh.constants import REPO_ROOT
from social_lsh.veo import GrokVideoConfig, VeoClient, VeoError, extract_media_url


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate news videos from news objects.")
    parser.add_argument("--news-json", default=str(REPO_ROOT / "jupyter" / "output" / "news" / "news_objects.json"))
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "jupyter" / "output" / "news" / "videos"))
    parser.add_argument("--duration", type=int, default=6, choices=[6, 10], help="Video length in seconds.")
    parser.add_argument("--aspect-ratio", default="16:9", choices=["2:3", "3:2", "16:9", "9:16", "1:1"])
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N items (0 = all).")
    parser.add_argument("--poll-interval", type=float, default=15.0)
    parser.add_argument("--timeout", type=float, default=900.0)
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")

    news_path = Path(args.news_json)
    if not news_path.exists():
        raise SystemExit(f"News objects not found: {news_path}. Run build_news_objects.py first.")
    items = json.loads(news_path.read_text(encoding="utf-8"))
    if args.limit > 0:
        items = items[: args.limit]

    client = VeoClient.from_env()
    config = GrokVideoConfig(duration=args.duration, aspect_ratio=args.aspect_ratio)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for item in items:
        cluster_id = item.get("cluster_id")
        prompt = item.get("video_prompt") or item.get("scene")
        if not prompt:
            print(f"cluster {cluster_id}: no prompt, skipping")
            continue

        print(f"\ncluster {cluster_id}: submitting Veo task...")
        entry = {
            "cluster_id": cluster_id,
            "headline": item.get("headline"),
            "prompt": prompt,
            "status": "pending",
        }
        try:
            def _log(status, info):
                print(f"  [{cluster_id}] {status} {info}")

            result = client.generate(
                prompt,
                config=config,
                interval_seconds=args.poll_interval,
                timeout_seconds=args.timeout,
                on_update=_log,
            )
            video_path = output_dir / f"cluster_{cluster_id}.mp4"
            client.download_media(result, video_path)
            entry["status"] = "success"
            entry["video_path"] = str(video_path)
            entry["video_url"] = extract_media_url(result)
            print(f"  [{cluster_id}] saved -> {video_path}")
        except VeoError as exc:
            entry["status"] = "failed"
            entry["error"] = str(exc)
            print(f"  [{cluster_id}] FAILED: {exc}")

        manifest.append(entry)

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    ok = sum(1 for entry in manifest if entry["status"] == "success")
    print(f"\nDone. {ok}/{len(manifest)} videos generated. Manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
