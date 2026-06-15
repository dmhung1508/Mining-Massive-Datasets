"""Generate illustrative images from news objects (YEScale gpt-image).

Reads news_objects.json (produced by build_news_objects.py), submits each
image_prompt to the image model, polls until completion, downloads the images,
and writes a manifest mapping clusters to image files.

Example:
    python scripts/media/generate_images.py --size 1024x1024 --quality low
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from social_lsh.constants import REPO_ROOT
from social_lsh.veo import ImageConfig, VeoClient, VeoError, extract_media_url


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate illustrative images from news objects.")
    parser.add_argument("--news-json", default=str(REPO_ROOT / "jupyter" / "output" / "news" / "news_objects.json"))
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "jupyter" / "output" / "news" / "images"))
    parser.add_argument("--size", default="1024x1024", choices=["1024x1024", "1024x1536", "1536x1024"])
    parser.add_argument("--quality", default="low", choices=["low", "medium", "high"])
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N items (0 = all).")
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")

    news_path = Path(args.news_json)
    if not news_path.exists():
        raise SystemExit(f"News objects not found: {news_path}. Run build_news_objects.py first.")
    items = json.loads(news_path.read_text(encoding="utf-8"))
    if args.limit > 0:
        items = items[: args.limit]

    client = VeoClient.for_images()
    config = ImageConfig(size=args.size, quality=args.quality)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for item in items:
        cluster_id = item.get("cluster_id")
        prompt = item.get("image_prompt") or item.get("scene")
        if not prompt:
            print(f"cluster {cluster_id}: no prompt, skipping")
            continue

        print(f"\ncluster {cluster_id}: submitting image task...")
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
            image_path = output_dir / f"cluster_{cluster_id}.png"
            client.download_media(result, image_path)
            entry["status"] = "success"
            entry["image_path"] = str(image_path)
            entry["image_url"] = extract_media_url(result)
            print(f"  [{cluster_id}] saved -> {image_path}")
        except VeoError as exc:
            entry["status"] = "failed"
            entry["error"] = str(exc)
            print(f"  [{cluster_id}] FAILED: {exc}")

        manifest.append(entry)

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    ok = sum(1 for entry in manifest if entry["status"] == "success")
    print(f"\nDone. {ok}/{len(manifest)} images generated. Manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
