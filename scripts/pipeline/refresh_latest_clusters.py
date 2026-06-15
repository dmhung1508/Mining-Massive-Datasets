"""Build the latest narrative clusters from a recent time window.

Instead of re-clustering the full historical corpus (jupyter/output/lsh_full),
this pulls only the most recent posts from every live source (Telegram realtime
+ the X collections in news_monitoring), then runs the lightweight LSH pipeline
on just that window. Run it on a schedule (e.g. hourly) to keep clusters fresh.

Example:
    python scripts/pipeline/refresh_latest_clusters.py --since-days 2 --scale-size 50000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from social_lsh.artifacts import artifact_path, ensure_artifact_dir, write_dataframe
from social_lsh.constants import BASELINE_SIZE, DEFAULT_SEED, OUTPUT_ROOT, REPO_ROOT, SCALE_SIZE
from social_lsh.datasets import build_recent_window_dataset
from social_lsh.pipeline import (
    build_shingles_artifacts,
    extract_subsets,
    run_baseline,
    run_lsh,
    verify_and_cluster,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh latest clusters from a recent time window.")
    parser.add_argument("--since-days", type=float, default=2.0, help="How far back to pull posts.")
    parser.add_argument("--artifact-dir", default=str(OUTPUT_ROOT / "lsh_latest"))
    parser.add_argument("--window-parquet", default=str(OUTPUT_ROOT / "recent_window.parquet"))
    parser.add_argument("--baseline-size", type=int, default=BASELINE_SIZE)
    parser.add_argument("--scale-size", type=int, default=SCALE_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    artifact_dir = ensure_artifact_dir(Path(args.artifact_dir))

    print(f"==> Pulling posts from the last {args.since_days} day(s)...")
    window = build_recent_window_dataset(
        output_path=Path(args.window_parquet),
        since_days=args.since_days,
        overwrite=True,
    )
    print(json.dumps(window, ensure_ascii=False, indent=2))
    if window["total_rows"] == 0:
        raise SystemExit("No recent posts found in the window. Nothing to cluster.")

    print("==> Extracting subsets...")
    extract_subsets(
        input_path=Path(args.window_parquet),
        artifact_dir=artifact_dir,
        seed=args.seed,
        baseline_size=args.baseline_size,
        scale_size=args.scale_size,
    )

    print("==> Building shingles...")
    build_shingles_artifacts(artifact_dir=artifact_dir)

    print("==> Running exact Jaccard baseline...")
    run_baseline(artifact_dir=artifact_dir)

    print("==> Running MinHash + LSH...")
    run_lsh(artifact_dir=artifact_dir, seed=args.seed)

    print("==> Verifying candidates and building clusters...")
    outputs = verify_and_cluster(artifact_dir=artifact_dir)

    print("\nDone. Latest clusters ready:")
    print(f"  artifact dir: {artifact_dir}")
    for name, path in outputs.items():
        print(f"  {name}: {path}")
    print("\nServe the broadcast from this window with:")
    print(f"  python scripts/api/serve_api.py --artifact-dir {artifact_dir}")


if __name__ == "__main__":
    main()
