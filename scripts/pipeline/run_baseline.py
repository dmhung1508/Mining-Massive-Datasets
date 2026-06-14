from __future__ import annotations

import argparse



from social_lsh import DEFAULT_ARTIFACT_DIR, DEFAULT_VERIFY_THRESHOLD
from social_lsh.pipeline import run_baseline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run exact Jaccard baseline on the baseline subset.")
    parser.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--threshold", type=float, default=DEFAULT_VERIFY_THRESHOLD)
    args = parser.parse_args()

    output_path, metrics = run_baseline(
        artifact_dir=args.artifact_dir,
        threshold=args.threshold,
    )
    print(f"baseline_pairs: {output_path}")
    print(metrics)


if __name__ == "__main__":
    main()
