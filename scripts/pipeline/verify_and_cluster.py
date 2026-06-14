from __future__ import annotations

import argparse



from social_lsh import DEFAULT_ARTIFACT_DIR, DEFAULT_VERIFY_THRESHOLD
from social_lsh.pipeline import verify_and_cluster


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify candidate pairs with exact Jaccard and build clusters.")
    parser.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--threshold", type=float, default=DEFAULT_VERIFY_THRESHOLD)
    args = parser.parse_args()

    outputs = verify_and_cluster(
        artifact_dir=args.artifact_dir,
        threshold=args.threshold,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
