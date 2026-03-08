from __future__ import annotations

import argparse

from _bootstrap import add_src_to_path

add_src_to_path()

from uk_russia_lsh import DEFAULT_ARTIFACT_DIR, DEFAULT_SHINGLE_SIZE
from uk_russia_lsh.pipeline import build_shingles_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Build shingle artifacts for extracted subsets.")
    parser.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--shingle-size", type=int, default=DEFAULT_SHINGLE_SIZE)
    args = parser.parse_args()

    outputs = build_shingles_artifacts(
        artifact_dir=args.artifact_dir,
        shingle_size=args.shingle_size,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
