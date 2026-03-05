from __future__ import annotations

import argparse

from _bootstrap import add_src_to_path

add_src_to_path()

from uk_russia_lsh import BASELINE_SIZE, DEFAULT_ARTIFACT_DIR, DEFAULT_INPUT_PARQUET, DEFAULT_SEED, SCALE_SIZE
from uk_russia_lsh.pipeline import extract_subsets


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract deterministic baseline and scale subsets.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT_PARQUET))
    parser.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--baseline-size", type=int, default=BASELINE_SIZE)
    parser.add_argument("--scale-size", type=int, default=SCALE_SIZE)
    args = parser.parse_args()

    outputs = extract_subsets(
        input_path=args.input,
        artifact_dir=args.artifact_dir,
        seed=args.seed,
        baseline_size=args.baseline_size,
        scale_size=args.scale_size,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
