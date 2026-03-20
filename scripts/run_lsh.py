from __future__ import annotations

import argparse

from _bootstrap import add_src_to_path

add_src_to_path()

from uk_russia_lsh import DEFAULT_ARTIFACT_DIR, DEFAULT_SEED
from uk_russia_lsh.pipeline import run_lsh


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune LSH configs and produce scale-set candidate pairs.")
    parser.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    output_path, metrics = run_lsh(
        artifact_dir=args.artifact_dir,
        seed=args.seed,
    )
    print(f"candidates: {output_path}")
    print(metrics["selected_config"])


if __name__ == "__main__":
    main()
