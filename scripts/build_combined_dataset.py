from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_src_to_path

repo_root = add_src_to_path()

from uk_russia_lsh.datasets import build_combined_dataset, default_combined_dataset_path, default_twitter_dataset_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge Twitter parquet data with Telegram messages into one dataset.")
    parser.add_argument(
        "--twitter-input",
        default=str(default_twitter_dataset_path() or ""),
        help="Path to the Twitter parquet dataset (tweets_final.parquet).",
    )
    parser.add_argument("--output", default=str(default_combined_dataset_path()))
    parser.add_argument("--telegram-source", choices=["auto", "mongo", "jsonl"], default="auto")
    parser.add_argument("--telegram-mongo-uri")
    parser.add_argument("--telegram-db")
    parser.add_argument("--telegram-collection")
    parser.add_argument("--telegram-local-data-dir")
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.twitter_input:
        raise SystemExit("Twitter parquet dataset not found automatically. Pass --twitter-input explicitly.")

    metrics = build_combined_dataset(
        output_path=Path(args.output),
        twitter_input_path=Path(args.twitter_input),
        telegram_source=args.telegram_source,
        telegram_mongo_uri=args.telegram_mongo_uri,
        telegram_db_name=args.telegram_db,
        telegram_collection_name=args.telegram_collection,
        telegram_local_data_dir=args.telegram_local_data_dir,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
