from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_src_to_path

repo_root = add_src_to_path()

from uk_russia_lsh.datasets import default_telegram_export_path, export_telegram_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Telegram messages into the canonical parquet schema.")
    parser.add_argument("--output", default=str(default_telegram_export_path()))
    parser.add_argument("--source", choices=["auto", "mongo", "jsonl"], default="auto")
    parser.add_argument("--mongo-uri")
    parser.add_argument("--mongo-db")
    parser.add_argument("--mongo-collection")
    parser.add_argument("--local-data-dir")
    parser.add_argument("--metrics-output")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    metrics = export_telegram_dataset(
        output_path=Path(args.output),
        source=args.source,
        mongo_uri=args.mongo_uri,
        db_name=args.mongo_db,
        collection_name=args.mongo_collection,
        local_data_dir=args.local_data_dir,
        overwrite=args.overwrite,
    )
    if args.metrics_output:
        metrics_path = Path(args.metrics_output)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
