from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


repo_root = Path(__file__).resolve().parents[2]


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(repo_root / ".env")


def _settings() -> tuple[str, str, str]:
    _load_dotenv_if_available()
    mongo_uri = os.getenv("MONGO_URI", "").strip()
    db_name = os.getenv("MONGO_DB_NAME", os.getenv("MONGO_DB", "telegram_data")).strip()
    collection_name = os.getenv("MONGO_COLLECTION_NAME", os.getenv("MONGO_COLLECTION", "messages")).strip()
    if not mongo_uri:
        raise SystemExit("Missing MONGO_URI. Put it in .env or set it in the shell environment.")
    return mongo_uri, db_name, collection_name


def _aggregate_counts(mongo_uri: str, db_name: str, collection_name: str) -> dict[str, int]:
    try:
        from pymongo import MongoClient
    except ImportError as exc:
        raise SystemExit("pymongo is required. Install requirements-telegram.txt first.") from exc

    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=8000)
    try:
        collection = client[db_name][collection_name]
        pipeline: list[dict[str, Any]] = [
            {
                "$group": {
                    "_id": {"$ifNull": ["$war_classification.label", "missing"]},
                    "count": {"$sum": 1},
                }
            },
            {"$sort": {"count": -1}},
        ]
        rows = list(collection.aggregate(pipeline))
    finally:
        client.close()

    return {str(row["_id"]): int(row["count"]) for row in rows}


def _write_markdown(counts: dict[str, int], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = sum(counts.values())
    ordered_labels = ["unrelated", "russia_ukraine_war", "us_iran_war", "missing"]

    lines = [
        "# Telegram Topic Classification Summary",
        "",
        "## Source",
        "",
        "- Database: `telegram_data`",
        "- Collection: `messages`",
        "- Classification field: `war_classification.label`",
        "",
        "## Label counts",
        "",
        "| Label | Count | Share |",
        "| --- | ---: | ---: |",
    ]

    seen = set()
    for label in ordered_labels:
        if label not in counts:
            continue
        seen.add(label)
        count = counts[label]
        share = count / total if total else 0.0
        lines.append(f"| `{label}` | {count:,} | {share:.2%} |")

    for label, count in sorted(counts.items()):
        if label in seen:
            continue
        share = count / total if total else 0.0
        lines.append(f"| `{label}` | {count:,} | {share:.2%} |")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `russia_ukraine_war`: Telegram messages classified as Russia-Ukraine conflict content.",
            "- `us_iran_war`: Telegram messages classified as US-Iran conflict content.",
            "- `unrelated`: Messages not clearly belonging to either conflict topic.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Telegram war classification labels from MongoDB.")
    parser.add_argument("--output-json", default="docs/telegram_topic_counts.json")
    parser.add_argument("--output-md", default="docs/telegram_topic_classification.md")
    args = parser.parse_args()

    counts = _aggregate_counts(*_settings())

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(counts, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(counts, Path(args.output_md))

    print(json.dumps({"counts": counts, "output_json": str(output_json), "output_md": args.output_md}, indent=2))


if __name__ == "__main__":
    main()
