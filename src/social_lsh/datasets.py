from __future__ import annotations

import hashlib
import json
import runpy
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from .constants import (
    DEFAULT_COMBINED_DATASET,
    DEFAULT_INPUT_PARQUET,
    DEFAULT_TELEGRAM_EXPORT,
    REPO_ROOT,
)


CANONICAL_COLUMNS = [
    "tweet_id",
    "user_id",
    "text",
    "timestamp",
    "date",
    "source",
    "source_item_id",
    "source_user_id",
    "source_channel_id",
    "media_type",
    "forward_from_user_id",
    "forward_from_username",
    "topic_label",
    "topic_confidence",
    "topic_reason",
]

CANONICAL_ARROW_SCHEMA = pa.schema(
    [
        pa.field("tweet_id", pa.int64(), nullable=False),
        pa.field("user_id", pa.int64(), nullable=False),
        pa.field("text", pa.string(), nullable=False),
        pa.field("timestamp", pa.timestamp("ns"), nullable=False),
        pa.field("date", pa.string(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("source_item_id", pa.string(), nullable=False),
        pa.field("source_user_id", pa.string(), nullable=False),
        pa.field("source_channel_id", pa.string()),
        pa.field("media_type", pa.string()),
        pa.field("forward_from_user_id", pa.string()),
        pa.field("forward_from_username", pa.string()),
        pa.field("topic_label", pa.string()),
        pa.field("topic_confidence", pa.float64()),
        pa.field("topic_reason", pa.string()),
    ]
)

STRING_METADATA_COLUMNS = [
    "source",
    "source_item_id",
    "source_user_id",
    "source_channel_id",
    "media_type",
    "forward_from_user_id",
    "forward_from_username",
    "topic_label",
    "topic_reason",
]


def _stable_positive_int64(value: str) -> int:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False) & 0x7FFF_FFFF_FFFF_FFFF


def _collapse_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return " ".join(str(value).split())


def _parse_timestamps(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, utc=True, errors="coerce")
    return parsed.dt.tz_localize(None)


def _coerce_int_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def _empty_canonical_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=CANONICAL_COLUMNS)


def _normalise_string_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    normalised = frame.copy()
    for column in STRING_METADATA_COLUMNS:
        values = normalised[column].astype("string")
        normalised[column] = values.where(values.notna(), None)
    return normalised


def _write_canonical_frame(frame: pd.DataFrame, output_path: Path) -> None:
    if frame.empty:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalised = _normalise_string_metadata(frame)
    table = pa.Table.from_pandas(
        normalised[CANONICAL_COLUMNS],
        schema=CANONICAL_ARROW_SCHEMA,
        preserve_index=False,
        safe=False,
    )
    pq.write_to_dataset(
        table,
        root_path=str(output_path),
        partition_cols=["date"],
        compression="snappy",
    )


def _prepare_output_path(output_path: Path | str, overwrite: bool) -> Path:
    target = Path(output_path)
    if target.exists():
        if not overwrite:
            raise FileExistsError(f"Output path already exists: {target}")
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def _load_repo_telegram_defaults() -> dict[str, Any]:
    config_path = REPO_ROOT / "telegram_crawler" / "config.py"
    if not config_path.exists():
        return {}

    previous_sys_path = list(sys.path)
    sys.path.insert(0, str(config_path.parent))
    try:
        namespace = runpy.run_path(str(config_path))
    except Exception:
        return {}
    finally:
        sys.path[:] = previous_sys_path

    return {
        "mongo_uri": namespace.get("MONGO_URI"),
        "mongo_db_name": namespace.get("MONGO_DB_NAME"),
        "mongo_collection_name": namespace.get("MONGO_COLLECTION_NAME"),
        "local_data_dir": namespace.get("LOCAL_DATA_DIR"),
    }


def default_twitter_dataset_path() -> Path | None:
    candidates = [
        DEFAULT_INPUT_PARQUET,
        REPO_ROOT / "tweets_final.parquet",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def default_telegram_export_path() -> Path:
    return DEFAULT_TELEGRAM_EXPORT


def default_combined_dataset_path() -> Path:
    return DEFAULT_COMBINED_DATASET


def iter_twitter_batches(input_path: Path | str, batch_size: int = 100_000) -> Iterator[pd.DataFrame]:
    dataset = ds.dataset(str(input_path), format="parquet", partitioning="hive")
    columns = [column for column in ["tweet_id", "user_id", "text", "timestamp", "date"] if column in dataset.schema.names]
    for batch in dataset.to_batches(columns=columns, batch_size=batch_size):
        yield batch.to_pandas()


def normalise_twitter_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return _empty_canonical_frame()

    working = frame.copy()
    working["tweet_id"] = _coerce_int_series(working["tweet_id"])
    working["user_id"] = _coerce_int_series(working["user_id"])
    working["text"] = working["text"].map(_collapse_text)
    working["timestamp"] = _parse_timestamps(working["timestamp"])

    if "date" in working.columns:
        working["date"] = pd.to_datetime(working["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    else:
        working["date"] = working["timestamp"].dt.strftime("%Y-%m-%d")

    working = working.loc[
        working["tweet_id"].notna()
        & working["user_id"].notna()
        & working["timestamp"].notna()
        & working["date"].notna()
        & working["text"].ne("")
    ].copy()

    if working.empty:
        return _empty_canonical_frame()

    working["tweet_id"] = working["tweet_id"].astype("int64")
    working["user_id"] = working["user_id"].astype("int64")
    working["source"] = "twitter"
    working["source_item_id"] = working["tweet_id"].astype("string")
    working["source_user_id"] = working["user_id"].astype("string")
    working["source_channel_id"] = None
    working["media_type"] = None
    working["forward_from_user_id"] = None
    working["forward_from_username"] = None
    working["topic_label"] = None
    working["topic_confidence"] = None
    working["topic_reason"] = None
    return working[CANONICAL_COLUMNS].reset_index(drop=True)


def _nested_value(value: object, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    return None


def _topic_label_from_classification(value: object) -> str | None:
    label = _nested_value(value, "label")
    if label is None:
        return None
    label = str(label).strip().lower()
    return label or None


def _topic_confidence_from_classification(value: object) -> float | None:
    confidence = _nested_value(value, "confidence")
    if confidence is None:
        return None
    try:
        return float(confidence)
    except (TypeError, ValueError):
        return None


def _topic_reason_from_classification(value: object) -> str | None:
    reason = _nested_value(value, "reason")
    if reason is None:
        return None
    reason = str(reason).strip()
    return reason or None


def _first_present(row: dict[str, Any], keys: list[str]) -> Any:
    """Return the first non-empty value among candidate keys (supports nesting)."""
    for key in keys:
        if "." in key:
            cur: Any = row
            for part in key.split("."):
                cur = _nested_value(cur, part) if isinstance(cur, dict) else None
                if cur is None:
                    break
            value = cur
        else:
            value = row.get(key)
        if value not in (None, "", [], {}):
            return value
    return None


# Field names for X posts in the news_monitoring MongoDB (confirmed from schema):
#   statusId, postedAt, account, monitorTopic, source, hash, url, hasMedia, media
# The post body field is not always visible in the sample, so `text` probes
# several likely keys.
_X_FIELD_CANDIDATES = {
    "id": ["statusId", "status_id", "tweet_id", "tweetid", "id_str", "id", "post_id"],
    "text": ["text", "content", "full_text", "tweet", "body", "postText", "rawText", "caption", "title"],
    "timestamp": ["postedAt", "crawledAt", "firstSeenAt", "created_at", "timestamp", "tweetcreatedts", "date"],
    "user_id": ["account", "user_id", "userid", "author_id", "username", "screen_name"],
    "username": ["account", "username", "screen_name", "author"],
    "url": ["url", "link", "permalink"],
}

# monitorTopic value on each document -> canonical topic label.
_X_MONITOR_TOPIC_MAP = {
    "russia_ukraine": "russia_ukraine_war",
    "russia_ukraina": "russia_ukraine_war",
    "ukraine_russia": "russia_ukraine_war",
    "us_iran": "us_iran_war",
    "iran_us": "us_iran_war",
}


def _x_topic_from_row(row: dict[str, Any], fallback: str | None) -> str | None:
    monitor = row.get("monitorTopic")
    if monitor:
        key = str(monitor).strip().lower()
        mapped = _X_MONITOR_TOPIC_MAP.get(key)
        if mapped:
            return mapped
        # Heuristic for unseen values.
        if "ukrain" in key or "russia" in key:
            return "russia_ukraine_war"
        if "iran" in key:
            return "us_iran_war"
    return (fallback or "").strip().lower() or None


def normalise_x_frame(frame: pd.DataFrame, topic_label: str | None = None) -> pd.DataFrame:
    """Normalise X posts (from the news_monitoring MongoDB) into the canonical schema.

    The per-document `monitorTopic` field drives the topic label; `topic_label`
    is only used as a fallback when `monitorTopic` is missing.
    """
    if frame.empty:
        return _empty_canonical_frame()

    records = frame.to_dict(orient="records")
    rows: list[dict[str, Any]] = []
    for row in records:
        raw_text = _first_present(row, _X_FIELD_CANDIDATES["text"])
        text = _collapse_text(raw_text)
        if not text:
            continue

        raw_ts = _first_present(row, _X_FIELD_CANDIDATES["timestamp"])
        timestamp = pd.to_datetime(raw_ts, utc=True, errors="coerce")
        if pd.isna(timestamp):
            continue
        timestamp = timestamp.tz_localize(None) if timestamp.tzinfo else timestamp

        raw_id = _first_present(row, _X_FIELD_CANDIDATES["id"])
        source_item_id = str(raw_id) if raw_id is not None else _stable_text_id(text + str(raw_ts))
        raw_user = _first_present(row, _X_FIELD_CANDIDATES["user_id"])
        source_user_id = str(raw_user) if raw_user is not None else "unknown"

        rows.append(
            {
                "tweet_id": _stable_positive_int64(f"x:{source_item_id}"),
                "user_id": _stable_positive_int64(f"x-user:{source_user_id}"),
                "text": text,
                "timestamp": timestamp,
                "date": timestamp.strftime("%Y-%m-%d"),
                "source": "x",
                "source_item_id": source_item_id,
                "source_user_id": source_user_id,
                "source_channel_id": None,
                "media_type": None,
                "forward_from_user_id": None,
                "forward_from_username": None,
                "topic_label": _x_topic_from_row(row, topic_label),
                "topic_confidence": None,
                "topic_reason": None,
            }
        )

    if not rows:
        return _empty_canonical_frame()
    return pd.DataFrame(rows)[CANONICAL_COLUMNS].reset_index(drop=True)


def _stable_text_id(value: str) -> str:
    return hashlib.blake2b(value.encode("utf-8"), digest_size=8).hexdigest()




def normalise_telegram_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return _empty_canonical_frame()

    working = frame.copy()
    working["channel_id"] = _coerce_int_series(working["channel_id"])
    working["message_id"] = _coerce_int_series(working["message_id"])
    working["text"] = working["text"].map(_collapse_text)
    working["timestamp"] = _parse_timestamps(working["timestamp"])
    working = working.loc[
        working["channel_id"].notna()
        & working["message_id"].notna()
        & working["timestamp"].notna()
        & working["text"].ne("")
    ].copy()

    if working.empty:
        return _empty_canonical_frame()

    channel_ids = working["channel_id"].astype("int64")
    message_ids = working["message_id"].astype("int64")

    working["tweet_id"] = [
        _stable_positive_int64(f"telegram:{channel_id}:{message_id}")
        for channel_id, message_id in zip(channel_ids, message_ids)
    ]
    working["user_id"] = [
        _stable_positive_int64(f"telegram-channel:{channel_id}")
        for channel_id in channel_ids
    ]
    working["date"] = working["timestamp"].dt.strftime("%Y-%m-%d")
    working["source"] = "telegram"
    working["source_item_id"] = message_ids.astype("string")
    working["source_user_id"] = channel_ids.astype("string")
    working["source_channel_id"] = channel_ids.astype("string")
    working["media_type"] = working.get("media", pd.Series([None] * len(working))).map(
        lambda value: _nested_value(value, "type")
    )
    working["forward_from_user_id"] = working.get(
        "forward_from",
        pd.Series([None] * len(working)),
    ).map(lambda value: _nested_value(value, "user_id"))
    working["forward_from_username"] = working.get(
        "forward_from",
        pd.Series([None] * len(working)),
    ).map(lambda value: _nested_value(value, "username"))
    classification = working.get("war_classification", pd.Series([None] * len(working)))
    working["topic_label"] = classification.map(_topic_label_from_classification)
    working["topic_confidence"] = classification.map(_topic_confidence_from_classification)
    working["topic_reason"] = classification.map(_topic_reason_from_classification)
    return working[CANONICAL_COLUMNS].reset_index(drop=True)


def load_telegram_messages_from_mongo(
    mongo_uri: str,
    db_name: str,
    collection_name: str,
    query: dict[str, Any] | None = None,
) -> pd.DataFrame:
    try:
        from pymongo import MongoClient
    except ImportError as exc:
        raise RuntimeError("pymongo is required to read Telegram data from MongoDB") from exc

    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    try:
        client.admin.command("ping")
        rows = list(client[db_name][collection_name].find(query or {}, {"_id": 0}))
    finally:
        client.close()
    return pd.DataFrame(rows)


def load_mongo_collection(
    mongo_uri: str,
    db_name: str,
    collection_name: str,
    since: "datetime | None" = None,
    timestamp_fields: list[str] | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """Load a MongoDB collection, optionally only documents newer than `since`.

    Because timestamp field names vary, when `since` is given we fetch and
    filter in Python against several candidate timestamp fields rather than
    relying on a single indexed field.
    """
    try:
        from pymongo import MongoClient
    except ImportError as exc:
        raise RuntimeError("pymongo is required to read data from MongoDB") from exc

    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=8000)
    try:
        client.admin.command("ping")
        cursor = client[db_name][collection_name].find({}, {"_id": 0})
        if limit:
            cursor = cursor.limit(int(limit))
        rows = list(cursor)
    finally:
        client.close()

    frame = pd.DataFrame(rows)
    if frame.empty or since is None:
        return frame

    fields = timestamp_fields or _X_FIELD_CANDIDATES["timestamp"]
    present = [f for f in fields if f in frame.columns]
    if not present:
        return frame
    parsed = None
    for field in present:
        col = pd.to_datetime(frame[field], utc=True, errors="coerce")
        parsed = col if parsed is None else parsed.fillna(col)
    since_ts = pd.Timestamp(since)
    if since_ts.tzinfo is None:
        since_ts = since_ts.tz_localize("UTC")
    return frame.loc[parsed.notna() & (parsed >= since_ts)].reset_index(drop=True)


def load_telegram_messages_from_jsonl(local_data_dir: Path | str) -> pd.DataFrame:
    directory = Path(local_data_dir)
    if not directory.exists():
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for path in sorted(directory.glob("channel_*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                payload = line.strip()
                if not payload:
                    continue
                rows.append(json.loads(payload))
    return pd.DataFrame(rows)


def load_telegram_messages(
    source: str = "auto",
    mongo_uri: str | None = None,
    db_name: str | None = None,
    collection_name: str | None = None,
    local_data_dir: Path | str | None = None,
) -> pd.DataFrame:
    if source not in {"auto", "mongo", "jsonl"}:
        raise ValueError(f"Unsupported Telegram source: {source}")

    defaults = _load_repo_telegram_defaults()
    resolved_mongo_uri = mongo_uri or defaults.get("mongo_uri")
    resolved_db_name = db_name or defaults.get("mongo_db_name") or "telegram_data"
    resolved_collection_name = collection_name or defaults.get("mongo_collection_name") or "messages"
    resolved_local_data_dir = Path(local_data_dir or defaults.get("local_data_dir") or (REPO_ROOT / "data"))

    if source in {"auto", "mongo"} and resolved_mongo_uri:
        try:
            frame = load_telegram_messages_from_mongo(
                mongo_uri=resolved_mongo_uri,
                db_name=resolved_db_name,
                collection_name=resolved_collection_name,
            )
            if source == "mongo" or not frame.empty:
                return frame
        except Exception:
            if source == "mongo":
                raise

    if source in {"auto", "jsonl"}:
        return load_telegram_messages_from_jsonl(resolved_local_data_dir)

    return pd.DataFrame()


def export_telegram_dataset(
    output_path: Path | str,
    source: str = "auto",
    mongo_uri: str | None = None,
    db_name: str | None = None,
    collection_name: str | None = None,
    local_data_dir: Path | str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    target = _prepare_output_path(output_path, overwrite=overwrite)
    telegram_raw = load_telegram_messages(
        source=source,
        mongo_uri=mongo_uri,
        db_name=db_name,
        collection_name=collection_name,
        local_data_dir=local_data_dir,
    )
    telegram_frame = normalise_telegram_frame(telegram_raw)
    _write_canonical_frame(telegram_frame, target)

    return {
        "output_path": str(target),
        "source": source,
        "raw_rows": int(len(telegram_raw)),
        "telegram_rows": int(len(telegram_frame)),
        "date_range": {
            "min": telegram_frame["date"].min() if not telegram_frame.empty else None,
            "max": telegram_frame["date"].max() if not telegram_frame.empty else None,
        },
    }


def build_combined_dataset(
    output_path: Path | str,
    twitter_input_path: Path | str | None = None,
    telegram_source: str = "auto",
    telegram_mongo_uri: str | None = None,
    telegram_db_name: str | None = None,
    telegram_collection_name: str | None = None,
    telegram_local_data_dir: Path | str | None = None,
    batch_size: int = 100_000,
    overwrite: bool = False,
) -> dict[str, Any]:
    resolved_twitter_path = twitter_input_path or default_twitter_dataset_path()
    if resolved_twitter_path is None:
        raise FileNotFoundError(
            "Twitter parquet dataset not found. Pass --twitter-input explicitly or place tweets_final.parquet in a known location."
        )
    twitter_path = Path(resolved_twitter_path)
    if not twitter_path.exists():
        raise FileNotFoundError(
            "Twitter parquet dataset not found. Pass --twitter-input explicitly or place tweets_final.parquet in a known location."
        )

    target = _prepare_output_path(output_path, overwrite=overwrite)

    twitter_rows = 0
    twitter_batches = 0
    for batch in iter_twitter_batches(twitter_path, batch_size=batch_size):
        normalised_batch = normalise_twitter_frame(batch)
        twitter_rows += len(normalised_batch)
        twitter_batches += 1
        _write_canonical_frame(normalised_batch, target)

    telegram_raw = load_telegram_messages(
        source=telegram_source,
        mongo_uri=telegram_mongo_uri,
        db_name=telegram_db_name,
        collection_name=telegram_collection_name,
        local_data_dir=telegram_local_data_dir,
    )
    telegram_frame = normalise_telegram_frame(telegram_raw)
    _write_canonical_frame(telegram_frame, target)

    total_rows = twitter_rows + len(telegram_frame)
    return {
        "output_path": str(target),
        "twitter_input_path": str(twitter_path),
        "twitter_batches": twitter_batches,
        "twitter_rows": int(twitter_rows),
        "telegram_rows": int(len(telegram_frame)),
        "telegram_raw_rows": int(len(telegram_raw)),
        "total_rows": int(total_rows),
        "source_counts": {
            "twitter": int(twitter_rows),
            "telegram": int(len(telegram_frame)),
        },
    }


# X (Twitter) collections in the news_monitoring database, mapped to topics.
# The per-document monitorTopic field is authoritative; this is the fallback.
DEFAULT_X_COLLECTIONS = {
    "x_russia_ukraine_posts": "russia_ukraine_war",
    "x_us_iran_posts": "us_iran_war",
}


def build_recent_window_dataset(
    output_path: Path | str,
    since_days: float = 2.0,
    mongo_uri: str | None = None,
    news_mongo_uri: str | None = None,
    news_db_name: str | None = None,
    x_collections: dict[str, str] | None = None,
    telegram_db_name: str | None = None,
    telegram_collection_name: str | None = None,
    telegram_local_data_dir: Path | str | None = None,
    overwrite: bool = True,
) -> dict[str, Any]:
    """Build a canonical parquet of only the most recent posts across sources.

    This powers the "latest clusters" flow: instead of re-clustering the full
    historical corpus (see lsh_full), pull a recent time window from Telegram +
    X and run the lightweight LSH pipeline on just that window.

    Connections:
    - X lives in `news_monitoring`; set NEWS_MONGO_URI if it is on a different
      cluster than Telegram. Falls back to MONGO_URI.
    - Telegram uses MONGO_URI.

    Sources:
    - Telegram: MongoDB collection (realtime), filtered to the recent window.
    - X: news_monitoring collections (one per topic), filtered to the window.
    """
    import os

    defaults = _load_repo_telegram_defaults()
    telegram_uri = mongo_uri or defaults.get("mongo_uri") or os.getenv("MONGO_URI")
    x_uri = news_mongo_uri or os.getenv("NEWS_MONGO_URI") or telegram_uri
    if not x_uri:
        raise ValueError(
            "A MongoDB URI is required. Set NEWS_MONGO_URI (for X) and/or MONGO_URI in .env."
        )
    resolved_news_db = news_db_name or os.getenv("NEWS_DB_NAME") or "news_monitoring"

    since = datetime.utcnow() - timedelta(days=since_days)
    collections = x_collections or DEFAULT_X_COLLECTIONS

    target = _prepare_output_path(output_path, overwrite=overwrite)
    frames: list[pd.DataFrame] = []
    source_counts: dict[str, int] = {}

    # --- X collections (one per topic) ---
    for collection_name, topic_label in collections.items():
        try:
            raw = load_mongo_collection(x_uri, resolved_news_db, collection_name, since=since)
        except Exception as exc:
            source_counts[collection_name] = 0
            print(f"  [warn] could not read {collection_name}: {exc}")
            continue
        normalised = normalise_x_frame(raw, topic_label=topic_label)
        if not normalised.empty:
            frames.append(normalised)
        source_counts[collection_name] = int(len(normalised))

    # --- Telegram (realtime) ---
    tg_db = telegram_db_name or defaults.get("mongo_db_name") or "telegram_data"
    tg_col = telegram_collection_name or defaults.get("mongo_collection_name") or "messages"
    telegram_raw = pd.DataFrame()
    try:
        telegram_raw = load_mongo_collection(
            telegram_uri, tg_db, tg_col, since=since, timestamp_fields=["timestamp", "date", "created_at"]
        )
    except Exception:
        # Fall back to local JSONL if Mongo is unavailable.
        telegram_raw = load_telegram_messages_from_jsonl(
            telegram_local_data_dir or defaults.get("local_data_dir") or (REPO_ROOT / "data")
        )
    telegram_frame = normalise_telegram_frame(telegram_raw)
    if not telegram_frame.empty:
        # Keep only the recent window for Telegram too.
        ts = pd.to_datetime(telegram_frame["timestamp"], errors="coerce")
        telegram_frame = telegram_frame.loc[ts >= pd.Timestamp(since)].reset_index(drop=True)
        if not telegram_frame.empty:
            frames.append(telegram_frame)
    source_counts["telegram"] = int(len(telegram_frame))

    combined = pd.concat(frames, ignore_index=True) if frames else _empty_canonical_frame()
    # Drop cross-source exact-id duplicates that may recur each run.
    if not combined.empty:
        combined = combined.drop_duplicates(subset=["tweet_id"]).reset_index(drop=True)
    _write_canonical_frame(combined, target)

    return {
        "output_path": str(target),
        "since": since.strftime("%Y-%m-%d %H:%M:%S"),
        "since_days": since_days,
        "total_rows": int(len(combined)),
        "source_counts": source_counts,
        "date_range": {
            "min": combined["date"].min() if not combined.empty else None,
            "max": combined["date"].max() if not combined.empty else None,
        },
    }
