from __future__ import annotations

import hashlib
import json
import runpy
import shutil
import sys
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from .constants import REPO_ROOT


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
        REPO_ROOT / "jupyter" / "output" / "tweets_final.parquet",
        Path("/home/anonymous/code/Mining_Massive_Dataset/tweets_final.parquet"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def default_telegram_export_path() -> Path:
    return REPO_ROOT / "jupyter" / "output" / "telegram_messages.parquet"


def default_combined_dataset_path() -> Path:
    return REPO_ROOT / "jupyter" / "output" / "combined_social.parquet"


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
    return working[CANONICAL_COLUMNS].reset_index(drop=True)


def _nested_value(value: object, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    return None


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
