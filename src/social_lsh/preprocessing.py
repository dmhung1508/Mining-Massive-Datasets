from __future__ import annotations

import json
import re
from typing import Iterable

import pandas as pd


URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?", re.IGNORECASE)


def normalize_text(text: object) -> str:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    normalized = str(text).lower()
    normalized = URL_RE.sub(" ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def tokenize_text(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def make_word_shingles(tokens: Iterable[str], k: int) -> list[str]:
    token_list = list(tokens)
    if len(token_list) < k:
        return []
    return [" ".join(token_list[idx : idx + k]) for idx in range(len(token_list) - k + 1)]


def build_shingles(df: pd.DataFrame, shingle_size: int = 3) -> pd.DataFrame:
    working = df.copy()
    working["norm_text"] = working["text"].map(normalize_text)
    working["tokens"] = working["norm_text"].map(tokenize_text)
    working["token_count"] = working["tokens"].map(len)
    working["shingles"] = working["tokens"].map(lambda tokens: make_word_shingles(tokens, shingle_size))
    working["shingle_count"] = working["shingles"].map(len)
    working = working.loc[working["token_count"] >= shingle_size].reset_index(drop=True)
    return working


def serialize_nested_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    serialised = df.copy()
    for column in columns:
        serialised[column] = serialised[column].map(json.dumps)
    return serialised


def deserialize_nested_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    hydrated = df.copy()
    for column in columns:
        hydrated[column] = hydrated[column].map(json.loads)
    return hydrated
