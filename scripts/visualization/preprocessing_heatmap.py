from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from matplotlib.gridspec import GridSpec


repo_root = Path(__file__).resolve().parents[2]

from social_lsh.datasets import (
    default_twitter_dataset_path,
    normalise_twitter_frame,
)
from social_lsh.preprocessing import build_shingles
from social_lsh.similarity import jaccard_similarity


BG = "#f7f3ea"
PANEL = "#fffdf8"
EDGE = "#d7c8b1"
TEXT = "#2d2218"
MUTED = "#7c6856"
ACCENT1 = "#1e5f74"
ACCENT2 = "#c26d3a"
ACCENT3 = "#7b8f4e"
ACCENT4 = "#c7a33c"
STRING_COLUMNS = [
    "text",
    "date",
    "source",
    "source_item_id",
    "source_user_id",
    "source_channel_id",
    "media_type",
    "forward_from_user_id",
    "forward_from_username",
]


def _sampling_score(tweet_id: int, seed: int) -> int:
    payload = f"{seed}:{tweet_id}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), byteorder="big", signed=False)


def _sample_smallest_scores(frame: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    if frame.empty or sample_size <= 0:
        return frame.iloc[0:0].copy()

    working = frame.copy()
    working["_score"] = working["tweet_id"].map(lambda value: _sampling_score(int(value), seed))
    sampled = working.nsmallest(min(sample_size, len(working)), ["_score", "tweet_id"]).drop(columns="_score")
    return sampled.reset_index(drop=True)


def _prepare_sample_frame(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    for column in STRING_COLUMNS:
        if column in working.columns:
            working[column] = working[column].astype("string")
    return working


def _load_telegram_frame(path: Path, sample_size: int, seed: int) -> tuple[pd.DataFrame, dict[str, int]]:
    frame = pd.read_parquet(path)
    raw_rows = int(len(frame))
    sampled = _sample_smallest_scores(frame, sample_size=sample_size, seed=seed)
    return sampled, {"raw_rows": raw_rows, "sampled_rows": int(len(sampled))}


def _sample_twitter_frame(
    path: Path,
    sample_size: int,
    seed: int,
    file_limit: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    parquet_files = sorted(path.rglob("*.parquet"))
    if not parquet_files:
        return pd.DataFrame(), {"raw_rows": 0, "sampled_rows": 0, "files_scanned": 0}

    def file_score(file_path: Path) -> tuple[int, str]:
        digest = hashlib.blake2b(f"{seed}:{file_path}".encode("utf-8"), digest_size=8).digest()
        score = int.from_bytes(digest, byteorder="big", signed=False)
        return score, str(file_path)

    selected_files = [path for _, path in sorted((file_score(file_path), file_path) for file_path in parquet_files)[:file_limit]]
    candidates: list[pd.DataFrame] = []
    raw_rows = 0

    for parquet_file in selected_files:
        frame = pd.read_parquet(parquet_file)
        if "date" not in frame.columns and parquet_file.parent.name.startswith("date="):
            frame["date"] = parquet_file.parent.name.replace("date=", "")

        normalised = normalise_twitter_frame(frame)
        if normalised.empty:
            continue
        raw_rows += len(normalised)
        candidates.append(_sample_smallest_scores(normalised, sample_size=sample_size, seed=seed))
        if len(candidates) >= 8:
            merged = pd.concat(candidates, ignore_index=True)
            candidates = [_sample_smallest_scores(merged, sample_size=sample_size, seed=seed)]

    if not candidates:
        return pd.DataFrame(), {"raw_rows": raw_rows, "sampled_rows": 0, "files_scanned": len(selected_files)}

    sampled = _sample_smallest_scores(pd.concat(candidates, ignore_index=True), sample_size=sample_size, seed=seed)
    return sampled, {"raw_rows": raw_rows, "sampled_rows": int(len(sampled)), "files_scanned": len(selected_files)}


def _style_axes(ax) -> None:
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_color(EDGE)
    ax.tick_params(colors=MUTED)
    ax.title.set_color(TEXT)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)


def _pairwise_jaccard(frame: pd.DataFrame) -> pd.DataFrame:
    if len(frame) < 2:
        return pd.DataFrame(columns=["tweet_id_left", "tweet_id_right", "jaccard"])

    records = []
    items = [
        (int(row.tweet_id), set(row.shingles))
        for row in frame[["tweet_id", "shingles"]].itertuples(index=False)
    ]
    for index, (left_id, left_set) in enumerate(items):
        for right_id, right_set in items[index + 1 :]:
            records.append(
                {
                    "tweet_id_left": left_id,
                    "tweet_id_right": right_id,
                    "jaccard": jaccard_similarity(left_set, right_set),
                }
            )
    pairs = pd.DataFrame(records)
    if pairs.empty:
        return pd.DataFrame(columns=["tweet_id_left", "tweet_id_right", "jaccard"])
    return pairs.sort_values(["jaccard", "tweet_id_left", "tweet_id_right"], ascending=[False, True, True]).reset_index(
        drop=True
    )


def _select_heatmap_frame(processed: pd.DataFrame, pairs: pd.DataFrame, heatmap_docs: int) -> pd.DataFrame:
    if processed.empty:
        return processed.copy()

    selected_ids: list[int] = []
    if not pairs.empty:
        positive_pairs = pairs.loc[pairs["jaccard"] > 0].copy()
        for row in positive_pairs.itertuples(index=False):
            for tweet_id in (int(row.tweet_id_left), int(row.tweet_id_right)):
                if tweet_id not in selected_ids:
                    selected_ids.append(tweet_id)
                if len(selected_ids) >= heatmap_docs:
                    break
            if len(selected_ids) >= heatmap_docs:
                break

    if len(selected_ids) < min(heatmap_docs, len(processed)):
        fallback_ids = (
            processed.sort_values(["token_count", "shingle_count", "tweet_id"], ascending=[False, False, True])[
                "tweet_id"
            ]
            .astype(int)
            .tolist()
        )
        for tweet_id in fallback_ids:
            if tweet_id not in selected_ids:
                selected_ids.append(tweet_id)
            if len(selected_ids) >= min(heatmap_docs, len(processed)):
                break

    lookup = processed.set_index("tweet_id")
    selected = lookup.loc[selected_ids].reset_index()
    selected["doc_label"] = [f"{row.source[:2].upper()}-{index + 1:02d}" for index, row in enumerate(selected.itertuples())]
    return selected


def _similarity_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    labels = frame["doc_label"].tolist()
    shingle_sets = [set(values) for values in frame["shingles"]]
    rows = []
    for left_set in shingle_sets:
        rows.append([jaccard_similarity(left_set, right_set) for right_set in shingle_sets])
    return pd.DataFrame(rows, index=labels, columns=labels)


def _serialize_preview(frame: pd.DataFrame) -> pd.DataFrame:
    preview = frame.copy()
    preview["tokens"] = preview["tokens"].map(lambda values: json.dumps(values, ensure_ascii=False))
    preview["shingles"] = preview["shingles"].map(lambda values: json.dumps(values, ensure_ascii=False))
    return preview[
        [
            "doc_label",
            "selected_for_heatmap",
            "source",
            "date",
            "tweet_id",
            "token_count",
            "shingle_count",
            "text",
            "norm_text",
            "tokens",
            "shingles",
        ]
    ]


def _source_summary(sampled: pd.DataFrame, processed: pd.DataFrame, base_stats: dict[str, dict[str, int]]) -> dict[str, dict[str, float | int]]:
    summary: dict[str, dict[str, float | int]] = {}
    for source, metrics in base_stats.items():
        source_sampled = sampled.loc[sampled["source"] == source]
        source_processed = processed.loc[processed["source"] == source]
        summary[source] = {
            **metrics,
            "processed_rows": int(len(source_processed)),
            "dropped_short_rows": int(len(source_sampled) - len(source_processed)),
            "avg_token_count": round(float(source_processed["token_count"].mean()), 2) if not source_processed.empty else 0.0,
            "avg_shingle_count": round(float(source_processed["shingle_count"].mean()), 2)
            if not source_processed.empty
            else 0.0,
        }
    return summary


def build_week2_dashboard(
    telegram_input: Path,
    output_png: Path,
    summary_json: Path,
    preview_csv: Path,
    matrix_csv: Path,
    twitter_input: Path | None = None,
    sample_per_source: int = 120,
    heatmap_docs: int = 12,
    shingle_size: int = 3,
    seed: int = 42,
    twitter_file_limit: int = 24,
) -> dict[str, object]:
    sampled_frames: list[pd.DataFrame] = []
    base_stats: dict[str, dict[str, int]] = {}

    if telegram_input.exists():
        telegram_sample, telegram_stats = _load_telegram_frame(telegram_input, sample_size=sample_per_source, seed=seed)
        if not telegram_sample.empty:
            sampled_frames.append(_prepare_sample_frame(telegram_sample))
        base_stats["telegram"] = telegram_stats

    if twitter_input and twitter_input.exists():
        twitter_sample, twitter_stats = _sample_twitter_frame(
            twitter_input,
            sample_size=sample_per_source,
            seed=seed + 1,
            file_limit=twitter_file_limit,
        )
        if not twitter_sample.empty:
            sampled_frames.append(_prepare_sample_frame(twitter_sample))
        base_stats["twitter"] = twitter_stats

    if not sampled_frames:
        raise FileNotFoundError("No usable Telegram/Twitter data found for week 2 preprocessing.")

    sampled = pd.concat(sampled_frames, ignore_index=True)
    processed = build_shingles(sampled, shingle_size=shingle_size)
    if processed.empty:
        raise ValueError("Week 2 preprocessing kept 0 rows after shingling. Increase sample size or lower shingle size.")
    pairs = _pairwise_jaccard(processed)
    heatmap_frame = _select_heatmap_frame(processed, pairs, heatmap_docs=heatmap_docs)
    matrix = _similarity_matrix(heatmap_frame)

    label_lookup = dict(zip(heatmap_frame["tweet_id"].astype(int), heatmap_frame["doc_label"]))
    processed["doc_label"] = processed["tweet_id"].map(lambda value: label_lookup.get(int(value)))
    processed["selected_for_heatmap"] = processed["doc_label"].notna()

    preview = _serialize_preview(processed)
    preview_csv.parent.mkdir(parents=True, exist_ok=True)
    preview.to_csv(preview_csv, index=False)
    matrix_csv.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(matrix_csv)

    source_stats = _source_summary(sampled, processed, base_stats)
    top_tokens = Counter(token for tokens in processed["tokens"] for token in tokens).most_common(10)
    top_pairs = []
    if not pairs.empty:
        for row in pairs.head(8).itertuples(index=False):
            top_pairs.append(
                {
                    "tweet_id_left": int(row.tweet_id_left),
                    "tweet_id_right": int(row.tweet_id_right),
                    "jaccard": round(float(row.jaccard), 4),
                }
            )

    fig = plt.figure(figsize=(18, 12), facecolor=BG)
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.25, 1.0], hspace=0.28, wspace=0.18)
    fig.suptitle(
        "Week 2 Preprocessing Dashboard",
        fontsize=22,
        fontweight="bold",
        color=TEXT,
        y=0.97,
    )
    fig.text(
        0.5,
        0.94,
        "Normalize + tokenize + shingles + small Jaccard heatmap on a mixed Telegram/Twitter sample",
        ha="center",
        color=MUTED,
        fontsize=11,
    )

    ax1 = fig.add_subplot(gs[:, 0])
    _style_axes(ax1)
    heatmap = ax1.imshow(matrix.values, cmap="YlOrBr", vmin=0.0, vmax=1.0)
    ax1.set_title("Jaccard Similarity Heatmap", fontsize=15, pad=12)
    ax1.set_xticks(range(len(matrix.columns)))
    ax1.set_yticks(range(len(matrix.index)))
    ax1.set_xticklabels(matrix.columns, rotation=45, ha="right", color=TEXT)
    ax1.set_yticklabels(matrix.index, color=TEXT)
    for row_index in range(len(matrix.index)):
        for col_index in range(len(matrix.columns)):
            ax1.text(
                col_index,
                row_index,
                f"{matrix.iloc[row_index, col_index]:.2f}",
                ha="center",
                va="center",
                color=TEXT if matrix.iloc[row_index, col_index] < 0.65 else PANEL,
                fontsize=8,
            )
    plt.colorbar(heatmap, ax=ax1, fraction=0.046, pad=0.04)

    ax2 = fig.add_subplot(gs[0, 1])
    _style_axes(ax2)
    sources = list(source_stats.keys())
    sampled_counts = [source_stats[source]["sampled_rows"] for source in sources]
    processed_counts = [source_stats[source]["processed_rows"] for source in sources]
    positions = list(range(len(sources)))
    ax2.bar([pos - 0.18 for pos in positions], sampled_counts, width=0.36, color=ACCENT2, label="sampled")
    ax2.bar([pos + 0.18 for pos in positions], processed_counts, width=0.36, color=ACCENT1, label="kept after shingles")
    ax2.set_title("Sample Coverage By Source", fontsize=15, pad=12)
    ax2.set_xticks(positions)
    ax2.set_xticklabels([source.title() for source in sources], color=TEXT)
    ax2.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    ax2.set_ylabel("Rows")
    ax2.legend(frameon=False)
    ax2.grid(axis="y", color=EDGE, alpha=0.6)

    ax3 = fig.add_subplot(gs[1, 1])
    _style_axes(ax3)
    token_labels = [token for token, _ in top_tokens]
    token_values = [count for _, count in top_tokens]
    ax3.barh(token_labels[::-1], token_values[::-1], color=ACCENT3)
    ax3.set_title("Top Tokens In Sample", fontsize=15, pad=12)
    ax3.set_xlabel("Frequency")
    ax3.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=170, bbox_inches="tight")
    plt.close(fig)

    non_diagonal_max = 0.0
    if len(matrix.index) > 1:
        off_diagonal = matrix.copy()
        for label in off_diagonal.index:
            off_diagonal.loc[label, label] = None
        non_diagonal_value = off_diagonal.max().max()
        non_diagonal_max = round(float(non_diagonal_value), 4) if pd.notna(non_diagonal_value) else 0.0

    summary = {
        "config": {
            "sample_per_source": sample_per_source,
            "heatmap_docs": int(len(heatmap_frame)),
            "shingle_size": shingle_size,
            "seed": seed,
        },
        "sources": source_stats,
        "processed_rows_total": int(len(processed)),
        "top_tokens": [{"token": token, "count": count} for token, count in top_tokens],
        "top_pairs": top_pairs,
        "heatmap": {
            "labels": heatmap_frame["doc_label"].tolist(),
            "max_off_diagonal_jaccard": non_diagonal_max,
        },
        "paths": {
            "telegram_input": str(telegram_input),
            "twitter_input": str(twitter_input) if twitter_input else None,
            "preview_csv": str(preview_csv),
            "matrix_csv": str(matrix_csv),
            "dashboard_png": str(output_png),
            "summary_json": str(summary_json),
        },
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Create week-2 preprocessing artifacts and similarity heatmap.")
    parser.add_argument(
        "--telegram-input",
        default=str(repo_root / "jupyter" / "output" / "visuals" / "telegram_messages.parquet"),
    )
    parser.add_argument(
        "--twitter-input",
        default=str(default_twitter_dataset_path() or ""),
    )
    parser.add_argument(
        "--sample-per-source",
        type=int,
        default=120,
    )
    parser.add_argument(
        "--heatmap-docs",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--shingle-size",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--twitter-file-limit",
        type=int,
        default=24,
    )
    parser.add_argument(
        "--output-png",
        default=str(repo_root / "jupyter" / "output" / "visuals" / "week2_overview.png"),
    )
    parser.add_argument(
        "--summary-json",
        default=str(repo_root / "jupyter" / "output" / "visuals" / "week2_summary.json"),
    )
    parser.add_argument(
        "--preview-csv",
        default=str(repo_root / "jupyter" / "output" / "visuals" / "week2_preprocessing_sample.csv"),
    )
    parser.add_argument(
        "--matrix-csv",
        default=str(repo_root / "jupyter" / "output" / "visuals" / "week2_similarity_matrix.csv"),
    )
    args = parser.parse_args()

    twitter_input = Path(args.twitter_input) if args.twitter_input else None
    summary = build_week2_dashboard(
        telegram_input=Path(args.telegram_input),
        twitter_input=twitter_input,
        output_png=Path(args.output_png),
        summary_json=Path(args.summary_json),
        preview_csv=Path(args.preview_csv),
        matrix_csv=Path(args.matrix_csv),
        sample_per_source=args.sample_per_source,
        heatmap_docs=args.heatmap_docs,
        shingle_size=args.shingle_size,
        seed=args.seed,
        twitter_file_limit=args.twitter_file_limit,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
