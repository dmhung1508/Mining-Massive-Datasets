from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from matplotlib.gridspec import GridSpec

from _bootstrap import add_src_to_path

repo_root = add_src_to_path()

from uk_russia_lsh.datasets import default_twitter_dataset_path, normalise_twitter_frame
from uk_russia_lsh.preprocessing import build_shingles
from uk_russia_lsh.similarity import exact_jaccard_pairs


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


def _sampling_score(value: int | str, seed: int) -> int:
    payload = f"{seed}:{value}".encode("utf-8")
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
    return _prepare_sample_frame(sampled), {"raw_rows": raw_rows, "sampled_rows": int(len(sampled))}


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
        return _sampling_score(str(file_path), seed), str(file_path)

    selected_files = [
        file_path for _, file_path in sorted((file_score(file_path), file_path) for file_path in parquet_files)[:file_limit]
    ]

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
    return _prepare_sample_frame(sampled), {
        "raw_rows": raw_rows,
        "sampled_rows": int(len(sampled)),
        "files_scanned": len(selected_files),
    }


def _style_axes(ax) -> None:
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_color(EDGE)
    ax.tick_params(colors=MUTED)
    ax.title.set_color(TEXT)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)


def _build_source_summary(
    sampled: pd.DataFrame,
    processed: pd.DataFrame,
    base_stats: dict[str, dict[str, int]],
) -> dict[str, dict[str, float | int]]:
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


def _benchmark_sizes(total_docs: int, explicit_sizes: list[int]) -> list[int]:
    sizes = {size for size in explicit_sizes if 0 < size <= total_docs}
    sizes.add(total_docs)
    return sorted(sizes)


def _benchmark_runtime(processed: pd.DataFrame, threshold: float, sizes: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = processed.copy()
    ordered["_benchmark_score"] = ordered["tweet_id"].map(lambda value: _sampling_score(int(value), 999))
    ordered = ordered.sort_values(["_benchmark_score", "tweet_id"]).drop(columns="_benchmark_score").reset_index(drop=True)

    benchmark_rows = []
    full_pairs = pd.DataFrame(columns=["tweet_id_left", "tweet_id_right", "jaccard"])
    for size in sizes:
        subset = ordered.iloc[:size].copy()
        pairs, metrics = exact_jaccard_pairs(subset, threshold=threshold)
        benchmark_rows.append(
            {
                "docs": size,
                "pairs_considered": int(metrics["pairs_considered"]),
                "positive_pairs": int(metrics["positive_pairs"]),
                "runtime_seconds": float(metrics["runtime_seconds"]),
                "pairs_per_second": round(
                    float(metrics["pairs_considered"]) / float(metrics["runtime_seconds"]),
                    2,
                )
                if float(metrics["runtime_seconds"]) > 0
                else None,
            }
        )
        if size == len(ordered):
            full_pairs = pairs

    return pd.DataFrame(benchmark_rows), full_pairs


def _pair_label(row: pd.Series) -> str:
    left_suffix = str(row["tweet_id_left"])[-6:]
    right_suffix = str(row["tweet_id_right"])[-6:]
    source_pair = str(row["source_pair"]).replace("twitter", "TW").replace("telegram", "TE").replace("cross", "X")
    return f"{source_pair} | {left_suffix}-{right_suffix}"


def _pair_source(left_source: str, right_source: str) -> str:
    if left_source == right_source:
        return f"{left_source}-{right_source}"
    return "cross-source"


def _enrich_pairs(pairs: pd.DataFrame, processed: pd.DataFrame) -> pd.DataFrame:
    if pairs.empty:
        return pd.DataFrame(
            columns=[
                "tweet_id_left",
                "tweet_id_right",
                "jaccard",
                "source_left",
                "source_right",
                "source_pair",
                "date_left",
                "date_right",
                "text_left",
                "text_right",
            ]
        )

    docs = processed[["tweet_id", "source", "date", "text", "token_count", "shingle_count"]].copy()
    left_meta = docs.rename(
        columns={
            "tweet_id": "tweet_id_left",
            "source": "source_left",
            "date": "date_left",
            "text": "text_left",
            "token_count": "token_count_left",
            "shingle_count": "shingle_count_left",
        }
    )
    right_meta = docs.rename(
        columns={
            "tweet_id": "tweet_id_right",
            "source": "source_right",
            "date": "date_right",
            "text": "text_right",
            "token_count": "token_count_right",
            "shingle_count": "shingle_count_right",
        }
    )
    enriched = pairs.merge(left_meta, on="tweet_id_left", how="left").merge(right_meta, on="tweet_id_right", how="left")
    enriched["source_pair"] = [
        _pair_source(str(left), str(right))
        for left, right in zip(enriched["source_left"], enriched["source_right"])
    ]
    enriched["pair_label"] = enriched.apply(_pair_label, axis=1)
    return enriched.sort_values(["jaccard", "tweet_id_left", "tweet_id_right"], ascending=[False, True, True]).reset_index(
        drop=True
    )


def _write_docs_csv(processed: pd.DataFrame, path: Path) -> None:
    export = processed.copy()
    export["tokens"] = export["tokens"].map(lambda values: json.dumps(values, ensure_ascii=False))
    export["shingles"] = export["shingles"].map(lambda values: json.dumps(values, ensure_ascii=False))
    path.parent.mkdir(parents=True, exist_ok=True)
    export.to_csv(
        path,
        index=False,
        columns=[
            "source",
            "date",
            "tweet_id",
            "token_count",
            "shingle_count",
            "text",
            "norm_text",
            "tokens",
            "shingles",
        ],
    )


def _draw_empty_panel(ax, title: str, message: str) -> None:
    _style_axes(ax)
    ax.set_title(title, fontsize=15, pad=12)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=11, color=MUTED)


def build_week3_dashboard(
    telegram_input: Path,
    output_png: Path,
    summary_json: Path,
    benchmark_csv: Path,
    pairs_csv: Path,
    docs_csv: Path,
    twitter_input: Path | None = None,
    sample_per_source: int = 200,
    threshold: float = 0.8,
    shingle_size: int = 3,
    seed: int = 42,
    twitter_file_limit: int = 24,
    benchmark_sizes: list[int] | None = None,
) -> dict[str, object]:
    sampled_frames: list[pd.DataFrame] = []
    base_stats: dict[str, dict[str, int]] = {}

    if telegram_input.exists():
        telegram_sample, telegram_stats = _load_telegram_frame(telegram_input, sample_size=sample_per_source, seed=seed)
        if not telegram_sample.empty:
            sampled_frames.append(telegram_sample)
        base_stats["telegram"] = telegram_stats

    if twitter_input and twitter_input.exists():
        twitter_sample, twitter_stats = _sample_twitter_frame(
            twitter_input,
            sample_size=sample_per_source,
            seed=seed + 1,
            file_limit=twitter_file_limit,
        )
        if not twitter_sample.empty:
            sampled_frames.append(twitter_sample)
        base_stats["twitter"] = twitter_stats

    if not sampled_frames:
        raise FileNotFoundError("No usable Telegram/Twitter data found for week 3 baseline run.")

    sampled = pd.concat(sampled_frames, ignore_index=True)
    processed = build_shingles(sampled, shingle_size=shingle_size)
    if processed.empty:
        raise ValueError("Week 3 preprocessing kept 0 rows after shingling. Increase sample size or lower shingle size.")

    _write_docs_csv(processed, docs_csv)
    source_stats = _build_source_summary(sampled, processed, base_stats)

    runtime_sizes = _benchmark_sizes(len(processed), benchmark_sizes or [100, 200, 300])
    benchmark_df, full_pairs = _benchmark_runtime(processed, threshold=threshold, sizes=runtime_sizes)
    pairs_with_meta = _enrich_pairs(full_pairs, processed)

    benchmark_csv.parent.mkdir(parents=True, exist_ok=True)
    benchmark_df.to_csv(benchmark_csv, index=False)
    pairs_csv.parent.mkdir(parents=True, exist_ok=True)
    pairs_with_meta.to_csv(pairs_csv, index=False)

    final_metrics = benchmark_df.loc[benchmark_df["docs"] == len(processed)].iloc[0]
    source_pair_counts = (
        pairs_with_meta["source_pair"].value_counts().rename_axis("source_pair").reset_index(name="count")
        if not pairs_with_meta.empty
        else pd.DataFrame(
            {
                "source_pair": ["twitter-twitter", "telegram-telegram", "cross-source"],
                "count": [0, 0, 0],
            }
        )
    )
    for source_pair in ["twitter-twitter", "telegram-telegram", "cross-source"]:
        if source_pair not in source_pair_counts["source_pair"].tolist():
            source_pair_counts.loc[len(source_pair_counts)] = [source_pair, 0]
    source_pair_counts = source_pair_counts.sort_values("count", ascending=False).reset_index(drop=True)

    top_pairs = []
    if not pairs_with_meta.empty:
        for row in pairs_with_meta.head(8).itertuples(index=False):
            top_pairs.append(
                {
                    "tweet_id_left": int(row.tweet_id_left),
                    "tweet_id_right": int(row.tweet_id_right),
                    "source_pair": str(row.source_pair),
                    "jaccard": round(float(row.jaccard), 4),
                    "text_left": str(row.text_left)[:180],
                    "text_right": str(row.text_right)[:180],
                }
            )

    fig = plt.figure(figsize=(18, 12), facecolor=BG)
    gs = GridSpec(2, 2, figure=fig, hspace=0.28, wspace=0.18)
    fig.suptitle(
        "Week 3 Exact Jaccard Baseline",
        fontsize=22,
        fontweight="bold",
        color=TEXT,
        y=0.97,
    )
    fig.text(
        0.5,
        0.94,
        "Brute-force similarity on a deterministic mixed subset with runtime benchmark and duplicate-pair summary",
        ha="center",
        color=MUTED,
        fontsize=11,
    )

    ax1 = fig.add_subplot(gs[0, 0])
    _style_axes(ax1)
    ax1.plot(benchmark_df["docs"], benchmark_df["runtime_seconds"], marker="o", linewidth=2, color=ACCENT1)
    ax1.set_title("Brute-force Runtime", fontsize=15, pad=12)
    ax1.set_xlabel("Documents")
    ax1.set_ylabel("Runtime (seconds)")
    ax1.grid(axis="y", color=EDGE, alpha=0.6)
    for row in benchmark_df.itertuples(index=False):
        ax1.annotate(
            f"{int(row.pairs_considered):,} pairs",
            (row.docs, row.runtime_seconds),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
            color=MUTED,
        )

    ax2 = fig.add_subplot(gs[0, 1])
    _style_axes(ax2)
    funnel_labels = ["Sampled", "Shingled", "Pairs", ">= threshold"]
    funnel_values = [
        int(len(sampled)),
        int(len(processed)),
        int(final_metrics["pairs_considered"]),
        int(final_metrics["positive_pairs"]),
    ]
    ax2.bar(funnel_labels, funnel_values, color=[ACCENT2, ACCENT1, ACCENT3, ACCENT4])
    ax2.set_yscale("log")
    ax2.set_title("Baseline Funnel", fontsize=15, pad=12)
    ax2.set_ylabel("Count (log scale)")
    ax2.grid(axis="y", color=EDGE, alpha=0.6)
    ax2.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))

    ax3 = fig.add_subplot(gs[1, 0])
    _style_axes(ax3)
    ax3.barh(
        source_pair_counts["source_pair"][::-1],
        source_pair_counts["count"][::-1],
        color=[ACCENT3, ACCENT2, ACCENT1][: len(source_pair_counts)],
    )
    ax3.set_title("Positive Pairs By Source Type", fontsize=15, pad=12)
    ax3.set_xlabel("Pairs")
    ax3.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))

    ax4 = fig.add_subplot(gs[1, 1])
    if pairs_with_meta.empty:
        _draw_empty_panel(ax4, "Top Duplicate Pairs", f"No pairs found with Jaccard >= {threshold:.2f}")
    else:
        _style_axes(ax4)
        top_plot = pairs_with_meta.head(8).copy().iloc[::-1]
        ax4.barh(top_plot["pair_label"], top_plot["jaccard"], color=ACCENT4)
        ax4.set_title("Top Duplicate Pairs", fontsize=15, pad=12)
        ax4.set_xlabel("Jaccard")
        ax4.set_xlim(0.0, 1.05)
        for row in top_plot.itertuples(index=False):
            ax4.text(
                row.jaccard + 0.015,
                row.pair_label,
                f"{row.jaccard:.2f}",
                va="center",
                ha="left",
                fontsize=9,
                color=TEXT,
            )

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=170, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "config": {
            "sample_per_source": sample_per_source,
            "threshold": threshold,
            "shingle_size": shingle_size,
            "seed": seed,
            "twitter_file_limit": twitter_file_limit,
            "benchmark_sizes": runtime_sizes,
        },
        "sources": source_stats,
        "final_run": {
            "documents": int(len(processed)),
            "pairs_considered": int(final_metrics["pairs_considered"]),
            "positive_pairs": int(final_metrics["positive_pairs"]),
            "runtime_seconds": round(float(final_metrics["runtime_seconds"]), 6),
            "pairs_per_second": round(float(final_metrics["pairs_per_second"]), 2)
            if pd.notna(final_metrics["pairs_per_second"])
            else None,
        },
        "source_pair_counts": [
            {"source_pair": str(row.source_pair), "count": int(row.count)}
            for row in source_pair_counts.itertuples(index=False)
        ],
        "top_pairs": top_pairs,
        "paths": {
            "telegram_input": str(telegram_input),
            "twitter_input": str(twitter_input) if twitter_input else None,
            "docs_csv": str(docs_csv),
            "pairs_csv": str(pairs_csv),
            "benchmark_csv": str(benchmark_csv),
            "dashboard_png": str(output_png),
            "summary_json": str(summary_json),
        },
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Create week-3 brute-force baseline artifacts and dashboard.")
    parser.add_argument("--telegram-input", default=str(repo_root / "datatele" / "telegram_messages.parquet"))
    parser.add_argument("--twitter-input", default=str(default_twitter_dataset_path() or ""))
    parser.add_argument("--sample-per-source", type=int, default=200)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--shingle-size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--twitter-file-limit", type=int, default=24)
    parser.add_argument("--benchmark-sizes", default="100,200,300")
    parser.add_argument("--output-png", default=str(repo_root / "datatele" / "week3_overview.png"))
    parser.add_argument("--summary-json", default=str(repo_root / "datatele" / "week3_summary.json"))
    parser.add_argument("--benchmark-csv", default=str(repo_root / "datatele" / "week3_runtime_benchmark.csv"))
    parser.add_argument("--pairs-csv", default=str(repo_root / "datatele" / "week3_baseline_pairs.csv"))
    parser.add_argument("--docs-csv", default=str(repo_root / "datatele" / "week3_baseline_docs.csv"))
    args = parser.parse_args()

    benchmark_sizes = [int(value.strip()) for value in args.benchmark_sizes.split(",") if value.strip()]
    twitter_input = Path(args.twitter_input) if args.twitter_input else None
    summary = build_week3_dashboard(
        telegram_input=Path(args.telegram_input),
        twitter_input=twitter_input,
        output_png=Path(args.output_png),
        summary_json=Path(args.summary_json),
        benchmark_csv=Path(args.benchmark_csv),
        pairs_csv=Path(args.pairs_csv),
        docs_csv=Path(args.docs_csv),
        sample_per_source=args.sample_per_source,
        threshold=args.threshold,
        shingle_size=args.shingle_size,
        seed=args.seed,
        twitter_file_limit=args.twitter_file_limit,
        benchmark_sizes=benchmark_sizes,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
