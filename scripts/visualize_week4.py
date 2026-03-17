from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

from _bootstrap import add_src_to_path

repo_root = add_src_to_path()

from uk_russia_lsh.datasets import default_twitter_dataset_path, normalise_twitter_frame
from uk_russia_lsh.minhash import hashed_shingles, signature_matrix
from uk_russia_lsh.preprocessing import build_shingles
from uk_russia_lsh.similarity import jaccard_similarity


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


def _pair_source(left_source: str, right_source: str) -> str:
    if left_source == right_source:
        return f"{left_source}-{right_source}"
    return "cross-source"


def _pair_metrics(processed: pd.DataFrame, signatures: np.ndarray) -> pd.DataFrame:
    rows = []
    records = processed[["tweet_id", "source", "text", "shingles"]].to_dict(orient="records")
    for left_index in range(len(records)):
        left_record = records[left_index]
        left_set = set(left_record["shingles"])
        left_signature = signatures[left_index]
        for right_index in range(left_index + 1, len(records)):
            right_record = records[right_index]
            true_jaccard = jaccard_similarity(left_set, set(right_record["shingles"]))
            minhash_estimate = float((left_signature == signatures[right_index]).mean())
            rows.append(
                {
                    "tweet_id_left": int(left_record["tweet_id"]),
                    "tweet_id_right": int(right_record["tweet_id"]),
                    "source_left": str(left_record["source"]),
                    "source_right": str(right_record["source"]),
                    "source_pair": _pair_source(str(left_record["source"]), str(right_record["source"])),
                    "true_jaccard": round(true_jaccard, 6),
                    "minhash_estimate": round(minhash_estimate, 6),
                    "abs_error": round(abs(true_jaccard - minhash_estimate), 6),
                    "text_left": str(left_record["text"]),
                    "text_right": str(right_record["text"]),
                }
            )
    metrics = pd.DataFrame(rows)
    if metrics.empty:
        return pd.DataFrame(
            columns=[
                "tweet_id_left",
                "tweet_id_right",
                "source_left",
                "source_right",
                "source_pair",
                "true_jaccard",
                "minhash_estimate",
                "abs_error",
                "text_left",
                "text_right",
            ]
        )
    return metrics.sort_values(
        ["true_jaccard", "minhash_estimate", "tweet_id_left", "tweet_id_right"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)


def _preview_doc_ids(processed: pd.DataFrame, pair_metrics: pd.DataFrame, preview_docs: int) -> list[int]:
    selected_ids: list[int] = []
    positive_pairs = pair_metrics.loc[pair_metrics["true_jaccard"] > 0].copy()
    for row in positive_pairs.itertuples(index=False):
        for tweet_id in (int(row.tweet_id_left), int(row.tweet_id_right)):
            if tweet_id not in selected_ids:
                selected_ids.append(tweet_id)
            if len(selected_ids) >= preview_docs:
                return selected_ids

    fallback_ids = (
        processed.sort_values(["token_count", "shingle_count", "tweet_id"], ascending=[False, False, True])["tweet_id"]
        .astype(int)
        .tolist()
    )
    for tweet_id in fallback_ids:
        if tweet_id not in selected_ids:
            selected_ids.append(tweet_id)
        if len(selected_ids) >= min(preview_docs, len(processed)):
            break
    return selected_ids


def _signature_frame(processed: pd.DataFrame, signatures: np.ndarray) -> pd.DataFrame:
    signature_df = pd.DataFrame(signatures, columns=[f"perm_{index + 1:03d}" for index in range(signatures.shape[1])])
    signature_df.insert(0, "tweet_id", processed["tweet_id"].astype(int).to_numpy())
    signature_df.insert(1, "source", processed["source"].astype(str).to_numpy())
    signature_df.insert(2, "date", processed["date"].astype(str).to_numpy())
    return signature_df


def _draw_summary_panel(ax, summary_lines: list[str]) -> None:
    _style_axes(ax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(
        0.03,
        0.97,
        "\n".join(summary_lines),
        ha="left",
        va="top",
        fontsize=10,
        color=TEXT,
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#fbf1e8", edgecolor=EDGE),
    )


def build_week4_dashboard(
    telegram_input: Path,
    output_png: Path,
    summary_json: Path,
    pair_metrics_csv: Path,
    pair_examples_csv: Path,
    signature_csv: Path,
    twitter_input: Path | None = None,
    sample_per_source: int = 200,
    shingle_size: int = 3,
    num_perm: int = 128,
    seed: int = 42,
    twitter_file_limit: int = 24,
    preview_docs: int = 12,
    preview_perms: int = 32,
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
        raise FileNotFoundError("No usable Telegram/Twitter data found for week 4 MinHash run.")

    sampled = pd.concat(sampled_frames, ignore_index=True)
    processed = build_shingles(sampled, shingle_size=shingle_size)
    if processed.empty:
        raise ValueError("Week 4 preprocessing kept 0 rows after shingling. Increase sample size or lower shingle size.")

    hashes = [hashed_shingles(shingles) for shingles in processed["shingles"]]
    signatures, signature_meta = signature_matrix(hashes, num_perm=num_perm, seed=seed)
    signature_df = _signature_frame(processed, signatures)
    signature_csv.parent.mkdir(parents=True, exist_ok=True)
    signature_df.to_csv(signature_csv, index=False)

    pair_metrics = _pair_metrics(processed, signatures)
    pair_metrics_export = pair_metrics.drop(columns=["text_left", "text_right"]).copy()
    pair_metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    pair_metrics_export.to_csv(pair_metrics_csv, index=False)

    pair_examples = pair_metrics.head(20).copy()
    pair_examples_csv.parent.mkdir(parents=True, exist_ok=True)
    pair_examples.to_csv(pair_examples_csv, index=False)

    source_stats = _build_source_summary(sampled, processed, base_stats)
    correlation = pair_metrics["true_jaccard"].corr(pair_metrics["minhash_estimate"]) if not pair_metrics.empty else 0.0
    mean_abs_error = float(pair_metrics["abs_error"].mean()) if not pair_metrics.empty else 0.0
    median_abs_error = float(pair_metrics["abs_error"].median()) if not pair_metrics.empty else 0.0

    preview_ids = _preview_doc_ids(processed, pair_metrics, preview_docs=preview_docs)
    preview_lookup = processed.set_index("tweet_id")
    preview_rows = preview_lookup.loc[preview_ids].reset_index()
    preview_rows["doc_label"] = [f"{row.source[:2].upper()}-{index + 1:02d}" for index, row in enumerate(preview_rows.itertuples())]
    preview_signature = signature_df.set_index("tweet_id").loc[preview_ids].reset_index(drop=False)
    preview_signature.index = preview_rows["doc_label"]
    preview_signature_values = preview_signature[[column for column in preview_signature.columns if column.startswith("perm_")]].iloc[
        :, :preview_perms
    ]

    summary_lines = [
        f"Documents: {len(processed):,}",
        f"Evaluated pairs: {len(pair_metrics):,}",
        f"num_perm: {num_perm}",
        f"Signature runtime: {signature_meta['runtime_seconds']:.6f}s",
        f"Correlation: {0.0 if pd.isna(correlation) else correlation:.4f}",
        f"Mean abs error: {mean_abs_error:.4f}",
        f"Median abs error: {median_abs_error:.4f}",
        f"Pairs with true Jaccard >= 0.8: {int((pair_metrics['true_jaccard'] >= 0.8).sum())}",
    ]

    fig = plt.figure(figsize=(18, 12), facecolor=BG)
    gs = GridSpec(2, 2, figure=fig, hspace=0.28, wspace=0.18)
    fig.suptitle(
        "Week 4 MinHash vs Jaccard",
        fontsize=22,
        fontweight="bold",
        color=TEXT,
        y=0.97,
    )
    fig.text(
        0.5,
        0.94,
        "Signature matrix preview and pairwise comparison between MinHash estimate and exact Jaccard",
        ha="center",
        color=MUTED,
        fontsize=11,
    )

    ax1 = fig.add_subplot(gs[0, 0])
    _style_axes(ax1)
    if pair_metrics.empty:
        ax1.text(0.5, 0.5, "No comparison pairs available", ha="center", va="center", fontsize=11, color=MUTED)
        ax1.set_xticks([])
        ax1.set_yticks([])
    else:
        ax1.scatter(pair_metrics["true_jaccard"], pair_metrics["minhash_estimate"], alpha=0.25, s=16, color=ACCENT1)
        ax1.plot([0, 1], [0, 1], linestyle="--", color=ACCENT2, linewidth=2)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.set_title("MinHash Estimate vs Exact Jaccard", fontsize=15, pad=12)
        ax1.set_xlabel("Exact Jaccard")
        ax1.set_ylabel("MinHash estimate")
        ax1.grid(color=EDGE, alpha=0.5)

    ax2 = fig.add_subplot(gs[0, 1])
    _style_axes(ax2)
    if pair_metrics.empty:
        ax2.text(0.5, 0.5, "No error distribution available", ha="center", va="center", fontsize=11, color=MUTED)
        ax2.set_xticks([])
        ax2.set_yticks([])
    else:
        ax2.hist(pair_metrics["abs_error"], bins=20, color=ACCENT3, edgecolor=BG)
        ax2.set_title("Absolute Error Distribution", fontsize=15, pad=12)
        ax2.set_xlabel("|Exact - MinHash|")
        ax2.set_ylabel("Pair count")
        ax2.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
        ax2.grid(axis="y", color=EDGE, alpha=0.5)

    ax3 = fig.add_subplot(gs[1, 0])
    _style_axes(ax3)
    heatmap = ax3.imshow(preview_signature_values.to_numpy(), aspect="auto", cmap="YlGnBu")
    ax3.set_title("Signature Matrix Preview", fontsize=15, pad=12)
    ax3.set_xlabel("MinHash permutation")
    ax3.set_ylabel("Document")
    ax3.set_xticks(range(preview_signature_values.shape[1]))
    ax3.set_xticklabels(range(1, preview_signature_values.shape[1] + 1), rotation=45, ha="right", fontsize=8)
    ax3.set_yticks(range(len(preview_signature_values.index)))
    ax3.set_yticklabels(preview_signature_values.index, fontsize=9, color=TEXT)
    plt.colorbar(heatmap, ax=ax3, fraction=0.046, pad=0.04)

    ax4 = fig.add_subplot(gs[1, 1])
    _draw_summary_panel(ax4, summary_lines)
    ax4.set_title("Week 4 Summary", fontsize=15, pad=12)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=170, bbox_inches="tight")
    plt.close(fig)

    top_pairs = []
    if not pair_metrics.empty:
        for row in pair_metrics.head(8).itertuples(index=False):
            top_pairs.append(
                {
                    "tweet_id_left": int(row.tweet_id_left),
                    "tweet_id_right": int(row.tweet_id_right),
                    "source_pair": str(row.source_pair),
                    "true_jaccard": round(float(row.true_jaccard), 4),
                    "minhash_estimate": round(float(row.minhash_estimate), 4),
                    "abs_error": round(float(row.abs_error), 4),
                    "text_left": str(row.text_left)[:180],
                    "text_right": str(row.text_right)[:180],
                }
            )

    summary = {
        "config": {
            "sample_per_source": sample_per_source,
            "shingle_size": shingle_size,
            "num_perm": num_perm,
            "seed": seed,
            "twitter_file_limit": twitter_file_limit,
            "preview_docs": int(len(preview_rows)),
            "preview_perms": int(preview_signature_values.shape[1]),
        },
        "sources": source_stats,
        "signature": {
            "num_documents": int(signature_meta["num_documents"]),
            "num_perm": int(signature_meta["num_perm"]),
            "runtime_seconds": round(float(signature_meta["runtime_seconds"]), 6),
        },
        "comparison": {
            "evaluated_pairs": int(len(pair_metrics)),
            "pearson_correlation": round(0.0 if pd.isna(correlation) else float(correlation), 6),
            "mean_abs_error": round(mean_abs_error, 6),
            "median_abs_error": round(median_abs_error, 6),
            "pairs_true_jaccard_ge_0_8": int((pair_metrics["true_jaccard"] >= 0.8).sum()) if not pair_metrics.empty else 0,
        },
        "top_pairs": top_pairs,
        "paths": {
            "telegram_input": str(telegram_input),
            "twitter_input": str(twitter_input) if twitter_input else None,
            "pair_metrics_csv": str(pair_metrics_csv),
            "pair_examples_csv": str(pair_examples_csv),
            "signature_csv": str(signature_csv),
            "dashboard_png": str(output_png),
            "summary_json": str(summary_json),
        },
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Create week-4 MinHash comparison artifacts and dashboard.")
    parser.add_argument("--telegram-input", default=str(repo_root / "datatele" / "telegram_messages.parquet"))
    parser.add_argument("--twitter-input", default=str(default_twitter_dataset_path() or ""))
    parser.add_argument("--sample-per-source", type=int, default=200)
    parser.add_argument("--shingle-size", type=int, default=3)
    parser.add_argument("--num-perm", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--twitter-file-limit", type=int, default=24)
    parser.add_argument("--preview-docs", type=int, default=12)
    parser.add_argument("--preview-perms", type=int, default=32)
    parser.add_argument("--output-png", default=str(repo_root / "datatele" / "week4_overview.png"))
    parser.add_argument("--summary-json", default=str(repo_root / "datatele" / "week4_summary.json"))
    parser.add_argument("--pair-metrics-csv", default=str(repo_root / "datatele" / "week4_pair_metrics.csv"))
    parser.add_argument("--pair-examples-csv", default=str(repo_root / "datatele" / "week4_pair_examples.csv"))
    parser.add_argument("--signature-csv", default=str(repo_root / "datatele" / "week4_signature_matrix.csv"))
    args = parser.parse_args()

    twitter_input = Path(args.twitter_input) if args.twitter_input else None
    summary = build_week4_dashboard(
        telegram_input=Path(args.telegram_input),
        twitter_input=twitter_input,
        output_png=Path(args.output_png),
        summary_json=Path(args.summary_json),
        pair_metrics_csv=Path(args.pair_metrics_csv),
        pair_examples_csv=Path(args.pair_examples_csv),
        signature_csv=Path(args.signature_csv),
        sample_per_source=args.sample_per_source,
        shingle_size=args.shingle_size,
        num_perm=args.num_perm,
        seed=args.seed,
        twitter_file_limit=args.twitter_file_limit,
        preview_docs=args.preview_docs,
        preview_perms=args.preview_perms,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
