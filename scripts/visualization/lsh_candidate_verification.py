from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from matplotlib.gridspec import GridSpec


repo_root = Path(__file__).resolve().parents[2]

from social_lsh.datasets import default_twitter_dataset_path, normalise_twitter_frame
from social_lsh.minhash import generate_candidate_pairs, hashed_shingles, signature_matrix
from social_lsh.preprocessing import build_shingles
from social_lsh.similarity import exact_jaccard_pairs, jaccard_similarity, verify_candidate_pairs


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


def _enrich_pair_frame(
    pair_frame: pd.DataFrame,
    processed: pd.DataFrame,
    shingles_lookup: dict[int, set[str]],
    threshold: float,
) -> pd.DataFrame:
    if pair_frame.empty:
        return pd.DataFrame(
            columns=[
                "tweet_id_left",
                "tweet_id_right",
                "source_left",
                "source_right",
                "source_pair",
                "date_left",
                "date_right",
                "exact_jaccard",
                "verified",
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
    enriched = pair_frame.merge(left_meta, on="tweet_id_left", how="left").merge(right_meta, on="tweet_id_right", how="left")
    enriched["source_pair"] = [
        _pair_source(str(left), str(right))
        for left, right in zip(enriched["source_left"], enriched["source_right"])
    ]
    enriched["exact_jaccard"] = [
        round(
            jaccard_similarity(shingles_lookup[int(left_id)], shingles_lookup[int(right_id)]),
            6,
        )
        for left_id, right_id in zip(enriched["tweet_id_left"], enriched["tweet_id_right"])
    ]
    enriched["verified"] = enriched["exact_jaccard"] >= threshold
    return enriched.sort_values(
        ["exact_jaccard", "tweet_id_left", "tweet_id_right"],
        ascending=[False, True, True],
    ).reset_index(drop=True)


def _summary_panel_lines(
    documents: int,
    config: dict[str, int | float],
    exact_metrics: dict[str, float | int],
    lsh_metrics: dict[str, float | int],
    debug_metrics: dict[str, float | int],
    verify_metrics: dict[str, float | int],
) -> list[str]:
    return [
        f"Documents: {documents:,}",
        f"Config: bands={config['bands']}, rows={config['rows']}, num_perm={config['num_perm']}",
        f"Exact pairs >= threshold: {int(exact_metrics['positive_pairs']):,}",
        f"Exact runtime: {float(exact_metrics['runtime_seconds']):.6f}s",
        f"Signature runtime: {float(lsh_metrics['signature_runtime_seconds']):.6f}s",
        f"Candidate generation runtime: {float(lsh_metrics['candidate_runtime_seconds']):.6f}s",
        f"Verify runtime: {float(verify_metrics['runtime_seconds']):.6f}s",
        f"Candidates: {int(lsh_metrics['candidate_pairs']):,}",
        f"Verified after Jaccard: {int(debug_metrics['verified_pairs']):,}",
        f"False positives before verify: {int(debug_metrics['false_positive_candidates']):,}",
        f"Missed exact pairs: {int(debug_metrics['missed_ground_truth_pairs']):,}",
        f"Candidate precision: {float(debug_metrics['candidate_precision']):.4f}",
        f"Candidate recall: {float(debug_metrics['candidate_recall']):.4f}",
    ]


def _draw_summary_panel(ax, lines: list[str]) -> None:
    _style_axes(ax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(
        0.03,
        0.97,
        "\n".join(lines),
        ha="left",
        va="top",
        fontsize=10,
        color=TEXT,
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#fbf1e8", edgecolor=EDGE),
    )


def _empty_panel(ax, title: str, message: str) -> None:
    _style_axes(ax)
    ax.set_title(title, fontsize=15, pad=12)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=11, color=MUTED)


def build_week5_dashboard(
    telegram_input: Path,
    output_png: Path,
    summary_json: Path,
    candidate_csv: Path,
    verified_csv: Path,
    exact_pairs_csv: Path,
    twitter_input: Path | None = None,
    sample_per_source: int = 200,
    threshold: float = 0.8,
    shingle_size: int = 3,
    num_perm: int = 128,
    bands: int = 32,
    rows: int = 4,
    seed: int = 42,
    twitter_file_limit: int = 24,
) -> dict[str, object]:
    if bands * rows != num_perm:
        raise ValueError("bands * rows must equal num_perm for week 5 LSH.")

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
        raise FileNotFoundError("No usable Telegram/Twitter data found for week 5 LSH run.")

    sampled = pd.concat(sampled_frames, ignore_index=True)
    processed = build_shingles(sampled, shingle_size=shingle_size)
    if processed.empty:
        raise ValueError("Week 5 preprocessing kept 0 rows after shingling. Increase sample size or lower shingle size.")

    source_stats = _build_source_summary(sampled, processed, base_stats)
    exact_pairs, exact_metrics = exact_jaccard_pairs(processed, threshold=threshold)

    hash_lists = [hashed_shingles(shingles) for shingles in processed["shingles"]]
    signatures, signature_meta = signature_matrix(hash_lists, num_perm=num_perm, seed=seed)

    candidate_started = time.perf_counter()
    candidate_pairs, candidate_meta = generate_candidate_pairs(
        signatures=signatures,
        tweet_ids=processed["tweet_id"].astype(int).tolist(),
        bands=bands,
        rows=rows,
    )
    candidate_runtime = round(time.perf_counter() - candidate_started, 6)

    shingles_lookup = {
        int(row.tweet_id): set(row.shingles)
        for row in processed[["tweet_id", "shingles"]].itertuples(index=False)
    }
    candidate_enriched = _enrich_pair_frame(
        candidate_pairs,
        processed=processed,
        shingles_lookup=shingles_lookup,
        threshold=threshold,
    )
    verified_core, verify_metrics = verify_candidate_pairs(
        shingles_lookup=shingles_lookup,
        candidate_pairs=candidate_pairs,
        threshold=threshold,
    )
    verified_pairs = candidate_enriched.loc[candidate_enriched["verified"]].copy().reset_index(drop=True)
    if len(verified_core) != len(verified_pairs):
        raise RuntimeError("Verified pair count mismatch between pipeline verify step and enriched candidate table.")

    exact_pairs_csv.parent.mkdir(parents=True, exist_ok=True)
    exact_pairs.to_csv(exact_pairs_csv, index=False)
    candidate_csv.parent.mkdir(parents=True, exist_ok=True)
    candidate_enriched.to_csv(candidate_csv, index=False)
    verified_csv.parent.mkdir(parents=True, exist_ok=True)
    verified_pairs.to_csv(verified_csv, index=False)

    candidate_set = {
        (int(row.tweet_id_left), int(row.tweet_id_right))
        for row in candidate_pairs[["tweet_id_left", "tweet_id_right"]].itertuples(index=False)
    }
    ground_truth_set = {
        (int(row.tweet_id_left), int(row.tweet_id_right))
        for row in exact_pairs[["tweet_id_left", "tweet_id_right"]].itertuples(index=False)
    }
    verified_set = {
        (int(row.tweet_id_left), int(row.tweet_id_right))
        for row in verified_pairs[["tweet_id_left", "tweet_id_right"]].itertuples(index=False)
    }

    true_positive_candidates = len(candidate_set & ground_truth_set)
    false_positive_candidates = len(candidate_set - ground_truth_set)
    missed_ground_truth_pairs = len(ground_truth_set - candidate_set)
    debug_metrics = {
        "verified_pairs": len(verified_set),
        "true_positive_candidates": true_positive_candidates,
        "false_positive_candidates": false_positive_candidates,
        "missed_ground_truth_pairs": missed_ground_truth_pairs,
        "candidate_precision": round(true_positive_candidates / len(candidate_set), 6) if candidate_set else 0.0,
        "candidate_recall": round(true_positive_candidates / len(ground_truth_set), 6) if ground_truth_set else 0.0,
    }
    lsh_metrics = {
        "signature_runtime_seconds": float(signature_meta["runtime_seconds"]),
        "candidate_runtime_seconds": candidate_runtime,
        "num_documents": int(signature_meta["num_documents"]),
        "num_perm": int(signature_meta["num_perm"]),
        **candidate_meta,
    }

    stage_counts = pd.DataFrame(
        [
            {"stage": "Documents", "count": len(processed)},
            {"stage": "Exact pairs", "count": int(exact_metrics["positive_pairs"])},
            {"stage": "Candidates", "count": int(candidate_meta["candidate_pairs"])},
            {"stage": "Verified", "count": int(len(verified_pairs))},
        ]
    )

    source_counts = []
    for source_pair in ["twitter-twitter", "telegram-telegram", "cross-source"]:
        source_counts.append(
            {
                "source_pair": source_pair,
                "candidates": int((candidate_enriched["source_pair"] == source_pair).sum()),
                "verified": int((verified_pairs["source_pair"] == source_pair).sum()),
            }
        )
    source_counts_df = pd.DataFrame(source_counts)

    fig = plt.figure(figsize=(18, 12), facecolor=BG)
    gs = GridSpec(2, 2, figure=fig, hspace=0.28, wspace=0.18)
    fig.suptitle(
        "Week 5 LSH Candidate Pipeline",
        fontsize=22,
        fontweight="bold",
        color=TEXT,
        y=0.97,
    )
    fig.text(
        0.5,
        0.94,
        "LSH banding -> candidate pairs -> exact Jaccard verify on a deterministic mixed subset",
        ha="center",
        color=MUTED,
        fontsize=11,
    )

    ax1 = fig.add_subplot(gs[0, 0])
    _style_axes(ax1)
    ax1.bar(stage_counts["stage"], stage_counts["count"], color=[ACCENT1, ACCENT2, ACCENT3, ACCENT4])
    ax1.set_yscale("log")
    ax1.set_title("Pipeline Counts", fontsize=15, pad=12)
    ax1.set_ylabel("Count (log scale)")
    ax1.grid(axis="y", color=EDGE, alpha=0.6)
    ax1.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))

    ax2 = fig.add_subplot(gs[0, 1])
    if candidate_enriched.empty:
        _empty_panel(ax2, "Candidate Jaccard Distribution", "LSH produced no candidate pairs")
    else:
        _style_axes(ax2)
        ax2.hist(candidate_enriched["exact_jaccard"], bins=20, color=ACCENT2, edgecolor=BG)
        ax2.axvline(threshold, color=ACCENT1, linestyle="--", linewidth=2, label=f"threshold={threshold:.2f}")
        ax2.set_title("Candidate Exact Jaccard Distribution", fontsize=15, pad=12)
        ax2.set_xlabel("Exact Jaccard")
        ax2.set_ylabel("Candidate count")
        ax2.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
        ax2.grid(axis="y", color=EDGE, alpha=0.6)
        ax2.legend(frameon=False)

    ax3 = fig.add_subplot(gs[1, 0])
    _style_axes(ax3)
    positions = list(range(len(source_counts_df)))
    ax3.bar([position - 0.18 for position in positions], source_counts_df["candidates"], width=0.36, color=ACCENT3, label="candidates")
    ax3.bar([position + 0.18 for position in positions], source_counts_df["verified"], width=0.36, color=ACCENT1, label="verified")
    ax3.set_xticks(positions)
    ax3.set_xticklabels(source_counts_df["source_pair"], color=TEXT)
    ax3.set_title("Candidate vs Verified By Source Type", fontsize=15, pad=12)
    ax3.set_ylabel("Pairs")
    ax3.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    ax3.grid(axis="y", color=EDGE, alpha=0.6)
    ax3.legend(frameon=False)

    ax4 = fig.add_subplot(gs[1, 1])
    summary_lines = _summary_panel_lines(
        documents=len(processed),
        config={"bands": bands, "rows": rows, "num_perm": num_perm},
        exact_metrics=exact_metrics,
        lsh_metrics=lsh_metrics,
        debug_metrics=debug_metrics,
        verify_metrics=verify_metrics,
    )
    _draw_summary_panel(ax4, summary_lines)
    ax4.set_title("Week 5 Summary", fontsize=15, pad=12)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=170, bbox_inches="tight")
    plt.close(fig)

    top_verified = []
    for row in verified_pairs.head(8).itertuples(index=False):
        top_verified.append(
            {
                "tweet_id_left": int(row.tweet_id_left),
                "tweet_id_right": int(row.tweet_id_right),
                "source_pair": str(row.source_pair),
                "exact_jaccard": round(float(row.exact_jaccard), 4),
                "text_left": str(row.text_left)[:180],
                "text_right": str(row.text_right)[:180],
            }
        )

    top_rejected = []
    rejected = candidate_enriched.loc[~candidate_enriched["verified"]].copy()
    for row in rejected.head(8).itertuples(index=False):
        top_rejected.append(
            {
                "tweet_id_left": int(row.tweet_id_left),
                "tweet_id_right": int(row.tweet_id_right),
                "source_pair": str(row.source_pair),
                "exact_jaccard": round(float(row.exact_jaccard), 4),
                "text_left": str(row.text_left)[:180],
                "text_right": str(row.text_right)[:180],
            }
        )

    summary = {
        "config": {
            "sample_per_source": sample_per_source,
            "threshold": threshold,
            "shingle_size": shingle_size,
            "num_perm": num_perm,
            "bands": bands,
            "rows": rows,
            "seed": seed,
            "twitter_file_limit": twitter_file_limit,
        },
        "sources": source_stats,
        "exact_baseline": {
            "pairs_considered": int(exact_metrics["pairs_considered"]),
            "positive_pairs": int(exact_metrics["positive_pairs"]),
            "runtime_seconds": round(float(exact_metrics["runtime_seconds"]), 6),
        },
        "lsh": {
            "signature_runtime_seconds": round(float(lsh_metrics["signature_runtime_seconds"]), 6),
            "candidate_runtime_seconds": round(float(lsh_metrics["candidate_runtime_seconds"]), 6),
            "bucket_count": int(lsh_metrics["bucket_count"]),
            "candidate_pairs": int(lsh_metrics["candidate_pairs"]),
        },
        "debug": debug_metrics,
        "verify": {
            "candidate_pairs_checked": int(verify_metrics["candidate_pairs_checked"]),
            "verified_pairs": int(verify_metrics["verified_pairs"]),
            "runtime_seconds": round(float(verify_metrics["runtime_seconds"]), 6),
        },
        "top_verified": top_verified,
        "top_rejected": top_rejected,
        "paths": {
            "telegram_input": str(telegram_input),
            "twitter_input": str(twitter_input) if twitter_input else None,
            "candidate_csv": str(candidate_csv),
            "verified_csv": str(verified_csv),
            "exact_pairs_csv": str(exact_pairs_csv),
            "dashboard_png": str(output_png),
            "summary_json": str(summary_json),
        },
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Create week-5 LSH candidate artifacts and dashboard.")
    parser.add_argument("--telegram-input", default=str(repo_root / "jupyter" / "output" / "visuals" / "telegram_messages.parquet"))
    parser.add_argument("--twitter-input", default=str(default_twitter_dataset_path() or ""))
    parser.add_argument("--sample-per-source", type=int, default=200)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--shingle-size", type=int, default=3)
    parser.add_argument("--num-perm", type=int, default=128)
    parser.add_argument("--bands", type=int, default=32)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--twitter-file-limit", type=int, default=24)
    parser.add_argument("--output-png", default=str(repo_root / "jupyter" / "output" / "visuals" / "week5_overview.png"))
    parser.add_argument("--summary-json", default=str(repo_root / "jupyter" / "output" / "visuals" / "week5_summary.json"))
    parser.add_argument("--candidate-csv", default=str(repo_root / "jupyter" / "output" / "visuals" / "week5_candidates.csv"))
    parser.add_argument("--verified-csv", default=str(repo_root / "jupyter" / "output" / "visuals" / "week5_verified_pairs.csv"))
    parser.add_argument("--exact-pairs-csv", default=str(repo_root / "jupyter" / "output" / "visuals" / "week5_exact_pairs.csv"))
    args = parser.parse_args()

    twitter_input = Path(args.twitter_input) if args.twitter_input else None
    summary = build_week5_dashboard(
        telegram_input=Path(args.telegram_input),
        twitter_input=twitter_input,
        output_png=Path(args.output_png),
        summary_json=Path(args.summary_json),
        candidate_csv=Path(args.candidate_csv),
        verified_csv=Path(args.verified_csv),
        exact_pairs_csv=Path(args.exact_pairs_csv),
        sample_per_source=args.sample_per_source,
        threshold=args.threshold,
        shingle_size=args.shingle_size,
        num_perm=args.num_perm,
        bands=args.bands,
        rows=args.rows,
        seed=args.seed,
        twitter_file_limit=args.twitter_file_limit,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
