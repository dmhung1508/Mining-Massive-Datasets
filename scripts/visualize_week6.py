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

from uk_russia_lsh.constants import CONFIG_GRID, LSHConfig
from uk_russia_lsh.datasets import default_twitter_dataset_path, normalise_twitter_frame
from uk_russia_lsh.minhash import evaluate_candidates, rank_configs, run_config
from uk_russia_lsh.preprocessing import build_shingles
from uk_russia_lsh.similarity import exact_jaccard_pairs, jaccard_similarity


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
    processed_by_shingle: dict[int, pd.DataFrame],
    base_stats: dict[str, dict[str, int]],
) -> dict[str, dict[str, float | int]]:
    summary: dict[str, dict[str, float | int]] = {}
    primary_processed = processed_by_shingle[min(processed_by_shingle)]
    for source, metrics in base_stats.items():
        source_sampled = sampled.loc[sampled["source"] == source]
        source_processed = primary_processed.loc[primary_processed["source"] == source]
        summary[source] = {
            **metrics,
            "processed_rows_k3": int(len(source_processed)),
            "dropped_short_rows_k3": int(len(source_sampled) - len(source_processed)),
            "avg_token_count_k3": round(float(source_processed["token_count"].mean()), 2) if not source_processed.empty else 0.0,
            "avg_shingle_count_k3": round(float(source_processed["shingle_count"].mean()), 2)
            if not source_processed.empty
            else 0.0,
        }
    return summary


def _pair_source(left_source: str, right_source: str) -> str:
    if left_source == right_source:
        return f"{left_source}-{right_source}"
    return "cross-source"


def _pair_frame_from_pairs(pairs: set[tuple[int, int]]) -> pd.DataFrame:
    if not pairs:
        return pd.DataFrame(columns=["tweet_id_left", "tweet_id_right"])
    return pd.DataFrame(sorted(pairs), columns=["tweet_id_left", "tweet_id_right"])


def _enrich_pairs(
    pair_frame: pd.DataFrame,
    processed: pd.DataFrame,
    exact_lookup: dict[tuple[int, int], float],
    shingles_lookup: dict[int, set[str]],
) -> pd.DataFrame:
    if pair_frame.empty:
        return pd.DataFrame(
            columns=[
                "tweet_id_left",
                "tweet_id_right",
                "source_left",
                "source_right",
                "source_pair",
                "exact_jaccard",
                "text_left",
                "text_right",
            ]
        )

    docs = processed[["tweet_id", "source", "date", "text"]].copy()
    left_meta = docs.rename(
        columns={
            "tweet_id": "tweet_id_left",
            "source": "source_left",
            "date": "date_left",
            "text": "text_left",
        }
    )
    right_meta = docs.rename(
        columns={
            "tweet_id": "tweet_id_right",
            "source": "source_right",
            "date": "date_right",
            "text": "text_right",
        }
    )
    enriched = pair_frame.merge(left_meta, on="tweet_id_left", how="left").merge(right_meta, on="tweet_id_right", how="left")
    enriched["source_pair"] = [
        _pair_source(str(left), str(right))
        for left, right in zip(enriched["source_left"], enriched["source_right"])
    ]
    enriched["exact_jaccard"] = [
        exact_lookup.get(
            (int(left_id), int(right_id)),
            round(jaccard_similarity(shingles_lookup[int(left_id)], shingles_lookup[int(right_id)]), 6),
        )
        for left_id, right_id in zip(enriched["tweet_id_left"], enriched["tweet_id_right"])
    ]
    return enriched.sort_values(
        ["exact_jaccard", "tweet_id_left", "tweet_id_right"],
        ascending=[False, True, True],
    ).reset_index(drop=True)


def _summary_lines(selected: dict[str, float | int | str]) -> list[str]:
    return [
        f"Selected config: {selected['config_name']}",
        f"Shingle size: {int(selected['shingle_size'])}",
        f"num_perm={int(selected['num_perm'])}, bands={int(selected['bands'])}, rows={int(selected['rows'])}",
        f"Processed docs: {int(selected['processed_rows']):,}",
        f"Ground truth pairs: {int(selected['ground_truth_pairs']):,}",
        f"Candidate pairs: {int(selected['candidate_pairs']):,}",
        f"True positives: {int(selected['true_positives']):,}",
        f"False positives: {int(selected['false_positives']):,}",
        f"False negatives: {int(selected['false_negatives']):,}",
        f"Precision: {float(selected['precision']):.4f}",
        f"Recall: {float(selected['recall']):.4f}",
        f"Signature runtime: {float(selected['signature_runtime_seconds']):.6f}s",
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


def build_week6_dashboard(
    telegram_input: Path,
    output_png: Path,
    summary_json: Path,
    config_results_csv: Path,
    false_positive_csv: Path,
    false_negative_csv: Path,
    twitter_input: Path | None = None,
    sample_per_source: int = 200,
    threshold: float = 0.8,
    seed: int = 42,
    twitter_file_limit: int = 24,
    configs: tuple[LSHConfig, ...] = CONFIG_GRID,
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
        raise FileNotFoundError("No usable Telegram/Twitter data found for week 6 tuning.")

    sampled = pd.concat(sampled_frames, ignore_index=True)
    dataset_cache: dict[int, dict[str, object]] = {}
    config_rows: list[dict[str, float | int | str]] = []

    for config in configs:
        cache_key = int(config.shingle_size)
        if cache_key not in dataset_cache:
            processed = build_shingles(sampled, shingle_size=config.shingle_size)
            if processed.empty:
                raise ValueError(f"No rows left after shingling with k={config.shingle_size}")
            ground_truth, exact_metrics = exact_jaccard_pairs(processed, threshold=threshold)
            dataset_cache[cache_key] = {
                "processed": processed,
                "ground_truth": ground_truth,
                "exact_metrics": exact_metrics,
            }

        processed = dataset_cache[cache_key]["processed"]
        ground_truth = dataset_cache[cache_key]["ground_truth"]

        candidate_pairs, run_metrics = run_config(processed, config=config, seed=seed)
        eval_metrics = evaluate_candidates(candidate_pairs, ground_truth)

        candidate_set = {
            (int(row.tweet_id_left), int(row.tweet_id_right))
            for row in candidate_pairs[["tweet_id_left", "tweet_id_right"]].itertuples(index=False)
        }
        ground_truth_set = {
            (int(row.tweet_id_left), int(row.tweet_id_right))
            for row in ground_truth[["tweet_id_left", "tweet_id_right"]].itertuples(index=False)
        }
        config_rows.append(
            {
                "config_name": config.name,
                **config.as_dict(),
                "processed_rows": int(len(processed)),
                **run_metrics,
                **eval_metrics,
                "false_positives": int(len(candidate_set - ground_truth_set)),
                "false_negatives": int(len(ground_truth_set - candidate_set)),
            }
        )

    ranked = rank_configs(config_rows)
    config_results = pd.DataFrame(ranked)
    selected = ranked[0]
    selected_config = next(config for config in configs if config.name == selected["config_name"])
    analysis_target = next(
        (row for row in ranked if int(row["false_positives"]) > 0 or int(row["false_negatives"]) > 0),
        selected,
    )
    analysis_config = next(config for config in configs if config.name == analysis_target["config_name"])

    analysis_bundle = dataset_cache[int(analysis_config.shingle_size)]
    analysis_processed = analysis_bundle["processed"]
    analysis_ground_truth = analysis_bundle["ground_truth"]
    analysis_candidates, _ = run_config(analysis_processed, config=analysis_config, seed=seed)

    analysis_candidate_set = {
        (int(row.tweet_id_left), int(row.tweet_id_right))
        for row in analysis_candidates[["tweet_id_left", "tweet_id_right"]].itertuples(index=False)
    }
    analysis_ground_truth_set = {
        (int(row.tweet_id_left), int(row.tweet_id_right))
        for row in analysis_ground_truth[["tweet_id_left", "tweet_id_right"]].itertuples(index=False)
    }
    false_positive_pairs = analysis_candidate_set - analysis_ground_truth_set
    false_negative_pairs = analysis_ground_truth_set - analysis_candidate_set

    shingles_lookup = {
        int(row.tweet_id): set(row.shingles)
        for row in analysis_processed[["tweet_id", "shingles"]].itertuples(index=False)
    }
    exact_lookup = {
        (int(row.tweet_id_left), int(row.tweet_id_right)): float(row.jaccard)
        for row in analysis_ground_truth[["tweet_id_left", "tweet_id_right", "jaccard"]].itertuples(index=False)
    }
    false_positive_frame = _enrich_pairs(
        _pair_frame_from_pairs(false_positive_pairs),
        processed=analysis_processed,
        exact_lookup=exact_lookup,
        shingles_lookup=shingles_lookup,
    )
    false_negative_frame = _enrich_pairs(
        _pair_frame_from_pairs(false_negative_pairs),
        processed=analysis_processed,
        exact_lookup=exact_lookup,
        shingles_lookup=shingles_lookup,
    )

    config_results_csv.parent.mkdir(parents=True, exist_ok=True)
    config_results.to_csv(config_results_csv, index=False)
    false_positive_csv.parent.mkdir(parents=True, exist_ok=True)
    false_positive_frame.to_csv(false_positive_csv, index=False)
    false_negative_csv.parent.mkdir(parents=True, exist_ok=True)
    false_negative_frame.to_csv(false_negative_csv, index=False)

    source_summary = _build_source_summary(sampled, {3: dataset_cache[min(dataset_cache)]["processed"]}, base_stats)

    fig = plt.figure(figsize=(18, 12), facecolor=BG)
    gs = GridSpec(2, 2, figure=fig, hspace=0.28, wspace=0.18)
    fig.suptitle(
        "Week 6 LSH Tuning",
        fontsize=22,
        fontweight="bold",
        color=TEXT,
        y=0.97,
    )
    fig.text(
        0.5,
        0.94,
        "Config sweep with precision/recall, false positive/false negative analysis, and final config selection",
        ha="center",
        color=MUTED,
        fontsize=11,
    )

    plot_df = config_results.copy()
    plot_df["selected"] = plot_df["config_name"].eq(selected["config_name"])

    ax1 = fig.add_subplot(gs[0, 0])
    _style_axes(ax1)
    x_positions = list(range(len(plot_df)))
    ax1.bar([position - 0.18 for position in x_positions], plot_df["precision"], width=0.36, color=ACCENT1, label="precision")
    ax1.bar([position + 0.18 for position in x_positions], plot_df["recall"], width=0.36, color=ACCENT3, label="recall")
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(plot_df["config_name"], rotation=15, ha="right", fontsize=9, color=TEXT)
    ax1.set_ylim(0, 1.05)
    ax1.set_title("Precision / Recall By Config", fontsize=15, pad=12)
    ax1.set_ylabel("Score")
    ax1.grid(axis="y", color=EDGE, alpha=0.6)
    ax1.legend(frameon=False)

    ax2 = fig.add_subplot(gs[0, 1])
    _style_axes(ax2)
    scatter = ax2.scatter(
        plot_df["candidate_pairs"],
        plot_df["precision"],
        s=160,
        c=plot_df["recall"],
        cmap="YlGnBu",
        edgecolors="black",
    )
    for row in plot_df.itertuples(index=False):
        label = f"{row.config_name}{' *' if row.selected else ''}"
        ax2.annotate(label, (row.candidate_pairs, row.precision), xytext=(6, 6), textcoords="offset points", fontsize=9)
    ax2.set_title("Config Trade-off", fontsize=15, pad=12)
    ax2.set_xlabel("Candidate pairs")
    ax2.set_ylabel("Precision")
    ax2.grid(color=EDGE, alpha=0.5)
    plt.colorbar(scatter, ax=ax2, fraction=0.046, pad=0.04, label="Recall")

    ax3 = fig.add_subplot(gs[1, 0])
    _style_axes(ax3)
    ax3.bar([position - 0.18 for position in x_positions], plot_df["false_positives"], width=0.36, color=ACCENT2, label="false positives")
    ax3.bar([position + 0.18 for position in x_positions], plot_df["false_negatives"], width=0.36, color=ACCENT4, label="false negatives")
    ax3.set_xticks(x_positions)
    ax3.set_xticklabels(plot_df["config_name"], rotation=15, ha="right", fontsize=9, color=TEXT)
    ax3.set_title("False Positives / False Negatives", fontsize=15, pad=12)
    ax3.set_ylabel("Pairs")
    ax3.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    ax3.grid(axis="y", color=EDGE, alpha=0.6)
    ax3.legend(frameon=False)

    ax4 = fig.add_subplot(gs[1, 1])
    _draw_summary_panel(ax4, _summary_lines(selected))
    ax4.set_title("Selected Config", fontsize=15, pad=12)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=170, bbox_inches="tight")
    plt.close(fig)

    top_false_positives = []
    for row in false_positive_frame.head(8).itertuples(index=False):
        top_false_positives.append(
            {
                "tweet_id_left": int(row.tweet_id_left),
                "tweet_id_right": int(row.tweet_id_right),
                "source_pair": str(row.source_pair),
                "exact_jaccard": round(float(row.exact_jaccard), 4),
                "text_left": str(row.text_left)[:180],
                "text_right": str(row.text_right)[:180],
            }
        )

    top_false_negatives = []
    for row in false_negative_frame.head(8).itertuples(index=False):
        top_false_negatives.append(
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
            "seed": seed,
            "twitter_file_limit": twitter_file_limit,
            "config_names": [config.name for config in configs],
        },
        "sources": source_summary,
        "selected_config": selected,
        "error_analysis_config": analysis_target,
        "false_positive_count": int(len(false_positive_frame)),
        "false_negative_count": int(len(false_negative_frame)),
        "top_false_positives": top_false_positives,
        "top_false_negatives": top_false_negatives,
        "paths": {
            "telegram_input": str(telegram_input),
            "twitter_input": str(twitter_input) if twitter_input else None,
            "config_results_csv": str(config_results_csv),
            "false_positive_csv": str(false_positive_csv),
            "false_negative_csv": str(false_negative_csv),
            "dashboard_png": str(output_png),
            "summary_json": str(summary_json),
        },
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Create week-6 tuning artifacts and dashboard.")
    parser.add_argument("--telegram-input", default=str(repo_root / "datatele" / "telegram_messages.parquet"))
    parser.add_argument("--twitter-input", default=str(default_twitter_dataset_path() or ""))
    parser.add_argument("--sample-per-source", type=int, default=200)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--twitter-file-limit", type=int, default=24)
    parser.add_argument("--output-png", default=str(repo_root / "datatele" / "week6_overview.png"))
    parser.add_argument("--summary-json", default=str(repo_root / "datatele" / "week6_summary.json"))
    parser.add_argument("--config-results-csv", default=str(repo_root / "datatele" / "week6_config_results.csv"))
    parser.add_argument("--false-positive-csv", default=str(repo_root / "datatele" / "week6_false_positives.csv"))
    parser.add_argument("--false-negative-csv", default=str(repo_root / "datatele" / "week6_false_negatives.csv"))
    args = parser.parse_args()

    twitter_input = Path(args.twitter_input) if args.twitter_input else None
    summary = build_week6_dashboard(
        telegram_input=Path(args.telegram_input),
        twitter_input=twitter_input,
        output_png=Path(args.output_png),
        summary_json=Path(args.summary_json),
        config_results_csv=Path(args.config_results_csv),
        false_positive_csv=Path(args.false_positive_csv),
        false_negative_csv=Path(args.false_negative_csv),
        sample_per_source=args.sample_per_source,
        threshold=args.threshold,
        seed=args.seed,
        twitter_file_limit=args.twitter_file_limit,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
