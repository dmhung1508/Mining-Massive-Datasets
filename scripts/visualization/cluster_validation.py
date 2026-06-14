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


repo_root = Path(__file__).resolve().parents[2]

from social_lsh.clustering import connected_components
from social_lsh.constants import CONFIG_GRID, LSHConfig
from social_lsh.datasets import default_twitter_dataset_path, normalise_twitter_frame
from social_lsh.minhash import run_config
from social_lsh.preprocessing import build_shingles
from social_lsh.similarity import jaccard_similarity, verify_candidate_pairs


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


def _resolve_selected_config(default_seed: int) -> tuple[LSHConfig, dict[str, int | float | str] | None]:
    summary_path = repo_root / "jupyter" / "output" / "visuals" / "week6_summary.json"
    if summary_path.exists():
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        selected = payload.get("selected_config")
        if isinstance(selected, dict):
            return (
                LSHConfig(
                    shingle_size=int(selected["shingle_size"]),
                    num_perm=int(selected["num_perm"]),
                    bands=int(selected["bands"]),
                    rows=int(selected["rows"]),
                ),
                selected,
            )
    fallback = CONFIG_GRID[0]
    return fallback, {
        "config_name": fallback.name,
        "shingle_size": fallback.shingle_size,
        "num_perm": fallback.num_perm,
        "bands": fallback.bands,
        "rows": fallback.rows,
        "seed": default_seed,
    }


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


def _enrich_pair_frame(pair_frame: pd.DataFrame, processed: pd.DataFrame, threshold: float) -> pd.DataFrame:
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
                "jaccard",
                "is_near_duplicate",
                "text_left",
                "text_right",
            ]
        )

    docs = processed[["tweet_id", "source", "date", "text", "shingles"]].copy()
    left_meta = docs.rename(
        columns={
            "tweet_id": "tweet_id_left",
            "source": "source_left",
            "date": "date_left",
            "text": "text_left",
            "shingles": "shingles_left",
        }
    )
    right_meta = docs.rename(
        columns={
            "tweet_id": "tweet_id_right",
            "source": "source_right",
            "date": "date_right",
            "text": "text_right",
            "shingles": "shingles_right",
        }
    )
    enriched = pair_frame.merge(left_meta, on="tweet_id_left", how="left").merge(right_meta, on="tweet_id_right", how="left")
    enriched["source_pair"] = [
        _pair_source(str(left), str(right))
        for left, right in zip(enriched["source_left"], enriched["source_right"])
    ]
    enriched["jaccard"] = [
        round(jaccard_similarity(set(left), set(right)), 6)
        for left, right in zip(enriched["shingles_left"], enriched["shingles_right"])
    ]
    enriched["is_near_duplicate"] = enriched["jaccard"] >= threshold
    return enriched.drop(columns=["shingles_left", "shingles_right"]).sort_values(
        ["jaccard", "tweet_id_left", "tweet_id_right"],
        ascending=[False, True, True],
    ).reset_index(drop=True)


def _cluster_validation(clusters: pd.DataFrame, processed: pd.DataFrame, verified_pairs: pd.DataFrame) -> pd.DataFrame:
    docs = processed[["tweet_id", "source", "date", "text"]].copy()
    members = clusters.merge(docs, on="tweet_id", how="left")
    verified_lookup = {
        tuple(sorted((int(row.tweet_id_left), int(row.tweet_id_right)))): float(row.jaccard)
        for row in verified_pairs[["tweet_id_left", "tweet_id_right", "jaccard"]].itertuples(index=False)
    }

    rows = []
    for cluster_id, group in members.groupby("cluster_id", sort=True):
        cluster_size = int(group["cluster_size"].iloc[0])
        tweet_ids = group["tweet_id"].astype(int).tolist()
        pair_scores = [
            score
            for (left_id, right_id), score in verified_lookup.items()
            if left_id in tweet_ids and right_id in tweet_ids
        ]
        source_mix = ", ".join(
            f"{source}:{count}"
            for source, count in group["source"].astype(str).value_counts().sort_index().items()
        )
        example_text = str(group.sort_values("tweet_id").iloc[0]["text"])[:220]
        rows.append(
            {
                "cluster_id": int(cluster_id),
                "cluster_size": cluster_size,
                "is_repeated_cluster": cluster_size > 1,
                "members": len(tweet_ids),
                "verified_edges": len(pair_scores),
                "avg_verified_jaccard": round(sum(pair_scores) / len(pair_scores), 6) if pair_scores else None,
                "min_verified_jaccard": round(min(pair_scores), 6) if pair_scores else None,
                "max_verified_jaccard": round(max(pair_scores), 6) if pair_scores else None,
                "source_mix": source_mix,
                "example_text": example_text,
            }
        )
    validation = pd.DataFrame(rows)
    if validation.empty:
        return validation
    return validation.sort_values(["cluster_size", "cluster_id"], ascending=[False, True]).reset_index(drop=True)


def _summary_lines(
    selected_config: LSHConfig,
    lsh_metrics: dict[str, float | int],
    verify_metrics: dict[str, float | int],
    clusters: pd.DataFrame,
    validation: pd.DataFrame,
    threshold: float,
) -> list[str]:
    repeat_clusters = int((validation["cluster_size"] > 1).sum()) if not validation.empty else 0
    singleton_clusters = int((validation["cluster_size"] == 1).sum()) if not validation.empty else 0
    largest_cluster = int(validation["cluster_size"].max()) if not validation.empty else 0
    return [
        f"Selected config: {selected_config.name}",
        f"Threshold: {threshold:.2f}",
        f"Candidates: {int(lsh_metrics['candidate_pairs']):,}",
        f"Candidate buckets: {int(lsh_metrics['bucket_count']):,}",
        f"Verified near-duplicates: {int(verify_metrics['verified_pairs']):,}",
        f"Verify runtime: {float(verify_metrics['runtime_seconds']):.6f}s",
        f"Total clusters: {int(clusters['cluster_id'].nunique()) if not clusters.empty else 0:,}",
        f"Repeated-content clusters: {repeat_clusters:,}",
        f"Singleton clusters: {singleton_clusters:,}",
        f"Largest cluster: {largest_cluster:,}",
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


def build_week7_dashboard(
    telegram_input: Path,
    output_png: Path,
    summary_json: Path,
    candidate_csv: Path,
    verified_csv: Path,
    clusters_csv: Path,
    validation_csv: Path,
    twitter_input: Path | None = None,
    sample_per_source: int = 200,
    threshold: float = 0.8,
    seed: int = 42,
    twitter_file_limit: int = 24,
) -> dict[str, object]:
    selected_config, selected_config_payload = _resolve_selected_config(default_seed=seed)

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
        raise FileNotFoundError("No usable Telegram/Twitter data found for week 7 clustering.")

    sampled = pd.concat(sampled_frames, ignore_index=True)
    processed = build_shingles(sampled, shingle_size=selected_config.shingle_size)
    if processed.empty:
        raise ValueError("Week 7 preprocessing kept 0 rows after shingling. Increase sample size or lower shingle size.")

    source_summary = _build_source_summary(sampled, processed, base_stats)
    candidate_pairs, lsh_metrics = run_config(processed, config=selected_config, seed=seed)

    shingles_lookup = {
        int(row.tweet_id): set(row.shingles)
        for row in processed[["tweet_id", "shingles"]].itertuples(index=False)
    }
    verified_pairs, verify_metrics = verify_candidate_pairs(
        shingles_lookup=shingles_lookup,
        candidate_pairs=candidate_pairs,
        threshold=threshold,
    )
    clusters = connected_components(processed["tweet_id"].astype(int).tolist(), verified_pairs)
    validation = _cluster_validation(clusters, processed, verified_pairs)

    candidate_enriched = _enrich_pair_frame(candidate_pairs, processed=processed, threshold=threshold)
    verified_enriched = candidate_enriched.loc[candidate_enriched["is_near_duplicate"]].copy().reset_index(drop=True)
    if len(verified_enriched) != len(verified_pairs):
        raise RuntimeError("Week 7 verified pair mismatch between enriched and core tables.")

    candidate_csv.parent.mkdir(parents=True, exist_ok=True)
    candidate_enriched.to_csv(candidate_csv, index=False)
    verified_csv.parent.mkdir(parents=True, exist_ok=True)
    verified_enriched.to_csv(verified_csv, index=False)
    clusters_csv.parent.mkdir(parents=True, exist_ok=True)
    clusters.to_csv(clusters_csv, index=False)
    validation_csv.parent.mkdir(parents=True, exist_ok=True)
    validation.to_csv(validation_csv, index=False)

    repeated_validation = validation.loc[validation["is_repeated_cluster"]].copy() if not validation.empty else validation
    top_clusters = (repeated_validation if not repeated_validation.empty else validation).head(10).copy()
    cluster_size_df = validation[["cluster_id", "cluster_size"]].copy() if not validation.empty else pd.DataFrame(columns=["cluster_id", "cluster_size"])
    repeat_cluster_sizes = cluster_size_df.loc[cluster_size_df["cluster_size"] > 1, "cluster_size"]

    fig = plt.figure(figsize=(18, 12), facecolor=BG)
    gs = GridSpec(2, 2, figure=fig, hspace=0.28, wspace=0.18)
    fig.suptitle(
        "Week 7 Verified Pairs and Clusters",
        fontsize=22,
        fontweight="bold",
        color=TEXT,
        y=0.97,
    )
    fig.text(
        0.5,
        0.94,
        "Verified near-duplicates, connected components clustering, and cluster validation preview",
        ha="center",
        color=MUTED,
        fontsize=11,
    )

    ax1 = fig.add_subplot(gs[0, 0])
    _style_axes(ax1)
    funnel_df = pd.DataFrame(
        [
            {"stage": "Documents", "count": len(processed)},
            {"stage": "Candidates", "count": len(candidate_enriched)},
            {"stage": "Near-duplicates", "count": len(verified_enriched)},
            {"stage": "Clusters", "count": int(clusters["cluster_id"].nunique()) if not clusters.empty else 0},
        ]
    )
    ax1.bar(funnel_df["stage"], funnel_df["count"], color=[ACCENT1, ACCENT2, ACCENT3, ACCENT4])
    ax1.set_yscale("log")
    ax1.set_title("Verify and Cluster Funnel", fontsize=15, pad=12)
    ax1.set_ylabel("Count (log scale)")
    ax1.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    ax1.grid(axis="y", color=EDGE, alpha=0.6)

    ax2 = fig.add_subplot(gs[0, 1])
    _style_axes(ax2)
    if repeat_cluster_sizes.empty:
        ax2.text(0.5, 0.5, "No repeated-content clusters", ha="center", va="center", fontsize=11, color=MUTED)
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_title("Repeated Cluster Size Distribution", fontsize=15, pad=12)
    else:
        bins = sorted(set([2, 3, 4, 5, 6, 8, 10, int(repeat_cluster_sizes.max()) + 1]))
        ax2.hist(repeat_cluster_sizes, bins=bins, color=ACCENT3, edgecolor=BG)
        ax2.set_title("Repeated Cluster Size Distribution", fontsize=15, pad=12)
        ax2.set_xlabel("Cluster size")
        ax2.set_ylabel("Cluster count")
        ax2.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
        ax2.grid(axis="y", color=EDGE, alpha=0.6)

    ax3 = fig.add_subplot(gs[1, 0])
    _style_axes(ax3)
    if top_clusters.empty:
        ax3.text(0.5, 0.5, "No clusters available", ha="center", va="center", fontsize=11, color=MUTED)
        ax3.set_xticks([])
        ax3.set_yticks([])
        ax3.set_title("Top Clusters", fontsize=15, pad=12)
    else:
        plot_df = top_clusters.iloc[::-1]
        ax3.barh(plot_df["cluster_id"].astype(str), plot_df["cluster_size"], color=ACCENT1)
        ax3.set_title("Top Clusters By Size", fontsize=15, pad=12)
        ax3.set_xlabel("Cluster size")
        ax3.set_ylabel("Cluster ID")
        ax3.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))

    ax4 = fig.add_subplot(gs[1, 1])
    _draw_summary_panel(
        ax4,
        _summary_lines(
            selected_config=selected_config,
            lsh_metrics=lsh_metrics,
            verify_metrics=verify_metrics,
            clusters=clusters,
            validation=validation,
            threshold=threshold,
        ),
    )
    ax4.set_title("Week 7 Summary", fontsize=15, pad=12)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=170, bbox_inches="tight")
    plt.close(fig)

    top_validated_clusters = []
    validation_preview = repeated_validation if not repeated_validation.empty else validation
    for row in validation_preview.head(8).itertuples(index=False):
        top_validated_clusters.append(
            {
                "cluster_id": int(row.cluster_id),
                "cluster_size": int(row.cluster_size),
                "verified_edges": int(row.verified_edges),
                "avg_verified_jaccard": None if pd.isna(row.avg_verified_jaccard) else round(float(row.avg_verified_jaccard), 4),
                "source_mix": str(row.source_mix),
                "example_text": str(row.example_text),
            }
        )

    top_near_duplicates = []
    for row in verified_enriched.head(8).itertuples(index=False):
        top_near_duplicates.append(
            {
                "tweet_id_left": int(row.tweet_id_left),
                "tweet_id_right": int(row.tweet_id_right),
                "source_pair": str(row.source_pair),
                "jaccard": round(float(row.jaccard), 4),
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
        },
        "selected_config": selected_config_payload,
        "sources": source_summary,
        "lsh": {
            "candidate_pairs": int(lsh_metrics["candidate_pairs"]),
            "bucket_count": int(lsh_metrics["bucket_count"]),
            "signature_runtime_seconds": round(float(lsh_metrics["signature_runtime_seconds"]), 6),
        },
        "verify": {
            "candidate_pairs_checked": int(verify_metrics["candidate_pairs_checked"]),
            "verified_pairs": int(verify_metrics["verified_pairs"]),
            "runtime_seconds": round(float(verify_metrics["runtime_seconds"]), 6),
        },
        "clusters": {
            "total_clusters": int(clusters["cluster_id"].nunique()) if not clusters.empty else 0,
            "repeated_clusters": int((validation["cluster_size"] > 1).sum()) if not validation.empty else 0,
            "singleton_clusters": int((validation["cluster_size"] == 1).sum()) if not validation.empty else 0,
            "largest_cluster_size": int(validation["cluster_size"].max()) if not validation.empty else 0,
        },
        "top_near_duplicates": top_near_duplicates,
        "top_validated_clusters": top_validated_clusters,
        "paths": {
            "telegram_input": str(telegram_input),
            "twitter_input": str(twitter_input) if twitter_input else None,
            "candidate_csv": str(candidate_csv),
            "verified_csv": str(verified_csv),
            "clusters_csv": str(clusters_csv),
            "validation_csv": str(validation_csv),
            "dashboard_png": str(output_png),
            "summary_json": str(summary_json),
        },
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Create week-7 verify-and-cluster artifacts and dashboard.")
    parser.add_argument("--telegram-input", default=str(repo_root / "jupyter" / "output" / "visuals" / "telegram_messages.parquet"))
    parser.add_argument("--twitter-input", default=str(default_twitter_dataset_path() or ""))
    parser.add_argument("--sample-per-source", type=int, default=200)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--twitter-file-limit", type=int, default=24)
    parser.add_argument("--output-png", default=str(repo_root / "jupyter" / "output" / "visuals" / "week7_overview.png"))
    parser.add_argument("--summary-json", default=str(repo_root / "jupyter" / "output" / "visuals" / "week7_summary.json"))
    parser.add_argument("--candidate-csv", default=str(repo_root / "jupyter" / "output" / "visuals" / "week7_candidates.csv"))
    parser.add_argument("--verified-csv", default=str(repo_root / "jupyter" / "output" / "visuals" / "week7_near_duplicates.csv"))
    parser.add_argument("--clusters-csv", default=str(repo_root / "jupyter" / "output" / "visuals" / "week7_clusters.csv"))
    parser.add_argument("--validation-csv", default=str(repo_root / "jupyter" / "output" / "visuals" / "week7_cluster_validation.csv"))
    args = parser.parse_args()

    twitter_input = Path(args.twitter_input) if args.twitter_input else None
    summary = build_week7_dashboard(
        telegram_input=Path(args.telegram_input),
        twitter_input=twitter_input,
        output_png=Path(args.output_png),
        summary_json=Path(args.summary_json),
        candidate_csv=Path(args.candidate_csv),
        verified_csv=Path(args.verified_csv),
        clusters_csv=Path(args.clusters_csv),
        validation_csv=Path(args.validation_csv),
        sample_per_source=args.sample_per_source,
        threshold=args.threshold,
        seed=args.seed,
        twitter_file_limit=args.twitter_file_limit,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
