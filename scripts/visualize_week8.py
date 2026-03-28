from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from matplotlib.gridspec import GridSpec

from _bootstrap import add_src_to_path

repo_root = add_src_to_path()

from uk_russia_lsh.clustering import connected_components
from uk_russia_lsh.constants import CONFIG_GRID, LSHConfig
from uk_russia_lsh.datasets import default_twitter_dataset_path, normalise_twitter_frame
from uk_russia_lsh.minhash import run_config
from uk_russia_lsh.preprocessing import build_shingles
from uk_russia_lsh.similarity import exact_jaccard_pairs, jaccard_similarity, verify_candidate_pairs


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
HASHTAG_RE = re.compile(r"#(\w+)")
STOPWORDS = {
    "the",
    "and",
    "for",
    "that",
    "with",
    "this",
    "from",
    "have",
    "will",
    "your",
    "they",
    "their",
    "about",
    "amid",
    "into",
    "near",
    "over",
    "under",
    "latest",
    "today",
    "just",
    "into",
    "than",
    "would",
    "there",
    "were",
    "been",
    "what",
    "when",
    "where",
    "which",
    "while",
    "also",
    "after",
    "between",
    "against",
    "amp",
    "not",
    "you",
    "american",
    "america",
    "russian",
    "ukraine",
    "russia",
    "iran",
    "usa",
}


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


def _resolve_selected_config(default_seed: int) -> tuple[LSHConfig, dict[str, int | float | str] | None]:
    summary_path = repo_root / "datatele" / "week6_summary.json"
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


def _sample_twitter_frame(
    path: Path,
    sample_size: int,
    seed: int,
    file_limit: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    parquet_files = sorted(path.rglob("*.parquet"))
    if not parquet_files:
        return pd.DataFrame(), {"source_rows_seen": 0, "analysis_rows": 0, "files_scanned": 0}

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
        return pd.DataFrame(), {"source_rows_seen": raw_rows, "analysis_rows": 0, "files_scanned": len(selected_files)}

    sampled = _sample_smallest_scores(pd.concat(candidates, ignore_index=True), sample_size=sample_size, seed=seed)
    sampled["dataset_name"] = "russia_ukraine"
    return _prepare_sample_frame(sampled), {
        "source_rows_seen": raw_rows,
        "analysis_rows": int(len(sampled)),
        "files_scanned": len(selected_files),
    }


def _load_us_iran_frame(path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"US-Iran dataset not found: {path}")

    frame = pd.read_csv(path)
    working = pd.DataFrame(
        {
            "tweet_id": pd.to_numeric(frame["tweet_id"], errors="coerce").astype("Int64"),
            "user_id": pd.to_numeric(frame["user_id"], errors="coerce").astype("Int64"),
            "text": frame["content"].astype("string"),
            "timestamp": pd.to_datetime(frame["created_at"], utc=True, errors="coerce").dt.tz_localize(None),
        }
    )
    working["date"] = working["timestamp"].dt.strftime("%Y-%m-%d")
    working["source"] = "twitter"
    working["source_item_id"] = working["tweet_id"].astype("string")
    working["source_user_id"] = working["user_id"].astype("string")
    working["source_channel_id"] = None
    working["media_type"] = None
    working["forward_from_user_id"] = None
    working["forward_from_username"] = None
    working["dataset_name"] = "us_iran"
    working = working.loc[
        working["tweet_id"].notna()
        & working["user_id"].notna()
        & working["timestamp"].notna()
        & working["date"].notna()
        & working["text"].notna()
    ].copy()
    working["tweet_id"] = working["tweet_id"].astype("int64")
    working["user_id"] = working["user_id"].astype("int64")
    return _prepare_sample_frame(working), {
        "source_rows_seen": int(len(frame)),
        "analysis_rows": int(len(working)),
        "is_simulated_demo": True,
        "dataset_path": str(path),
        "date_min": working["date"].min() if not working.empty else None,
        "date_max": working["date"].max() if not working.empty else None,
    }


def _style_axes(ax) -> None:
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_color(EDGE)
    ax.tick_params(colors=MUTED)
    ax.title.set_color(TEXT)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)


def _extract_top_terms(processed: pd.DataFrame, limit: int = 8) -> list[dict[str, object]]:
    tokens = []
    for token_list in processed["tokens"]:
        for token in token_list:
            if len(token) < 3 or token.isdigit() or token in STOPWORDS:
                continue
            tokens.append(token)
    return [{"term": term, "count": count} for term, count in Counter(tokens).most_common(limit)]


def _extract_top_hashtags(frame: pd.DataFrame, limit: int = 8) -> list[dict[str, object]]:
    hashtags = []
    for text in frame["text"].astype(str):
        hashtags.extend(tag.lower() for tag in HASHTAG_RE.findall(text))
    return [{"hashtag": tag, "count": count} for tag, count in Counter(hashtags).most_common(limit)]


def _pair_source(left_source: str, right_source: str) -> str:
    if left_source == right_source:
        return f"{left_source}-{right_source}"
    return "cross-source"


def _enrich_pairs(pair_frame: pd.DataFrame, processed: pd.DataFrame, threshold: float) -> pd.DataFrame:
    if pair_frame.empty:
        return pd.DataFrame(
            columns=[
                "tweet_id_left",
                "tweet_id_right",
                "date_left",
                "date_right",
                "jaccard",
                "is_near_duplicate",
                "text_left",
                "text_right",
            ]
        )

    docs = processed[["tweet_id", "date", "text", "shingles"]].copy()
    left_meta = docs.rename(
        columns={
            "tweet_id": "tweet_id_left",
            "date": "date_left",
            "text": "text_left",
            "shingles": "shingles_left",
        }
    )
    right_meta = docs.rename(
        columns={
            "tweet_id": "tweet_id_right",
            "date": "date_right",
            "text": "text_right",
            "shingles": "shingles_right",
        }
    )
    enriched = pair_frame.merge(left_meta, on="tweet_id_left", how="left").merge(right_meta, on="tweet_id_right", how="left")
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
    docs = processed[["tweet_id", "date", "text"]].copy()
    members = clusters.merge(docs, on="tweet_id", how="left")
    verified_lookup = {
        tuple(sorted((int(row.tweet_id_left), int(row.tweet_id_right)))): float(row.jaccard)
        for row in verified_pairs[["tweet_id_left", "tweet_id_right", "jaccard"]].itertuples(index=False)
    }
    rows = []
    for cluster_id, group in members.groupby("cluster_id", sort=True):
        cluster_size = int(group["cluster_size"].iloc[0])
        tweet_ids = group["tweet_id"].astype(int).tolist()
        scores = [
            score
            for (left_id, right_id), score in verified_lookup.items()
            if left_id in tweet_ids and right_id in tweet_ids
        ]
        rows.append(
            {
                "cluster_id": int(cluster_id),
                "cluster_size": cluster_size,
                "is_repeated_cluster": cluster_size > 1,
                "verified_edges": len(scores),
                "avg_verified_jaccard": round(sum(scores) / len(scores), 6) if scores else float("nan"),
                "example_text": str(group.sort_values("tweet_id").iloc[0]["text"])[:220],
            }
        )
    validation = pd.DataFrame(rows)
    if validation.empty:
        return validation
    return validation.sort_values(["cluster_size", "cluster_id"], ascending=[False, True]).reset_index(drop=True)


def _dataset_analysis(
    dataset_name: str,
    input_frame: pd.DataFrame,
    selected_config: LSHConfig,
    threshold: float,
    seed: int,
) -> dict[str, object]:
    processed = build_shingles(input_frame, shingle_size=selected_config.shingle_size)
    if processed.empty:
        raise ValueError(f"No rows left after shingling for dataset {dataset_name}")

    exact_pairs, exact_metrics = exact_jaccard_pairs(processed, threshold=threshold)
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

    candidate_enriched = _enrich_pairs(candidate_pairs, processed=processed, threshold=threshold)
    verified_enriched = candidate_enriched.loc[candidate_enriched["is_near_duplicate"]].copy().reset_index(drop=True)
    if len(verified_enriched) != len(verified_pairs):
        raise RuntimeError(f"Verified pair mismatch for dataset {dataset_name}")

    exact_set = {
        (int(row.tweet_id_left), int(row.tweet_id_right))
        for row in exact_pairs[["tweet_id_left", "tweet_id_right"]].itertuples(index=False)
    }
    candidate_set = {
        (int(row.tweet_id_left), int(row.tweet_id_right))
        for row in candidate_pairs[["tweet_id_left", "tweet_id_right"]].itertuples(index=False)
    }

    metrics = {
        "dataset_name": dataset_name,
        "processed_rows": int(len(processed)),
        "candidate_pairs": int(len(candidate_pairs)),
        "verified_pairs": int(len(verified_pairs)),
        "exact_pairs": int(len(exact_pairs)),
        "total_clusters": int(clusters["cluster_id"].nunique()) if not clusters.empty else 0,
        "repeated_clusters": int((validation["cluster_size"] > 1).sum()) if not validation.empty else 0,
        "largest_cluster_size": int(validation["cluster_size"].max()) if not validation.empty else 0,
        "candidate_precision": round(len(candidate_set & exact_set) / len(candidate_set), 6) if candidate_set else 0.0,
        "candidate_recall": round(len(candidate_set & exact_set) / len(exact_set), 6) if exact_set else 0.0,
        "signature_runtime_seconds": round(float(lsh_metrics["signature_runtime_seconds"]), 6),
        "verify_runtime_seconds": round(float(verify_metrics["runtime_seconds"]), 6),
        "top_terms": _extract_top_terms(processed),
        "top_hashtags": _extract_top_hashtags(input_frame),
    }
    return {
        "processed": processed,
        "exact_pairs": exact_pairs,
        "candidate_pairs": candidate_enriched,
        "verified_pairs": verified_enriched,
        "clusters": clusters,
        "validation": validation,
        "metrics": metrics,
    }


def _format_top_terms_block(title: str, terms: list[dict[str, object]], hashtags: list[dict[str, object]]) -> str:
    lines = [title]
    if terms:
        lines.append("Top terms:")
        lines.extend(f"- {item['term']} ({item['count']})" for item in terms[:5])
    if hashtags:
        lines.append("Top hashtags:")
        lines.extend(f"- #{item['hashtag']} ({item['count']})" for item in hashtags[:5])
    if len(lines) == 1:
        lines.append("- no strong repeated terms/hashtags")
    return "\n".join(lines)


def build_week8_dashboard(
    russia_ukraine_input: Path,
    us_iran_input: Path,
    output_png: Path,
    summary_json: Path,
    comparison_csv: Path,
    russia_pairs_csv: Path,
    us_iran_pairs_csv: Path,
    cluster_csv: Path,
    twitter_file_limit: int = 24,
    matched_sample_size: int = 50,
    threshold: float = 0.8,
    seed: int = 42,
) -> dict[str, object]:
    selected_config, selected_payload = _resolve_selected_config(default_seed=seed)
    russia_frame, russia_meta = _sample_twitter_frame(
        russia_ukraine_input,
        sample_size=matched_sample_size,
        seed=seed,
        file_limit=twitter_file_limit,
    )
    us_iran_frame, us_iran_meta = _load_us_iran_frame(us_iran_input)
    us_iran_frame = _sample_smallest_scores(us_iran_frame, sample_size=matched_sample_size, seed=seed + 1)
    us_iran_frame = _prepare_sample_frame(us_iran_frame)
    us_iran_meta["analysis_rows"] = int(len(us_iran_frame))

    russia_analysis = _dataset_analysis(
        dataset_name="russia_ukraine",
        input_frame=russia_frame,
        selected_config=selected_config,
        threshold=threshold,
        seed=seed,
    )
    us_iran_analysis = _dataset_analysis(
        dataset_name="us_iran",
        input_frame=us_iran_frame,
        selected_config=selected_config,
        threshold=threshold,
        seed=seed,
    )

    comparison_rows = []
    for meta, analysis in [(russia_meta, russia_analysis), (us_iran_meta, us_iran_analysis)]:
        comparison_rows.append(
            {
                "dataset_name": analysis["metrics"]["dataset_name"],
                "source_rows_seen": meta["source_rows_seen"],
                "analysis_rows": meta["analysis_rows"],
                "processed_rows": analysis["metrics"]["processed_rows"],
                "exact_pairs": analysis["metrics"]["exact_pairs"],
                "candidate_pairs": analysis["metrics"]["candidate_pairs"],
                "verified_pairs": analysis["metrics"]["verified_pairs"],
                "repeated_clusters": analysis["metrics"]["repeated_clusters"],
                "largest_cluster_size": analysis["metrics"]["largest_cluster_size"],
                "candidate_precision": analysis["metrics"]["candidate_precision"],
                "candidate_recall": analysis["metrics"]["candidate_recall"],
            }
        )
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_csv.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(comparison_csv, index=False)

    russia_pairs_csv.parent.mkdir(parents=True, exist_ok=True)
    russia_analysis["verified_pairs"].to_csv(russia_pairs_csv, index=False)
    us_iran_pairs_csv.parent.mkdir(parents=True, exist_ok=True)
    us_iran_analysis["verified_pairs"].to_csv(us_iran_pairs_csv, index=False)

    cluster_frames = []
    if not russia_analysis["validation"].empty:
        cluster_frames.append(russia_analysis["validation"].assign(dataset_name="russia_ukraine"))
    if not us_iran_analysis["validation"].empty:
        cluster_frames.append(us_iran_analysis["validation"].assign(dataset_name="us_iran"))
    cluster_export = pd.concat(cluster_frames, ignore_index=True) if cluster_frames else pd.DataFrame()
    cluster_csv.parent.mkdir(parents=True, exist_ok=True)
    cluster_export.to_csv(cluster_csv, index=False)

    fig = plt.figure(figsize=(18, 12), facecolor=BG)
    gs = GridSpec(2, 2, figure=fig, hspace=0.28, wspace=0.18)
    fig.suptitle(
        "Week 8 Dataset Comparison",
        fontsize=22,
        fontweight="bold",
        color=TEXT,
        y=0.97,
    )
    fig.text(
        0.5,
        0.94,
        "Russia–Ukraine vs US–Iran comparison on matched sample size with duplicate and cluster statistics",
        ha="center",
        color=MUTED,
        fontsize=11,
    )

    ax1 = fig.add_subplot(gs[0, 0])
    _style_axes(ax1)
    labels = ["Russia-Ukraine", "US-Iran"]
    positions = [0, 1]
    ax1.bar([position - 0.18 for position in positions], comparison_df["verified_pairs"], width=0.36, color=ACCENT1, label="verified pairs")
    ax1.bar([position + 0.18 for position in positions], comparison_df["repeated_clusters"], width=0.36, color=ACCENT2, label="repeated clusters")
    ax1.set_xticks(positions)
    ax1.set_xticklabels(labels, color=TEXT)
    ax1.set_title("Duplicate and Cluster Counts", fontsize=15, pad=12)
    ax1.set_ylabel("Count")
    ax1.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    ax1.grid(axis="y", color=EDGE, alpha=0.6)
    ax1.legend(frameon=False)

    ax2 = fig.add_subplot(gs[0, 1])
    _style_axes(ax2)
    ax2.bar([position - 0.18 for position in positions], comparison_df["candidate_precision"], width=0.36, color=ACCENT3, label="precision")
    ax2.bar([position + 0.18 for position in positions], comparison_df["candidate_recall"], width=0.36, color=ACCENT4, label="recall")
    ax2.set_xticks(positions)
    ax2.set_xticklabels(labels, color=TEXT)
    ax2.set_ylim(0, 1.05)
    ax2.set_title("LSH Quality On Matched Samples", fontsize=15, pad=12)
    ax2.set_ylabel("Score")
    ax2.grid(axis="y", color=EDGE, alpha=0.6)
    ax2.legend(frameon=False)

    ax3 = fig.add_subplot(gs[1, 0])
    _style_axes(ax3)
    ax3.set_xticks([])
    ax3.set_yticks([])
    left_block = _format_top_terms_block(
        "Russia–Ukraine",
        russia_analysis["metrics"]["top_terms"],
        russia_analysis["metrics"]["top_hashtags"],
    )
    right_block = _format_top_terms_block(
        "US–Iran",
        us_iran_analysis["metrics"]["top_terms"],
        us_iran_analysis["metrics"]["top_hashtags"],
    )
    ax3.text(
        0.03,
        0.97,
        left_block,
        ha="left",
        va="top",
        fontsize=10,
        color=TEXT,
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#eef4f6", edgecolor=EDGE),
    )
    ax3.text(
        0.53,
        0.97,
        right_block,
        ha="left",
        va="top",
        fontsize=10,
        color=TEXT,
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#fbf1e8", edgecolor=EDGE),
    )
    ax3.set_title("Top Terms and Hashtags", fontsize=15, pad=12)

    ax4 = fig.add_subplot(gs[1, 1])
    _style_axes(ax4)
    ax4.set_xticks([])
    ax4.set_yticks([])
    summary_lines = [
        f"Selected config: {selected_config.name}",
        f"Matched sample size: {matched_sample_size} docs per dataset",
        f"Russia–Ukraine verified pairs: {int(russia_analysis['metrics']['verified_pairs'])}",
        f"Russia–Ukraine largest cluster: {int(russia_analysis['metrics']['largest_cluster_size'])}",
        f"US–Iran verified pairs: {int(us_iran_analysis['metrics']['verified_pairs'])}",
        f"US–Iran largest cluster: {int(us_iran_analysis['metrics']['largest_cluster_size'])}",
        f"US–Iran source type: simulated demo CSV",
        "Interpretation caveat: comparison is useful for pipeline behavior and rough topical contrast,",
        "not for claiming real-world prevalence between the two conflicts.",
    ]
    ax4.text(
        0.03,
        0.97,
        "\n".join(summary_lines),
        ha="left",
        va="top",
        fontsize=10,
        color=TEXT,
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#fbf1e8", edgecolor=EDGE),
    )
    ax4.set_title("Insights and Caveats", fontsize=15, pad=12)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=170, bbox_inches="tight")
    plt.close(fig)

    russia_repeated = russia_analysis["validation"].loc[russia_analysis["validation"]["is_repeated_cluster"]]
    us_iran_repeated = us_iran_analysis["validation"].loc[us_iran_analysis["validation"]["is_repeated_cluster"]]

    summary = {
        "config": {
            "matched_sample_size": matched_sample_size,
            "threshold": threshold,
            "seed": seed,
            "twitter_file_limit": twitter_file_limit,
        },
        "selected_config": selected_payload,
        "datasets": {
            "russia_ukraine": {
                **russia_meta,
                **russia_analysis["metrics"],
                "top_cluster_example": (
                    russia_repeated.iloc[0]["example_text"]
                    if not russia_repeated.empty
                    else None
                ),
            },
            "us_iran": {
                **us_iran_meta,
                **us_iran_analysis["metrics"],
                "top_cluster_example": (
                    us_iran_repeated.iloc[0]["example_text"]
                    if not us_iran_repeated.empty
                    else None
                ),
            },
        },
        "comparison": {
            "same_sample_size": True,
            "us_iran_is_simulated_demo": True,
            "recommended_report_note": "Use this as a matched-sample technical comparison, not a real-world prevalence claim.",
        },
        "paths": {
            "russia_ukraine_input": str(russia_ukraine_input),
            "us_iran_input": str(us_iran_input),
            "comparison_csv": str(comparison_csv),
            "russia_pairs_csv": str(russia_pairs_csv),
            "us_iran_pairs_csv": str(us_iran_pairs_csv),
            "cluster_csv": str(cluster_csv),
            "dashboard_png": str(output_png),
            "summary_json": str(summary_json),
        },
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Create week-8 dataset comparison artifacts and dashboard.")
    parser.add_argument("--russia-ukraine-input", default=str(default_twitter_dataset_path() or ""))
    parser.add_argument(
        "--us-iran-input",
        default="/home/anonymous/code/Mining_Massive_Dataset/dataset/us_iran_war_tweets.csv",
    )
    parser.add_argument("--matched-sample-size", type=int, default=50)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--twitter-file-limit", type=int, default=24)
    parser.add_argument("--output-png", default=str(repo_root / "datatele" / "week8_overview.png"))
    parser.add_argument("--summary-json", default=str(repo_root / "datatele" / "week8_summary.json"))
    parser.add_argument("--comparison-csv", default=str(repo_root / "datatele" / "week8_dataset_comparison.csv"))
    parser.add_argument("--russia-pairs-csv", default=str(repo_root / "datatele" / "week8_russia_ukraine_pairs.csv"))
    parser.add_argument("--us-iran-pairs-csv", default=str(repo_root / "datatele" / "week8_us_iran_pairs.csv"))
    parser.add_argument("--cluster-csv", default=str(repo_root / "datatele" / "week8_cluster_comparison.csv"))
    args = parser.parse_args()

    if not args.russia_ukraine_input:
        raise FileNotFoundError("Russia–Ukraine parquet dataset not found. Pass --russia-ukraine-input explicitly.")

    summary = build_week8_dashboard(
        russia_ukraine_input=Path(args.russia_ukraine_input),
        us_iran_input=Path(args.us_iran_input),
        output_png=Path(args.output_png),
        summary_json=Path(args.summary_json),
        comparison_csv=Path(args.comparison_csv),
        russia_pairs_csv=Path(args.russia_pairs_csv),
        us_iran_pairs_csv=Path(args.us_iran_pairs_csv),
        cluster_csv=Path(args.cluster_csv),
        twitter_file_limit=args.twitter_file_limit,
        matched_sample_size=args.matched_sample_size,
        threshold=args.threshold,
        seed=args.seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
