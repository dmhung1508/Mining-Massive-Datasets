from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from social_lsh.artifacts import artifact_path, read_dataframe
from social_lsh.constants import DEFAULT_ARTIFACT_DIR


def full_search_database(artifact_dir: str) -> Path:
    return Path(artifact_dir) / "full_search.duckdb"


def query_full_search(artifact_dir: str, sql: str, parameters: list | None = None) -> pd.DataFrame:
    import duckdb

    connection = duckdb.connect(str(full_search_database(artifact_dir)), read_only=True)
    try:
        return connection.execute(sql, parameters or []).fetchdf()
    finally:
        connection.close()


@st.cache_data(show_spinner=False)
def load_metrics(artifact_dir: str) -> dict:
    metrics_path = Path(artifact_dir) / "metrics.json"
    if not metrics_path.exists():
        return {}
    return json.loads(metrics_path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_clusters(artifact_dir: str) -> pd.DataFrame:
    return read_dataframe(artifact_path("clusters", Path(artifact_dir)))


@st.cache_data(show_spinner=False)
def load_scale_subset(artifact_dir: str) -> pd.DataFrame:
    frame = read_dataframe(artifact_path("scale_subset", Path(artifact_dir)))
    columns = [
        "tweet_id",
        "user_id",
        "source",
        "source_item_id",
        "source_user_id",
        "source_channel_id",
        "topic_label",
        "text",
        "timestamp",
        "date",
    ]
    return frame[[column for column in columns if column in frame.columns]].copy()


@st.cache_data(show_spinner=False)
def load_verified_pairs(artifact_dir: str) -> pd.DataFrame:
    return read_dataframe(artifact_path("verified_pairs", Path(artifact_dir)))


def compact_text(value: object, limit: int = 180) -> str:
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def top_cluster_examples(artifact_dir: str, limit: int) -> pd.DataFrame:
    if full_search_database(artifact_dir).exists():
        return query_full_search(
            artifact_dir,
            """
            WITH top_clusters AS (
                SELECT cluster_id, max(cluster_size) AS cluster_size
                FROM documents
                WHERE cluster_size > 1
                GROUP BY cluster_id
                ORDER BY cluster_size DESC, cluster_id
                LIMIT ?
            ),
            ranked AS (
                SELECT
                    documents.cluster_id,
                    documents.source,
                    documents.tweet_id,
                    documents.text,
                    row_number() OVER (
                        PARTITION BY documents.cluster_id ORDER BY documents.tweet_id
                    ) AS example_rank
                FROM documents
                JOIN top_clusters USING (cluster_id)
            )
            SELECT
                top_clusters.cluster_id,
                top_clusters.cluster_size,
                string_agg(DISTINCT ranked.source, ', ') AS sources,
                string_agg(ranked.tweet_id::VARCHAR, ', ' ORDER BY ranked.tweet_id)
                    FILTER (WHERE ranked.example_rank <= 3) AS sample_tweet_ids,
                string_agg(left(ranked.text, 180), ' | ' ORDER BY ranked.tweet_id)
                    FILTER (WHERE ranked.example_rank <= 3) AS sample_text
            FROM top_clusters
            JOIN ranked USING (cluster_id)
            GROUP BY top_clusters.cluster_id, top_clusters.cluster_size
            ORDER BY top_clusters.cluster_size DESC, top_clusters.cluster_id
            """,
            [limit],
        )

    clusters = load_clusters(artifact_dir)
    scale = load_scale_subset(artifact_dir)
    merged = clusters.merge(scale, on="tweet_id", how="left")
    top = (
        clusters[["cluster_id", "cluster_size"]]
        .drop_duplicates()
        .sort_values(["cluster_size", "cluster_id"], ascending=[False, True])
        .head(limit)
    )

    rows = []
    for cluster in top.itertuples(index=False):
        examples = (
            merged.loc[merged["cluster_id"].eq(cluster.cluster_id)]
            .sort_values(["tweet_id"])
            .head(3)
        )
        rows.append(
            {
                "cluster_id": int(cluster.cluster_id),
                "cluster_size": int(cluster.cluster_size),
                "sources": ", ".join(sorted(examples.get("source", pd.Series(dtype=str)).dropna().astype(str).unique())),
                "sample_tweet_ids": ", ".join(examples["tweet_id"].astype(str).tolist()),
                "sample_text": " | ".join(compact_text(text) for text in examples["text"].tolist()),
            }
        )
    return pd.DataFrame(rows)


def render_metrics(metrics: dict) -> None:
    full = metrics.get("full_validation")
    if full:
        st.subheader("Full-corpus pipeline summary")
        cols = st.columns(4)
        cols[0].metric("Indexed documents", f"{int(full.get('source_documents', 0)):,}")
        cols[1].metric("Narrative clusters", f"{int(full.get('cluster_count', 0)):,}")
        cols[2].metric(
            "Docs in repeated clusters",
            f"{int(full.get('documents_in_repeated_clusters', 0)):,}",
        )
        cols[3].metric("Largest cluster", f"{int(full.get('largest_cluster', 0)):,}")
        return

    extract = metrics.get("extract_subsets", {})
    baseline = metrics.get("baseline", {})
    scale = metrics.get("run_lsh", {}).get("scale_run", {})
    verify = metrics.get("verify_and_cluster", {})

    st.subheader("Pipeline summary")
    cols = st.columns(4)
    cols[0].metric("Baseline rows", f"{int(extract.get('baseline_rows', 0)):,}")
    cols[1].metric("Scale rows", f"{int(extract.get('scale_rows', 0)):,}")
    cols[2].metric("LSH candidates", f"{int(scale.get('candidate_pairs', 0)):,}")
    cols[3].metric("Verified pairs", f"{int(verify.get('verified_pairs', 0)):,}")

    cols = st.columns(4)
    cols[0].metric("Clusters", f"{int(verify.get('clusters', 0)):,}")
    cols[1].metric("Largest cluster", f"{int(verify.get('largest_cluster_size', 0)):,}")
    cols[2].metric("Baseline runtime", f"{float(baseline.get('runtime_seconds', 0.0)):.3f}s")
    cols[3].metric("LSH runtime", f"{float(scale.get('runtime_seconds', 0.0)):.3f}s")


def metric_source_counts(metrics: dict) -> pd.DataFrame:
    source_counts = (
        metrics.get("incremental_update", {}).get("source_counts")
        or metrics.get("source_counts")
        or {}
    )
    if not source_counts:
        return pd.DataFrame()
    frame = pd.DataFrame(
        [{"source": source, "reported_rows": int(rows)} for source, rows in source_counts.items()]
    )
    return frame.sort_values("source").reset_index(drop=True)


def render_source_mix(artifact_dir: str, metrics: dict) -> None:
    reported_counts = metric_source_counts(metrics)
    if not reported_counts.empty:
        st.subheader("Reported source totals")
        st.caption(
            "These totals come from metrics.json and are used for the presentation summary. "
            "Trending still reads the current DuckDB index."
        )
        st.dataframe(reported_counts, width="stretch", hide_index=True)
        return

    if full_search_database(artifact_dir).exists():
        source_counts = query_full_search(
            artifact_dir,
            "SELECT source, count(*) AS indexed_rows FROM documents GROUP BY source ORDER BY source",
        )
        st.subheader("Full search-index source mix")
        st.dataframe(source_counts, width="stretch", hide_index=True)
        return

    scale = load_scale_subset(artifact_dir)
    if "source" not in scale.columns:
        return
    source_counts = scale["source"].fillna("unknown").value_counts().reset_index()
    source_counts.columns = ["source", "rows_in_search_sample"]
    st.subheader("Search sample source mix")
    st.dataframe(source_counts, width="stretch", hide_index=True)


def render_config_table(metrics: dict) -> None:
    results = metrics.get("run_lsh", {}).get("config_results", [])
    if not results:
        return
    frame = pd.DataFrame(results)
    selected = metrics.get("run_lsh", {}).get("selected_config", {}).get("config_name")
    frame["selected"] = frame["config_name"].eq(selected)
    columns = [
        "selected",
        "config_name",
        "shingle_size",
        "num_perm",
        "bands",
        "rows",
        "candidate_pairs",
        "precision",
        "recall",
        "signature_runtime_seconds",
    ]
    st.subheader("LSH config tuning")
    st.dataframe(frame[columns], width="stretch", hide_index=True)


def render_top_clusters(artifact_dir: str) -> None:
    st.subheader("Top narrative clusters")
    limit = st.slider("Top clusters", min_value=3, max_value=20, value=10, step=1)
    st.dataframe(top_cluster_examples(artifact_dir, limit), width="stretch", hide_index=True)


def render_trending(artifact_dir: str) -> None:
    st.subheader("Trending narrative clusters")
    st.caption(
        "Ranked from the latest timestamp in the dataset using post volume, growth, "
        "source/author diversity, recency, and a spam penalty."
    )
    controls = st.columns(3)
    limit = controls[0].slider("Top trending", 5, 50, 15)
    repeated_only = controls[1].checkbox("Repeated clusters only", value=True)
    min_posts = controls[2].number_input("Minimum posts", 1, 100, 2 if repeated_only else 1)
    condition = "posts_lookback >= ?" if repeated_only else "posts_lookback >= ?"
    try:
        trends = query_full_search(
            artifact_dir,
            f"""
            SELECT
                cluster_id,
                trend_score,
                posts_lookback,
                posts_recent,
                posts_previous,
                round(growth_rate, 3) AS growth_rate,
                source_count,
                author_count,
                round(spam_ratio, 3) AS spam_ratio,
                first_seen,
                last_seen,
                sample_source,
                topic_label,
                sample_text
            FROM cluster_trends
            WHERE {condition}
            ORDER BY trend_score DESC, posts_lookback DESC
            LIMIT ?
            """,
            [int(min_posts), int(limit)],
        )
    except Exception:
        st.info("Trending tables are not ready. Run scripts/reporting/refresh_trending.py.")
        return
    if trends.empty:
        st.info("No clusters match the selected filters.")
        return

    visible_columns = ["cluster_id", "first_seen", "sample_source", "topic_label", "sample_text"]
    display = trends[visible_columns].copy()
    display["sample_text"] = display["sample_text"].map(lambda value: compact_text(value, 220))
    st.dataframe(display, width="stretch", hide_index=True)

    selected_cluster = st.selectbox(
        "Inspect cluster timeline",
        trends["cluster_id"].astype(int).tolist(),
        format_func=lambda value: f"Cluster {value}",
    )
    timeline = query_full_search(
        artifact_dir,
        """
        SELECT time_bucket, post_count, source_count, author_count
        FROM cluster_timeline
        WHERE cluster_id = ?
        ORDER BY time_bucket
        """,
        [int(selected_cluster)],
    )
    source_mix = query_full_search(
        artifact_dir,
        """
        SELECT source, count(*) AS posts
        FROM documents
        WHERE cluster_id = ?
        GROUP BY source
        ORDER BY posts DESC
        """,
        [int(selected_cluster)],
    )
    cols = st.columns([3, 1])
    with cols[0]:
        st.markdown("**Posts over time**")
        st.line_chart(timeline.set_index("time_bucket")[["post_count"]], width="stretch")
    with cols[1]:
        st.markdown("**Source mix**")
        st.dataframe(source_mix, width="stretch", hide_index=True)


def main() -> None:
    st.set_page_config(page_title="LSH Narrative Cluster Dashboard", layout="wide")
    st.title("LSH Narrative Cluster Dashboard")

    artifact_dir = st.sidebar.text_input("Artifact directory", value=str(DEFAULT_ARTIFACT_DIR))
    artifact_path_value = Path(artifact_dir)
    if not artifact_path_value.exists():
        st.error(f"Artifact directory not found: {artifact_dir}")
        return

    metrics_path = (
        artifact_path_value / "full_optimized_metrics.json"
        if full_search_database(artifact_dir).exists()
        else artifact_path_value / "metrics.json"
    )
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    if not metrics:
        st.error(f"metrics.json not found in {artifact_dir}")
        return

    st.sidebar.caption(f"Input: {metrics.get('extract_subsets', {}).get('input_path', 'unknown')}")

    tab_overview, tab_trending, tab_clusters, tab_raw = st.tabs(
        ["Overview", "Trending", "Clusters", "Raw metrics"]
    )

    with tab_overview:
        render_metrics(metrics)
        render_source_mix(artifact_dir, metrics)
        render_config_table(metrics)

    with tab_clusters:
        render_top_clusters(artifact_dir)

    with tab_trending:
        render_trending(artifact_dir)

    with tab_raw:
        st.json(metrics)


if __name__ == "__main__":
    main()
