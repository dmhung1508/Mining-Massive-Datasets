from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from social_lsh.artifacts import artifact_path, read_dataframe
from social_lsh.constants import DEFAULT_ARTIFACT_DIR
from social_lsh.search import search_similar_tweets


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


def render_source_mix(artifact_dir: str) -> None:
    scale = load_scale_subset(artifact_dir)
    if "source" not in scale.columns:
        return
    source_counts = scale["source"].fillna("unknown").value_counts().reset_index()
    source_counts.columns = ["source", "rows_in_search_sample"]
    st.subheader("Search sample source mix")
    st.dataframe(source_counts, use_container_width=True, hide_index=True)


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
    st.dataframe(frame[columns], use_container_width=True, hide_index=True)


def render_top_clusters(artifact_dir: str) -> None:
    st.subheader("Top narrative clusters")
    limit = st.slider("Top clusters", min_value=3, max_value=20, value=10, step=1)
    st.dataframe(top_cluster_examples(artifact_dir, limit), use_container_width=True, hide_index=True)


def render_search(artifact_dir: str) -> None:
    st.subheader("Similar post search")
    query = st.text_area(
        "Query text",
        value="What is happening in Kaliningrad today is genocide by the Russian regime.",
        height=90,
    )
    cols = st.columns([1, 1, 2])
    top_k = cols[0].number_input("Top K", min_value=1, max_value=20, value=5, step=1)
    min_jaccard = cols[1].number_input(
        "Min Jaccard",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05,
        help="Use 0.0 for exploratory search. Use 0.8 to show only verified near-duplicate style matches.",
    )
    run = cols[2].button("Search", type="primary", use_container_width=True)

    if not run:
        return

    with st.spinner("Searching similar posts..."):
        results, metadata = search_similar_tweets(
            query_text=query,
            artifact_dir=Path(artifact_dir),
            top_k=int(top_k),
            min_jaccard=float(min_jaccard),
        )

    st.json(metadata)
    if metadata.get("retrieval_mode") == "bruteforce_fallback":
        st.warning(
            "LSH did not find a bucket match for this query, so the app scanned the demo sample "
            "and returned the closest rows above the Jaccard threshold."
        )
    if results.empty:
        st.info("No similar posts found.")
        return

    printable = results.copy()
    max_jaccard = float(printable["jaccard"].max()) if "jaccard" in printable.columns and not printable.empty else 0.0
    if max_jaccard < 0.1:
        st.info(
            "Returned rows are weak fallback matches. For a clean duplicate demo, use a query copied "
            "from an existing cluster or set Min Jaccard around 0.8."
        )
    printable["text"] = printable["text"].map(lambda value: compact_text(value, limit=260))
    printable = printable.rename(
        columns={
            "tweet_id": "internal_id",
            "user_id": "internal_user_id",
            "source_item_id": "source_post_id",
            "source_user_id": "source_author_or_channel",
            "source_channel_id": "telegram_channel_id",
        }
    )
    st.dataframe(printable, use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(page_title="LSH Narrative Cluster Dashboard", layout="wide")
    st.title("LSH Narrative Cluster Dashboard")

    artifact_dir = st.sidebar.text_input("Artifact directory", value=str(DEFAULT_ARTIFACT_DIR))
    artifact_path_value = Path(artifact_dir)
    if not artifact_path_value.exists():
        st.error(f"Artifact directory not found: {artifact_dir}")
        return

    metrics = load_metrics(artifact_dir)
    if not metrics:
        st.error(f"metrics.json not found in {artifact_dir}")
        return

    st.sidebar.caption(f"Input: {metrics.get('extract_subsets', {}).get('input_path', 'unknown')}")

    tab_overview, tab_clusters, tab_search, tab_raw = st.tabs(
        ["Overview", "Clusters", "Search", "Raw metrics"]
    )

    with tab_overview:
        render_metrics(metrics)
        render_source_mix(artifact_dir)
        render_config_table(metrics)

    with tab_clusters:
        render_top_clusters(artifact_dir)

    with tab_search:
        render_search(artifact_dir)

    with tab_raw:
        st.json(metrics)


if __name__ == "__main__":
    main()
