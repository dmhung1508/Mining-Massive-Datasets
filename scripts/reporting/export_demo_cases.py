from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd



from social_lsh.artifacts import artifact_path, ensure_artifact_dir, read_dataframe
from social_lsh.constants import DEFAULT_ARTIFACT_DIR
from social_lsh.preprocessing import deserialize_nested_columns


def _compact_text(value: object, limit: int = 220) -> str:
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _load_scale(artifact_dir: Path) -> pd.DataFrame:
    frame = read_dataframe(artifact_path("scale_shingles", artifact_dir))
    frame = deserialize_nested_columns(frame, ["tokens", "shingles"])
    return frame[["tweet_id", "user_id", "timestamp", "text", "shingle_count"]].copy()


def _cluster_examples(scale: pd.DataFrame, clusters: pd.DataFrame, top_n: int) -> pd.DataFrame:
    merged = clusters.merge(scale, on="tweet_id", how="left")
    ranked_clusters = (
        clusters[["cluster_id", "cluster_size"]]
        .drop_duplicates()
        .sort_values(["cluster_size", "cluster_id"], ascending=[False, True])
        .head(top_n)
    )
    rows: list[dict[str, object]] = []
    for cluster in ranked_clusters.itertuples(index=False):
        members = (
            merged.loc[merged["cluster_id"].eq(cluster.cluster_id)]
            .sort_values(["shingle_count", "tweet_id"], ascending=[False, True])
            .head(3)
        )
        rows.append(
            {
                "cluster_id": int(cluster.cluster_id),
                "cluster_size": int(cluster.cluster_size),
                "sample_tweet_ids": ", ".join(members["tweet_id"].astype(str).tolist()),
                "sample_text": " | ".join(_compact_text(text, limit=160) for text in members["text"].tolist()),
            }
        )
    return pd.DataFrame(rows)


def _pair_examples(scale: pd.DataFrame, pairs: pd.DataFrame, top_n: int) -> pd.DataFrame:
    lookup = scale.set_index("tweet_id")
    rows: list[dict[str, object]] = []
    ordered = pairs.sort_values(["jaccard", "tweet_id_left", "tweet_id_right"], ascending=[False, True, True])
    for row in ordered.head(top_n).itertuples(index=False):
        left = lookup.loc[int(row.tweet_id_left)]
        right = lookup.loc[int(row.tweet_id_right)]
        rows.append(
            {
                "tweet_id_left": int(row.tweet_id_left),
                "tweet_id_right": int(row.tweet_id_right),
                "jaccard": round(float(row.jaccard), 6),
                "left_text": _compact_text(left["text"]),
                "right_text": _compact_text(right["text"]),
            }
        )
    return pd.DataFrame(rows)


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"

    columns = [str(column) for column in frame.columns]
    rows = []
    string_frame = frame.astype("object").where(frame.notna(), "").astype(str)
    for record in string_frame.to_dict(orient="records"):
        rows.append([record[column] for column in frame.columns])

    def clean(value: str) -> str:
        return value.replace("\n", " ").replace("|", "\\|")

    header = "| " + " | ".join(clean(column) for column in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(clean(value) for value in row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def _write_report(
    top_clusters: pd.DataFrame,
    exact_pairs: pd.DataFrame,
    near_pairs: pd.DataFrame,
    output_md: Path,
) -> None:
    lines = [
        "# Week 11 Demo Cases - Hung",
        "",
        "## Muc tieu",
        "",
        "- Chon san cac case de demo pipeline LSH va narrative clusters.",
        "- Co vi du cluster lon, exact duplicate, va near duplicate.",
        "- Dung cho task tuan 11: chuan bi demo cases va test demo.",
        "",
        "## Top clusters",
        "",
        _markdown_table(top_clusters),
        "",
        "## Exact duplicate examples",
        "",
        _markdown_table(exact_pairs),
        "",
        "## Near duplicate examples",
        "",
        _markdown_table(near_pairs),
        "",
        "## Demo script ngan",
        "",
        "1. Mo bang top clusters de chi ra cum noi dung lap lon nhat.",
        "2. Chon mot exact duplicate pair de giai thich Jaccard = 1.0.",
        "3. Chon mot near duplicate pair de giai thich vi sao raw text khac nhung shingles van overlap cao.",
        "4. Chay query demo voi mot cau trong exact duplicate de tra ve cac bai viet tuong tu.",
        "5. Ket luan bang precision/recall va candidate reduction trong benchmark.",
        "",
    ]
    output_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export week-11 demo cases for Hung.")
    parser.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--output-dir", default="docs/hung")
    parser.add_argument("--top-n", type=int, default=5)
    args = parser.parse_args()

    artifact_dir = ensure_artifact_dir(Path(args.artifact_dir))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scale = _load_scale(artifact_dir)
    clusters = read_dataframe(artifact_path("clusters", artifact_dir))
    pairs = read_dataframe(artifact_path("verified_pairs", artifact_dir))

    top_clusters = _cluster_examples(scale, clusters, top_n=args.top_n)
    exact_pairs = _pair_examples(scale, pairs.loc[pairs["jaccard"].ge(0.999999)], top_n=args.top_n)
    near_pairs = _pair_examples(
        scale,
        pairs.loc[pairs["jaccard"].ge(0.8) & pairs["jaccard"].lt(0.999999)],
        top_n=args.top_n,
    )

    outputs = {
        "top_clusters": output_dir / "top_clusters.csv",
        "exact_pairs": output_dir / "exact_duplicate_examples.csv",
        "near_pairs": output_dir / "near_duplicate_examples.csv",
        "report": output_dir / "demo_cases.md",
    }
    top_clusters.to_csv(outputs["top_clusters"], index=False)
    exact_pairs.to_csv(outputs["exact_pairs"], index=False)
    near_pairs.to_csv(outputs["near_pairs"], index=False)
    _write_report(top_clusters, exact_pairs, near_pairs, outputs["report"])

    print(json.dumps({key: str(path) for key, path in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
