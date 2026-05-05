from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from _bootstrap import add_src_to_path

add_src_to_path()

from uk_russia_lsh.artifacts import artifact_path, ensure_artifact_dir, read_dataframe, read_metrics
from uk_russia_lsh.constants import DEFAULT_ARTIFACT_DIR, DEFAULT_SEED, DEFAULT_VERIFY_THRESHOLD, LSHConfig
from uk_russia_lsh.minhash import run_config
from uk_russia_lsh.preprocessing import deserialize_nested_columns
from uk_russia_lsh.similarity import exact_jaccard_pairs


def _load_shingles(name: str, artifact_dir: Path) -> pd.DataFrame:
    frame = read_dataframe(artifact_path(name, artifact_dir))
    return deserialize_nested_columns(frame, ["tokens", "shingles"])


def _selected_config(artifact_dir: Path) -> LSHConfig:
    metrics = read_metrics(artifact_dir / "metrics.json")
    payload = metrics.get("run_lsh", {}).get("selected_config")
    if not payload:
        raise FileNotFoundError("Selected LSH config not found. Run scripts/run_lsh.py first.")
    return LSHConfig(
        shingle_size=int(payload["shingle_size"]),
        num_perm=int(payload["num_perm"]),
        bands=int(payload["bands"]),
        rows=int(payload["rows"]),
    )


def _sample(frame: pd.DataFrame, size: int, seed: int) -> pd.DataFrame:
    if size >= len(frame):
        return frame.reset_index(drop=True)
    sampled = frame.sample(n=size, random_state=seed)
    return sampled.sort_values("tweet_id").reset_index(drop=True)


def _baseline_rows(
    baseline_frame: pd.DataFrame,
    sizes: list[int],
    threshold: float,
    seed: int,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for size in sizes:
        subset = _sample(baseline_frame, size=size, seed=seed + size)
        pairs, metrics = exact_jaccard_pairs(subset, threshold=threshold)
        rows.append(
            {
                "stage": "brute_force_jaccard",
                "documents": int(len(subset)),
                "pairs_considered": int(metrics["pairs_considered"]),
                "positive_pairs": int(len(pairs)),
                "candidate_pairs": "",
                "runtime_seconds": float(metrics["runtime_seconds"]),
                "pairs_per_second": round(
                    float(metrics["pairs_considered"]) / float(metrics["runtime_seconds"]),
                    2,
                )
                if float(metrics["runtime_seconds"]) > 0
                else 0.0,
            }
        )
    return rows


def _lsh_rows(
    scale_frame: pd.DataFrame,
    sizes: list[int],
    config: LSHConfig,
    seed: int,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for size in sizes:
        subset = _sample(scale_frame, size=size, seed=seed + size)
        started_at = time.perf_counter()
        candidates, metrics = run_config(subset, config=config, seed=seed)
        runtime = round(time.perf_counter() - started_at, 6)
        total_pairs = len(subset) * (len(subset) - 1) // 2
        rows.append(
            {
                "stage": "minhash_lsh_candidates",
                "documents": int(len(subset)),
                "pairs_considered": int(total_pairs),
                "positive_pairs": "",
                "candidate_pairs": int(len(candidates)),
                "runtime_seconds": runtime,
                "candidate_reduction_ratio": round(1 - (len(candidates) / total_pairs), 8)
                if total_pairs
                else 0.0,
                "signature_runtime_seconds": float(metrics["signature_runtime_seconds"]),
                "bucket_count": int(metrics["bucket_count"]),
            }
        )
    return rows


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


def _write_report(rows: list[dict[str, object]], output_md: Path, config: LSHConfig) -> None:
    frame = pd.DataFrame(rows)
    brute = frame.loc[frame["stage"].eq("brute_force_jaccard")].copy()
    lsh = frame.loc[frame["stage"].eq("minhash_lsh_candidates")].copy()

    lines = [
        "# Scalability Benchmark - Hung",
        "",
        "## Muc tieu",
        "",
        "- Do scalability cua brute-force Jaccard va MinHash + LSH tren cac kich thuoc mau khac nhau.",
        "- Chung minh brute-force chi phu hop voi baseline subset, con LSH phu hop hon cho scale run.",
        "- Tao bang so sanh runtime de dua vao demo va bao cao.",
        "",
        "## Cau hinh LSH cuoi",
        "",
        f"- shingle_size: `{config.shingle_size}`",
        f"- num_perm: `{config.num_perm}`",
        f"- bands: `{config.bands}`",
        f"- rows: `{config.rows}`",
        "",
        "## Brute-force Jaccard",
        "",
        _markdown_table(brute),
        "",
        "## MinHash + LSH",
        "",
        _markdown_table(lsh),
        "",
        "## Ket luan",
        "",
        "- Brute-force co so cap tang theo `N * (N - 1) / 2`, nen chi dung de tao ground truth tren tap mau.",
        "- LSH sinh candidate pairs nho hon rat nhieu so voi tong so cap co the co.",
        "- Ket qua benchmark nay dung cho task scalability test, runtime comparison, va toi uu demo.",
        "",
    ]
    output_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create scalability benchmark artifacts for Hung.")
    parser.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--baseline-sizes", default="400,800,1200")
    parser.add_argument("--lsh-sizes", default="5000,10000,50000")
    parser.add_argument("--threshold", type=float, default=DEFAULT_VERIFY_THRESHOLD)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", default="docs/hung_deliverables")
    args = parser.parse_args()

    artifact_dir = ensure_artifact_dir(Path(args.artifact_dir))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_sizes = [int(item.strip()) for item in args.baseline_sizes.split(",") if item.strip()]
    lsh_sizes = [int(item.strip()) for item in args.lsh_sizes.split(",") if item.strip()]
    config = _selected_config(artifact_dir)

    baseline_frame = _load_shingles("baseline_shingles", artifact_dir)
    scale_frame = _load_shingles("scale_shingles", artifact_dir)

    rows = [
        *_baseline_rows(baseline_frame, baseline_sizes, threshold=args.threshold, seed=args.seed),
        *_lsh_rows(scale_frame, lsh_sizes, config=config, seed=args.seed),
    ]

    output_csv = output_dir / "scalability_benchmark.csv"
    output_json = output_dir / "scalability_benchmark.json"
    output_md = output_dir / "lsh_benchmark_report.md"

    pd.DataFrame(rows).to_csv(output_csv, index=False)
    output_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_report(rows, output_md, config)

    print(json.dumps({"csv": str(output_csv), "json": str(output_json), "report": str(output_md)}, indent=2))


if __name__ == "__main__":
    main()
