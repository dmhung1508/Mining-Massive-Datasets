from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd


BG = "#f7f3ea"
PANEL = "#fffdf8"
EDGE = "#d7c8b1"
TEXT = "#2d2218"
MUTED = "#7c6856"
ACCENT1 = "#1e5f74"
ACCENT2 = "#c26d3a"
ACCENT3 = "#7b8f4e"
ACCENT4 = "#c7a33c"


def _style_axes(ax) -> None:
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_color(EDGE)
    ax.tick_params(colors=MUTED)
    ax.title.set_color(TEXT)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _compact_text(value: object, limit: int = 160) -> str:
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _build_components(edges: list[tuple[int, int]]) -> list[list[int]]:
    adjacency: dict[int, set[int]] = defaultdict(set)
    for left, right in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)

    visited: set[int] = set()
    components: list[list[int]] = []
    for node in sorted(adjacency):
        if node in visited:
            continue
        stack = [node]
        current: list[int] = []
        while stack:
            item = stack.pop()
            if item in visited:
                continue
            visited.add(item)
            current.append(item)
            for neighbor in sorted(adjacency[item]):
                if neighbor not in visited:
                    stack.append(neighbor)
        components.append(sorted(current))
    return sorted(components, key=lambda values: (-len(values), values))


def _build_graph_layout(components: list[list[int]]) -> dict[int, tuple[float, float]]:
    if not components:
        return {}

    cols = max(1, math.ceil(math.sqrt(len(components))))
    positions: dict[int, tuple[float, float]] = {}
    for index, component in enumerate(components):
        center_x = (index % cols) * 5.5
        center_y = -(index // cols) * 5.5
        if len(component) == 1:
            positions[component[0]] = (center_x, center_y)
            continue
        radius = max(0.9, 0.45 * math.sqrt(len(component)))
        for item_index, node in enumerate(component):
            angle = 2 * math.pi * item_index / len(component)
            positions[node] = (
                center_x + radius * math.cos(angle),
                center_y + radius * math.sin(angle),
            )
    return positions


def _write_similarity_outputs(
    output_dir: Path,
    pair_sources: list[Path],
    cluster_source: Path,
) -> dict[str, object]:
    pair_frames: list[pd.DataFrame] = []
    for source in pair_sources:
        frame = _read_csv(source)
        if frame.empty:
            continue
        frame = frame.copy()
        frame["source_file"] = source.name
        pair_frames.append(frame)

    pairs = pd.concat(pair_frames, ignore_index=True) if pair_frames else pd.DataFrame()
    if not pairs.empty:
        pairs["jaccard"] = pd.to_numeric(pairs["jaccard"], errors="coerce")
        pairs = pairs.dropna(subset=["tweet_id_left", "tweet_id_right", "jaccard"]).copy()
        pairs["tweet_id_left"] = pairs["tweet_id_left"].astype("int64")
        pairs["tweet_id_right"] = pairs["tweet_id_right"].astype("int64")
        pairs = pairs.sort_values(["jaccard", "tweet_id_left", "tweet_id_right"], ascending=[False, True, True]).reset_index(
            drop=True
        )

    clusters = _read_csv(cluster_source)
    if not clusters.empty and "cluster_size" in clusters.columns:
        clusters["cluster_size"] = pd.to_numeric(clusters["cluster_size"], errors="coerce")
        clusters = clusters.dropna(subset=["cluster_size"]).copy()
        clusters["cluster_size"] = clusters["cluster_size"].astype("int64")

    histogram_png = output_dir / "similarity_histogram.png"
    graph_png = output_dir / "cluster_graph.png"
    summary_json = output_dir / "similarity_summary.json"
    notes_md = output_dir / "similarity_notes.md"

    fig_hist, ax_hist = plt.subplots(figsize=(10.5, 6.2), facecolor=BG)
    _style_axes(ax_hist)
    if pairs.empty:
        ax_hist.text(0.5, 0.5, "No pair data available", ha="center", va="center", color=MUTED, fontsize=12)
        ax_hist.set_xticks([])
        ax_hist.set_yticks([])
    else:
        bins = [0.8, 0.85, 0.9, 0.95, 0.98, 1.0001]
        ax_hist.hist(pairs["jaccard"], bins=bins, color=ACCENT1, edgecolor=BG)
        ax_hist.axvline(0.8, color=ACCENT2, linestyle="--", linewidth=2, label="verify threshold = 0.8")
        ax_hist.set_xlim(0.78, 1.01)
        ax_hist.set_xlabel("Jaccard similarity")
        ax_hist.set_ylabel("Pair count")
        ax_hist.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
        ax_hist.grid(axis="y", color=EDGE, alpha=0.6)
        ax_hist.legend(frameon=False)
    ax_hist.set_title("Similarity Histogram", fontsize=15, pad=12)
    fig_hist.tight_layout()
    fig_hist.savefig(histogram_png, dpi=170, bbox_inches="tight")
    plt.close(fig_hist)

    fig_graph, (ax_graph, ax_cluster) = plt.subplots(1, 2, figsize=(15, 6.2), facecolor=BG)
    _style_axes(ax_graph)
    _style_axes(ax_cluster)

    edges = []
    if not pairs.empty:
        for row in pairs[["tweet_id_left", "tweet_id_right"]].drop_duplicates().itertuples(index=False):
            left_id = int(min(row.tweet_id_left, row.tweet_id_right))
            right_id = int(max(row.tweet_id_left, row.tweet_id_right))
            edges.append((left_id, right_id))

    if not edges:
        ax_graph.text(0.5, 0.5, "No graph edges available", ha="center", va="center", color=MUTED, fontsize=12)
        ax_graph.set_xticks([])
        ax_graph.set_yticks([])
    else:
        components = _build_components(edges)
        positions = _build_graph_layout(components)
        degree: dict[int, int] = defaultdict(int)
        for left_id, right_id in edges:
            degree[left_id] += 1
            degree[right_id] += 1

        for left_id, right_id in edges:
            x1, y1 = positions[left_id]
            x2, y2 = positions[right_id]
            ax_graph.plot([x1, x2], [y1, y2], color=EDGE, linewidth=1.5, zorder=1)

        node_order = sorted(positions.keys(), key=lambda node: (-degree[node], node))
        node_sizes = [90 + 50 * degree[node] for node in node_order]
        ax_graph.scatter(
            [positions[node][0] for node in node_order],
            [positions[node][1] for node in node_order],
            s=node_sizes,
            c=ACCENT3,
            edgecolors=ACCENT1,
            linewidths=1.0,
            zorder=2,
        )
        for node in node_order[:10]:
            x_pos, y_pos = positions[node]
            ax_graph.text(x_pos, y_pos, str(node)[-4:], ha="center", va="center", fontsize=8, color=TEXT, zorder=3)
        ax_graph.set_xticks([])
        ax_graph.set_yticks([])
        ax_graph.set_aspect("equal")

    ax_graph.set_title("Cluster Graph (pair examples)", fontsize=15, pad=12)

    if clusters.empty:
        ax_cluster.text(0.5, 0.5, "No cluster summary file available", ha="center", va="center", color=MUTED, fontsize=12)
        ax_cluster.set_xticks([])
        ax_cluster.set_yticks([])
    else:
        top_clusters = clusters.sort_values(["cluster_size", "cluster_id"], ascending=[False, True]).head(8).copy()
        top_clusters = top_clusters.iloc[::-1]
        ax_cluster.barh(top_clusters["cluster_id"].astype(str), top_clusters["cluster_size"], color=ACCENT4)
        ax_cluster.set_xlabel("Cluster size")
        ax_cluster.set_ylabel("Cluster ID")
        ax_cluster.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
        ax_cluster.grid(axis="x", color=EDGE, alpha=0.6)
    ax_cluster.set_title("Top Clusters By Size", fontsize=15, pad=12)
    fig_graph.tight_layout()
    fig_graph.savefig(graph_png, dpi=170, bbox_inches="tight")
    plt.close(fig_graph)

    component_count = len(_build_components(edges)) if edges else 0
    largest_component_size = max((len(component) for component in _build_components(edges)), default=0) if edges else 0
    summary = {
        "pairs_total": int(len(pairs)),
        "pairs_exact": int((pairs["jaccard"] >= 0.999999).sum()) if not pairs.empty else 0,
        "pairs_near_duplicate": int((pairs["jaccard"] >= 0.8).sum()) if not pairs.empty else 0,
        "graph_components": int(component_count),
        "largest_graph_component": int(largest_component_size),
        "cluster_rows": int(len(clusters)),
        "pair_sources": [str(path) for path in pair_sources],
        "cluster_source": str(cluster_source),
        "histogram_png": str(histogram_png),
        "graph_png": str(graph_png),
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Similarity Deliverables (Bao)",
        "",
        "## Muc tieu",
        "",
        "- Ve histogram similarity de minh hoa phan bo Jaccard cua cac pair tieu bieu.",
        "- Ve cluster graph de trinh bay lien ket giua cac pair gan trung lap.",
        "- Tao tom tat nhanh de dua vao slide phan trinh bay.",
        "",
        "## Ket qua chinh",
        "",
        f"- Tong so pair duoc tong hop: `{summary['pairs_total']}`",
        f"- Pair exact duplicate (Jaccard ~= 1.0): `{summary['pairs_exact']}`",
        f"- Pair near duplicate (Jaccard >= 0.8): `{summary['pairs_near_duplicate']}`",
        f"- So connected components trong graph: `{summary['graph_components']}`",
        f"- Kich thuoc component lon nhat: `{summary['largest_graph_component']}`",
        "",
        "## Artifact",
        "",
        f"- Histogram: `{histogram_png}`",
        f"- Cluster graph: `{graph_png}`",
        f"- Summary json: `{summary_json}`",
        "",
    ]
    notes_md.write_text("\n".join(lines), encoding="utf-8")
    summary["notes_md"] = str(notes_md)
    summary["summary_json"] = str(summary_json)
    return summary


def _write_performance_outputs(
    output_dir: Path,
    benchmark_source: Path,
) -> dict[str, object]:
    benchmark = _read_csv(benchmark_source)
    if benchmark.empty:
        raise FileNotFoundError(f"Benchmark csv not found or empty: {benchmark_source}")

    benchmark["documents"] = pd.to_numeric(benchmark["documents"], errors="coerce")
    benchmark["pairs_considered"] = pd.to_numeric(benchmark["pairs_considered"], errors="coerce")
    benchmark["runtime_seconds"] = pd.to_numeric(benchmark["runtime_seconds"], errors="coerce")
    benchmark["candidate_pairs"] = pd.to_numeric(benchmark["candidate_pairs"], errors="coerce")
    benchmark["candidate_reduction_ratio"] = pd.to_numeric(benchmark["candidate_reduction_ratio"], errors="coerce")
    benchmark = benchmark.dropna(subset=["documents", "pairs_considered", "runtime_seconds"]).copy()

    brute = benchmark.loc[benchmark["stage"].eq("brute_force_jaccard")].copy().sort_values("documents")
    lsh = benchmark.loc[benchmark["stage"].eq("minhash_lsh_candidates")].copy().sort_values("documents")

    brute["pairs_per_second_est"] = brute["pairs_considered"] / brute["runtime_seconds"]
    lsh["pairs_per_second_est"] = lsh["pairs_considered"] / lsh["runtime_seconds"]

    dashboard_png = output_dir / "performance_dashboard.png"
    summary_json = output_dir / "performance_summary.json"
    notes_md = output_dir / "performance_notes.md"
    table_csv = output_dir / "performance_table.csv"

    fig = plt.figure(figsize=(16, 11), facecolor=BG)
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.2)
    fig.suptitle("Performance Visualization", fontsize=20, fontweight="bold", color=TEXT, y=0.98)

    ax1 = fig.add_subplot(gs[0, 0])
    _style_axes(ax1)
    if not brute.empty:
        ax1.plot(brute["documents"], brute["runtime_seconds"], marker="o", color=ACCENT2, linewidth=2, label="brute-force")
    if not lsh.empty:
        ax1.plot(lsh["documents"], lsh["runtime_seconds"], marker="o", color=ACCENT1, linewidth=2, label="minhash + lsh")
    ax1.set_title("Runtime by document count", fontsize=14, pad=10)
    ax1.set_xlabel("Documents")
    ax1.set_ylabel("Runtime (seconds)")
    ax1.grid(axis="y", color=EDGE, alpha=0.6)
    ax1.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.2f}"))
    if not brute.empty or not lsh.empty:
        ax1.legend(frameon=False)

    ax2 = fig.add_subplot(gs[0, 1])
    _style_axes(ax2)
    if not brute.empty:
        ax2.plot(
            brute["pairs_considered"],
            brute["runtime_seconds"],
            marker="o",
            color=ACCENT2,
            linewidth=2,
            label="brute-force",
        )
    if not lsh.empty:
        ax2.plot(
            lsh["pairs_considered"],
            lsh["runtime_seconds"],
            marker="o",
            color=ACCENT1,
            linewidth=2,
            label="minhash + lsh",
        )
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_title("Runtime vs pairs considered (log-log)", fontsize=14, pad=10)
    ax2.set_xlabel("Pairs considered")
    ax2.set_ylabel("Runtime (seconds)")
    ax2.grid(color=EDGE, alpha=0.5)
    if not brute.empty or not lsh.empty:
        ax2.legend(frameon=False)

    ax3 = fig.add_subplot(gs[1, 0])
    _style_axes(ax3)
    if lsh.empty:
        ax3.text(0.5, 0.5, "No LSH rows", ha="center", va="center", color=MUTED, fontsize=11)
        ax3.set_xticks([])
        ax3.set_yticks([])
    else:
        ax3.bar(
            lsh["documents"].astype(int).astype(str),
            lsh["candidate_reduction_ratio"] * 100.0,
            color=ACCENT3,
        )
        ax3.set_title("Candidate reduction ratio", fontsize=14, pad=10)
        ax3.set_xlabel("Documents")
        ax3.set_ylabel("Reduction (%)")
        ax3.set_ylim(99.9, 100.001)
        ax3.grid(axis="y", color=EDGE, alpha=0.6)

    ax4 = fig.add_subplot(gs[1, 1])
    _style_axes(ax4)
    if brute.empty and lsh.empty:
        ax4.text(0.5, 0.5, "No speed rows", ha="center", va="center", color=MUTED, fontsize=11)
        ax4.set_xticks([])
        ax4.set_yticks([])
    else:
        labels: list[str] = []
        values: list[float] = []
        colors: list[str] = []
        if not brute.empty:
            labels.append("Brute-force")
            values.append(float(brute["pairs_per_second_est"].mean()))
            colors.append(ACCENT2)
        if not lsh.empty:
            labels.append("MinHash+LSH")
            values.append(float(lsh["pairs_per_second_est"].mean()))
            colors.append(ACCENT1)
        ax4.bar(labels, values, color=colors)
        ax4.set_title("Average pairs processed per second", fontsize=14, pad=10)
        ax4.set_ylabel("Pairs/second")
        ax4.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
        ax4.grid(axis="y", color=EDGE, alpha=0.6)

    fig.subplots_adjust(left=0.06, right=0.98, top=0.93, bottom=0.07, hspace=0.32, wspace=0.22)
    fig.savefig(dashboard_png, dpi=170, bbox_inches="tight")
    plt.close(fig)

    benchmark.to_csv(table_csv, index=False)
    summary = {
        "benchmark_source": str(benchmark_source),
        "rows_total": int(len(benchmark)),
        "bruteforce_rows": int(len(brute)),
        "lsh_rows": int(len(lsh)),
        "max_documents": int(benchmark["documents"].max()) if not benchmark.empty else 0,
        "max_pairs_considered": int(benchmark["pairs_considered"].max()) if not benchmark.empty else 0,
        "max_runtime_seconds": float(benchmark["runtime_seconds"].max()) if not benchmark.empty else 0.0,
        "lsh_avg_candidate_reduction_percent": round(
            float((lsh["candidate_reduction_ratio"] * 100.0).mean()) if not lsh.empty else 0.0,
            6,
        ),
        "bruteforce_avg_pairs_per_second": round(float(brute["pairs_per_second_est"].mean()) if not brute.empty else 0.0, 2),
        "lsh_avg_pairs_per_second": round(float(lsh["pairs_per_second_est"].mean()) if not lsh.empty else 0.0, 2),
        "dashboard_png": str(dashboard_png),
        "table_csv": str(table_csv),
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Performance Deliverables (Bao)",
        "",
        "## Muc tieu",
        "",
        "- Ve bieu do hieu nang de so sanh luong tinh toan giua brute-force va MinHash + LSH.",
        "- Minh hoa kha nang giam candidate pairs cua LSH khi scale len du lieu lon.",
        "- Chuan bi visual ngắn gon de dua thang vao slide.",
        "",
        "## Ket qua chinh",
        "",
        f"- So dong benchmark: `{summary['rows_total']}`",
        f"- So document lon nhat duoc test: `{summary['max_documents']}`",
        f"- Candidate reduction trung binh cua LSH: `{summary['lsh_avg_candidate_reduction_percent']}%`",
        f"- Toc do trung binh brute-force: `{summary['bruteforce_avg_pairs_per_second']}` pairs/s",
        f"- Toc do trung binh MinHash+LSH: `{summary['lsh_avg_pairs_per_second']}` pairs/s",
        "",
        "## Artifact",
        "",
        f"- Dashboard: `{dashboard_png}`",
        f"- Benchmark table (copy): `{table_csv}`",
        f"- Summary json: `{summary_json}`",
        "",
    ]
    notes_md.write_text("\n".join(lines), encoding="utf-8")
    summary["notes_md"] = str(notes_md)
    summary["summary_json"] = str(summary_json)
    return summary


def _write_slides_outputs(
    output_dir: Path,
    similarity_summary: dict[str, object],
    performance_summary: dict[str, object],
    top_clusters_csv: Path,
    exact_examples_csv: Path,
    near_examples_csv: Path,
) -> dict[str, object]:
    top_clusters = _read_csv(top_clusters_csv)
    exact_examples = _read_csv(exact_examples_csv)
    near_examples = _read_csv(near_examples_csv)

    slides_md = output_dir / "slides_outline.md"
    tables_csv = output_dir / "charts_tables.csv"
    summary_json = output_dir / "slides_summary.json"

    chart_table_rows = [
        {
            "slide": "3",
            "asset_type": "chart",
            "asset_path": str(similarity_summary.get("histogram_png", "")),
            "message": "Similarity histogram cho thay phan bo pair near-duplicate",
        },
        {
            "slide": "4",
            "asset_type": "chart",
            "asset_path": str(similarity_summary.get("graph_png", "")),
            "message": "Cluster graph minh hoa cac nhom noi dung lap",
        },
        {
            "slide": "5",
            "asset_type": "chart",
            "asset_path": str(performance_summary.get("dashboard_png", "")),
            "message": "So sanh hieu nang brute-force va MinHash + LSH",
        },
        {
            "slide": "6",
            "asset_type": "table",
            "asset_path": str(top_clusters_csv),
            "message": "Top clusters duoc dung lam case demo",
        },
        {
            "slide": "7",
            "asset_type": "table",
            "asset_path": str(exact_examples_csv),
            "message": "Exact duplicate examples (Jaccard = 1.0)",
        },
        {
            "slide": "8",
            "asset_type": "table",
            "asset_path": str(near_examples_csv),
            "message": "Near duplicate examples (0.8 <= Jaccard < 1.0)",
        },
    ]
    chart_table_frame = pd.DataFrame(chart_table_rows)
    chart_table_frame.to_csv(tables_csv, index=False)

    lines = [
        "# Slide Outline (Bao)",
        "",
        "## Slide 1 - Title",
        "- Mining Massive Datasets: Near-Duplicate Detection with MinHash + LSH",
        "",
        "## Slide 2 - Problem and pipeline",
        "- Bai toan: tim noi dung gan trung lap tren social stream.",
        "- Pipeline: preprocessing -> shingling -> MinHash -> LSH -> verify -> cluster.",
        "",
        "## Slide 3 - Similarity distribution",
        f"- Dung `{similarity_summary.get('histogram_png', '')}` de trinh bay phan bo similarity.",
        "- Nhan manh nguong xac minh Jaccard >= 0.8.",
        "",
        "## Slide 4 - Cluster graph",
        f"- Dung `{similarity_summary.get('graph_png', '')}` de mo ta cau truc cum.",
        "- Chi ra so component va component lon nhat.",
        "",
        "## Slide 5 - Scalability and runtime",
        f"- Dung `{performance_summary.get('dashboard_png', '')}` de so sanh hieu nang.",
        "- Ket luan: brute-force chi hop cho subset, LSH moi scale duoc.",
        "",
        "## Slide 6 - Top cluster examples",
        f"- Nguon bang: `{top_clusters_csv}`",
        "- Chon 2-3 cum co y nghia de minh hoa story.",
        "",
        "## Slide 7 - Exact duplicate demo",
        f"- Nguon bang: `{exact_examples_csv}`",
        "- Giai thich vi sao Jaccard = 1.0 va tai sao can dedup.",
        "",
        "## Slide 8 - Near duplicate demo",
        f"- Nguon bang: `{near_examples_csv}`",
        "- Giai thich overlap shingles du text khac nhau.",
        "",
        "## Slide 9 - Risks and limitations",
        "- Nguong Jaccard co the bo sot paraphrase xa.",
        "- Sampling va nguon du lieu co the gay lech nhan xet.",
        "",
        "## Slide 10 - Final takeaway",
        "- LSH giam candidate rat manh, giu lai kha nang tim duplicate.",
        "- KQ du tot de support demo va bao cao cuoi ky.",
        "",
    ]
    slides_md.write_text("\n".join(lines), encoding="utf-8")

    summary = {
        "top_clusters_rows": int(len(top_clusters)),
        "exact_examples_rows": int(len(exact_examples)),
        "near_examples_rows": int(len(near_examples)),
        "slides_md": str(slides_md),
        "charts_tables_csv": str(tables_csv),
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["summary_json"] = str(summary_json)
    return summary


def _write_report_outputs(
    output_dir: Path,
    similarity_summary: dict[str, object],
    performance_summary: dict[str, object],
    slides_summary: dict[str, object],
) -> dict[str, object]:
    report_md = output_dir / "dataset_results_report.md"
    practice_md = output_dir / "demo_practice_checklist.md"
    summary_json = output_dir / "report_summary.json"

    lines_report = [
        "# Dataset and Results (Bao)",
        "",
        "## 1) Dataset mo ta ngan",
        "",
        "- Du lieu duoc xu ly theo pipeline LSH cua nhom (Twitter + Telegram sau khi chuan hoa).",
        "- Tap benchmark va demo cases duoc lay tu artifact da precompute de dam bao tai lap.",
        "- Cac bang case duoc dung: top clusters, exact duplicates, near duplicates.",
        "",
        "## 2) Ket qua thuc nghiem",
        "",
        f"- Similarity pairs tong hop: `{similarity_summary.get('pairs_total', 0)}`",
        f"- Connected components: `{similarity_summary.get('graph_components', 0)}`",
        f"- Max documents benchmark: `{performance_summary.get('max_documents', 0)}`",
        f"- Candidate reduction trung binh: `{performance_summary.get('lsh_avg_candidate_reduction_percent', 0)}%`",
        f"- Exact demo examples: `{slides_summary.get('exact_examples_rows', 0)}`",
        f"- Near-duplicate demo examples: `{slides_summary.get('near_examples_rows', 0)}`",
        "",
        "## 3) Insight",
        "",
        "- LSH giup cat giam candidate pairs rat lon so voi toan bo cap co the.",
        "- Van giu duoc cac cap duplicate/near-duplicate co chat luong de phuc vu demo.",
        "- Cluster lon giup minh hoa cac luong noi dung lap tren social stream.",
        "",
        "## 4) Han che",
        "",
        "- Nguong Jaccard co the bo sot noi dung paraphrase xa.",
        "- Ket qua benchmark va demo phu thuoc vao chat luong preprocess.",
        "- Cac case dua vao slide nen duoc loc theo do ro rang ve noi dung va encoding.",
        "",
        "## 5) Kien nghi cho final report",
        "",
        "- Dat histogram similarity + cluster graph o phan ket qua.",
        "- Dat benchmark runtime ngay sau pipeline de giai thich ly do chon LSH.",
        "- Dung 1 exact case + 1 near case + 1 top cluster de chot thong diep.",
        "",
    ]
    report_md.write_text("\n".join(lines_report), encoding="utf-8")

    lines_practice = [
        "# Practice Demo Checklist (Bao)",
        "",
        "## Truoc buoi demo",
        "- [ ] Mo san cac file chart/table trong docs/bao",
        "- [ ] Mo san case files trong docs/hung",
        "- [ ] Kiem tra lai thu tu slide va luong ke chuyen",
        "",
        "## Dry-run 1",
        "- [ ] Trinh bay duoc bai toan trong <= 60 giay",
        "- [ ] Di qua pipeline va threshold verify ro rang",
        "- [ ] Minh hoa 1 exact duplicate va 1 near duplicate",
        "- [ ] Giai thich duoc vi sao LSH nhanh hon brute-force",
        "",
        "## Dry-run 2",
        "- [ ] Giu tong thoi luong trong 8-10 phut",
        "- [ ] Chuyen slide khong bi ngat quang",
        "- [ ] Tra loi duoc 3 cau hoi thuong gap ve precision/recall, false positive, tuning",
        "",
        "## Q&A quick answers",
        "- Tai sao threshold 0.8? -> Can bang giua precision va recall cho near-duplicate.",
        "- Tai sao khong brute-force full data? -> Do O(N^2), khong practical khi scale lon.",
        "- LSH co bo sot khong? -> Co, vi vay can buoc verify exact Jaccard sau candidate generation.",
        "",
    ]
    practice_md.write_text("\n".join(lines_practice), encoding="utf-8")

    summary = {
        "report_md": str(report_md),
        "practice_md": str(practice_md),
        "pairs_total": int(similarity_summary.get("pairs_total", 0)),
        "max_documents": int(performance_summary.get("max_documents", 0)),
        "demo_rows": int(slides_summary.get("exact_examples_rows", 0))
        + int(slides_summary.get("near_examples_rows", 0)),
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["summary_json"] = str(summary_json)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Bao deliverables from existing benchmark and demo cases.")
    parser.add_argument("--output-dir", default="docs/bao")
    parser.add_argument("--benchmark-csv", default="docs/hung/scalability_benchmark.csv")
    parser.add_argument("--top-clusters-csv", default="docs/hung/top_clusters.csv")
    parser.add_argument("--exact-examples-csv", default="docs/hung/exact_duplicate_examples.csv")
    parser.add_argument("--near-examples-csv", default="docs/hung/near_duplicate_examples.csv")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    similarity = _write_similarity_outputs(
        output_dir=output_dir,
        pair_sources=[Path(args.exact_examples_csv), Path(args.near_examples_csv)],
        cluster_source=Path(args.top_clusters_csv),
    )
    performance = _write_performance_outputs(
        output_dir=output_dir,
        benchmark_source=Path(args.benchmark_csv),
    )
    slides = _write_slides_outputs(
        output_dir=output_dir,
        similarity_summary=similarity,
        performance_summary=performance,
        top_clusters_csv=Path(args.top_clusters_csv),
        exact_examples_csv=Path(args.exact_examples_csv),
        near_examples_csv=Path(args.near_examples_csv),
    )
    report = _write_report_outputs(
        output_dir=output_dir,
        similarity_summary=similarity,
        performance_summary=performance,
        slides_summary=slides,
    )

    payload = {
        "similarity": similarity,
        "performance": performance,
        "slides": slides,
        "report": report,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
