from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import pyarrow.dataset as ds
from matplotlib.gridspec import GridSpec

from _bootstrap import add_src_to_path

repo_root = add_src_to_path()

from uk_russia_lsh.datasets import default_twitter_dataset_path


BG = "#f7f3ea"
PANEL = "#fffdf8"
EDGE = "#d7c8b1"
TEXT = "#2d2218"
MUTED = "#7c6856"
ACCENT1 = "#1e5f74"
ACCENT2 = "#c26d3a"
ACCENT3 = "#7b8f4e"
ACCENT4 = "#c7a33c"


def _load_telegram_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if frame.empty:
        return frame
    frame["timestamp"] = pd.Series(
        pd.to_datetime(frame["timestamp"].astype("string"), errors="coerce"),
        index=frame.index,
    )
    frame["date"] = pd.Series(
        pd.to_datetime(frame["date"].astype("string"), errors="coerce").dt.normalize(),
        index=frame.index,
    )
    frame["text_len"] = frame["text"].map(lambda value: len(str(value)))
    return frame


def _telegram_summary(frame: pd.DataFrame) -> dict[str, object]:
    if frame.empty:
        return {
            "rows": 0,
            "channels": 0,
            "date_min": None,
            "date_max": None,
            "media_ratio": 0.0,
            "forward_ratio": 0.0,
        }

    date_series = frame["date"].dropna()
    date_min = date_series.min().strftime("%Y-%m-%d") if not date_series.empty else None
    date_max = date_series.max().strftime("%Y-%m-%d") if not date_series.empty else None

    return {
        "rows": int(len(frame)),
        "channels": int(frame["source_channel_id"].nunique(dropna=True)),
        "date_min": date_min,
        "date_max": date_max,
        "media_ratio": round(float(frame["media_type"].notna().mean()), 4),
        "forward_ratio": round(float(frame["forward_from_username"].notna().mean()), 4),
    }


def _twitter_summary(twitter_path: Path | None) -> dict[str, object]:
    if twitter_path is None or not twitter_path.exists():
        return {
            "exists": False,
            "csv_files": 0,
            "partitions": 0,
            "parquet_files": 0,
            "size_gb": 0.0,
            "date_min": None,
            "date_max": None,
        }

    partition_dirs = sorted(path.name for path in twitter_path.glob("date=*") if path.is_dir())
    parquet_files = list(twitter_path.rglob("*.parquet"))
    size_gb = sum(path.stat().st_size for path in parquet_files) / (1024**3)

    csv_root = Path("/home/anonymous/code/Mining_Massive_Dataset/dataset")
    csv_files = list(csv_root.glob("*_UkraineCombinedTweetsDeduped.csv")) if csv_root.exists() else []

    return {
        "exists": True,
        "csv_files": len(csv_files),
        "partitions": len(partition_dirs),
        "parquet_files": len(parquet_files),
        "size_gb": round(size_gb, 2),
        "date_min": partition_dirs[0].replace("date=", "") if partition_dirs else None,
        "date_max": partition_dirs[-1].replace("date=", "") if partition_dirs else None,
    }


def _style_axes(ax) -> None:
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_color(EDGE)
    ax.tick_params(colors=MUTED)
    ax.title.set_color(TEXT)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)


def _draw_summary_panel(ax, telegram_meta: dict[str, object], twitter_meta: dict[str, object]) -> None:
    _style_axes(ax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    telegram_lines = [
        "Telegram",
        f"Usable rows: {telegram_meta['rows']:,}",
        f"Channels: {telegram_meta['channels']:,}",
        f"Date range: {telegram_meta['date_min']} -> {telegram_meta['date_max']}",
        f"Media ratio: {telegram_meta['media_ratio'] * 100:.1f}%",
        f"Forward ratio: {telegram_meta['forward_ratio'] * 100:.1f}%",
    ]
    twitter_lines = [
        "Twitter",
        f"CSV files: {twitter_meta['csv_files']:,}",
        f"Date partitions: {twitter_meta['partitions']:,}",
        f"Parquet files: {twitter_meta['parquet_files']:,}",
        f"Size: {twitter_meta['size_gb']:.2f} GB",
        f"Date range: {twitter_meta['date_min']} -> {twitter_meta['date_max']}",
        "Canonical merge: ready via build_combined_dataset.py",
    ]

    ax.text(
        0.05,
        0.92,
        "\n".join(telegram_lines),
        va="top",
        ha="left",
        fontsize=10,
        color=TEXT,
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#eef4f6", edgecolor=EDGE),
    )
    ax.text(
        0.53,
        0.92,
        "\n".join(twitter_lines),
        va="top",
        ha="left",
        fontsize=10,
        color=TEXT,
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#fbf1e8", edgecolor=EDGE),
    )


def build_week1_dashboard(
    telegram_input: Path,
    output_png: Path,
    summary_json: Path,
    twitter_input: Path | None = None,
) -> dict[str, object]:
    telegram_frame = _load_telegram_frame(telegram_input)
    if telegram_frame.empty:
        raise ValueError(f"Telegram parquet is empty: {telegram_input}")

    telegram_meta = _telegram_summary(telegram_frame)
    twitter_meta = _twitter_summary(twitter_input)

    daily = telegram_frame.groupby("date").size().reset_index(name="count").sort_values("date")
    top_channels = (
        telegram_frame.groupby("source_channel_id", dropna=True)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(8)
    )
    media_counts = (
        telegram_frame["media_type"]
        .fillna("text_only")
        .value_counts()
        .rename_axis("media_type")
        .reset_index(name="count")
    )

    fig = plt.figure(figsize=(18, 12), facecolor=BG)
    gs = GridSpec(2, 2, figure=fig, hspace=0.24, wspace=0.16)
    fig.suptitle(
        "Week 1 Data Readiness Dashboard",
        fontsize=22,
        fontweight="bold",
        color=TEXT,
        y=0.97,
    )
    fig.text(
        0.5,
        0.94,
        "Telegram export in datatele + Twitter dataset/parquet status for collection and cleaning phase",
        ha="center",
        color=MUTED,
        fontsize=11,
    )

    ax1 = fig.add_subplot(gs[0, 0])
    _style_axes(ax1)
    ax1.fill_between(daily["date"], daily["count"], color=ACCENT1, alpha=0.18)
    ax1.plot(daily["date"], daily["count"], color=ACCENT1, linewidth=2.0)
    ax1.set_title("Telegram Messages By Day", fontsize=15, pad=12)
    ax1.set_ylabel("Messages")
    ax1.grid(axis="y", color=EDGE, alpha=0.6)

    ax2 = fig.add_subplot(gs[0, 1])
    _style_axes(ax2)
    bar_df = top_channels.sort_values("count", ascending=True)
    ax2.barh(bar_df["source_channel_id"].astype(str), bar_df["count"], color=ACCENT2)
    ax2.set_title("Top Telegram Channels", fontsize=15, pad=12)
    ax2.set_xlabel("Messages")
    ax2.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))

    ax3 = fig.add_subplot(gs[1, 0])
    _style_axes(ax3)
    wedges, _, autotexts = ax3.pie(
        media_counts["count"],
        labels=media_counts["media_type"],
        colors=[ACCENT3, ACCENT4, ACCENT2, ACCENT1][: len(media_counts)],
        autopct="%1.1f%%",
        startangle=140,
        wedgeprops=dict(edgecolor=BG, linewidth=2),
        textprops=dict(color=TEXT, fontsize=10),
    )
    for text in autotexts:
        text.set_color(TEXT)
        text.set_fontsize(10)
    ax3.set_title("Telegram Media Mix", fontsize=15, pad=12)

    ax4 = fig.add_subplot(gs[1, 1])
    _draw_summary_panel(ax4, telegram_meta, twitter_meta)
    ax4.set_title("Week 1 Summary", fontsize=15, pad=12)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=170, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "telegram": telegram_meta,
        "twitter": twitter_meta,
        "paths": {
            "telegram_input": str(telegram_input),
            "twitter_input": str(twitter_input) if twitter_input else None,
            "dashboard_png": str(output_png),
            "summary_json": str(summary_json),
        },
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a week-1 data collection dashboard.")
    parser.add_argument(
        "--telegram-input",
        default=str(repo_root / "datatele" / "telegram_messages.parquet"),
    )
    parser.add_argument(
        "--twitter-input",
        default=str(default_twitter_dataset_path() or ""),
    )
    parser.add_argument(
        "--output-png",
        default=str(repo_root / "datatele" / "week1_overview.png"),
    )
    parser.add_argument(
        "--summary-json",
        default=str(repo_root / "datatele" / "week1_summary.json"),
    )
    args = parser.parse_args()

    twitter_input = Path(args.twitter_input) if args.twitter_input else None
    summary = build_week1_dashboard(
        telegram_input=Path(args.telegram_input),
        twitter_input=twitter_input,
        output_png=Path(args.output_png),
        summary_json=Path(args.summary_json),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
