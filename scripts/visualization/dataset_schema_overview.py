import os
from pathlib import Path

import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless (không cần display)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore")

from social_lsh.constants import DEFAULT_INPUT_PARQUET, DEFAULT_VISUALS_DIR

# ─────────────────────────────────────────────────────────
PARQUET_DIR = str(DEFAULT_INPUT_PARQUET)
OUTPUT_PNG  = str(DEFAULT_VISUALS_DIR / "dataset_schema_overview.png")
# ─────────────────────────────────────────────────────────

# ── Palette ──────────────────────────────────────────────
BG       = "#0d1117"
PANEL    = "#161b22"
ACCENT1  = "#58a6ff"   # blue
ACCENT2  = "#3fb950"   # green
ACCENT3  = "#f78166"   # red / coral
ACCENT4  = "#d2a8ff"   # purple
ACCENT5  = "#ffa657"   # orange
TEXT     = "#e6edf3"
SUBTEXT  = "#8b949e"
GRID     = "#21262d"


def set_style():
    plt.rcParams.update({
        "figure.facecolor":  BG,
        "axes.facecolor":    PANEL,
        "axes.edgecolor":    GRID,
        "axes.labelcolor":   TEXT,
        "axes.titlecolor":   TEXT,
        "xtick.color":       SUBTEXT,
        "ytick.color":       SUBTEXT,
        "text.color":        TEXT,
        "grid.color":        GRID,
        "grid.linewidth":    0.6,
        "font.family":       "DejaVu Sans",
        "font.size":         10,
        "axes.titlesize":    13,
        "axes.labelsize":    10,
        "legend.facecolor":  PANEL,
        "legend.edgecolor":  GRID,
        "legend.labelcolor": TEXT,
    })


# ── Load data ─────────────────────────────────────────────
def load_data():
    print("📂  Đọc tweets_final.parquet ...")
    dataset = pq.ParquetDataset(PARQUET_DIR)
    # chỉ cần vài cột để nhanh
    table = dataset.read(columns=["tweet_id", "user_id", "text", "timestamp", "date"])
    df = table.to_pandas()

    # Partition columns from pyarrow come back as Categorical — cast to str first
    df["date"]      = pd.to_datetime(df["date"].astype(str))
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(str), errors="coerce")
    df["text_len"]  = df["text"].str.len()
    print(f"   ✓ Loaded {len(df):,} rows — {df['date'].min().date()} → {df['date'].max().date()}")
    return df


# ── Aggregations ──────────────────────────────────────────
def compute_stats(df):
    daily    = df.groupby("date").size().reset_index(name="count").sort_values("date")
    top_usr  = df.groupby("user_id").size().nlargest(15).reset_index(name="count")
    # heatmap: weekday × week-of-year
    df2      = df.copy()
    df2["weekday"] = df2["date"].dt.day_name()
    df2["month"]   = df2["date"].dt.strftime("%Y-%m")
    hm = df2.groupby(["month", "weekday"]).size().unstack(
        fill_value=0
    ).reindex(columns=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
    return daily, top_usr, hm


# Notable war events to annotate
EVENTS = {
    "2022-09-06": "🏙️ Kharkiv\ncounter-offensive",
    "2022-10-10": "💥 Kyiv\nmissile strikes",
    "2022-11-15": "🏘️ Kherson\nliberated",
    "2023-02-24": "📅 War\n1-year mark",
}


# ── Plot ──────────────────────────────────────────────────
def make_figure(df, daily, top_usr, hm):
    set_style()

    fig = plt.figure(figsize=(22, 18), facecolor=BG)
    fig.suptitle(
        "Ukraine War Twitter Dataset  •  Intelligent Post & User Recommendation",
        fontsize=18, fontweight="bold", color=TEXT, y=0.98
    )

    gs = GridSpec(
        3, 3,
        figure=fig,
        hspace=0.45, wspace=0.35,
        left=0.06, right=0.97, top=0.93, bottom=0.06
    )

    # ── [0,0:3]  Time-series (full width) ────────────────
    ax1 = fig.add_subplot(gs[0, :])
    dates  = daily["date"].values
    counts = daily["count"].values

    # gradient fill
    ax1.fill_between(dates, counts, alpha=0.18, color=ACCENT1)
    ax1.plot(dates, counts, color=ACCENT1, linewidth=1.5, zorder=3)

    # 7-day rolling avg
    roll = daily["count"].rolling(7, center=True).mean()
    ax1.plot(dates, roll, color=ACCENT2, linewidth=2.2,
             linestyle="--", label="7-day rolling avg", zorder=4)

    ax1.set_title("📈  Tweets Per Day  (2022-08-19 → 2023-06-14)", pad=10)
    ax1.set_ylabel("Tweet count")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.grid(True, axis="y", alpha=0.4)
    ax1.legend(framealpha=0.3, fontsize=9)

    # annotate events
    for date_str, label in EVENTS.items():
        dt  = pd.Timestamp(date_str)
        row = daily[daily["date"] == dt]
        if row.empty:
            continue
        y = row["count"].values[0]
        ax1.annotate(
            label, xy=(dt, y), xytext=(0, 22),
            textcoords="offset points",
            fontsize=7.5, color=ACCENT3, ha="center",
            arrowprops=dict(arrowstyle="-|>", color=ACCENT3, lw=0.8),
            bbox=dict(boxstyle="round,pad=0.3", fc=PANEL, ec=ACCENT3, alpha=0.7)
        )

    # peak label
    peak_idx = daily["count"].idxmax()
    peak_row = daily.loc[peak_idx]
    ax1.annotate(
        f"Peak\n{int(peak_row['count']):,}",
        xy=(peak_row["date"], peak_row["count"]),
        xytext=(15, -30), textcoords="offset points",
        fontsize=8, color=ACCENT5,
        arrowprops=dict(arrowstyle="-|>", color=ACCENT5, lw=0.8),
    )

    # ── [1,0]  Schema overview (table visual) ─────────────
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_axis_off()
    ax2.set_facecolor(PANEL)

    schema_rows = [
        ("tweet_id",    "Long",       "NOT NULL", "Tweet ID"),
        ("user_id",     "Long",       "NOT NULL", "hash(username)"),
        ("text",        "String",     "NOT NULL", "Cleaned tweet text"),
        ("timestamp",   "Timestamp",  "NOT NULL", "Created at (UTC)"),
        ("embeddings",  "String",     "NULL ✦",   "Placeholder – NLP"),
        ("topic_label", "String",     "NULL ✦",   "Placeholder – ML"),
        ("date",        "String",     "NOT NULL", "Partition key"),
    ]
    col_hdrs = ["Column", "Type", "Nullable", "Description"]
    col_widths = [0.20, 0.18, 0.16, 0.46]
    row_h = 0.10
    x_starts = [0.01]
    for w in col_widths[:-1]:
        x_starts.append(x_starts[-1] + w)

    ax2.set_title("🗂️  Data Schema", pad=8)
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)

    # header
    for x, hdr, w in zip(x_starts, col_hdrs, col_widths):
        ax2.text(x + w/2, 0.94, hdr, ha="center", va="center",
                 fontsize=8.5, fontweight="bold", color=ACCENT1)

    ax2.axhline(0.90, color=ACCENT1, linewidth=0.8, alpha=0.6)

    colors_row = [ACCENT2 if r[2] == "NOT NULL" else ACCENT3 for r in schema_rows]
    for i, (row, rc) in enumerate(zip(schema_rows, colors_row)):
        y = 0.85 - i * row_h
        if i % 2 == 0:
            ax2.add_patch(mpatches.FancyBboxPatch(
                (0, y - 0.04), 1, row_h,
                boxstyle="square,pad=0", fc="#1f2937", ec="none", zorder=0
            ))
        vals = list(row)
        for x, val, w, ci in zip(x_starts, vals, col_widths, range(4)):
            color = rc if ci == 2 else TEXT
            fw = "bold" if ci == 0 else "normal"
            ax2.text(x + w/2, y, val, ha="center", va="center",
                     fontsize=7.5, color=color, fontweight=fw)

    ax2.text(0.5, 0.02, "✦ NULL = placeholder for later pipeline stages",
             ha="center", fontsize=7, color=SUBTEXT, style="italic")

    # ── [1,1]  Tweet volume summary (donut) ──────────────
    ax3 = fig.add_subplot(gs[1, 1])
    total  = len(df)
    n_days = daily["date"].nunique()
    avg    = total / n_days
    peak   = daily["count"].max()

    # mini donut: showing % of tweets in top-spike days (>2× avg)
    spike_days  = daily[daily["count"] > 2 * avg]["count"].sum()
    normal_days = total - spike_days

    wedges, texts, autotexts = ax3.pie(
        [spike_days, normal_days],
        labels=["Spike days\n(>2× avg)", "Normal days"],
        colors=[ACCENT3, ACCENT1],
        autopct="%1.1f%%",
        startangle=140,
        wedgeprops=dict(width=0.5, edgecolor=BG, linewidth=2),
        textprops=dict(color=TEXT, fontsize=8.5)
    )
    for at in autotexts:
        at.set_fontsize(9)
        at.set_color(BG)
        at.set_fontweight("bold")

    ax3.text(0, 0, f"{total/1e6:.2f}M\ntweets", ha="center", va="center",
             fontsize=11, fontweight="bold", color=TEXT)
    ax3.set_title("📊  Volume Distribution", pad=8)

    # stats annotation
    stats_txt = (
        f"Total tweets : {total:,}\n"
        f"Date range   : {n_days} days\n"
        f"Avg / day    : {avg:,.0f}\n"
        f"Peak / day   : {peak:,}\n"
        f"Unique users : {df['user_id'].nunique():,}"
    )
    ax3.text(0, -1.55, stats_txt, ha="center", va="center",
             fontsize=8, color=TEXT,
             bbox=dict(boxstyle="round,pad=0.5", fc="#1f2937", ec=GRID, alpha=0.9))

    # ── [1,2]  Top 15 users ───────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    y_pos = range(len(top_usr) - 1, -1, -1)
    bars  = ax4.barh(
        list(y_pos), top_usr["count"].values,
        color=[ACCENT4] * len(top_usr), edgecolor=BG, linewidth=0.5
    )
    # color gradient
    norm_vals = top_usr["count"].values / top_usr["count"].max()
    cmap = plt.cm.get_cmap("cool")
    for bar, nv in zip(bars, reversed(norm_vals)):
        bar.set_facecolor(cmap(nv))

    ax4.set_yticks(list(y_pos))
    ax4.set_yticklabels(
        [f"user_{uid}" for uid in top_usr["user_id"].values],
        fontsize=7.5
    )
    ax4.set_title("👥  Top 15 Most Active Users", pad=8)
    ax4.set_xlabel("Number of tweets")
    ax4.grid(True, axis="x", alpha=0.3)
    for bar in bars:
        w = bar.get_width()
        ax4.text(w + 10, bar.get_y() + bar.get_height()/2,
                 f"{int(w):,}", va="center", fontsize=7, color=TEXT)

    # ── [2,0:2]  Tweet length distribution ───────────────
    ax5 = fig.add_subplot(gs[2, 0:2])
    lens = df["text_len"].clip(upper=500)
    ax5.hist(lens, bins=80, color=ACCENT2, edgecolor=BG,
             linewidth=0.3, alpha=0.85)
    ax5.axvline(lens.median(), color=ACCENT3, linewidth=1.8,
                linestyle="--", label=f"Median: {lens.median():.0f} chars")
    ax5.axvline(lens.mean(), color=ACCENT5, linewidth=1.8,
                linestyle="-.", label=f"Mean: {lens.mean():.0f} chars")
    # Twitter 280-char limit line
    ax5.axvline(280, color=ACCENT4, linewidth=1.2,
                linestyle=":", alpha=0.7, label="280-char limit")
    ax5.set_title("📝  Tweet Text Length Distribution", pad=8)
    ax5.set_xlabel("Characters per tweet")
    ax5.set_ylabel("Number of tweets")
    ax5.legend(fontsize=9, framealpha=0.3)
    ax5.grid(True, axis="y", alpha=0.3)

    # ── [2,2]  Weekday bar ────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 2])
    wd_order   = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    wd_counts  = df["timestamp"].dt.day_name().value_counts().reindex(wd_order, fill_value=0)
    bar_colors = [ACCENT1 if d not in ("Saturday","Sunday") else ACCENT5 for d in wd_order]
    ax6.bar(range(7), wd_counts.values, color=bar_colors, edgecolor=BG, linewidth=0.5)
    ax6.set_xticks(range(7))
    ax6.set_xticklabels(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"], fontsize=9)
    ax6.set_title("📅  Tweets by Weekday", pad=8)
    ax6.set_ylabel("Number of tweets")
    ax6.grid(True, axis="y", alpha=0.3)
    for i, v in enumerate(wd_counts.values):
        ax6.text(i, v + max(wd_counts)*0.01, f"{v/1000:.0f}k",
                 ha="center", fontsize=8, color=TEXT)

    # ── Footer ───────────────────────────────────────────
    fig.text(
        0.5, 0.005,
        "Schema: tweet_id | user_id | text | timestamp | embeddings (null) | topic_label (null) | date   •   "
        "Partitioned by date (300 partitions, 2022-08-19 → 2023-06-14)   •   Total: 11,099,751 tweets",
        ha="center", fontsize=8, color=SUBTEXT
    )

    return fig


def main():
    print("="*60)
    print("  VISUALIZATION — tweets_final.parquet")
    print("="*60)

    df              = load_data()
    daily, top_usr, hm = compute_stats(df)

    print("🎨  Vẽ biểu đồ ...")
    fig = make_figure(df, daily, top_usr, hm)

    Path(OUTPUT_PNG).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"✓  Đã lưu → {OUTPUT_PNG}")
    print("="*60)


if __name__ == "__main__":
    main()