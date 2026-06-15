# Social LSH — Near-Duplicate Detection on War-Related Social Media

Detect near-duplicate posts and narrative clusters across the Ukraine-Russia Twitter
dataset and Telegram messages using **MinHash + LSH**. The pipeline follows a 10-step flow:
collect → canonical schema → text preprocessing → shingling → MinHash → LSH banding →
exact Jaccard verification → clustering → API → dashboard.

## Project structure

```
src/social_lsh/        Core library (pip-installable package)
scripts/
  data/                Build combined dataset, export Telegram
  pipeline/            extract_subsets, build_shingles, run_baseline, run_lsh, verify_and_cluster
  search/              Similar-post search CLI
  api/                 FastAPI backend (app.py) + uvicorn launcher (serve_api.py)
  visualization/       Per-stage matplotlib dashboards
  reporting/           Benchmarks, demo cases, deliverables
  run_full_combined.sh End-to-end runner (.ps1 for Windows)
telegram_crawler/      Telethon crawler (config.py, crawler.py, utils.py)
dashboard/             Streamlit dashboard
tests/                 pytest suite
docs/                  Deliverables (hung/, bao/) + telegram topic summary
jupyter/               Notebooks + generated output/ (gitignored)
```

## Setup

```bash
pip install -e ".[dev]"                          # core + tests
pip install -e ".[api,dashboard,telegram]"       # optional extras
```

## Run the pipeline

End-to-end:

```bash
bash scripts/run_full_combined.sh
```

Or step by step (artifacts land in `jupyter/output/lsh_combined/`):

```bash
python scripts/data/build_combined_dataset.py --overwrite
python scripts/pipeline/extract_subsets.py
python scripts/pipeline/build_shingles.py
python scripts/pipeline/run_baseline.py
python scripts/pipeline/run_lsh.py
python scripts/pipeline/verify_and_cluster.py
```

## Incremental Telegram/X updates

Export fresh Mongo data and rebuild the canonical combined dataset:

```powershell
.\.venv-search\Scripts\python.exe scripts\data\export_telegram_dataset.py --source mongo --overwrite
.\.venv-search\Scripts\python.exe scripts\data\export_x_dataset.py --overwrite
.\.venv-search\Scripts\python.exe scripts\data\build_combined_dataset.py --telegram-source mongo --overwrite
```

Stop Streamlit/API, then add only unseen documents to the full search index:

```powershell
.\.venv-search\Scripts\python.exe scripts\pipeline\update_full_search_index.py `
  --input jupyter/output/combined_social.parquet `
  --artifact-dir jupyter/output/lsh_full `
  --batch-size 5000
```

The updater is idempotent. It records short rejected posts, attaches exact/near
duplicates to existing clusters, creates clusters for new content, and merges
clusters when a new post connects them. It also refreshes the hourly cluster
timeline and trending ranking.

Refresh trending without importing new posts:

```powershell
.\.venv-search\Scripts\python.exe scripts\reporting\refresh_trending.py `
  --artifact-dir jupyter/output/lsh_full `
  --lookback-days 7 `
  --recent-hours 24
```

Trending uses the latest timestamp in the dataset as its reference time. Its
score combines recent post volume, growth, source/author diversity, recency,
and a penalty when one author dominates a cluster.

Generate a 10-item video-news brief from the top trending clusters:

```powershell
.\.venv-search\Scripts\python.exe scripts\reporting\generate_daily_news_brief.py `
  --artifact-dir jupyter/output/lsh_full `
  --output-dir jupyter/output/news_brief `
  --limit 10 `
  --min-posts 2
```

Outputs:

- `jupyter/output/news_brief/daily_news_brief.json`
- `jupyter/output/news_brief/daily_news_brief.md`

The brief is extractive by default: it selects representative posts, metadata,
timeline, and a voice-over draft for each trending cluster. Add an LLM step
after this stage if you want more polished wording.

## Serve results

FastAPI backend:

```bash
python scripts/api/serve_api.py --port 8765
# GET /health  /metrics  /clusters/top?limit=10  /search?text=...&top_k=5&min_jaccard=0.8
# Interactive docs at http://127.0.0.1:8765/docs
```

Streamlit dashboard:

```bash
streamlit run dashboard/streamlit_app.py
```

## Latest clusters (realtime refresh)

The full historical corpus is clustered offline into `jupyter/output/lsh_full`
(11M+ posts, DuckDB-backed). For fresh "what's trending now" clusters, do **not**
re-cluster everything. Instead pull a recent time window from every live source
(Telegram realtime + the X collections `x_russia_ukraina_posts` /
`x_us_iran_posts` in the `news_monitoring` MongoDB) and run the lightweight LSH
pipeline on just that window:

```bash
python scripts/pipeline/refresh_latest_clusters.py --since-days 2 --scale-size 50000
# -> writes clusters to jupyter/output/lsh_latest
```

Then serve the broadcast from the fresh window:

```bash
python scripts/api/serve_api.py --artifact-dir jupyter/output/lsh_latest
```

To serve from the **full historical corpus** instead (the big offline run):

```bash
python scripts/api/serve_api.py --artifact-dir jupyter/output/lsh_full
```

The broadcast reads columns selectively, so it works against the multi-GB
`lsh_full` artifacts (top clusters returned in ~15s).

Run the refresh on a schedule (cron / systemd timer) to keep clusters current.

Before the API prepares a broadcast it runs a local preflight check (artifact
size, parquet rows, available RAM/disk, and cache status). Large artifacts such
as `lsh_full` are read in streaming mode for broadcast prep, but the first cache
miss can still take time because it scans the parquet row groups. Use
`lsh_combined` for a quick demo, or pre-generate once while the machine is idle:

```bash
python scripts/media/pregen_broadcast.py --artifact-dir jupyter/output/lsh_combined --top-n 5
```

Broadcast scripts and image references are persisted in
`jupyter/output/news/metadata/`; generated images are stored under
`jupyter/output/news/images/<cache-id>/`. If the source parquet artifacts change,
the metadata cache is automatically invalidated and rebuilt.

## Generate news media (images / videos)

After the pipeline produces clusters, turn the top narrative clusters into
illustrative media. Step 1 summarises each cluster into a structured news object
(uses Grok from `.env`, falls back to a template offline) and builds both an
image prompt and a video prompt. Step 2 sends prompts to the YEScale API.

```bash
# 1) clusters -> news objects + English image/video prompts
python scripts/media/build_news_objects.py --top-n 5 --language Vietnamese

# 2a) news objects -> illustrative images (gpt-image, works today)
python scripts/media/generate_images.py --size 1024x1024 --quality low

# 2b) news objects -> videos (veo3.1; needs the provider to be available)
python scripts/media/generate_videos.py --size 720p --aspect-ratio 16:9
```

Needs `API_VEO` in `.env` (shared by both image and video models). Outputs land
in `jupyter/output/news/` (`news_objects.json`, `images/`, `videos/`, `manifest.json`).
Set `--no-llm` on step 1 to skip Grok and use the deterministic template.

## Key defaults (`src/social_lsh/constants.py`)

- Output root: `jupyter/output/`
- Artifact dir: `jupyter/output/lsh_combined`
- Near-duplicate threshold: Jaccard ≥ 0.8
- Shingle size k=3, num_perm=128, tuning grid in `CONFIG_GRID`

## Tests

```bash
pytest
```
