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
