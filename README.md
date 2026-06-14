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

## Key defaults (`src/social_lsh/constants.py`)

- Output root: `jupyter/output/`
- Artifact dir: `jupyter/output/lsh_combined`
- Near-duplicate threshold: Jaccard ≥ 0.8
- Shingle size k=3, num_perm=128, tuning grid in `CONFIG_GRID`

## Tests

```bash
pytest
```
