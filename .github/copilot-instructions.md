# Project Guidelines — Social LSH (Near-Duplicate Detection)

## What this project does

Detects near-duplicate posts and narrative clusters across war-related social media
(Twitter Ukraine-Russia dataset + Telegram messages) using **MinHash + LSH**. The pipeline
mirrors the 10-step diagram in the project plan: collect → canonical schema → text
preprocessing → shingling → MinHash → LSH banding → exact Jaccard verify → clustering →
API → dashboard.

## Layout

```
src/social_lsh/          # core library (importable package)
  constants.py           # paths, seeds, thresholds, LSHConfig, CONFIG_GRID
  datasets.py            # canonical schema, Twitter/Telegram normalisation, merge
  preprocessing.py       # normalize_text, tokenize, word shingles
  similarity.py          # exact Jaccard baseline + candidate verification
  minhash.py             # MinHash signatures, LSH candidate generation, eval
  clustering.py          # Union-Find connected components
  pipeline.py            # orchestrates extract -> shingles -> baseline -> lsh -> verify
  search.py              # LSH bucket index + similar-post search
  artifacts.py           # parquet/JSON artifact IO + metrics merge

scripts/                 # CLI entry points, grouped by function
  data/                  # build_combined_dataset, export_telegram_dataset
  pipeline/              # extract_subsets, build_shingles, run_baseline, run_lsh, verify_and_cluster
  search/                # search_similar
  api/                   # app.py (FastAPI), serve_api.py (uvicorn launcher)
  visualization/         # per-stage matplotlib dashboards
  reporting/             # benchmark, demo cases, deliverables, telegram topic summary
  run_full_combined.sh / .ps1   # end-to-end runner

telegram_crawler/        # Telethon crawler (config.py, crawler.py, utils.py)
streamlit_app.py         # dashboard
tests/                   # pytest suite
```

## Install and test

```bash
pip install -e ".[dev]"          # core + pytest
pip install -e ".[api,dashboard,telegram]"   # optional extras as needed
pytest                            # runs the full suite (src/ is on pythonpath via pyproject)
```

## Conventions

- The package is pip-installed editable; scripts import `social_lsh` directly. Do **not**
  re-add a `sys.path` bootstrap shim.
- All generated files live under `jupyter/output/`. Key locations come from
  `social_lsh.constants`: `DEFAULT_ARTIFACT_DIR` = `jupyter/output/lsh_combined`,
  `DEFAULT_VISUALS_DIR` = `jupyter/output/visuals`. Avoid hardcoding absolute paths.
- Default near-duplicate threshold is Jaccard ≥ 0.8 (`DEFAULT_VERIFY_THRESHOLD`).
- Default shingle size k=3, num_perm=128; tuning grid is `CONFIG_GRID`.
- Two subset sizes: `BASELINE_SIZE` (brute-force ground truth) and `SCALE_SIZE` (LSH run).
- Artifacts are parquet; metrics accumulate in `metrics.json` via `artifacts.merge_metrics`.

## Data

- Twitter source: `tweets_final.parquet` (date-partitioned, hive layout) at repo root.
- Canonical columns: `tweet_id, user_id, text, timestamp, date` plus source metadata
  (`source`, `source_item_id`, `topic_label`, ...). See `datasets.CANONICAL_COLUMNS`.
- Prefer pyarrow dataset streaming for large scans; Spark is used opportunistically in
  `pipeline._load_with_spark` with a pyarrow fallback.

## Pipeline order

1. `scripts/data/build_combined_dataset.py` — merge Twitter + Telegram into canonical parquet
2. `scripts/pipeline/extract_subsets.py` — deterministic baseline + scale subsets
3. `scripts/pipeline/build_shingles.py` — tokenize + word shingles
4. `scripts/pipeline/run_baseline.py` — exact Jaccard ground truth (baseline subset)
5. `scripts/pipeline/run_lsh.py` — tune CONFIG_GRID, generate scale-set candidates
6. `scripts/pipeline/verify_and_cluster.py` — verify pairs, Union-Find clusters
7. `scripts/api/serve_api.py` or `streamlit_app.py` — serve results

Or run everything: `scripts/run_full_combined.sh`.
