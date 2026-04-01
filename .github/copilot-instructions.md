# Project Guidelines — Ukraine Conflict Twitter NLP

## Architecture

Research project analysing ~290 daily Twitter CSV files (Aug 2022 – mid 2023, ~13.75 GB total) on the Ukraine-Russia conflict. The goal is intelligent tweet & user recommendation using the pipeline described in `Intelligent Post and User Recommendation from War-Related Discourse.pdf`.

**Two notebooks:**
- `Data/code/pipeline.ipynb` — full ML pipeline (primary)
- `Data/mine_tweets.ipynb` — exploratory data mining

**Pipeline stages (pipeline.ipynb):**
1. **Preprocessing** — Polars, English-only filter, strip trailing hashtags, cross-file dedup by `tweetid`
2. **Doc2Vec** (gensim) — 100-dim tweet embeddings, saved to `output/doc2vec.model`
3. **UMAP** (umap-learn) — 100D → 10D for clustering, 100D → 2D for visualisation
4. **DBSCAN** (scikit-learn) — density-based clustering, cosine metric
5. **Zero-shot topic labeling** — `facebook/bart-large-mnli` via HuggingFace `transformers`, top-50 words/cluster as input
6. **Personalized PageRank** (networkx) — k-NN similarity graph + PPR for tweet recommendation

## Data

- Location: `F:\UK-Russia\Data\` (glob: `*_UkraineCombinedTweetsDeduped.csv`)
- Filename formats: `MMDD_...csv` (2022) and `YYYYMMDD_...csv` (2023)
- Key columns used: `tweetid`, `text`, `language`, `username`, `retweetcount`, `is_retweet`, `tweetcreatedts`, `hashtags`, `followers`
- Always read with `ignore_errors=True, low_memory=True` to handle malformed rows
- Prefer Polars (`pl.scan_csv` with `streaming=True`) over pandas for large scans

## Build and Test

```powershell
# Activate the project venv
& "F:\UK-Russia\Data\.venv\Scripts\Activate.ps1"

# Install all dependencies (run inside notebook cell 1, or manually):
pip install polars pyarrow gensim umap-learn scikit-learn transformers torch networkx tqdm matplotlib seaborn
```

Outputs land in `F:\UK-Russia\Data\code\output\`:
- `tweets_clean.parquet` — preprocessed data
- `doc2vec.model` — trained Doc2Vec
- `umap_clusters.png`, `topic_distribution.png` — visualisations

## Project Conventions

- **Test on a subset first**: set `MAX_FILES = 5` and `MAX_SAMPLE = 10_000` before running full dataset
- Key tunable constants are declared in the Config cell (cell 2 of `pipeline.ipynb`): `MAX_FILES`, `CHUNK_SIZE`, `MAX_SAMPLE`, `DBSCAN_EPS`, `DBSCAN_MIN_SAMP`, `UMAP_N_NEIGHBORS`, `UMAP_N_COMPONENTS`, `KNN_K`, `PPR_ALPHA`
- UMAP is reused at inference time (`reducer.transform()`), so the reducer must stay in scope for `recommend_ppr()`
- Cross-file deduplication uses an in-memory `seen_ids: set` — do not reset between files
- The k-NN graph is built in batches of 2000 nodes to avoid OOM on the similarity matrix

## Integration Points

- `bart-large-mnli` (~1.6 GB) is downloaded on first run from HuggingFace; set `device=0` if a CUDA GPU is available (default is CPU `device=-1`)
- Doc2Vec inference for new queries uses `model_d2v.infer_vector(tokens, epochs=30)` then `reducer.transform()` for UMAP projection
