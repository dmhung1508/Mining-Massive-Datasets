#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-C:/Users/HungDM/AppData/Local/Python/pythoncore-3.14-64/python.exe}"
TELEGRAM_OUTPUT="${TELEGRAM_OUTPUT:-jupyter/output/telegram_messages.parquet}"
COMBINED_OUTPUT="${COMBINED_OUTPUT:-jupyter/output/combined_social.parquet}"
ARTIFACT_DIR="${ARTIFACT_DIR:-jupyter/output/lsh_combined}"
BASELINE_SIZE="${BASELINE_SIZE:-3000}"
SCALE_SIZE="${SCALE_SIZE:-50000}"
SKIP_MONGO_EXPORT="${SKIP_MONGO_EXPORT:-0}"
SKIP_MERGE="${SKIP_MERGE:-0}"
REBUILD_SEARCH_INDEX="${REBUILD_SEARCH_INDEX:-0}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

run_step() {
  local name="$1"
  shift
  printf '\n==> %s\n' "$name"
  "$PYTHON_BIN" "$@"
}

if [[ "$SKIP_MONGO_EXPORT" != "1" ]]; then
  run_step "Export Telegram Mongo -> Parquet" \
    scripts/export_telegram_dataset.py \
    --source mongo \
    --output "$TELEGRAM_OUTPUT" \
    --overwrite
fi

if [[ "$SKIP_MERGE" != "1" ]]; then
  run_step "Merge Twitter + Telegram -> Combined Parquet" \
    scripts/build_combined_dataset.py \
    --telegram-source mongo \
    --output "$COMBINED_OUTPUT" \
    --overwrite
fi

run_step "Extract deterministic subsets" \
  scripts/extract_subsets.py \
  --input "$COMBINED_OUTPUT" \
  --artifact-dir "$ARTIFACT_DIR" \
  --baseline-size "$BASELINE_SIZE" \
  --scale-size "$SCALE_SIZE"

run_step "Build shingles" \
  scripts/build_shingles.py \
  --artifact-dir "$ARTIFACT_DIR"

run_step "Run exact Jaccard baseline" \
  scripts/run_baseline.py \
  --artifact-dir "$ARTIFACT_DIR"

run_step "Run MinHash + LSH" \
  scripts/run_lsh.py \
  --artifact-dir "$ARTIFACT_DIR"

run_step "Verify candidates and build clusters" \
  scripts/verify_and_cluster.py \
  --artifact-dir "$ARTIFACT_DIR"

SEARCH_ARGS=(
  scripts/search_similar.py
  --artifact-dir "$ARTIFACT_DIR"
  --text "Russia Ukraine war update"
  --top-k 5
)

if [[ "$REBUILD_SEARCH_INDEX" == "1" ]]; then
  SEARCH_ARGS+=(--rebuild-index)
fi

run_step "Build/check search index" "${SEARCH_ARGS[@]}"

printf '\nDone. Combined LSH artifacts:\n  %s\n' "$ARTIFACT_DIR"
printf '\nOpen metrics:\n  cat %s/metrics.json\n' "$ARTIFACT_DIR"
printf '\nServe API:\n  "%s" scripts/serve_api.py --artifact-dir %s --port 8765\n' "$PYTHON_BIN" "$ARTIFACT_DIR"
