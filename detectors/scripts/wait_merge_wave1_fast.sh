#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${DETECTORS_PYTHON:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  else
    echo "No Python interpreter found. Set DETECTORS_PYTHON to the benchmark environment Python." >&2
    exit 2
  fi
fi

OUTPUT_ROOT=""
DETECTORS="openai_roberta,chatgpt_d,argugpt,radar,mage_d,detectllm_lrr,mfd"
SHARDS=16
EXPECTED_ROWS=200000
POLL_SECONDS=300
BOOTSTRAP=0
SUMMARY_NAME="wave1_merge_summary.json"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --detectors) DETECTORS="$2"; shift 2 ;;
    --shards) SHARDS="$2"; shift 2 ;;
    --expected-rows) EXPECTED_ROWS="$2"; shift 2 ;;
    --poll-seconds) POLL_SECONDS="$2"; shift 2 ;;
    --bootstrap) BOOTSTRAP="$2"; shift 2 ;;
    --summary-name) SUMMARY_NAME="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$OUTPUT_ROOT" ]]; then
  echo "Usage: $0 --output-root <dir> [--detectors a,b] [--shards 16] [--expected-rows 200000]" >&2
  exit 2
fi

cd "$ROOT"
export PYTHONPATH="$ROOT/src"
mkdir -p "$OUTPUT_ROOT/logs"
log_file="$OUTPUT_ROOT/logs/wait_merge_wave1_fast.log"

IFS=',' read -r -a DETECTOR_LIST <<<"$DETECTORS"
expected_total=$((${#DETECTOR_LIST[@]} * SHARDS))

echo "[wait-merge] START $(date -Is) detectors=$DETECTORS shards=$SHARDS expected_total=$expected_total" | tee -a "$log_file"
while true; do
  done_total=0
  missing=()
  for detector in "${DETECTOR_LIST[@]}"; do
    for shard_idx in $(seq 0 $((SHARDS - 1))); do
      done_file="$OUTPUT_ROOT/status/$(printf "%s.s%03d.done" "$detector" "$shard_idx")"
      if [[ -f "$done_file" ]]; then
        done_total=$((done_total + 1))
      else
        missing+=("$(printf "%s.s%03d" "$detector" "$shard_idx")")
      fi
    done
  done

  echo "[wait-merge] $(date -Is) done=$done_total/$expected_total missing=${#missing[@]}" | tee -a "$log_file"
  if [[ "$done_total" -eq "$expected_total" ]]; then
    break
  fi
  sleep "$POLL_SECONDS"
done

echo "[wait-merge] MERGE $(date -Is)" | tee -a "$log_file"
"$PYTHON_BIN" scripts/merge_wave1_fast.py \
  --output-root "$OUTPUT_ROOT" \
  --detectors "$DETECTORS" \
  --shards "$SHARDS" \
  --expected-rows "$EXPECTED_ROWS" \
  --summary-name "$SUMMARY_NAME" \
  --bootstrap "$BOOTSTRAP" 2>&1 | tee -a "$log_file"
echo "[wait-merge] DONE $(date -Is)" | tee -a "$log_file"
