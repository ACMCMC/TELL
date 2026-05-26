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
DETECTORS=""
SHARDS=16
EXPECTED_ROWS=0
POLL_SECONDS=300
BOOTSTRAP=0
SUMMARY_NAME="merge_summary.json"
ALLOW_FAILED=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --detectors) DETECTORS="$2"; shift 2 ;;
    --shards) SHARDS="$2"; shift 2 ;;
    --expected-rows) EXPECTED_ROWS="$2"; shift 2 ;;
    --poll-seconds) POLL_SECONDS="$2"; shift 2 ;;
    --bootstrap) BOOTSTRAP="$2"; shift 2 ;;
    --summary-name) SUMMARY_NAME="$2"; shift 2 ;;
    --allow-failed) ALLOW_FAILED=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$OUTPUT_ROOT" || -z "$DETECTORS" || "$EXPECTED_ROWS" -le 0 ]]; then
  cat >&2 <<'USAGE'
Usage:
  wait_merge_detector_suite.sh \
    --output-root <dir> \
    --detectors detector_a,detector_b \
    --shards <n> \
    --expected-rows <n>
USAGE
  exit 2
fi

cd "$ROOT"
export PYTHONPATH="$ROOT/src"
mkdir -p "$OUTPUT_ROOT/logs"
log_file="$OUTPUT_ROOT/logs/wait_merge_detector_suite.log"

IFS=',' read -r -a DETECTOR_LIST <<<"$DETECTORS"
expected_total=$((${#DETECTOR_LIST[@]} * SHARDS))

echo "[wait-merge-suite] START $(date -Is) detectors=$DETECTORS shards=$SHARDS expected_total=$expected_total" | tee -a "$log_file"
while true; do
  done_total=0
  failed=()
  missing=()
  for detector in "${DETECTOR_LIST[@]}"; do
    for shard_idx in $(seq 0 $((SHARDS - 1))); do
      key="$(printf "%s.s%03d" "$detector" "$shard_idx")"
      done_file="$OUTPUT_ROOT/status/$key.done"
      failed_file="$OUTPUT_ROOT/status/$key.failed"
      if [[ -f "$done_file" ]]; then
        done_total=$((done_total + 1))
      elif [[ -f "$failed_file" ]]; then
        failed+=("$key")
      else
        missing+=("$key")
      fi
    done
  done

  echo "[wait-merge-suite] $(date -Is) done=$done_total/$expected_total failed=${#failed[@]} missing=${#missing[@]}" | tee -a "$log_file"
  if [[ ${#failed[@]} -gt 0 && "$ALLOW_FAILED" -ne 1 ]]; then
    printf "[wait-merge-suite] FAILED shards: %s\n" "${failed[*]}" | tee -a "$log_file"
    exit 1
  fi
  if [[ "$done_total" -eq "$expected_total" ]]; then
    break
  fi
  sleep "$POLL_SECONDS"
done

echo "[wait-merge-suite] MERGE $(date -Is)" | tee -a "$log_file"
"$PYTHON_BIN" scripts/merge_wave1_fast.py \
  --output-root "$OUTPUT_ROOT" \
  --detectors "$DETECTORS" \
  --shards "$SHARDS" \
  --expected-rows "$EXPECTED_ROWS" \
  --summary-name "$SUMMARY_NAME" \
  --bootstrap "$BOOTSTRAP" 2>&1 | tee -a "$log_file"

"$PYTHON_BIN" -m detectors_bench.summarize_results \
  --metrics-dir "$OUTPUT_ROOT/metrics" \
  --output "$OUTPUT_ROOT/summary.tsv" 2>&1 | tee -a "$log_file"
echo "[wait-merge-suite] DONE $(date -Is) summary=$OUTPUT_ROOT/summary.tsv" | tee -a "$log_file"
