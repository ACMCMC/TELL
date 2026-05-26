#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 <session> <detector> <input.jsonl> <output-dir> [cuda_devices]" >&2
  exit 2
fi

SESSION="$1"
DETECTOR="$2"
INPUT="$3"
OUTDIR="$4"
CUDA_DEVICES="${5:-0}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${DETECTORS_PYTHON:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python"
fi

mkdir -p "$ROOT/$OUTDIR" "$ROOT/logs"

tmux new-session -d -s "$SESSION" \
  "cd '$ROOT' && export CUDA_VISIBLE_DEVICES='$CUDA_DEVICES' && export PYTHONPATH='$ROOT/src' && '$PYTHON_BIN' -m detectors_bench.run_detector --detector '$DETECTOR' --input '$INPUT' --output '$OUTDIR/${DETECTOR}.predictions.jsonl' 2>&1 | tee 'logs/${SESSION}.log'; '$PYTHON_BIN' -m detectors_bench.run_benchmark --predictions '$OUTDIR/${DETECTOR}.predictions.jsonl' --output '$OUTDIR/${DETECTOR}.metrics.json' 2>&1 | tee -a 'logs/${SESSION}.log'"

echo "Started tmux session $SESSION for $DETECTOR on CUDA_VISIBLE_DEVICES=$CUDA_DEVICES"
