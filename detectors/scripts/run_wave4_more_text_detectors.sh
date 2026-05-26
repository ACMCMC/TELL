#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

INPUT=""
OUTPUT_ROOT=""
GPUS="0,1"
SHARDS=16
BOOTSTRAP=0
POLL_SECONDS=300
SESSION_PREFIX="det_wave4"
DETECTORS="aigc_mpu_env3,meld,t5_sentinel,logrank_gpt2_medium,phd_roberta,binoculars"
DRY_RUN=0
ALLOW_FAILED=0

usage() {
  cat <<'USAGE'
Prepare or launch the added detector wave on one JSONL file.

This is a thin, restart-safe wrapper around run_full_suite_2gpu.sh for:
  aigc_mpu_env3, meld, t5_sentinel, logrank_gpt2_medium, phd_roberta, binoculars

Required:
  --input <jsonl>          Input examples. Required fields: id, text. For metrics, include label where 0=human, 1=AI.
  --output-root <dir>     Output directory for shards, logs, predictions, metrics, and manifests.

Common:
  --gpus 0,1              Exactly two CUDA device IDs. Default: 0,1.
  --shards 16             Deterministic runtime shards. Default: 16.
  --detectors a,b,c       Override detector list.
  --session-prefix name   tmux session prefix. Default: det_wave4.
  --bootstrap n           Bootstrap replicates during final metrics. Default: 0.
  --poll-seconds n        Merge watcher poll interval. Default: 300.
  --allow-failed          Let merge watcher continue despite failed shards. Not recommended for paper tables.
  --dry-run               Write scripts/manifests but do not start tmux sessions.

Environment:
  DETECTORS_PYTHON             Python env for the main harness.
  DETECTORS_BINOCULARS_PYTHON  Optional Python env for Binoculars/transformers==4.31.
  HF_TOKEN                     Required if any gated HF model is used by the selected detectors.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input) INPUT="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --shards) SHARDS="$2"; shift 2 ;;
    --detectors) DETECTORS="$2"; shift 2 ;;
    --session-prefix) SESSION_PREFIX="$2"; shift 2 ;;
    --bootstrap) BOOTSTRAP="$2"; shift 2 ;;
    --poll-seconds) POLL_SECONDS="$2"; shift 2 ;;
    --allow-failed) ALLOW_FAILED=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$INPUT" || -z "$OUTPUT_ROOT" ]]; then
  usage >&2
  exit 2
fi

args=(
  --input "$INPUT"
  --output-root "$OUTPUT_ROOT"
  --gpus "$GPUS"
  --shards "$SHARDS"
  --detectors "$DETECTORS"
  --session-prefix "$SESSION_PREFIX"
  --bootstrap "$BOOTSTRAP"
  --poll-seconds "$POLL_SECONDS"
)

if [[ "$ALLOW_FAILED" -eq 1 ]]; then
  args+=(--allow-failed)
fi
if [[ "$DRY_RUN" -eq 1 ]]; then
  args+=(--dry-run)
fi

exec bash "$ROOT/scripts/run_full_suite_2gpu.sh" "${args[@]}"
