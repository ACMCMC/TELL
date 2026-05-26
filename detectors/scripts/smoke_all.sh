#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MODE="${2:-quick}"
ROOT="$(cd "$ROOT" && pwd)"
cd "$ROOT"
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false

if [[ -n "${DETECTORS_PYTHON:-}" ]]; then
  PYTHON_BIN="$DETECTORS_PYTHON"
elif [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT/.venv/bin/python"
else
  PYTHON_BIN="python"
fi

mkdir -p results/smoke logs

QUICK_DETECTORS=(
  openai_roberta
  chatgpt_d
  argugpt
  radar
  mage_d
  detectllm_lrr
)

ALL_DETECTORS=(
  binoculars
  detectgpt
  fast_detectgpt
  mage_d
  openai_roberta
  argugpt
  detectllm_lrr
  detectllm_npr
  radar
  pangram_editlens_llama
  dnagpt
  mfd
  chatgpt_d
  aigc_mpu_env3
  meld
  t5_sentinel
  logrank_gpt2_medium
  phd_roberta
)

WAVE4_DETECTORS=(
  aigc_mpu_env3
  meld
  t5_sentinel
  logrank_gpt2_medium
  phd_roberta
  binoculars
)

if [[ "$MODE" == "all" ]]; then
  DETECTORS=("${ALL_DETECTORS[@]}")
elif [[ "$MODE" == "wave4" ]]; then
  DETECTORS=("${WAVE4_DETECTORS[@]}")
else
  DETECTORS=("${QUICK_DETECTORS[@]}")
fi

for detector in "${DETECTORS[@]}"; do
  echo "[smoke] $detector"
  DETECTOR_PYTHON="$PYTHON_BIN"
  INPUT="smoke/smoke.jsonl"
  if [[ "$detector" == "binoculars" ]]; then
    DETECTOR_PYTHON="${DETECTORS_BINOCULARS_PYTHON:-$PYTHON_BIN}"
    INPUT="smoke/smoke_long.jsonl"
  elif [[ "$detector" == "phd_roberta" ]]; then
    INPUT="smoke/smoke_long.jsonl"
  elif [[ "$detector" == "ghostbuster" ]]; then
    DETECTOR_PYTHON="${DETECTORS_GHOSTBUSTER_PYTHON:-$PYTHON_BIN}"
  fi

  PYTHONPATH="$ROOT/src" "$DETECTOR_PYTHON" -m detectors_bench.run_detector \
    --detector "$detector" \
    --input "$INPUT" \
    --output "results/smoke/${detector}.predictions.jsonl" \
    --allow-init-error \
    --limit 4 \
    >"logs/${detector}.smoke.log" 2>&1 || {
      echo "[smoke] FAILED $detector; see logs/${detector}.smoke.log"
      continue
    }
  PYTHONPATH="$ROOT/src" "$PYTHON_BIN" -m detectors_bench.run_benchmark \
    --predictions "results/smoke/${detector}.predictions.jsonl" \
    --output "results/smoke/${detector}.metrics.json" \
    --bootstrap 0 \
    >>"logs/${detector}.smoke.log" 2>&1 || {
      echo "[metrics] FAILED $detector; see logs/${detector}.smoke.log"
      continue
    }
done

PYTHONPATH="$ROOT/src" "$PYTHON_BIN" -m detectors_bench.summarize_results \
  --metrics-dir results/smoke \
  --output results/smoke/summary.tsv

echo "[smoke] wrote results/smoke/summary.tsv"
