#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$(pwd)}"
DETECTORS_ROOT="$ROOT/detectors"

cd "$ROOT"

echo "[setup] host=$(hostname)"
echo "[setup] root=$ROOT"
df -h "$ROOT" /data || true
nvidia-smi -L || true

git submodule update --init \
  detectors/vendor/binoculars \
  detectors/vendor/detectgpt \
  detectors/vendor/fast_detectgpt \
  detectors/vendor/mage \
  detectors/vendor/argugpt \
  detectors/vendor/detectllm \
  detectors/vendor/radar \
  detectors/vendor/dnagpt \
  detectors/vendor/chatgpt_detection \
  detectors/vendor/ghostbuster \
  detectors/vendor/imgtb \
  detectors/vendor/gptid \
  detectors/vendor/t5_sentinel \
  detectors/vendor/aigc_mpu \
  detectors/vendor/llm_detector_eval

cd "$DETECTORS_ROOT"
if [[ -n "${DETECTORS_PYTHON:-}" ]]; then
  PYTHON_BIN="$DETECTORS_PYTHON"
  echo "[setup] using DETECTORS_PYTHON=$PYTHON_BIN"
elif python3 -m venv .venv; then
  PYTHON_BIN="$DETECTORS_ROOT/.venv/bin/python"
else
  cat >&2 <<'MSG'
[setup] python3-venv is unavailable. Install python3-venv or rerun with:
  DETECTORS_PYTHON=/path/to/existing/python bash detectors/scripts/setup_server.sh <repo-root>
MSG
  exit 1
fi

"$PYTHON_BIN" -m pip install --upgrade pip
"$PYTHON_BIN" -m pip install -e ".[hf,dev]"

mkdir -p cache/huggingface results/smoke results/models logs

cat <<'MSG'
[setup] Base harness installed.
[setup] Heavy official detector dependencies are intentionally installed per-detector:
  python -m venv /path/to/binoculars431 && /path/to/binoculars431/bin/pip install 'transformers[torch]==4.31.0' torch accelerate sentencepiece numpy scikit-learn pandas pyyaml joblib scipy
  pip install -r vendor/mage/requirements.txt
  pip install -r vendor/fast_detectgpt/requirements.txt
  pip install -r vendor/detectllm/requirements.txt
  pip install -r vendor/dnagpt/requirements.txt
  pip install -r vendor/ghostbuster/requirements.txt
  pip install -e ".[pangram]"

[setup] Added detector wave dependencies:
  - AIGC MPU: included in the main HF environment via yuchuantian/AIGC_detector_env3.
  - MELD: included in the main HF environment; downloads anon-review-meld-2026/meld and jhu-clsp/ettin-encoder-400m on first run.
  - T5Sentinel: included in the main HF environment; wrapper downloads T5Sentinel.0613.pt into cache/t5_sentinel on first run.
  - PHD and LogRank: included in the main HF environment; PHD uses vendor/gptid and RoBERTa-base embeddings.

Wave-4 setup smoke:
  cd detectors
  DETECTORS_PYTHON=/path/to/detectors/python bash scripts/smoke_all.sh . wave4
MSG
