#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${DETECTORS_PYTHON:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python"
fi

SHARD_DIR=""
OUTPUT_ROOT=""
GPUS="2,3"
DETECTORS="openai_roberta,chatgpt_d,argugpt,radar,mage_d,detectllm_lrr,mfd"
SHARDS=16
SESSION_PREFIX="paper_w1_gpu"
LOG_LABEL="wave1"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --shard-dir) SHARD_DIR="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --detectors) DETECTORS="$2"; shift 2 ;;
    --shards) SHARDS="$2"; shift 2 ;;
    --session-prefix) SESSION_PREFIX="$2"; shift 2 ;;
    --log-label) LOG_LABEL="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$SHARD_DIR" || -z "$OUTPUT_ROOT" ]]; then
  echo "Usage: $0 --shard-dir <dir> --output-root <dir> [--gpus 2,3] [--detectors a,b] [--shards 16]" >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT"/{predictions_sharded,logs,status,task_scripts}

IFS=',' read -r -a GPU_LIST <<<"$GPUS"
IFS=',' read -r -a DETECTOR_LIST <<<"$DETECTORS"
if [[ ${#GPU_LIST[@]} -eq 0 ]]; then
  echo "At least one GPU is required." >&2
  exit 2
fi

for gpu in "${GPU_LIST[@]}"; do
  task_script="$OUTPUT_ROOT/task_scripts/gpu_${gpu}.sh"
  cat >"$task_script" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$ROOT"
export PYTHONPATH="$ROOT/src"
export CUDA_VISIBLE_DEVICES="$gpu"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/status"
EOF
  chmod +x "$task_script"
done

task_idx=0
for detector in "${DETECTOR_LIST[@]}"; do
  mkdir -p "$OUTPUT_ROOT/predictions_sharded/$detector"
  for shard_idx in $(seq 0 $((SHARDS - 1))); do
    shard_name="$(printf "main_eval.%03d.jsonl" "$shard_idx")"
    input="$SHARD_DIR/$shard_name"
    output="$OUTPUT_ROOT/predictions_sharded/$detector/$(printf "%s.s%03d.predictions.jsonl" "$detector" "$shard_idx")"
    done_file="$OUTPUT_ROOT/status/$(printf "%s.s%03d.done" "$detector" "$shard_idx")"
    log_file="$OUTPUT_ROOT/logs/$(printf "%s.s%03d.log" "$detector" "$shard_idx")"
    if [[ -f "$done_file" && -s "$output" ]]; then
      continue
    fi
    gpu="${GPU_LIST[$((task_idx % ${#GPU_LIST[@]}))]}"
    task_script="$OUTPUT_ROOT/task_scripts/gpu_${gpu}.sh"
    cat >>"$task_script" <<EOF
echo "[$LOG_LABEL] START detector=$detector shard=$shard_idx gpu=$gpu \$(date -Is)" | tee -a "$log_file"
"$PYTHON_BIN" -m detectors_bench.run_detector --detector "$detector" --input "$input" --output "$output" 2>&1 | tee -a "$log_file"
touch "$done_file"
echo "[$LOG_LABEL] DONE detector=$detector shard=$shard_idx gpu=$gpu \$(date -Is)" | tee -a "$log_file"
EOF
    task_idx=$((task_idx + 1))
  done
done

for gpu in "${GPU_LIST[@]}"; do
  session="${SESSION_PREFIX}${gpu}"
  task_script="$OUTPUT_ROOT/task_scripts/gpu_${gpu}.sh"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux session already exists: $session"
  else
    tmux new-session -d -s "$session" "bash '$task_script' 2>&1 | tee '$OUTPUT_ROOT/logs/${session}.log'"
    echo "Started tmux session $session with $task_script"
  fi
done

echo "Queued $task_idx shard jobs across GPUs: $GPUS"
