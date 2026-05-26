#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CALLER_CWD="$(pwd)"
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

INPUT=""
OUTPUT_ROOT=""
GPUS="0,1"
SHARDS=16
BOOTSTRAP=0
POLL_SECONDS=300
SESSION_PREFIX="det_suite"
MFD_TRAIN=""
MFD_MODEL="$ROOT/results/models/mfd.joblib"
MFD_MODEL_USER_SET=0
INCLUDE_DETECTGPT=0
ALLOW_FAILED=0
DRY_RUN=0

# MFD is off by default: it needs a logistic fit on labeled non-test data (--mfd-train) or an existing
# results/models/mfd.joblib. To run it, append mfd to --detectors (or pass a full list that includes mfd).
DEFAULT_DETECTORS="binoculars,openai_roberta,chatgpt_d,argugpt,radar,mage_d,detectllm_lrr,fast_detectgpt,pangram_editlens_llama,detectllm_npr,dnagpt"
DETECTORS="$DEFAULT_DETECTORS"

usage() {
  cat <<'USAGE'
Run the external detector benchmark suite on one JSONL file using two GPUs.

Required:
  --input <jsonl>          Input examples. Required fields: id, text. For metrics, include label where 0=human, 1=AI.
  --output-root <dir>     Output directory for shards, logs, predictions, metrics, and manifests.

Common:
  --gpus 0,1              Two CUDA device IDs. Default: 0,1.
  --shards 16             Number of deterministic runtime shards. Default: 16.
  --mfd-train <jsonl>     Non-test labeled data used to fit MFD if results/models/mfd.joblib is missing.
  --mfd-model <path>      Optional MFD model path; symlinked into the registry's default model path.
  --detectors a,b,c       Override detector list (default omits mfd; add mfd here to enable it).
  --include-detectgpt     Append DetectGPT. Off by default because it was impractically slow in paper runs.
  --session-prefix name   tmux session prefix. Default: det_suite.
  --bootstrap n           Bootstrap replicates during final metrics. Default: 0.
  --poll-seconds n        Merge watcher poll interval. Default: 300.
  --allow-failed          Let merge watcher continue despite failed shards. Not recommended for paper tables.
  --dry-run               Write scripts/manifests but do not start tmux sessions.

Environment:
  DETECTORS_PYTHON             Python env for the main harness.
  DETECTORS_BINOCULARS_PYTHON  Optional Python env for Binoculars/transformers==4.31.
  HF_TOKEN                     Required for gated HF detectors such as Pangram EditLens.

Outputs:
  <output-root>/summary.tsv
  <output-root>/metrics/*.metrics.json
  <output-root>/merged_predictions/*.predictions.jsonl
  <output-root>/logs/
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input) INPUT="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --shards) SHARDS="$2"; shift 2 ;;
    --detectors) DETECTORS="$2"; shift 2 ;;
    --mfd-train) MFD_TRAIN="$2"; shift 2 ;;
    --mfd-model) MFD_MODEL="$2"; MFD_MODEL_USER_SET=1; shift 2 ;;
    --include-detectgpt) INCLUDE_DETECTGPT=1; shift ;;
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
if [[ ! -f "$INPUT" ]]; then
  echo "Input JSONL not found: $INPUT" >&2
  exit 2
fi
if [[ "$SHARDS" -le 0 ]]; then
  echo "--shards must be positive." >&2
  exit 2
fi

IFS=',' read -r -a GPU_LIST <<<"$GPUS"
if [[ ${#GPU_LIST[@]} -ne 2 ]]; then
  echo "This runner expects exactly two GPUs, for example --gpus 0,1." >&2
  exit 2
fi
for gpu in "${GPU_LIST[@]}"; do
  if [[ -z "$gpu" ]]; then
    echo "Invalid --gpus value: $GPUS" >&2
    exit 2
  fi
done

if [[ "$INCLUDE_DETECTGPT" -eq 1 && ",$DETECTORS," != *",detectgpt,"* ]]; then
  DETECTORS="$DETECTORS,detectgpt"
fi

mkdir -p "$OUTPUT_ROOT"/{input_shards,predictions_sharded,merged_predictions,metrics,manifests,logs,status,claims,task_scripts}
cd "$ROOT"
export PYTHONPATH="$ROOT/src"

INPUT_ABS="$("$PYTHON_BIN" - <<'PY' "$CALLER_CWD" "$INPUT"
import sys
from pathlib import Path
base = Path(sys.argv[1])
path = Path(sys.argv[2]).expanduser()
if not path.is_absolute():
    path = base / path
print(path.resolve())
PY
)"
OUTPUT_ABS="$("$PYTHON_BIN" - <<'PY' "$CALLER_CWD" "$OUTPUT_ROOT"
import sys
from pathlib import Path
base = Path(sys.argv[1])
path = Path(sys.argv[2]).expanduser()
if not path.is_absolute():
    path = base / path
print(path.resolve())
PY
)"
MFD_MODEL_ABS="$("$PYTHON_BIN" - <<'PY' "$CALLER_CWD" "$MFD_MODEL"
import sys
from pathlib import Path
base = Path(sys.argv[1])
path = Path(sys.argv[2]).expanduser()
if not path.is_absolute():
    path = base / path
print(path.resolve())
PY
)"
MFD_CONFIG_MODEL_ABS="$("$PYTHON_BIN" - <<'PY' "$ROOT/results/models/mfd.joblib"
import sys
from pathlib import Path
print(Path(sys.argv[1]).resolve())
PY
)"
MFD_TRAIN_ABS=""
if [[ -n "$MFD_TRAIN" ]]; then
  MFD_TRAIN_ABS="$("$PYTHON_BIN" - <<'PY' "$CALLER_CWD" "$MFD_TRAIN"
import sys
from pathlib import Path
base = Path(sys.argv[1])
path = Path(sys.argv[2]).expanduser()
if not path.is_absolute():
    path = base / path
print(path.resolve())
PY
)"
fi

echo "[suite] root=$ROOT"
echo "[suite] input=$INPUT_ABS"
echo "[suite] output_root=$OUTPUT_ABS"
echo "[suite] detectors=$DETECTORS"
echo "[suite] gpus=$GPUS shards=$SHARDS"

if [[ ",$DETECTORS," == *",mfd,"* ]]; then
  if [[ ! -f "$MFD_MODEL_ABS" ]]; then
    if [[ -z "$MFD_TRAIN" ]]; then
      cat >&2 <<MSG
MFD is requested but no fitted model exists at:
  $MFD_MODEL_ABS

Pass --mfd-train <non-test-labeled.jsonl> to fit it, or remove mfd from --detectors.
Do not fit MFD on the evaluation/test JSONL.
MSG
      exit 2
    fi
    echo "[suite] fitting MFD model on non-test data: $MFD_TRAIN_ABS"
    "$PYTHON_BIN" -m detectors_bench.fit_mfd --train "$MFD_TRAIN_ABS" --output "$MFD_MODEL_ABS"
  else
    echo "[suite] using existing MFD model: $MFD_MODEL_ABS"
  fi
  if [[ "$MFD_MODEL_USER_SET" -eq 1 && "$MFD_MODEL_ABS" != "$MFD_CONFIG_MODEL_ABS" ]]; then
    mkdir -p "$(dirname "$MFD_CONFIG_MODEL_ABS")"
    ln -sfn "$MFD_MODEL_ABS" "$MFD_CONFIG_MODEL_ABS"
    echo "[suite] linked MFD registry path $MFD_CONFIG_MODEL_ABS -> $MFD_MODEL_ABS"
  fi
fi

row_count="$("$PYTHON_BIN" - <<'PY' "$INPUT_ABS" "$OUTPUT_ABS/input_shards" "$SHARDS"
import json
import sys
from pathlib import Path

inp = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
shards = int(sys.argv[3])
out_dir.mkdir(parents=True, exist_ok=True)
handles = [(out_dir / f"input.{i:03d}.jsonl").open("w", encoding="utf-8") for i in range(shards)]
counts = [0 for _ in range(shards)]
total = 0
try:
    with inp.open(encoding="utf-8") as src:
        for line_no, line in enumerate(src, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "id" not in row or "text" not in row:
                raise ValueError(f"{inp}:{line_no} missing required id/text field")
            shard = total % shards
            handles[shard].write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            counts[shard] += 1
            total += 1
finally:
    for handle in handles:
        handle.close()

manifest = {
    "input": str(inp),
    "shards": shards,
    "total_rows": total,
    "shard_counts": counts,
    "method": "deterministic round-robin by non-empty JSONL row order",
}
(out_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(total)
PY
)"
if [[ "$row_count" -le 0 ]]; then
  echo "Input contains no examples: $INPUT_ABS" >&2
  exit 2
fi
echo "[suite] split rows=$row_count into $SHARDS shards"

IFS=',' read -r -a DETECTOR_LIST <<<"$DETECTORS"
ACTIVE_DETECTORS=()
HAS_BINOCULARS=0
for detector in "${DETECTOR_LIST[@]}"; do
  detector="${detector//[[:space:]]/}"
  [[ -z "$detector" ]] && continue
  if [[ "$detector" == "ghostbuster" ]]; then
    echo "ghostbuster is disabled for planned paper runs; remove it or run manually with --include-disabled." >&2
    exit 2
  fi
  if [[ "$detector" == "binoculars" ]]; then
    HAS_BINOCULARS=1
  fi
  ACTIVE_DETECTORS+=("$detector")
done
if [[ ${#ACTIVE_DETECTORS[@]} -eq 0 ]]; then
  echo "No active detectors requested." >&2
  exit 2
fi
DETECTORS="$(IFS=,; echo "${ACTIVE_DETECTORS[*]}")"

printf "%s\n" "${ACTIVE_DETECTORS[@]}" >"$OUTPUT_ABS/task_scripts/detectors.txt"

queue="$OUTPUT_ABS/task_scripts/single_gpu_queue.tsv"
: >"$queue"
for detector in "${ACTIVE_DETECTORS[@]}"; do
  mkdir -p "$OUTPUT_ABS/predictions_sharded/$detector"
  if [[ "$detector" == "binoculars" ]]; then
    continue
  fi
  for shard_idx in $(seq 0 $((SHARDS - 1))); do
    printf "%s\t%s\n" "$detector" "$shard_idx" >>"$queue"
  done
done

cat >"$OUTPUT_ABS/task_scripts/worker_single_gpu.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
ROOT="$ROOT"
OUT="$OUTPUT_ABS"
PY="$PYTHON_BIN"
GPU="\$1"
cd "\$ROOT"
export PYTHONPATH="\$ROOT/src"
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="\$GPU"
mkdir -p "\$OUT/logs" "\$OUT/status" "\$OUT/claims"
while true; do
  task=""
  {
    flock -x 9
    while IFS=\$'\\t' read -r detector shard; do
      key="\$(printf "%s.s%03d" "\$detector" "\$shard")"
      if [[ ! -e "\$OUT/status/\$key.done" && ! -e "\$OUT/status/\$key.failed" && ! -e "\$OUT/claims/\$key.claimed" ]]; then
        touch "\$OUT/claims/\$key.claimed"
        task="\$detector \$shard \$key"
        break
      fi
    done < "\$OUT/task_scripts/single_gpu_queue.tsv"
  } 9>"\$OUT/task_scripts/single_gpu_queue.lock"
  if [[ -z "\$task" ]]; then
    echo "[suite-worker] gpu=\$GPU no tasks left \$(date -Is)"
    exit 0
  fi
  read -r detector shard key <<<"\$task"
  input="\$OUT/input_shards/\$(printf "input.%03d.jsonl" "\$shard")"
  outdir="\$OUT/predictions_sharded/\$detector"
  mkdir -p "\$outdir"
  output="\$outdir/\$(printf "%s.s%03d.predictions.jsonl" "\$detector" "\$shard")"
  log="\$OUT/logs/\$(printf "%s.s%03d.log" "\$detector" "\$shard")"
  echo "[suite-worker] START detector=\$detector shard=\$shard gpu=\$GPU \$(date -Is)" | tee -a "\$log"
  if "\$PY" -m detectors_bench.run_detector --detector "\$detector" --input "\$input" --output "\$output" 2>&1 | tee -a "\$log"; then
    touch "\$OUT/status/\$key.done"
    echo "[suite-worker] DONE detector=\$detector shard=\$shard gpu=\$GPU \$(date -Is)" | tee -a "\$log"
  else
    touch "\$OUT/status/\$key.failed"
    echo "[suite-worker] FAILED detector=\$detector shard=\$shard gpu=\$GPU \$(date -Is)" | tee -a "\$log"
  fi
  rm -f "\$OUT/claims/\$key.claimed"
done
EOF
chmod +x "$OUTPUT_ABS/task_scripts/worker_single_gpu.sh"

if [[ "$HAS_BINOCULARS" -eq 1 ]]; then
  BINOCULARS_PY="${DETECTORS_BINOCULARS_PYTHON:-$PYTHON_BIN}"
  cat >"$OUTPUT_ABS/task_scripts/run_binoculars.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
ROOT="$ROOT"
OUT="$OUTPUT_ABS"
PY="$BINOCULARS_PY"
GPUS="$GPUS"
cd "\$ROOT"
export PYTHONPATH="\$ROOT/src"
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="\$GPUS"
mkdir -p "\$OUT/logs" "\$OUT/status" "\$OUT/predictions_sharded/binoculars"
for shard in \$(seq 0 $((SHARDS - 1))); do
  key="\$(printf "binoculars.s%03d" "\$shard")"
  input="\$OUT/input_shards/\$(printf "input.%03d.jsonl" "\$shard")"
  output="\$OUT/predictions_sharded/binoculars/\$(printf "binoculars.s%03d.predictions.jsonl" "\$shard")"
  log="\$OUT/logs/\$(printf "binoculars.s%03d.log" "\$shard")"
  if [[ -f "\$OUT/status/\$key.done" && -s "\$output" ]]; then
    echo "[suite-binoculars] SKIP shard=\$shard \$(date -Is)" | tee -a "\$log"
    continue
  fi
  echo "[suite-binoculars] START shard=\$shard gpus=\$GPUS \$(date -Is)" | tee -a "\$log"
  if "\$PY" -m detectors_bench.run_detector --detector binoculars --input "\$input" --output "\$output" 2>&1 | tee -a "\$log"; then
    touch "\$OUT/status/\$key.done"
    echo "[suite-binoculars] DONE shard=\$shard gpus=\$GPUS \$(date -Is)" | tee -a "\$log"
  else
    touch "\$OUT/status/\$key.failed"
    echo "[suite-binoculars] FAILED shard=\$shard gpus=\$GPUS \$(date -Is)" | tee -a "\$log"
    exit 1
  fi
done
touch "\$OUT/status/binoculars.all.done"
echo "[suite-binoculars] ALL DONE \$(date -Is)"
EOF
  chmod +x "$OUTPUT_ABS/task_scripts/run_binoculars.sh"
fi

cat >"$OUTPUT_ABS/task_scripts/run_after_binoculars.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
OUT="$OUTPUT_ABS"
GPU="\$1"
if [[ $HAS_BINOCULARS -eq 1 ]]; then
  echo "[suite-wait] gpu=\$GPU waiting for binoculars.all.done \$(date -Is)"
  while [[ ! -f "\$OUT/status/binoculars.all.done" ]]; do
    if find "\$OUT/status" -maxdepth 1 -name 'binoculars.s*.failed' | grep -q .; then
      echo "[suite-wait] gpu=\$GPU binoculars failed; not starting single-GPU queue"
      exit 1
    fi
    sleep 60
  done
fi
exec "\$OUT/task_scripts/worker_single_gpu.sh" "\$GPU"
EOF
chmod +x "$OUTPUT_ABS/task_scripts/run_after_binoculars.sh"

merge_args=(
  --output-root "$OUTPUT_ABS"
  --detectors "$DETECTORS"
  --shards "$SHARDS"
  --expected-rows "$row_count"
  --poll-seconds "$POLL_SECONDS"
  --bootstrap "$BOOTSTRAP"
  --summary-name "merge_summary.json"
)
if [[ "$ALLOW_FAILED" -eq 1 ]]; then
  merge_args+=(--allow-failed)
fi
printf "%q " "${merge_args[@]}" >"$OUTPUT_ABS/task_scripts/merge_args.txt"
printf "\n" >>"$OUTPUT_ABS/task_scripts/merge_args.txt"

"$PYTHON_BIN" - <<'PY' "$OUTPUT_ABS/run_manifest.json" "$INPUT_ABS" "$OUTPUT_ABS" "$GPUS" "$SHARDS" "$row_count" "$DETECTORS" "$SESSION_PREFIX" "$PYTHON_BIN" "$MFD_MODEL_ABS"
import json
import platform
import subprocess
import sys
from pathlib import Path

out, inp, root, gpus, shards, rows, detectors, prefix, py, mfd = sys.argv[1:]
try:
    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
except Exception:
    git_sha = ""
manifest = {
    "input": inp,
    "output_root": root,
    "gpus": gpus,
    "shards": int(shards),
    "expected_rows": int(rows),
    "detectors": [x for x in detectors.split(",") if x],
    "session_prefix": prefix,
    "python": py,
    "mfd_model": mfd,
    "project_git_sha": git_sha,
    "platform": platform.platform(),
}
Path(out).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "[suite] dry-run complete; scripts written under $OUTPUT_ABS/task_scripts"
  exit 0
fi

if [[ "$HAS_BINOCULARS" -eq 1 ]]; then
  session="${SESSION_PREFIX}_binoculars"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "[suite] tmux session already exists: $session"
  else
    tmux new-session -d -s "$session" "bash -lc \"bash '$OUTPUT_ABS/task_scripts/run_binoculars.sh' 2>&1 | tee -a '$OUTPUT_ABS/logs/${session}.log'\""
    echo "[suite] started $session"
  fi
fi

for gpu in "${GPU_LIST[@]}"; do
  session="${SESSION_PREFIX}_gpu${gpu}"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "[suite] tmux session already exists: $session"
  else
    tmux new-session -d -s "$session" "bash -lc \"bash '$OUTPUT_ABS/task_scripts/run_after_binoculars.sh' '$gpu' 2>&1 | tee -a '$OUTPUT_ABS/logs/${session}.log'\""
    echo "[suite] started $session"
  fi
done

merge_session="${SESSION_PREFIX}_merge"
if tmux has-session -t "$merge_session" 2>/dev/null; then
  echo "[suite] tmux session already exists: $merge_session"
else
  tmux new-session -d -s "$merge_session" "bash -lc \"cd '$ROOT' && DETECTORS_PYTHON='$PYTHON_BIN' bash scripts/wait_merge_detector_suite.sh $(cat "$OUTPUT_ABS/task_scripts/merge_args.txt") 2>&1 | tee -a '$OUTPUT_ABS/logs/${merge_session}.log'\""
  echo "[suite] started $merge_session"
fi

echo "[suite] launched. Monitor with:"
echo "  tmux ls | grep '$SESSION_PREFIX'"
echo "  find '$OUTPUT_ABS/status' -maxdepth 1 -type f | sed 's#.*/##' | sort | awk -F. '{print \$1, \$NF}' | sort | uniq -c"
echo "  tail -f '$OUTPUT_ABS/logs/${merge_session}.log'"
