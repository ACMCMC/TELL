#!/usr/bin/env bash
# End-to-end training data preparation.
#
# Generates 50/50-balanced train/val/test splits from the unified TELL parquet,
# adds an adversarial slice (domain="adversarial") from RAID, scores all documents
# with MAGE, and writes final JSONL splits to data/balanced-splits-v1/final/.
#
# Usage (run from repo root, inside tmux so it survives disconnects):
#   tmux new-session -d -s prep_data \
#     "bash scripts/run_prepare_training_data.sh 2>&1 | tee data/prepare.log"

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_DIR="$REPO_ROOT/scripts"
PYTHON="$(which python)"

PARQUET="$REPO_ROOT/data/unified-v3/unified_tell_dataset.parquet"
OUTPUT_DIR="$REPO_ROOT/data/balanced-splits-v1"
HARNESS_DIR="$REPO_ROOT/detectors"
HARNESS_PYTHON="${HARNESS_PYTHON:-$PYTHON}"

TRAIN_PER_CLASS=100000
VAL_PER_CLASS=10000
TEST_PER_CLASS=10000
ADV_TRAIN=50000
ADV_VAL=5000
ADV_TEST=5000
SHARDS=16
GPUS=0,1
SEED=20260428

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# ── Preflight checks ──────────────────────────────────────────────────────────
if [ ! -f "$PARQUET" ]; then
    echo "ERROR: parquet not found at $PARQUET"
    echo "Run scripts/build_unified_dataset.py first:"
    echo "  python scripts/build_unified_dataset.py --output-dir data/unified-v3"
    exit 1
fi
if [ ! -d "$HARNESS_DIR" ]; then
    echo "ERROR: detectors harness not found at $HARNESS_DIR"
    echo "Expected the in-repo harness at $REPO_ROOT/detectors."
    exit 1
fi
HARNESS_DIR="$(cd "$HARNESS_DIR" && pwd)"
if [ ! -x "$HARNESS_DIR/scripts/launch_wave1_fast.sh" ] || [ ! -x "$HARNESS_DIR/scripts/wait_merge_wave1_fast.sh" ]; then
    echo "ERROR: detector harness scripts are missing or not executable under $HARNESS_DIR/scripts"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# ── Stage 1: generate splits ─────────────────────────────────────────────────
log "── Stage 1/4: generating splits ──────────────────────────────────────"
if [ -f "$OUTPUT_DIR/split_summary.json" ]; then
    log "split_summary.json exists, skipping"
else
    "$PYTHON" "$SCRIPT_DIR/sample_balanced_splits.py" \
        --parquet         "$PARQUET" \
        --output-dir      "$OUTPUT_DIR" \
        --train-per-class "$TRAIN_PER_CLASS" \
        --val-per-class   "$VAL_PER_CLASS" \
        --test-per-class  "$TEST_PER_CLASS" \
        --adv-train       "$ADV_TRAIN" \
        --adv-val         "$ADV_VAL" \
        --adv-test        "$ADV_TEST" \
        --shards          "$SHARDS" \
        --seed            "$SEED"
fi

# ── Stage 2: MAGE scoring (clean) ────────────────────────────────────────────
log "── Stage 2/4: MAGE scoring (clean) ───────────────────────────────────"
for SPLIT in train val test; do
    DONE_MARKER="$OUTPUT_DIR/mage_scores/$SPLIT/.done"
    if [ -f "$DONE_MARKER" ]; then
        log "mage/$SPLIT: already done, skipping"
        continue
    fi

    EXPECTED=$("$PYTHON" -c "
import json
print(json.load(open('$OUTPUT_DIR/split_summary.json'))['splits']['$SPLIT']['total'])
")
    log "mage/$SPLIT: launching ($EXPECTED rows, shards=$SHARDS, gpus=$GPUS) ..."

    DETECTORS_PYTHON="$HARNESS_PYTHON" \
        bash "$HARNESS_DIR/scripts/launch_wave1_fast.sh" \
            --shard-dir    "$OUTPUT_DIR/shards/$SPLIT" \
            --output-root  "$OUTPUT_DIR/mage_scores/$SPLIT" \
            --gpus         "$GPUS" \
            --detectors    mage_d \
            --shards       "$SHARDS"

    log "mage/$SPLIT: waiting for completion ..."
    DETECTORS_PYTHON="$HARNESS_PYTHON" \
        bash "$HARNESS_DIR/scripts/wait_merge_wave1_fast.sh" \
            --output-root   "$OUTPUT_DIR/mage_scores/$SPLIT" \
            --detectors     mage_d \
            --shards        "$SHARDS" \
            --expected-rows "$EXPECTED" \
            --poll-seconds  60 \
            --bootstrap     0

    touch "$DONE_MARKER"
    log "mage/$SPLIT: done"
done

# ── Stage 3: MAGE scoring (adversarial) ──────────────────────────────────────
log "── Stage 3/4: MAGE scoring (adversarial) ─────────────────────────────"
for SPLIT in train val test; do
    DONE_MARKER="$OUTPUT_DIR/adversarial_mage_scores/$SPLIT/.done"
    if [ -f "$DONE_MARKER" ]; then
        log "mage_adv/$SPLIT: already done, skipping"
        continue
    fi

    EXPECTED=$("$PYTHON" -c "
import json
print(json.load(open('$OUTPUT_DIR/split_summary.json'))['splits']['$SPLIT']['adversarial_ai'])
")
    log "mage_adv/$SPLIT: launching ($EXPECTED rows, shards=$SHARDS, gpus=$GPUS) ..."

    DETECTORS_PYTHON="$HARNESS_PYTHON" \
        bash "$HARNESS_DIR/scripts/launch_wave1_fast.sh" \
            --shard-dir    "$OUTPUT_DIR/adversarial_shards/$SPLIT" \
            --output-root  "$OUTPUT_DIR/adversarial_mage_scores/$SPLIT" \
            --gpus         "$GPUS" \
            --detectors    mage_d \
            --shards       "$SHARDS"

    log "mage_adv/$SPLIT: waiting for completion ..."
    DETECTORS_PYTHON="$HARNESS_PYTHON" \
        bash "$HARNESS_DIR/scripts/wait_merge_wave1_fast.sh" \
            --output-root   "$OUTPUT_DIR/adversarial_mage_scores/$SPLIT" \
            --detectors     mage_d \
            --shards        "$SHARDS" \
            --expected-rows "$EXPECTED" \
            --poll-seconds  60 \
            --bootstrap     0

    touch "$DONE_MARKER"
    log "mage_adv/$SPLIT: done"
done

# ── Stage 4: join scores ──────────────────────────────────────────────────────
log "── Stage 4/4: joining MAGE scores ────────────────────────────────────"
if [ -f "$OUTPUT_DIR/final/mage_join_summary.json" ]; then
    log "final/ exists, skipping"
else
    "$PYTHON" "$SCRIPT_DIR/join_mage_scores.py" \
        --splits-dir              "$OUTPUT_DIR/splits" \
        --mage-dir                "$OUTPUT_DIR/mage_scores" \
        --output-dir              "$OUTPUT_DIR/final" \
        --adversarial-splits-dir  "$OUTPUT_DIR/adversarial_splits" \
        --adversarial-mage-dir    "$OUTPUT_DIR/adversarial_mage_scores"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
log "── Pipeline complete ──────────────────────────────────────────────────"
"$PYTHON" -c "
import json
s = json.load(open('$OUTPUT_DIR/split_summary.json'))
for split, c in s['splits'].items():
    print(f'  {split}: {c[\"total\"]:,} clean ({c[\"human\"]:,} human + {c[\"ai\"]:,} AI) + {c[\"adversarial_ai\"]:,} adversarial AI')
m = json.load(open('$OUTPUT_DIR/final/mage_join_summary.json'))
print(f'  mage_score range (train): [{m[\"score_min\"]:.4f}, {m[\"score_max\"]:.4f}]')
if m.get('missing_rows'):
    print(f'  WARNING: {m[\"missing_rows\"]} docs missing mage_score')
"
