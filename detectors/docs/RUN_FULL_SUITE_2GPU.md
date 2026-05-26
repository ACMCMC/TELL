# Running the Full Detector Suite on New Data

Use this when a collaborator wants to benchmark the external detector baselines
on a new JSONL dataset with two strong GPUs. The script is:

```bash
detectors/scripts/run_full_suite_2gpu.sh
```

It creates deterministic shards, launches detached `tmux` jobs, writes rich logs
and manifests, merges predictions, and computes the paper metrics.

## Input Contract

Input is JSONL with one document per line:

```json
{"id":"doc_001","text":"document text","label":1,"dataset":"my_dataset","domain":"news","generator":"gpt-4","attack":"none"}
```

Required fields are `id` and `text`. `label` is optional for scoring but required
for AUROC and the paper metrics, with `0=human` and `1=AI`. Metadata fields such
as `dataset`, `domain`, `generator`, and `attack` are preserved for subgroup
metrics.

## Setup

From a fresh checkout on the GPU server:

```bash
cd /data/$USER/rl-detector
bash detectors/scripts/setup_server.sh "$PWD"
cd detectors
```

Use the benchmark Python environment explicitly:

```bash
export DETECTORS_PYTHON=/path/to/benchmark/python
```

If running Binoculars with its separate upstream-compatible environment:

```bash
export DETECTORS_BINOCULARS_PYTHON=/path/to/binoculars/python
```

If running gated Hugging Face detectors such as Pangram EditLens:

```bash
export HF_TOKEN=...
```

Do not write tokens into tracked files.

## Standard Command

Run the paper-practical suite on two GPUs:

```bash
cd /data/$USER/rl-detector/detectors
bash scripts/run_full_suite_2gpu.sh \
  --input /path/to/new_eval.jsonl \
  --output-root /path/to/output/full_detector_suite \
  --gpus 0,1 \
  --shards 32 \
  --mfd-train /path/to/non_test_mfd_train.jsonl \
  --session-prefix newdata_detectors
```

The default detector set is (MFD is omitted by default because it requires fitting on non-test labeled data; add `mfd` via `--detectors` when needed):

```text
binoculars,openai_roberta,chatgpt_d,argugpt,radar,mage_d,detectllm_lrr,fast_detectgpt,pangram_editlens_llama,detectllm_npr,dnagpt
```

`detectgpt` is not included by default because it was empirically impractical on
the 200k paper split. To run it anyway:

```bash
bash scripts/run_full_suite_2gpu.sh ... --include-detectgpt
```

Ghostbuster is intentionally disabled for planned paper runs because current
execution would be a paid OpenAI compatibility port for retired upstream models.

## Execution Model

The script assumes exactly two visible GPUs.

1. If `binoculars` is requested, it runs first with both GPUs visible because the
   official implementation uses one model per GPU.
2. After Binoculars completes, two single-GPU workers start and process the
   remaining detector/shard queue using file locks.
3. A merge watcher waits for all requested detector shards, fails if any shard
   has a `.failed` marker, then writes metrics and `summary.tsv`.

This is restartable. Re-running the same command with the same `--output-root`
skips shards that already have `.done` markers and non-empty prediction files.

## Outputs

Important files under `--output-root`:

```text
input_shards/input.NNN.jsonl
input_shards/split_manifest.json
predictions_sharded/<detector>/<detector>.sNNN.predictions.jsonl
merged_predictions/<detector>.predictions.jsonl
metrics/<detector>.metrics.json
metrics/<detector>.roc_points.tsv
manifests/*.manifest.json
logs/*.log
status/*.done
status/*.failed
summary.tsv
merge_summary.json
run_manifest.json
```

`summary.tsv` is the compact table to inspect first. The per-detector metric JSON
files contain AUROC, AUPRC, accuracy, balanced accuracy, F1, MCC, low-FPR TPRs,
FPR-at-TPR, calibration metrics, subgroup metrics, row error counts, and optional
bootstrap confidence intervals.

## Monitoring

```bash
tmux ls | grep newdata_detectors
find /path/to/output/full_detector_suite/status -maxdepth 1 -type f \
  | sed 's#.*/##' | sort | awk -F. '{print $1, $NF}' | sort | uniq -c
tail -f /path/to/output/full_detector_suite/logs/newdata_detectors_merge.log
```

Inspect failed shards:

```bash
find /path/to/output/full_detector_suite/status -name '*.failed' -print
```

## MFD Rule

MFD is a trained logistic model over log-likelihood, log-rank, entropy, and LLM
deviation. For paper-valid results, fit it only on train/validation data and
never on the evaluation/test JSONL. The launcher enforces this by requiring
either an existing `detectors/results/models/mfd.joblib` or an explicit
`--mfd-train` file.

## Practical Defaults

Use `--bootstrap 0` for the unattended run, then rerun metric computation with
bootstrap confidence intervals only after predictions are complete. Heavy
methods can take hours to days depending on data size. Row-level errors are kept
in the prediction JSONL and counted in metrics; do not silently drop them.
