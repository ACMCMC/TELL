# Multi-domain HF test benchmark (5k)

External detector baselines on **`acmc/multi_domain_ai_human_text`** **`test`** split (5000 balanced rows).

## How this was produced

```bash
cd /workspace/rl-detector
bash setup_env_package.sh   # once per machine
bash detectors/scripts/launch_multi_domain_hf5k_benchmark.sh
```

Launcher: `detectors/scripts/launch_multi_domain_hf5k_benchmark.sh`  
Input JSONL: `detectors/results/input/multi_domain_ai_human_text_test.jsonl`  
Runs detached in **tmux** (`hf5k_bench_gpu0`, `hf5k_bench_merge`).

**Detectors (no MFD, no Binoculars on 1 GPU):**  
`openai_roberta`, `chatgpt_d`, `argugpt`, `radar`, `mage_d`, `detectllm_lrr`, `fast_detectgpt`, `pangram_editlens_llama`, `detectllm_npr`, `dnagpt`

## Published artifacts (commit these)

| Path | Description |
|------|-------------|
| `merged_predictions/<detector>.predictions.jsonl` | Per-doc scores + `features` (raw harness output) |
| `metrics/<detector>.metrics.json` | AUROC, TPR@FPR, etc. |
| `summary.tsv` | Compact table (after merge completes) |
| `run_manifest.json` | Git SHA, detector list, shard count |
| `PROVENANCE.json` | Dataset + policy metadata |

Shard-level files under `predictions_sharded/` are for restart/debug; optional to commit.

## Metrics note

`TPR@FPR≤1%` in metrics JSON is **`tpr_at_fpr_0.01`**: max TPR on the ROC with FPR ≤ 1% (not validation-tuned threshold).

## Citation

Dataset: `acmc/multi_domain_ai_human_text` — see `docs/DATASET_REPORT.md`.
