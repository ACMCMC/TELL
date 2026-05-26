# Win-rate eval

Listwise blind ranking: TELL model explanation vs 5 human annotators per doc, 5-judge panel, document-level win rate.

**Script:** `eval_human_detectors_winrate.py` · **Judges:** `winrate_judges.py`

## Setup

```bash
source .env   # OPENAI_API_KEY, TRITONAI_API_KEY, DEEPINFRA_API_KEY, Tinker creds
```

## Full run (test split, 200 docs)

```bash
python experiments/winrate_eval/eval_human_detectors_winrate.py \
  --dataset-url "hf://suraj-ranganath/tell-human-detectors/test" \
  --checkpoint-path "tinker://.../weights/best-step-235" \
  --sample-size 200 --workers 8 --seed 2242 \
  --run-name my_run \
  --judge-max-tokens 512
```

Paper split is **test** (200 docs), not validation (100).

## Caches (default: read, never overwrite)

Under `data/winrate_eval/`:

| Stage | Path |
|-------|------|
| Rollouts | `rollouts/{dataset}/{rollout_key}/{doc_id}.json` |
| Style | `style_rewrites/...` or frozen bank `tell_human_detectors_style_paraphrases_v3.json` |
| Judges | `judge_rankings/{dataset}/{rollout_key}/{judge_id}/{doc_id}.json` |

Existing cache files are **not** overwritten. Stale `cache_key` → error. To refresh:

```bash
--invalidate-judges    # alias --force-judges
--invalidate-rollouts
--invalidate-style
```

Missing cache files are still written on first compute.

Cache filenames use `{source_id}_row{index}` (HF `id` alone is not unique in this dataset).

No judge fallbacks: no salvage parse, no json_object retry after structured parse, no imputed ranks/scores.

## Partial runs

```bash
--only-style          # build style bank only
--skip-rollouts       # need rollout caches
--skip-style          # need style bank or style caches
--only-judge          # judges only (rollouts + style from cache/bank)
```

## Outputs

- `data/winrate_eval/experiments/{run_name}/results.json` (share)
- `data/winrate_eval/experiments/{run_name}/audit.jsonl` (full replay)
- `results/winrate_eval/{run_name}/` mirror

## Smoke

```bash
--sample-size 5 --workers 2
```

Debug judge raw responses: `inspect_judge_responses.py`

## LaTeX table

After each run:
- `winrate_table.tex` — full table (win rate %, 95% CI, permutation $p$, Wilcoxon $p$)
- `winrate_table_compact.tex` — win rate % and 95% CI in one column only

CIs are **document-level bootstrap** ($B=10000$, resample documents with replacement). Permutation $p$ is sign-flip on per-doc rates vs 50%; Wilcoxon is signed-rank vs 50%.

Regenerate from an existing summary:

```bash
python experiments/winrate_eval/eval_human_detectors_winrate.py \
  --only-export-tex results/winrate_eval/judge_panel_step235_test200_v2/summary.json
```
