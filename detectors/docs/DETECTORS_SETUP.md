# External Detector Benchmark Setup

This subsystem benchmarks TELL against external AI-text detectors without touching the training code in `src/rl_detector`. Inputs and outputs use one JSONL schema, while detector-specific wrappers preserve raw scores and features for paper analysis.

## What Is Included

Pinned upstream repos live in `detectors/vendor/` as git submodules:

| Detector | Upstream | Notes |
|---|---|---|
| Binoculars | `ahans30/Binoculars` | Uses the official Falcon observer/performer score; lower raw score means more AI. |
| DetectGPT | `eric-mitchell/detect-gpt` | Implemented through the same perturbation statistic: original log-likelihood minus perturbed mean, normalized by perturbed std. |
| Fast-DetectGPT | `baoguangsheng/fast-detect-gpt` | Imports the official `FastDetectGPT` local inference class. |
| MAGE-D | `yafuly/MAGE` | Uses the official `yaful/MAGE` Longformer detector and preprocessing. |
| OpenAI RoBERTa | HF `openai-community/roberta-base-openai-detector` | Hugging Face classifier wrapper. |
| ArguGPT | `huhailinguist/ArguGPT` | Uses `SJTU-CL/RoBERTa-large-ArguGPT`. |
| DetectLLM-LRR | `mbzuai-nlp/DetectLLM` | Implements the official LRR formula, `-log p(x) / log-rank(x)`. |
| DetectLLM-NPR | `mbzuai-nlp/DetectLLM` | Implements the official perturbation ratio over log-rank. |
| RADAR | `IBM/RADAR` | Uses `TrustSafeAI/RADAR-Vicuna-7B`; class index 0 is AI probability per upstream README. |
| Pangram EditLens | HF `pangram/editlens_Llama-3.2-3B` | Gated PEFT adapter on `meta-llama/Llama-3.2-3B`; outputs expected EditLens bucket score in `[0, 1]`. |
| AIGC MPU | `YuchuanTian/AIGC_text_detector` | Uses the official latest English v3 RoBERTa checkpoint `yuchuantian/AIGC_detector_env3`; the released HF Space maps `id2label=['Human', 'AI']`, so class 1 is AI. |
| MELD | HF `anon-review-meld-2026/meld` | Uses the released checkpoint's main AI/Human head; long documents are chunked with overlap and mean aggregation. |
| T5Sentinel | `MarkChenYutian/T5-Sentinel-public` | Uses the official T5-small sentinel checkpoint and the released special-token label surface. |
| LogRank | Ippolito et al. | Mean token log-rank under `gpt2-medium`; raw values are lower for more model-like text, so `score_ai` is inverted. |
| PHD | `ArGintum/GPTID` | Persistent-homology intrinsic dimension over RoBERTa-base token embeddings using NAACL detector-eval defaults. |
| DNA-GPT | `Xianjun-Yang/DNA-GPT` | Local-regeneration variant using the official divergent n-gram idea. Closed OpenAI regeneration can be added as a separate config. |
| MFD | Wu and Xiang / IMGTB | Four features: log-likelihood, log-rank, entropy, LLM-deviation. Fit the logistic model on train/validation only. |
| ChatGPT-D | `Hello-SimpleAI/chatgpt-comparison-detection` | Uses `Hello-SimpleAI/chatgpt-detector-roberta`. |
| Ghostbuster | `vivek3141/ghostbuster` | Optional only; excluded from planned paper runs because current execution requires paid OpenAI echo/logprobs calls and is a compatibility port for retired `ada`/`davinci`. |
| IMGTB | `kinit-sk/IMGTB` | Secondary reference harness for implementation checks. Its repo has a broken nested test gitlink, so do not use recursive submodule init. |

## Server Setup

From a fresh checkout, run the setup script from the repo root (it initializes vendor submodules and installs the harness):

```bash
cd /path/to/TELL
bash detectors/scripts/setup_server.sh "$PWD"
```

If `python3-venv` is unavailable, point the harness at an existing GPU environment:

```bash
DETECTORS_PYTHON=/path/to/existing/python \
  bash detectors/scripts/setup_server.sh /path/to/TELL
```

For smoke tests after setup:

```bash
cd detectors
bash scripts/smoke_all.sh                     # quick mode: no API keys needed
bash scripts/smoke_all.sh . all               # all enabled detectors
bash scripts/smoke_all.sh . wave4             # AIGC MPU, T5Sentinel, LogRank, PHD, Binoculars
```

The default smoke mode is `quick` and covers detectors that do not need perturbation loops or external API keys. The `all` mode attempts all enabled detectors and logs failures explicitly; use it after installing each detector's official dependency file and fitting MFD.
The `wave4` mode covers `aigc_mpu_env3`, `t5_sentinel`, `logrank_gpt2_medium`, `phd_roberta`, and `binoculars`.

Binoculars requires a separate venv pinned to `transformers==4.31.0`:

```bash
python -m venv /path/to/binoculars431
/path/to/binoculars431/bin/pip install 'transformers[torch]==4.31.0' torch accelerate sentencepiece numpy scikit-learn pandas pyyaml joblib scipy
# then run with:
DETECTORS_BINOCULARS_PYTHON=/path/to/binoculars431/bin/python bash scripts/smoke_all.sh . wave4
```

Pangram EditLens requires accepted HuggingFace access for both `pangram/editlens_Llama-3.2-3B` and `meta-llama/Llama-3.2-3B`; export `HF_TOKEN` in the shell running the detector, never commit it.

For long detector jobs, use the tmux wrapper:

```bash
cd detectors
bash scripts/run_detector_tmux.sh SESSION_NAME DETECTOR_NAME path/to/input.jsonl results/output GPU_ID
```

For the full paper detector suite on a new JSONL dataset with two GPUs, use the restartable launcher:

```bash
cd detectors
bash scripts/run_full_suite_2gpu.sh \
  --input /path/to/new_eval.jsonl \
  --output-root /path/to/output/full_detector_suite \
  --gpus 0,1 \
  --shards 32 \
  --mfd-train /path/to/non_test_mfd_train.jsonl \
  --session-prefix newdata_detectors
```

See `docs/RUN_FULL_SUITE_2GPU.md` for the exact execution model, outputs, and monitoring commands.

For the added detector wave only:

```bash
cd detectors
bash scripts/run_wave4_more_text_detectors.sh \
  --input /path/to/new_eval.jsonl \
  --output-root /path/to/output/wave4_more_text_detectors \
  --gpus 0,1 \
  --shards 32 \
  --session-prefix wave4_detectors
```

Heavy detectors should be run in separate tmux sessions with explicit `CUDA_VISIBLE_DEVICES`. Binoculars, Pangram EditLens, Fast-DetectGPT, DetectGPT, DetectLLM-NPR, DNA-GPT, and MFD can download multi-GB models; check disk before launch.

## Data Contract

Input JSONL:

```json
{"id":"doc_1","text":"...","label":1,"split":"test","dataset":"raid","domain":"news","generator":"gpt-4","attack":"none"}
```

Labels are `0=human`, `1=AI`. Extra metadata fields are allowed and preserved in manifests or subgroup metrics when explicitly supported.

Prediction JSONL:

```json
{"id":"doc_1","detector":"openai_roberta","score_ai":0.93,"raw_score":0.93,"raw_label":"Fake","pred_builtin":1,"features":{},"runtime_s":0.12,"error":null}
```

`score_ai` always means higher is more AI-generated, even when the upstream detector uses the opposite raw orientation.

## Metrics For Paper Tables

The benchmark reports the metrics commonly used in recent detector evaluations, including the detector suite in arXiv:2502.19614, RAID, MGTBench, and IMGTB:

- AUROC and AUPRC for threshold-free ranking.
- Accuracy, balanced accuracy, precision, recall, F1, MCC, specificity, and confusion matrix at a fixed or validation-selected threshold.
- TPR at FPR `0.001`, `0.005`, `0.01`, and `0.05`.
- FPR at TPR `0.8`, `0.9`, and `0.95`.
- Brier score and ECE for probability-like outputs.
- Subgroup metrics by split, dataset, domain, generator, and attack when metadata exists.
- Paired bootstrap confidence intervals for final paper numbers.

Thresholds must be selected on validation only. For final reporting, use the test split once with the frozen threshold and frozen detector configs.

## Reproducibility Rules

- Do not edit vendor code for paper results. If a patch is unavoidable, add it as a wrapper-level adapter and document it here.
- MAGE-D uses a wrapper-level HF config sanitizer because `yaful/MAGE` publishes `id2label` values as integers; newer Transformers releases reject that config although the official weights are unchanged.
- Pangram EditLens uses the official EditLens inference semantics: official text cleaning, `meta-llama/Llama-3.2-3B` tokenizer, 4-bit QLoRA sequence-classification adapter, `NormedLinear` score head, bucket probabilities, and expected bucket score. It is gated on Hugging Face and licensed CC BY-NC-SA 4.0, so treat it as noncommercial unless a different license is obtained.
- AIGC MPU uses the latest upstream English v3 checkpoint for the primary paper-ready setup. The released HF Space uses `id2label=['Human', 'AI']`, so `aigc_mpu_env3` records class 1 as `score_ai`. If an exact older ICLR-era checkpoint comparison is needed, add `env1` or `env2` as explicit appendix configs instead of silently changing `aigc_mpu_env3`.
- MELD is pinned by HF model revision because the anonymous code URL linked from the paper does not expose a cloneable git remote. The released model card provides a self-contained inference path; the wrapper reconstructs the released architecture from `meld_config.json`, loads `model.safetensors`, uses the main AI/Human head with `[logit_human, logit_ai]`, applies the released 2048-token sequence length, and mean-aggregates overlapping chunks for long documents.
- T5Sentinel uses the official `T5Sentinel.0613.pt` checkpoint URL from the upstream release. The wrapper scores `1 - P(Human)` over the released sentinel label tokens: Human, ChatGPT, PaLM, LLaMA, and GPT2.
- LogRank and PHD have raw scores where lower is more AI-like. The wrappers preserve `raw_score` and `features.score_direction`, but invert the value through a monotone sigmoid so the common `score_ai` convention still means higher is more AI.
- PHD uses the official estimator from `ArGintum/GPTID` with defaults aligned to `LeiLiLab/llm-detector-eval`: RoBERTa-base token embeddings, minimum subsample 40, seven intermediate subsample points, alpha 1, nine PHD subsamples per scale, and Euclidean distances. The wrapper fixes the NumPy seed per example id so reruns are reproducible enough for benchmark auditing. PHD is undefined for very short documents under this setting: after RoBERTa special tokens are removed, the wrapper requires at least 42 usable token embeddings (`min_subsample + 2`). Rows below that threshold are marked as explicit errors and excluded from PHD aggregate metrics rather than assigned imputed scores.
- Binoculars should be smoke-tested on sufficiently long texts, for example `smoke/smoke_long.jsonl`, because the official perplexity ratio can become non-finite on short or degenerate inputs. The wrapper computes the official perplexity / cross-perplexity ratio directly in float32 to avoid bfloat16 numerical failures on the server GPUs while preserving the upstream Falcon models and thresholds; if finite logits still produce a zero denominator from numerical collapse, the wrapper uses the `log(vocab_size)` cross-entropy limit. It also validates tokenizer ids before model forward passes and re-raises fatal CUDA errors so a bad shard fails cleanly instead of poisoning the process and writing misleading error rows.
- Record submodule SHAs, HF model revisions, command line, GPU, Python version, and package environment through the generated `*.manifest.json`.
- Keep `configs/vendor_locks.tsv` synchronized with submodule SHAs before reporting final paper results.
- Keep raw method features in `features` so later analysis can recompute detector-specific diagnostics.
- Mark failures explicitly in `error`; do not drop failed examples silently.

## Ghostbuster Compatibility

The original Ghostbuster release (Verma et al., NAACL 2024) is not a transformer checkpoint detector; it is a feature-based classifier over token log-probability streams and symbolic n-gram features. The upstream repo's `classify.py` uses OpenAI Completion echo/logprobs calls with `model="ada"` and `model="davinci"`, truncates with the `davinci` tokenizer, featurizes the `davinci`, `ada`, trigram, and unigram streams, and applies the released `model/features.txt`, `model/model`, `model/mu`, and `model/sigma` artifacts. Its released datasets also store per-document `X-ada.txt` and `X-davinci.txt` logprob files.

Those original GPT-3 base models are no longer callable: OpenAI retired `ada`, `babbage`, `curie`, and `davinci` on January 4, 2024, and lists `babbage-002` as the replacement for `ada`/`babbage` and `davinci-002` as the replacement for `curie`/`davinci`. Our citable runnable setup therefore uses a compatibility variant:

- `original_ada_model=ada`, `original_davinci_model=davinci`
- `ada_model=babbage-002`, `davinci_model=davinci-002`
- the same Completion API shape as upstream: `max_tokens=0`, `echo=True`, `logprobs=1`
- the official Ghostbuster classifier artifacts and symbolic feature definitions, without vendor-code edits
- wrapper-level alignment of replacement-model logprob streams by common prefix before official vector feature algebra

Report this as "Ghostbuster with OpenAI base-model replacements" or "Ghostbuster compatibility port." Do not describe it as an exact historical reproduction unless using cached original `ada`/`davinci` logprob files from the Ghostbuster release or from pre-retirement API runs. The wrapper records `compatibility_variant`, original model names, replacement model names, raw stream lengths, and aligned feature length in each prediction's `features` field so this caveat survives into paper logs.

Recent benchmark precedent supports this compatibility choice. Stowe et al. (2025/2026), a bias evaluation of 16 machine-generated-text detectors, used the official Ghostbuster implementation but modified it to address outdated OpenAI model references and explicitly reports `davinci-002` and `babbage-002`. Zeng et al. (NeurIPS 2025) also evaluate GhostBuster as a baseline and state that they use the official implementation with an updated OpenAI API model, but their appendix does not name the exact Ghostbuster replacement pair. We therefore use `babbage-002` for the original `ada` role and `davinci-002` for the original `davinci` role, because that is both OpenAI's documented base-model replacement mapping and the most explicit recent paper setup found.

Decision for this project: do not include Ghostbuster in the planned paper benchmark table. It is disabled in `configs/detectors.yaml`, removed from `scripts/smoke_all.sh all`, and `run_detector` refuses it unless `--include-disabled` is passed. This avoids accidental OpenAI spend and avoids presenting a compatibility port as a fully faithful reproduction. If an appendix comparison is later needed, run it explicitly with `DETECTORS_GHOSTBUSTER_PYTHON=/path/to/ghostbuster312/bin/python` and `--include-disabled`, and report the compatibility caveat above.

## References

- DetectGPT: Mitchell et al., 2023, https://arxiv.org/abs/2301.11305
- DetectLLM: Su et al., 2023, https://arxiv.org/abs/2306.05540
- DNA-GPT: Yang et al., 2023, https://arxiv.org/abs/2305.17359
- Ghostbuster: Verma et al., 2024, https://aclanthology.org/2024.naacl-long.95/
- Ghostbuster BAIR implementation note: https://bair.berkeley.edu/blog/2023/11/14/ghostbuster/
- Ghostbuster upstream implementation: https://github.com/vivek3141/ghostbuster
- Recent Ghostbuster compatibility precedent: Stowe et al., 2025/2026, https://arxiv.org/abs/2512.09292
- Recent Ghostbuster baseline precedent: Zeng et al., NeurIPS 2025, https://arxiv.org/abs/2510.08602
- OpenAI model deprecations and base-model replacements: https://platform.openai.com/docs/deprecations and https://platform.openai.com/docs/models/davinci-002
- Pangram EditLens model: https://huggingface.co/pangram/editlens_Llama-3.2-3B
- Pangram EditLens repository: https://github.com/pangramlabs/EditLens
- EditLens paper: Thai et al., 2025, https://arxiv.org/abs/2510.03154
- PHD: Tulchinskii et al., 2023, https://arxiv.org/abs/2306.04723
- PHD upstream implementation: https://github.com/ArGintum/GPTID
- LogRank / generation detection by sampling: Ippolito et al., 2020, https://doi.org/10.18653/v1/2020.acl-main.164
- T5Sentinel: Chen et al., 2023, https://arxiv.org/abs/2311.08723
- T5Sentinel upstream implementation: https://github.com/MarkChenYutian/T5-Sentinel-public
- AIGC MPU: Tian et al., 2023, https://arxiv.org/abs/2305.18149
- AIGC MPU upstream implementation: https://github.com/YuchuanTian/AIGC_text_detector
- AIGC MPU latest English checkpoint: https://huggingface.co/yuchuantian/AIGC_detector_env3
- AIGC MPU HF Space inference mapping: https://huggingface.co/spaces/yuchuantian/AIGC_text_detector/blob/main/app.py
- MELD: Li et al., 2026, https://arxiv.org/abs/2605.06903
- MELD released checkpoint: https://huggingface.co/anon-review-meld-2026/meld
- RADAR: Hu et al., 2023, https://proceedings.neurips.cc/paper_files/paper/2023/file/30e15e5941ae0cdab7ef58cc8d59a4ca-Paper-Conference.pdf
- External detector benchmark reference: https://arxiv.org/pdf/2502.19614
- NAACL detector-eval reference harness: Tufts et al., 2025, https://aclanthology.org/2025.findings-naacl.271/
- NAACL detector-eval repository: https://github.com/LeiLiLab/llm-detector-eval
