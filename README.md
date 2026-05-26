# TELL: Show, Don't TELL — Explainable AI-Generated Text Detection

**Demo:** [ai-tells.tech](https://ai-tells.tech)  
**Model weights:** [acmc/TELL](https://huggingface.co/acmc/TELL)  
**Training data:** [acmc/multi_domain_ai_human_text](https://huggingface.co/datasets/acmc/multi_domain_ai_human_text)  
**Win-rate eval data:** [acmc/expert-annotated-TELL](https://huggingface.co/datasets/acmc/expert-annotated-TELL)

---

TELL is an explainable AI-generated text detector. Instead of returning a bare score, it annotates specific spans of the input and explains *why* each one is evidence of AI or human authorship. The aggregate of those span-level judgments produces the final verdict.

```
Input:
  "NFS [...] uses two servers [...] with one acting as both the server
   and the other as the client"

TELL output (86% AI):
  Span: "acting as both the server and the other as the client"
  Type: AI  |  Score: 0.82
  Why:  "clear contradiction — a server cannot act as both the server
         and the client in the same sentence; the model mixed up the
         architecture while trying to summarize it"
```

TELL achieves AUROC 0.927 on the paper benchmark while providing human-readable annotations that win 72.3% of blind comparisons against expert human annotations on concreteness, falsifiability, coherence, plausibility, and grounding.

---

## Table of contents

- [Infrastructure requirements](#infrastructure-requirements)
- [Setup](#setup)
- [Running inference](#running-inference)
- [Training from scratch](#training-from-scratch)
  - [Step 0 — Configure](#step-0--configure)
  - [Step 1 — Generate SFT data](#step-1--generate-sft-data)
  - [Step 2 — Run SFT](#step-2--run-sft)
  - [Step 3 — Run RL (GRPO)](#step-3--run-rl-grpo)
- [Reproducing the baseline detector benchmark](#reproducing-the-baseline-detector-benchmark)
- [Win-rate evaluation](#win-rate-evaluation)
- [Tests](#tests)
- [Repository layout](#repository-layout)

---

## Infrastructure requirements

TELL uses [Tinker](https://tinkercorp.com) for model serving and distributed training. You will need:

- A **Tinker API key** (`TINKER_API_KEY`) — required for all training and inference
- An **xAI or OpenAI API key** — required only during RL training when `training.use_rubric_scorer: true` (the frozen LLM judge)
- A **HuggingFace token** (`HF_TOKEN`) — required to access gated datasets and models (e.g. Pangram EditLens)
- Python 3.11+ and [`uv`](https://github.com/astral-sh/uv)

To run inference only against the released weights (`acmc/TELL`), you need only a Tinker API key.

---

## Setup

```bash
git clone https://github.com/ACMCMC/TELL
cd TELL
uv sync
```

Create a `.env` file in the repo root (it is git-ignored):

```bash
TINKER_API_KEY=...
XAI_API_KEY=...        # for the LLM judge during RL training
OPENAI_API_KEY=...     # alternative judge provider
HF_TOKEN=...           # for gated HuggingFace datasets/models
```

All scripts call `load_dotenv()` at startup, so this file is loaded automatically.

---

## Running inference

### Annotate from stdin

```bash
echo "Your text here." | uv run python -m rl_detector.annotate
```

The checkpoint used is whatever `model.checkpoint` is set to in `conf/config.yaml` (defaults to `"acmc/TELL"`). To use a different checkpoint, pass it as a positional argument:

```bash
echo "Your text here." | uv run python -m rl_detector.annotate "tinker://<run-id>:train:0/sampler_weights/final"
```

Output is JSON on stdout:

```json
{
  "verdict": "AI",
  "aggregate_score": 0.86,
  "indicators": [
    {
      "span_text": "acting as both the server and the other as the client",
      "type": "AI",
      "explanation": "clear contradiction ...",
      "signed_score": 0.82
    }
  ],
  "annotated_text": "..."
}
```

---

## Training from scratch

### Step 0 — Configure

Open `conf/config.yaml` and set the following before doing anything else:

| Key | What to set |
|-----|-------------|
| `model.base_model` | The pretrained base model to fine-tune (we used `"openai/gpt-oss-120b"`) |
| `model.checkpoint` | Same as `base_model` initially; will be updated to your SFT output after Step 2 |
| `run.resume` | `null` for a fresh run; or your SFT checkpoint path after Step 2 |
| `training.kl_reference_checkpoint` | Must point to the SFT checkpoint (set after Step 2, before Step 3) |
| `wandb.entity` | Your W&B username or team (or leave `null` to disable W&B) |
| `sft.expert_annot_dataset_path` | `"hf://acmc/expert-annotated-TELL"` (already set) |
| `data.train_docs_path` / `data.eval_docs_path` | `"hf://acmc/multi_domain_ai_human_text/train"` / `"…/validation"` (already set) |

All other hyperparameters match the paper (see Appendix C for the exact settings used).

### Step 1 — Generate SFT data

TELL's SFT stage requires annotated span examples. We generate them from two sources:

**Source A** — paired AI/human examples from the EditLens corpus:

```bash
uv run python -m rl_detector.sft.generate_editlens
```

Writes to `sft.output_examples` and `sft.output_flat` (see `sft_editlens` section of `conf/config.yaml`). Uses GPT-5.5 to generate span-level annotations; requires `OPENAI_API_KEY`.

**Source B** — human expert annotations from Russell et al. (2025):

```bash
uv run python -m rl_detector.sft.generate_human_annot_sft
```

Reads from `sft.expert_annot_dataset_path` and generates formatted SFT examples. Also requires `OPENAI_API_KEY`.

> **Tip:** You can skip Step 1 entirely and train only on the expert annotation dataset (`sft.dataset_path: ""` is already the default, which skips the EditLens portion). The expert annotation set is smaller but sufficient to learn the annotation format.

### Step 2 — Run SFT

```bash
uv run python -m rl_detector.sft.train_tinker_sft
```

Key Hydra overrides:

```bash
uv run python -m rl_detector.sft.train_tinker_sft \
  sft.epochs=1 \
  sft.batch_size=8 \
  wandb.name=tell_sft
```

When it finishes, note the checkpoint path printed at the end (a Tinker path like `tinker://<run-id>:train:0/weights/sft-final`). Then update `conf/config.yaml`:

```yaml
run:
  resume: "tinker://<your-sft-run-id>:train:0/weights/sft-final"
model:
  checkpoint: "tinker://<your-sft-run-id>:train:0/weights/sft-final"
training:
  kl_reference_checkpoint: "tinker://<your-sft-run-id>:train:0/weights/sft-final"
```

### Step 3 — Run RL (GRPO)

```bash
uv run python -m rl_detector.train
```

Override individual config values without editing the file:

```bash
uv run python -m rl_detector.train \
  run.run_name=tell_rl \
  training.max_steps=310 \
  wandb.name=tell_rl
```

Training checkpoints and audit logs are written to `runs/<datetime>_<run_name>/`. The best checkpoint (by validation AUROC) is saved as `runs/.../weights/best-step-N`.

To run evaluation only on an existing checkpoint:

```bash
uv run python -m rl_detector.train \
  run.eval_only=true \
  run.checkpoint="tinker://<run-id>:train:0/weights/best-step-390"
```

#### Building the training corpus from scratch

The corpus `acmc/multi_domain_ai_human_text` is published on HuggingFace and used directly by the training code. To rebuild it from the public source datasets yourself:

```bash
# Aggregate public source datasets into a single parquet
uv run python scripts/build_unified_dataset.py --output-dir data/unified-v3

# Generate balanced splits and score with MAGE (requires 2 GPUs)
bash scripts/run_prepare_training_data.sh
```

See [`scripts/run_prepare_training_data.sh`](scripts/run_prepare_training_data.sh) for the full pipeline.

---

## Reproducing the baseline detector benchmark

The `detectors/` directory is a self-contained harness for the 16 baseline detectors in Table 2 of the paper. Pre-computed predictions and per-domain metrics are already in `detectors/results/` — you do not need to rerun them to reproduce the paper numbers.

To run the full benchmark on a new evaluation set, first set up the harness:

```bash
cd detectors
bash scripts/setup_server.sh "$PWD"   # initializes vendor submodules + installs harness
```

Then run a single detector:

```bash
python -m detectors_bench.run_detector \
  --detector openai_roberta \
  --input path/to/eval.jsonl \
  --output results/openai_roberta.predictions.jsonl

python -m detectors_bench.run_benchmark \
  --predictions results/openai_roberta.predictions.jsonl \
  --output results/openai_roberta.metrics.json
```

For the full suite on two GPUs:

```bash
bash scripts/run_full_suite_2gpu.sh \
  --input path/to/eval.jsonl \
  --output-root path/to/output \
  --gpus 0,1 \
  --shards 32
```

Input JSONL format: `{"id":"doc_1","text":"...","label":1,"dataset":"raid","domain":"news"}` (label: 0=human, 1=AI).

See [`detectors/docs/DETECTORS_SETUP.md`](detectors/docs/DETECTORS_SETUP.md) for the full guide including vendor submodule setup, per-detector dependency installation, and smoke testing.

---

## Win-rate evaluation

This reproduces Table 3 of the paper: TELL's annotations vs. human expert annotations, judged blindly by a panel of LLMs.

```bash
uv run python -m rl_detector.eval_human_detectors \
  --checkpoint-path "tinker://<run-id>:train:0/weights/best-step-390" \
  --dataset-url "hf://acmc/expert-annotated-TELL/validation" \
  --sample-size 200 \
  --workers 32 \
  --judge-model grok-4-1-fast-reasoning \
  --output-dir results/winrate \
  --run-name tell_winrate
```

Key flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint-path` | base model | Tinker checkpoint or `null` for base model |
| `--dataset-url` | `hf://acmc/expert-annotated-TELL/validation` | HF dataset split to evaluate on |
| `--sample-size` | 25 | Number of documents to evaluate |
| `--judge-model` | `grok-4-1-fast-reasoning` | LLM judge model name |
| `--judge-base-url` | xAI API | Base URL for the judge API |
| `--workers` | 32 | Parallel workers |
| `--output-dir` | `human_detectors_eval` | Where to write results |

---

## Tests

```bash
uv run pytest tests/
```

The test suite covers annotation parsing, format fixing, reward computation, per-token advantage decomposition, and data sampling. No API keys are required.

---

## Repository layout

```
TELL/
├── conf/
│   └── config.yaml             # all hyperparameters (Hydra) — edit before training
├── src/rl_detector/
│   ├── train.py                # GRPO training loop
│   ├── data.py                 # dataset loading and stratified sampling
│   ├── rewards.py              # reward functions and per-token advantage computation
│   ├── rollouts.py             # rollout generation and token-type masking
│   ├── format_fix.py           # format-fixing pipeline for malformed rollouts
│   ├── frozen.py               # frozen LLM judge (rubric scorer) and self-scoring
│   ├── prompts.py              # prompt templates and few-shot examples
│   ├── annotate.py             # inference entry point
│   ├── eval_runner.py          # evaluation loop (called during training)
│   ├── eval_human_detectors.py # win-rate evaluation vs. human expert annotations
│   ├── tell_xml.py             # annotation format parsing and serialization
│   ├── format_fix.py           # format-fixing pipeline
│   └── sft/
│       ├── train_tinker_sft.py    # SFT training script
│       ├── generate_editlens.py   # generate SFT examples from EditLens pairs
│       └── generate_human_annot_sft.py  # generate SFT examples from human annotations
├── detectors/                  # baseline detector benchmark harness (self-contained)
│   ├── src/detectors_bench/    # harness library: wrappers, metrics, IO
│   ├── scripts/                # launch, merge, and smoke-test scripts
│   ├── results/                # pre-computed baseline predictions and metrics
│   └── docs/                   # setup guides (DETECTORS_SETUP.md, RUN_FULL_SUITE_2GPU.md)
├── scripts/                    # data preparation and analysis scripts
│   ├── build_unified_dataset.py    # aggregate public source datasets
│   ├── sample_balanced_splits.py   # create balanced train/val/test splits
│   ├── bootstrap_eval.py           # bootstrap CI computation for paper tables
│   └── run_prepare_training_data.sh  # end-to-end data pipeline
└── tests/                      # unit and integration tests
```
