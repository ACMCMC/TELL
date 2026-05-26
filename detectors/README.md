# Detector Benchmark Harness

This folder is intentionally isolated from `src/rl_detector`. It contains pinned upstream detector repos, a thin scoring harness, metrics, smoke tests, and setup docs for paper benchmarking.

Typical flow:

```bash
cd detectors
python -m venv .venv
source .venv/bin/activate
pip install -e ".[hf,dev]"
python -m detectors_bench.run_detector \
  --detector openai_roberta \
  --input smoke/smoke.jsonl \
  --output results/smoke/openai_roberta.predictions.jsonl
python -m detectors_bench.run_benchmark \
  --predictions results/smoke/openai_roberta.predictions.jsonl \
  --output results/smoke/openai_roberta.metrics.json
```

See `docs/DETECTORS_SETUP.md` for the server setup, detector registry, and paper logging policy.
For running the full benchmark suite on a new JSONL dataset with two GPUs, use
`scripts/run_full_suite_2gpu.sh`; the runbook is
`docs/RUN_FULL_SUITE_2GPU.md`.

Gated detectors such as Pangram EditLens require access tokens in the runtime environment, for example `HF_TOKEN`, but tokens must never be committed.

## Collaborator Setup

The detector harness is this `detectors/` directory; no separate sibling
harness checkout is required. From a fresh checkout on the server:

```bash
cd /data/$USER/rl-detector
bash detectors/scripts/setup_server.sh "$PWD"
cd detectors
DETECTORS_PYTHON=/path/to/python bash scripts/smoke_all.sh
```

If `DETECTORS_PYTHON` is omitted, setup creates `detectors/.venv` when
`python3-venv` is available. Use environment variables such as `HF_TOKEN` only
in the shell running gated detectors; do not write tokens into tracked files.
