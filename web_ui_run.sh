#!/usr/bin/env bash
set -euo pipefail

RL_DETECTOR_LOCAL_DEV=1 PYTHONPATH=src uv run --no-sync uvicorn rl_detector.webui.app:app --host 0.0.0.0 --port "${PORT:-8000}"