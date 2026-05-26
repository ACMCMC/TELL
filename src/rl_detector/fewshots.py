"""Few-shot banks for annotation prompts, with deterministic rotation."""

from __future__ import annotations

import hashlib

FEWSHOT_SEED = 2242

# Few-shot strings live in fewshots_data.py (XML spans)
from rl_detector.fewshots_data import FEWSHOT_EXAMPLES


def pick_fewshot(main_text: str) -> str:
    """Pick one few-shot example deterministically from prompt inputs and seed."""
    key = f"{FEWSHOT_SEED}|{main_text[:128]}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    idx = int(digest[:8], 16) % len(FEWSHOT_EXAMPLES)
    return FEWSHOT_EXAMPLES[idx]
