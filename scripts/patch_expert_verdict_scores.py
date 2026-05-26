#!/usr/bin/env python3
"""Recompute verdict scores from inner tell scores (softmax-weighted) on expert-annot JSONL / HF."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset

from rl_detector.sft.verdict_score_agg import (
    patch_sft_text_verdict_score,
    verdict_score_from_annotation,
)

logger = logging.getLogger(__name__)


def _patch_row(
    row: dict[str, Any],
    beta: float,
    scale: float,
    tau: float,
) -> dict[str, Any] | None:
    if row.get("skip") or not row.get("annotation") or not row.get("sft_text"):
        return None
    label = int(row.get("label") or 0)
    ann = str(row["annotation"])
    old = float(row.get("verdict_score") or 0.0)
    new = verdict_score_from_annotation(
        annotation=ann,
        label=label,
        beta=beta,
        scale=scale,
        tau=tau,
    )
    if abs(new - old) < 1e-4:
        return None
    out = dict(row)
    out["verdict_score"] = new
    out["sft_text"] = patch_sft_text_verdict_score(sft_text=str(row["sft_text"]), verdict_score=new)
    return out


def _patch_jsonl(path: Path, beta: float, scale: float, tau: float) -> tuple[int, int]:
    rows = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    n_changed = 0
    for i, row in enumerate(rows):
        patched = _patch_row(row=row, beta=beta, scale=scale, tau=tau)
        if patched is not None:
            rows[i] = patched
            n_changed += 1
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return len(rows), n_changed


def _to_hf_row(r: dict[str, Any]) -> dict[str, Any]:
    return {
        "text": r["text"],
        "annotation": r["annotation"],
        "sft_text": str(r.get("sft_text") or ""),
        "label": int(r["label"]),
        "verdict_type": str(r.get("verdict_type") or ""),
        "verdict_score": float(r.get("verdict_score") or 0.0),
        "verdict_why": str(r.get("verdict_why") or ""),
        "example_id": str(r.get("example_id", "")),
        "source_id": str(r.get("source_id", "")),
        "source_row_index": r.get("source_row_index"),
        "source_sha256": str(r.get("source_sha256", "")),
        "generation_model_src": str(r.get("generation_model_src", "")),
        "annotator_idx": int(r.get("annotator_idx", 0)),
        "ground_truth": str(r.get("ground_truth", "")),
        "expert_majority_vote": str(r.get("expert_majority_vote", "")),
        "annotator_comment": str(r.get("annotator_comment", "")),
        "annotator_confidence": float(r.get("annotator_confidence") or 0.0),
        "annotator_guess": str(r.get("annotator_guess", "")),
        "generation_model": str(r.get("generation_model", "")),
    }


def _patch_hub(
    hub_repo: str,
    beta: float,
    scale: float,
    tau: float,
    jsonl_paths: list[Path],
) -> None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    rows: list[dict[str, Any]] = []
    for p in jsonl_paths:
        rows.extend(json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip())
    if not rows:
        rows = [dict(r) for r in load_dataset(hub_repo, split="train", token=token)]
    n_changed = 0
    out_rows: list[dict[str, Any]] = []
    for row in rows:
        patched = _patch_row(row=row, beta=beta, scale=scale, tau=tau)
        if patched is not None:
            n_changed += 1
            out_rows.append(patched)
        else:
            out_rows.append(row)
    hf_rows = [_to_hf_row(r) for r in out_rows if not r.get("skip") and r.get("annotation")]
    ds = DatasetDict({"train": Dataset.from_list(hf_rows)})
    ds.push_to_hub(hub_repo, token=token)
    logger.info("Pushed %d rows (%d verdict_score updates) to %s", len(hf_rows), n_changed, hub_repo)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    p = argparse.ArgumentParser(description="Patch verdict scores from inner tell scores")
    p.add_argument("--beta", type=float, required=True, help="softmax weight temperature (3.0 = training default)")
    p.add_argument("--scale", type=float, required=True, help="tanh scale for directional verdict (0.45 default)")
    p.add_argument("--tau", type=float, required=True, help="tanh temperature on signed agg (1.25 default)")
    p.add_argument("--paths", nargs="*", default=[], help="JSONL files to patch in place")
    p.add_argument("--push-to-hub", action="store_true")
    p.add_argument("--hub-repo", default="acmc/expert-annotated-TELL")
    args = p.parse_args()
    paths = [Path(x) for x in args.paths]
    for path in paths:
        n, ch = _patch_jsonl(path=path, beta=args.beta, scale=args.scale, tau=args.tau)
        logger.info("%s: %d rows, %d verdict_score updated", path, n, ch)
    if args.push_to_hub:
        _patch_hub(
            hub_repo=args.hub_repo,
            beta=args.beta,
            scale=args.scale,
            tau=args.tau,
            jsonl_paths=paths,
        )


if __name__ == "__main__":
    main()
