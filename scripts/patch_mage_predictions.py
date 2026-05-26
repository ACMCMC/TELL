"""Append new MAGE predictions from rescore mini-shards into existing merged files.

After a rescore run (mini-shards for missing rows), this script:
  1. Loads new predictions from rescore_mage/*/predictions_sharded/mage_d/*.jsonl
  2. Appends them to the existing merged_predictions/mage_d.predictions.jsonl
  3. Deduplicates by ID (new rows win)

Usage:
    python scripts/patch_mage_predictions.py \
        --mage-dir data/balanced-splits-v1/mage_scores \
        --rescore-dir data/balanced-splits-v1/rescore_mage
"""
from __future__ import annotations
import argparse
import glob
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mage-dir", required=True, type=Path)
    parser.add_argument("--rescore-dir", required=True, type=Path)
    args = parser.parse_args()

    for split in ("train", "val", "test"):
        merged_path = args.mage_dir / split / "merged_predictions" / "mage_d.predictions.jsonl"
        if not merged_path.exists():
            print(f"{split}: merged predictions not found at {merged_path}, skipping")
            continue

        rescore_preds = list((args.rescore_dir / split / "predictions_sharded" / "mage_d").glob("*.jsonl"))
        if not rescore_preds:
            print(f"{split}: no rescore predictions found, skipping")
            continue

        new_scores: dict[str, dict] = {}
        for path in rescore_preds:
            with path.open() as fh:
                for line in fh:
                    row = json.loads(line)
                    new_scores[row["id"]] = row

        existing: dict[str, dict] = {}
        with merged_path.open() as fh:
            for line in fh:
                row = json.loads(line)
                existing[row["id"]] = row

        n_before = len(existing)
        existing.update(new_scores)
        n_after = len(existing)
        n_added = sum(1 for k in new_scores if k not in existing or True)

        with merged_path.open("w", encoding="utf-8") as fh:
            for row in existing.values():
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

        print(f"{split}: {n_before:,} → {n_after:,} rows (+{len(new_scores):,} new/updated)")


if __name__ == "__main__":
    main()
