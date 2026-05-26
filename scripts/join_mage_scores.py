"""Join MAGE predictions back to split JSONL files and normalize mage_score.

Reads merged mage_d predictions (id → score_ai) and attaches mage_score to
each row in the split JSONL files. Normalizes scores to [0,1] using the
training split min/max. If --adversarial-splits-dir and --adversarial-mage-dir
are provided, adversarial rows are appended to the final splits.

Usage:
    python scripts/join_mage_scores.py \\
        --splits-dir data/balanced-splits-v1/splits \\
        --mage-dir data/balanced-splits-v1/mage_scores \\
        --output-dir data/balanced-splits-v1/final \\
        [--adversarial-splits-dir data/balanced-splits-v1/adversarial_splits] \\
        [--adversarial-mage-dir data/balanced-splits-v1/adversarial_mage_scores]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_predictions(mage_dir: Path, split: str) -> dict[str, float]:
    merged = mage_dir / split / "merged_predictions" / "mage_d.predictions.jsonl"
    if not merged.exists():
        raise FileNotFoundError(f"Merged predictions not found: {merged}")
    scores: dict[str, float] = {}
    with merged.open() as fh:
        for line in fh:
            row = json.loads(line)
            scores[row["id"]] = float(row["score_ai"])
    print(f"  Loaded {len(scores):,} MAGE scores for {split}", flush=True)
    return scores


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  {len(rows):,} rows → {path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", required=True, type=Path)
    parser.add_argument("--mage-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--adversarial-splits-dir", type=Path, default=None)
    parser.add_argument("--adversarial-mage-dir", type=Path, default=None)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_scores: dict[str, dict[str, float]] = {}
    for split in ("train", "val", "test"):
        all_scores[split] = _load_predictions(args.mage_dir, split)

    train_scores = list(all_scores["train"].values())
    score_min = min(train_scores)
    score_max = max(train_scores)
    score_range = score_max - score_min
    print(f"Train mage_score range: [{score_min:.4f}, {score_max:.4f}]", flush=True)

    def normalize(score: float) -> float:
        if score_range == 0:
            return 0.5
        return (score - score_min) / score_range

    missing_total = 0
    for split in ("train", "val", "test"):
        rows = _load_jsonl(args.splits_dir / f"{split}.jsonl")
        scores = all_scores[split]
        missing = 0
        for row in rows:
            raw = scores.get(row["id"])
            if raw is None:
                missing += 1
                row["mage_score"] = None
            else:
                row["mage_score"] = normalize(raw)
        if missing:
            print(f"  WARNING: {missing} rows in {split} had no MAGE prediction", flush=True)
            missing_total += missing

        # Append adversarial rows if provided
        if args.adversarial_splits_dir and args.adversarial_mage_dir:
            adv_path = args.adversarial_splits_dir / f"{split}.jsonl"
            adv_scores = _load_predictions(args.adversarial_mage_dir, split)
            adv_rows = _load_jsonl(adv_path)
            adv_missing = 0
            for row in adv_rows:
                raw = adv_scores.get(row["id"])
                if raw is None:
                    adv_missing += 1
                    row["mage_score"] = None
                else:
                    row["mage_score"] = normalize(raw)
            if adv_missing:
                print(f"  WARNING: {adv_missing} adversarial rows in {split} had no MAGE prediction", flush=True)
                missing_total += adv_missing
            rows.extend(adv_rows)

        _write_jsonl(args.output_dir / f"{split}.jsonl", rows)

    summary = {
        "score_min": score_min,
        "score_max": score_max,
        "missing_rows": missing_total,
    }
    (args.output_dir / "mage_join_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
