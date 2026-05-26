from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = [
    "detector",
    "n_scored",
    "n_errors",
    "auroc",
    "auprc",
    "accuracy",
    "balanced_accuracy",
    "f1",
    "mcc",
    "tpr_at_fpr_0.01",
    "fpr_at_tpr_0.95",
]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Collect metric JSON files into one TSV.")
    parser.add_argument("--metrics-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    rows = []
    for path in sorted(args.metrics_dir.glob("*.metrics.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        overall = data.get("overall", {})
        detector = path.name.replace(".metrics.json", "")
        row = {"detector": detector}
        row.update({field: overall.get(field, "") for field in FIELDS if field != "detector"})
        rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
