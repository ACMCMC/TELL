from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_curve

from .calibration import choose_threshold_at_fpr, choose_threshold_by_f1
from .io import load_predictions, write_json
from .metrics import binary_metrics, bootstrap_ci, grouped_metrics


def _threshold_from_args(args) -> tuple[float, dict]:
    if args.threshold_policy == "fixed":
        return float(args.threshold), {"policy": "fixed", "threshold": float(args.threshold)}
    if not args.validation_predictions:
        raise ValueError("--validation-predictions is required for non-fixed threshold policies.")
    val = load_predictions(args.validation_predictions)
    if args.threshold_policy == "f1":
        t = choose_threshold_by_f1(val)
    elif args.threshold_policy == "fpr01":
        t = choose_threshold_at_fpr(val, 0.01)
    else:
        raise ValueError(f"Unknown threshold policy: {args.threshold_policy}")
    return t, {"policy": args.threshold_policy, "threshold": t, "validation_predictions": str(args.validation_predictions)}


def _write_roc_points(path: Path, predictions) -> None:
    rows = [
        p
        for p in predictions
        if p.label is not None and p.score_ai is not None and not p.error and np.isfinite(float(p.score_ai))
    ]
    if not rows:
        return
    y = np.asarray([p.label for p in rows], dtype=int)
    s = np.asarray([p.score_ai for p in rows], dtype=float)
    if len(np.unique(y)) < 2:
        return
    fpr, tpr, thresholds = roc_curve(y, s)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["fpr", "tpr", "threshold"])
        writer.writeheader()
        for a, b, c in zip(fpr, tpr, thresholds):
            writer.writerow({"fpr": float(a), "tpr": float(b), "threshold": float(c)})


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compute paper detector metrics from prediction JSONL.")
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--threshold-policy", choices=["fixed", "f1", "fpr01"], default="fixed")
    parser.add_argument("--validation-predictions", type=Path)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    predictions = load_predictions(args.predictions)
    threshold, threshold_info = _threshold_from_args(args)
    out = {
        "predictions": str(args.predictions),
        "threshold": threshold_info,
        "overall": binary_metrics(predictions, threshold=threshold),
        "groups": grouped_metrics(predictions, threshold=threshold),
    }
    if args.bootstrap > 0:
        out["bootstrap_ci"] = {
            "auroc": bootstrap_ci(predictions, "auroc", threshold, args.bootstrap, args.seed),
            "f1": bootstrap_ci(predictions, "f1", threshold, args.bootstrap, args.seed),
            "tpr_at_fpr_0.01": bootstrap_ci(predictions, "tpr_at_fpr_0.01", threshold, args.bootstrap, args.seed),
        }
    write_json(args.output, out)
    _write_roc_points(args.output.with_suffix(".roc_points.tsv"), predictions)


if __name__ == "__main__":
    main()
