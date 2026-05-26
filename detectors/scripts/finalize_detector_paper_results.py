from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any

from detectors_bench.calibration import choose_threshold_at_fpr, choose_threshold_by_f1
from detectors_bench.io import load_predictions
from detectors_bench.metrics import binary_metrics, bootstrap_ci, grouped_metrics
from detectors_bench.schemas import Prediction


def collect_predictions(roots: list[Path]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for root in roots:
        pred_dir = root / "merged_predictions"
        for path in sorted(pred_dir.glob("*.predictions.jsonl")):
            detector = path.name.removesuffix(".predictions.jsonl")
            out[detector] = path
    return out


def quantile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    pos = (len(xs) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return float(xs[lo] * (1.0 - frac) + xs[hi] * frac)


def runtime_summary(predictions: list[Prediction]) -> dict[str, float]:
    values = [
        float(row.runtime_s)
        for row in predictions
        if row.runtime_s is not None and row.error is None
    ]
    if not values:
        return {
            "runtime_mean_s": float("nan"),
            "runtime_median_s": float("nan"),
            "runtime_p90_s": float("nan"),
            "runtime_p95_s": float("nan"),
            "runtime_p99_s": float("nan"),
        }
    return {
        "runtime_mean_s": float(mean(values)),
        "runtime_median_s": quantile(values, 0.5),
        "runtime_p90_s": quantile(values, 0.9),
        "runtime_p95_s": quantile(values, 0.95),
        "runtime_p99_s": quantile(values, 0.99),
    }


def metric_at(metrics: dict[str, Any], key: str) -> Any:
    return metrics.get(key, "")


def build_detector_record(
    detector: str,
    test_path: Path,
    validation_path: Path | None,
    bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    test = load_predictions(test_path)
    validation = load_predictions(validation_path) if validation_path is not None else []

    fixed_threshold = 0.5
    f1_threshold = choose_threshold_by_f1(validation) if validation else fixed_threshold
    fpr01_threshold = choose_threshold_at_fpr(validation, 0.01) if validation else fixed_threshold

    test_fixed = binary_metrics(test, threshold=fixed_threshold)
    test_val_f1 = binary_metrics(test, threshold=f1_threshold)
    test_val_fpr01 = binary_metrics(test, threshold=fpr01_threshold)
    validation_fixed = binary_metrics(validation, threshold=fixed_threshold) if validation else {}
    validation_val_f1 = binary_metrics(validation, threshold=f1_threshold) if validation else {}
    validation_val_fpr01 = binary_metrics(validation, threshold=fpr01_threshold) if validation else {}

    ci = {}
    if bootstrap > 0:
        ci = {
            "auroc": bootstrap_ci(test, "auroc", fixed_threshold, bootstrap, seed),
            "f1_fixed": bootstrap_ci(test, "f1", fixed_threshold, bootstrap, seed),
            "tpr_at_fpr_0.01": bootstrap_ci(
                test,
                "tpr_at_fpr_0.01",
                fixed_threshold,
                bootstrap,
                seed,
            ),
        }

    return {
        "detector": detector,
        "test_predictions": str(test_path),
        "validation_predictions": str(validation_path) if validation_path is not None else "",
        "thresholds": {
            "fixed": fixed_threshold,
            "validation_f1": f1_threshold,
            "validation_fpr01": fpr01_threshold,
        },
        "test_fixed": test_fixed,
        "test_validation_f1_threshold": test_val_f1,
        "test_validation_fpr01_threshold": test_val_fpr01,
        "validation_fixed": validation_fixed,
        "validation_validation_f1_threshold": validation_val_f1,
        "validation_validation_fpr01_threshold": validation_val_fpr01,
        "test_groups_fixed": grouped_metrics(test, threshold=fixed_threshold),
        "runtime": runtime_summary(test),
        "bootstrap_ci": ci,
    }


def flatten_record(record: dict[str, Any]) -> dict[str, Any]:
    fixed = record["test_fixed"]
    val_f1 = record["test_validation_f1_threshold"]
    val_fpr01 = record["test_validation_fpr01_threshold"]
    validation = record["validation_fixed"]
    runtime = record["runtime"]
    ci = record["bootstrap_ci"]
    tpr01_ci = ci.get("tpr_at_fpr_0.01", {})
    auroc_ci = ci.get("auroc", {})
    f1_ci = ci.get("f1_fixed", {})
    val_fpr01_fpr = ""
    if "specificity" in val_fpr01:
        val_fpr01_fpr = 1.0 - float(val_fpr01["specificity"])

    return {
        "detector": record["detector"],
        "test_n_scored": metric_at(fixed, "n_scored"),
        "test_n_errors": metric_at(fixed, "n_errors"),
        "test_auroc": metric_at(fixed, "auroc"),
        "test_auroc_ci_lo": auroc_ci.get("lo", ""),
        "test_auroc_ci_hi": auroc_ci.get("hi", ""),
        "test_auprc": metric_at(fixed, "auprc"),
        "test_tpr_at_fpr_0.001": metric_at(fixed, "tpr_at_fpr_0.001"),
        "test_tpr_at_fpr_0.005": metric_at(fixed, "tpr_at_fpr_0.005"),
        "test_tpr_at_fpr_0.01": metric_at(fixed, "tpr_at_fpr_0.01"),
        "test_tpr_at_fpr_0.01_ci_lo": tpr01_ci.get("lo", ""),
        "test_tpr_at_fpr_0.01_ci_hi": tpr01_ci.get("hi", ""),
        "test_tpr_at_fpr_0.05": metric_at(fixed, "tpr_at_fpr_0.05"),
        "test_fpr_at_tpr_0.8": metric_at(fixed, "fpr_at_tpr_0.8"),
        "test_fpr_at_tpr_0.9": metric_at(fixed, "fpr_at_tpr_0.9"),
        "test_fpr_at_tpr_0.95": metric_at(fixed, "fpr_at_tpr_0.95"),
        "test_fixed_accuracy": metric_at(fixed, "accuracy"),
        "test_fixed_balanced_accuracy": metric_at(fixed, "balanced_accuracy"),
        "test_fixed_f1": metric_at(fixed, "f1"),
        "test_fixed_f1_ci_lo": f1_ci.get("lo", ""),
        "test_fixed_f1_ci_hi": f1_ci.get("hi", ""),
        "test_fixed_mcc": metric_at(fixed, "mcc"),
        "test_fixed_ece_10": metric_at(fixed, "ece_10"),
        "val_f1_threshold": record["thresholds"]["validation_f1"],
        "val_f1_test_accuracy": metric_at(val_f1, "accuracy"),
        "val_f1_test_balanced_accuracy": metric_at(val_f1, "balanced_accuracy"),
        "val_f1_test_f1": metric_at(val_f1, "f1"),
        "val_f1_test_mcc": metric_at(val_f1, "mcc"),
        "val_f1_test_precision": metric_at(val_f1, "precision"),
        "val_f1_test_recall": metric_at(val_f1, "recall"),
        "val_f1_test_specificity": metric_at(val_f1, "specificity"),
        "val_fpr01_threshold": record["thresholds"]["validation_fpr01"],
        "val_fpr01_test_recall": metric_at(val_fpr01, "recall"),
        "val_fpr01_test_fpr": val_fpr01_fpr,
        "val_fpr01_test_precision": metric_at(val_fpr01, "precision"),
        "val_fpr01_test_f1": metric_at(val_fpr01, "f1"),
        "validation_auroc": metric_at(validation, "auroc"),
        "validation_auprc": metric_at(validation, "auprc"),
        "validation_tpr_at_fpr_0.01": metric_at(validation, "tpr_at_fpr_0.01"),
        "runtime_mean_s": runtime["runtime_mean_s"],
        "runtime_median_s": runtime["runtime_median_s"],
        "runtime_p90_s": runtime["runtime_p90_s"],
        "runtime_p95_s": runtime["runtime_p95_s"],
        "runtime_p99_s": runtime["runtime_p99_s"],
        "test_predictions": record["test_predictions"],
        "validation_predictions": record["validation_predictions"],
    }


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else ["detector"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build final paper detector tables from merged predictions.")
    parser.add_argument("--test-root", action="append", required=True, type=Path)
    parser.add_argument("--validation-root", action="append", default=[], type=Path)
    parser.add_argument("--output-tsv", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--per-detector-dir", type=Path)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    test_paths = collect_predictions(args.test_root)
    validation_paths = collect_predictions(args.validation_root)
    records = [
        build_detector_record(
            detector,
            test_path,
            validation_paths.get(detector),
            bootstrap=args.bootstrap,
            seed=args.seed,
        )
        for detector, test_path in sorted(test_paths.items())
    ]
    flat = [flatten_record(record) for record in records]

    write_tsv(args.output_tsv, flat)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(
            {
                "test_roots": [str(path) for path in args.test_root],
                "validation_roots": [str(path) for path in args.validation_root],
                "bootstrap": args.bootstrap,
                "seed": args.seed,
                "detectors": records,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if args.per_detector_dir is not None:
        args.per_detector_dir.mkdir(parents=True, exist_ok=True)
        for record in records:
            out = args.per_detector_dir / f"{record['detector']}.paper_metrics.json"
            out.write_text(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
