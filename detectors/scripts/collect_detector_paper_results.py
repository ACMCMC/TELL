from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any


METRIC_FIELDS = [
    "n_total",
    "n_scored",
    "n_errors",
    "auroc",
    "auprc",
    "tpr_at_fpr_0.001",
    "tpr_at_fpr_0.005",
    "tpr_at_fpr_0.01",
    "tpr_at_fpr_0.05",
    "fpr_at_tpr_0.8",
    "fpr_at_tpr_0.9",
    "fpr_at_tpr_0.95",
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "specificity",
    "f1",
    "mcc",
    "brier",
    "ece_10",
    "tn",
    "fp",
    "fn",
    "tp",
]

RUNTIME_FIELDS = [
    "runtime_mean_s",
    "runtime_median_s",
    "runtime_p90_s",
    "runtime_p95_s",
    "runtime_p99_s",
]

CI_METRICS = ["auroc", "f1", "tpr_at_fpr_0.01"]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


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


def runtime_summary(predictions_path: Path | None) -> dict[str, float]:
    if predictions_path is None or not predictions_path.exists():
        return {field: float("nan") for field in RUNTIME_FIELDS}
    rows = read_jsonl(predictions_path)
    values = [
        float(row["runtime_s"])
        for row in rows
        if row.get("runtime_s") is not None and row.get("error") is None
    ]
    if not values:
        return {field: float("nan") for field in RUNTIME_FIELDS}
    return {
        "runtime_mean_s": float(mean(values)),
        "runtime_median_s": quantile(values, 0.5),
        "runtime_p90_s": quantile(values, 0.9),
        "runtime_p95_s": quantile(values, 0.95),
        "runtime_p99_s": quantile(values, 0.99),
    }


def collect_root(root: Path, split_name: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    metrics_dir = root / "metrics"
    pred_dir = root / "merged_predictions"
    for metric_path in sorted(metrics_dir.glob("*.metrics.json")):
        detector = metric_path.name.removesuffix(".metrics.json")
        data = read_json(metric_path)
        overall = data.get("overall", {})
        ci = data.get("bootstrap_ci", {})
        pred_path = pred_dir / f"{detector}.predictions.jsonl"
        row: dict[str, Any] = {
            "detector": detector,
            "split": split_name,
            "output_root": str(root),
            "metrics_json": str(metric_path),
            "predictions_jsonl": str(pred_path) if pred_path.exists() else "",
        }
        for field in METRIC_FIELDS:
            row[field] = overall.get(field, "")
        for metric in CI_METRICS:
            metric_ci = ci.get(metric, {})
            row[f"{metric}_ci_lo"] = metric_ci.get("lo", "")
            row[f"{metric}_ci_hi"] = metric_ci.get("hi", "")
        row.update(runtime_summary(pred_path))
        out[detector] = row
    return out


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "detector",
        "test_n_scored",
        "test_n_errors",
        "test_auroc",
        "test_auroc_ci_lo",
        "test_auroc_ci_hi",
        "test_auprc",
        "test_tpr_at_fpr_0.001",
        "test_tpr_at_fpr_0.005",
        "test_tpr_at_fpr_0.01",
        "test_tpr_at_fpr_0.01_ci_lo",
        "test_tpr_at_fpr_0.01_ci_hi",
        "test_tpr_at_fpr_0.05",
        "test_fpr_at_tpr_0.95",
        "test_accuracy",
        "test_balanced_accuracy",
        "test_f1",
        "test_f1_ci_lo",
        "test_f1_ci_hi",
        "test_mcc",
        "test_ece_10",
        "test_runtime_median_s",
        "test_runtime_p95_s",
        "validation_auroc",
        "validation_auprc",
        "validation_tpr_at_fpr_0.01",
        "validation_f1",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in sorted(rows, key=lambda r: r["detector"]):
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect detector benchmark metrics into paper-ready artifacts.")
    parser.add_argument("--test-root", action="append", required=True, type=Path)
    parser.add_argument("--validation-root", action="append", type=Path, default=[])
    parser.add_argument("--output-tsv", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()

    test_rows: dict[str, dict[str, Any]] = {}
    for root in args.test_root:
        test_rows.update(collect_root(root, "test"))

    validation_rows: dict[str, dict[str, Any]] = {}
    for root in args.validation_root:
        validation_rows.update(collect_root(root, "validation"))

    combined = []
    for detector, test in sorted(test_rows.items()):
        row: dict[str, Any] = {"detector": detector}
        for key, value in test.items():
            if key == "detector":
                continue
            row[f"test_{key}"] = value
        val = validation_rows.get(detector, {})
        for key, value in val.items():
            if key == "detector":
                continue
            row[f"validation_{key}"] = value
        combined.append(row)

    write_tsv(args.output_tsv, combined)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(
            {
                "test_roots": [str(p) for p in args.test_root],
                "validation_roots": [str(p) for p in args.validation_root],
                "detectors": combined,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
