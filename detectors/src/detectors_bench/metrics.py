from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from .schemas import Prediction


LOW_FPRS = (0.001, 0.005, 0.01, 0.05)
TARGET_TPRS = (0.8, 0.9, 0.95)


def _valid_arrays(predictions: list[Prediction]) -> tuple[np.ndarray, np.ndarray]:
    ys = []
    scores = []
    for p in predictions:
        if p.label is None or p.score_ai is None or p.error:
            continue
        if not np.isfinite(float(p.score_ai)):
            continue
        ys.append(int(p.label))
        scores.append(float(p.score_ai))
    return np.asarray(ys, dtype=int), np.asarray(scores, dtype=float)


def tpr_at_fpr(y_true: np.ndarray, scores: np.ndarray, target_fpr: float) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true, scores)
    valid = tpr[fpr <= target_fpr]
    return float(np.max(valid)) if len(valid) else 0.0


def fpr_at_tpr(y_true: np.ndarray, scores: np.ndarray, target_tpr: float) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true, scores)
    valid = fpr[tpr >= target_tpr]
    return float(np.min(valid)) if len(valid) else 1.0


def expected_calibration_error(y_true: np.ndarray, scores: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(scores)
    if total == 0:
        return float("nan")
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (scores >= lo) & (scores < hi if hi < 1.0 else scores <= hi)
        if not np.any(mask):
            continue
        conf = float(np.mean(scores[mask]))
        acc = float(np.mean(y_true[mask]))
        ece += (np.sum(mask) / total) * abs(acc - conf)
    return float(ece)


def binary_metrics(predictions: list[Prediction], threshold: float = 0.5) -> dict[str, Any]:
    y_true, scores = _valid_arrays(predictions)
    out: dict[str, Any] = {
        "n_total": len(predictions),
        "n_scored": int(len(scores)),
        "n_errors": int(sum(1 for p in predictions if p.error)),
        "threshold": threshold,
    }
    if len(scores) == 0:
        return out

    out.update(
        {
            "score_mean": float(np.mean(scores)),
            "score_std": float(np.std(scores)),
            "score_min": float(np.min(scores)),
            "score_max": float(np.max(scores)),
        }
    )
    if len(np.unique(y_true)) < 2:
        out["warning"] = "Only one label class present; ROC/PR metrics undefined."
        return out

    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    out.update(
        {
            "auroc": float(roc_auc_score(y_true, scores)),
            "auprc": float(average_precision_score(y_true, scores)),
            "accuracy": float(accuracy_score(y_true, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
            "precision": float(precision_score(y_true, pred, zero_division=0)),
            "recall": float(recall_score(y_true, pred, zero_division=0)),
            "f1": float(f1_score(y_true, pred, zero_division=0)),
            "mcc": float(matthews_corrcoef(y_true, pred)),
            "specificity": float(tn / max(tn + fp, 1)),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "brier": float(brier_score_loss(y_true, np.clip(scores, 0.0, 1.0))),
            "ece_10": expected_calibration_error(y_true, np.clip(scores, 0.0, 1.0), bins=10),
        }
    )
    for target in LOW_FPRS:
        out[f"tpr_at_fpr_{target:g}"] = tpr_at_fpr(y_true, scores, target)
    for target in TARGET_TPRS:
        out[f"fpr_at_tpr_{target:g}"] = fpr_at_tpr(y_true, scores, target)
    return out


def grouped_metrics(predictions: list[Prediction], threshold: float = 0.5) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[Prediction]] = defaultdict(list)
    for p in predictions:
        for field in ("split", "dataset", "domain", "generator", "attack"):
            value = getattr(p, field)
            if value is not None:
                groups[f"{field}={value}"].append(p)
    return {name: binary_metrics(rows, threshold=threshold) for name, rows in sorted(groups.items())}


def bootstrap_ci(
    predictions: list[Prediction],
    metric: str = "auroc",
    threshold: float = 0.5,
    n_bootstrap: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    usable = [p for p in predictions if p.label is not None and p.score_ai is not None and not p.error]
    if len(usable) < 2:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_bootstrap):
        sample = [usable[i] for i in rng.integers(0, len(usable), size=len(usable))]
        value = binary_metrics(sample, threshold=threshold).get(metric)
        if value is not None and np.isfinite(value):
            vals.append(float(value))
    if not vals:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
    arr = np.asarray(vals)
    return {
        "mean": float(np.mean(arr)),
        "lo": float(np.quantile(arr, 0.025)),
        "hi": float(np.quantile(arr, 0.975)),
    }
