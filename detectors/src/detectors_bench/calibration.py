from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_curve

from .schemas import Prediction


def choose_threshold_by_f1(predictions: list[Prediction]) -> float:
    rows = [p for p in predictions if p.label is not None and p.score_ai is not None and not p.error]
    if not rows:
        return 0.5
    y = np.asarray([p.label for p in rows], dtype=int)
    s = np.asarray([p.score_ai for p in rows], dtype=float)
    if len(np.unique(y)) < 2:
        return 0.5
    order = np.argsort(-s, kind="mergesort")
    sorted_scores = s[order]
    sorted_y = y[order]
    positives = int(np.sum(sorted_y))
    if positives == 0:
        return 0.5

    tp = np.cumsum(sorted_y)
    fp = np.cumsum(1 - sorted_y)
    score_change = np.r_[sorted_scores[1:] != sorted_scores[:-1], True]
    idx = np.where(score_change)[0]
    precision = tp[idx] / np.maximum(tp[idx] + fp[idx], 1)
    recall = tp[idx] / positives
    denom = precision + recall
    f1 = np.where(denom > 0, 2.0 * precision * recall / denom, 0.0)
    if len(f1) == 0:
        return 0.5
    return float(sorted_scores[idx[int(np.argmax(f1))]])


def choose_threshold_at_fpr(predictions: list[Prediction], target_fpr: float) -> float:
    rows = [p for p in predictions if p.label is not None and p.score_ai is not None and not p.error]
    if not rows:
        return 0.5
    y = np.asarray([p.label for p in rows], dtype=int)
    s = np.asarray([p.score_ai for p in rows], dtype=float)
    if len(np.unique(y)) < 2:
        return 0.5
    fpr, _, thresholds = roc_curve(y, s)
    valid = np.where(fpr <= target_fpr)[0]
    if len(valid) == 0:
        return float(np.max(s) + 1e-12)
    return float(thresholds[valid[-1]])
