"""Causal faithfulness metrics for TELL outputs.

The functions in this module are deliberately pure: they operate on parsed
examples and a caller-provided score_fn(texts) -> P(AI).  The CLI handles model
calls, audit-log parsing, and file output.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Callable, Iterable, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


@dataclass(frozen=True)
class Tell:
    start: int
    end: int
    polarity: int
    score: float
    span_text: str
    explanation: str
    tell_type: str

    @property
    def signed_score(self) -> float:
        return float(self.polarity) * float(self.score)


@dataclass(frozen=True)
class Example:
    doc_id: str
    text: str
    y: int
    tells: list[Tell]


GENERIC_PHRASES = [
    "too polished",
    "generic phrasing",
    "sounds generic",
    "ai-like",
    "lacks personality",
    "overly formal",
    "formulaic",
    "unnatural",
    "repetitive",
    "predictable structure",
    "typical of human writing",
    "typical of ai writing",
    "typical of ai-generated",
    "typical of machine-generated",
    "common in ai",
    "common in human",
]


def label_to_signed(label: int) -> int:
    return 1 if int(label) == 1 else -1


def label01(ys: Sequence[int]) -> np.ndarray:
    return (np.asarray(ys) == 1).astype(int)


def gold_confidence(ai_probs: Sequence[float], ys: Sequence[int]) -> np.ndarray:
    probs = np.asarray(ai_probs, dtype=float)
    signed = np.asarray(ys, dtype=int)
    return np.where(signed == 1, probs, 1.0 - probs)


def tpr_at_fpr(ys: Sequence[int], ai_probs: Sequence[float], target_fpr: float = 0.01) -> float:
    y01 = label01(ys)
    if len(np.unique(y01)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y01, np.asarray(ai_probs, dtype=float))
    vals = tpr[fpr <= target_fpr]
    return float(np.max(vals)) if len(vals) else 0.0


def classification_metrics(ys: Sequence[int], ai_probs: Sequence[float]) -> dict[str, float]:
    y01 = label01(ys)
    probs = np.asarray(ai_probs, dtype=float)
    if len(np.unique(y01)) < 2:
        return {"auroc": float("nan"), "tpr_at_fpr_0.01": float("nan")}
    return {
        "auroc": float(roc_auc_score(y01, probs)),
        "tpr_at_fpr_0.01": tpr_at_fpr(ys, probs, 0.01),
    }


def trapezoid_area(y: Sequence[float], x: Sequence[float]) -> float:
    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)
    if len(xs) < 2 or len(xs) != len(ys):
        return float("nan")
    widths = xs[1:] - xs[:-1]
    heights = 0.5 * (ys[1:] + ys[:-1])
    return float(np.sum(widths * heights))


def merge_intervals(intervals: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    clean = sorted((max(0, int(s)), max(0, int(e))) for s, e in intervals if int(e) > int(s))
    if not clean:
        return []
    merged = [clean[0]]
    for start, end in clean[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _overlaps(candidate: tuple[int, int], intervals: Sequence[tuple[int, int]]) -> bool:
    start, end = candidate
    return any(start < used_end and end > used_start for used_start, used_end in intervals)


def locate_tells(text: str, indicators: Sequence[dict]) -> list[Tell]:
    """Locate parsed tell spans in document text, preserving tell order.

    Duplicate span strings are common, so this first searches after the previous
    matched tell and falls back to the first non-overlapping global occurrence.
    """
    tells: list[Tell] = []
    used: list[tuple[int, int]] = []
    cursor = 0
    n = len(text)
    for ind in indicators:
        span = str(ind.get("span_text", ""))
        if not span:
            continue
        start = text.find(span, cursor)
        end = start + len(span) if start >= 0 else -1
        if start < 0 or _overlaps((start, end), used):
            start = -1
            search_at = 0
            while True:
                candidate = text.find(span, search_at)
                if candidate < 0:
                    break
                candidate_end = candidate + len(span)
                if not _overlaps((candidate, candidate_end), used):
                    start = candidate
                    end = candidate_end
                    break
                search_at = candidate + 1
        if start < 0:
            continue

        tell_type = str(ind.get("type") or "")
        polarity = 1 if tell_type == "AI" else (-1 if tell_type == "human" else 0)
        raw_score = ind.get("rubric_credibility", ind.get("model_score", ind.get("score", 0.0)))
        try:
            score = max(0.0, min(1.0, abs(float(raw_score))))
        except (TypeError, ValueError):
            score = 0.0
        tells.append(
            Tell(
                start=max(0, min(start, n)),
                end=max(0, min(end, n)),
                polarity=polarity,
                score=score,
                span_text=span,
                explanation=str(ind.get("explanation", "")),
                tell_type=tell_type,
            )
        )
        used.append((start, end))
        cursor = max(cursor, end)
    return tells


def spans_for(
    ex: Example,
    polarity: int | None = None,
    min_score: float | None = None,
    tells: Sequence[Tell] | None = None,
) -> list[tuple[int, int]]:
    items = tells if tells is not None else ex.tells
    spans = []
    for tell in items:
        if polarity is not None and tell.polarity != polarity:
            continue
        if min_score is not None and tell.score < min_score:
            continue
        spans.append((tell.start, tell.end))
    return merge_intervals(spans)


def extract_spans(text: str, spans: Sequence[tuple[int, int]], sep: str = " [...] ") -> str:
    pieces = [text[start:end].strip() for start, end in merge_intervals(spans) if text[start:end].strip()]
    return sep.join(pieces) if pieces else " "


def delete_spans(text: str, spans: Sequence[tuple[int, int]], mask_token: str | None = None) -> str:
    intervals = merge_intervals(spans)
    if not intervals:
        return text
    out: list[str] = []
    cursor = 0
    for start, end in intervals:
        out.append(text[cursor:start])
        if mask_token is not None:
            out.append(mask_token)
        cursor = end
    out.append(text[cursor:])
    return "".join(out).strip() or " "


def _softmax(xs: np.ndarray) -> np.ndarray:
    if len(xs) == 0:
        return xs
    shifted = xs - np.max(xs)
    exp = np.exp(shifted)
    total = np.sum(exp)
    return exp / total if total else np.ones_like(xs) / len(xs)


def aggregate_prob_from_tells(
    tells: Sequence[Tell],
    beta: float = 3.0,
    alpha: float = 4.0,
    center: float = 0.0,
    override_scores: Sequence[float] | None = None,
) -> float:
    if not tells:
        return 0.5
    strengths = np.asarray(override_scores if override_scores is not None else [t.score for t in tells], dtype=float)
    strengths = np.clip(np.abs(strengths), 0.0, 1.0)
    polarities = np.asarray([t.polarity for t in tells], dtype=float)
    weights = _softmax(beta * strengths)
    agg = float(np.sum(weights * polarities * strengths))
    return float(1.0 / (1.0 + math.exp(-alpha * (agg - center))))


def select_tells_by_char_budget(ex: Example, budget_frac: float) -> list[Tell]:
    if not ex.tells:
        return []
    max_chars = max(1, int(len(ex.text) * float(budget_frac)))
    selected: list[Tell] = []
    used = 0
    for tell in sorted(ex.tells, key=lambda item: item.score, reverse=True):
        span_len = max(0, tell.end - tell.start)
        if span_len <= 0:
            continue
        if used + span_len <= max_chars:
            selected.append(tell)
            used += span_len
    if not selected:
        selected = [min(ex.tells, key=lambda item: max(1, item.end - item.start))]
    return selected


def evaluate_causal_tells(
    examples: Sequence[Example],
    score_fn: Callable[[list[str]], Sequence[float]],
    budget_fracs: Sequence[float] = (0.01, 0.02, 0.05, 0.10, 0.20, 0.50),
    high_score_threshold: float = 0.5,
    mask_token: str | None = None,
) -> dict:
    ys = [ex.y for ex in examples]
    full_texts = [ex.text for ex in examples]
    tell_only_texts = [extract_spans(ex.text, spans_for(ex)) for ex in examples]
    removed_texts = [delete_spans(ex.text, spans_for(ex), mask_token=mask_token) for ex in examples]
    ai_removed_texts = [delete_spans(ex.text, spans_for(ex, polarity=1), mask_token=mask_token) for ex in examples]
    human_removed_texts = [delete_spans(ex.text, spans_for(ex, polarity=-1), mask_token=mask_token) for ex in examples]

    all_probe_texts = full_texts + tell_only_texts + removed_texts + ai_removed_texts + human_removed_texts
    all_probs = np.asarray(list(score_fn(all_probe_texts)), dtype=float)
    n = len(examples)
    full_probs = all_probs[:n]
    suff_probs = all_probs[n : 2 * n]
    removed_probs = all_probs[2 * n : 3 * n]
    ai_removed_probs = all_probs[3 * n : 4 * n]
    human_removed_probs = all_probs[4 * n : 5 * n]

    full_gold = gold_confidence(full_probs, ys)
    suff_gold = gold_confidence(suff_probs, ys)
    removed_gold = gold_confidence(removed_probs, ys)
    suff_drop = np.maximum(0.0, full_gold - suff_gold)
    comp_drop = full_gold - removed_gold

    has_ai_tell = np.asarray([bool(spans_for(ex, polarity=1)) for ex in examples])
    has_human_tell = np.asarray([bool(spans_for(ex, polarity=-1)) for ex in examples])
    ai_drop = full_probs - ai_removed_probs
    human_rise = human_removed_probs - full_probs

    budget_rows = []
    for frac in budget_fracs:
        budget_texts = []
        selected_counts = []
        selected_chars = []
        for ex in examples:
            selected = select_tells_by_char_budget(ex, frac)
            selected_counts.append(len(selected))
            selected_chars.append(sum(max(0, t.end - t.start) for t in selected))
            budget_texts.append(extract_spans(ex.text, spans_for(ex, tells=selected)))
        probs = np.asarray(list(score_fn(budget_texts)), dtype=float)
        metrics = classification_metrics(ys, probs)
        budget_rows.append(
            {
                "budget_frac": float(frac),
                "auroc": metrics["auroc"],
                "tpr_at_fpr_0.01": metrics["tpr_at_fpr_0.01"],
                "mean_selected_tells": float(np.mean(selected_counts)) if selected_counts else 0.0,
                "mean_selected_chars": float(np.mean(selected_chars)) if selected_chars else 0.0,
            }
        )

    budget_x = np.asarray([row["budget_frac"] for row in budget_rows], dtype=float)
    budget_auc = np.asarray([row["auroc"] for row in budget_rows], dtype=float)
    if len(budget_rows) >= 2 and not np.any(np.isnan(budget_auc)) and budget_x[-1] > budget_x[0]:
        area_under_budget_curve_auroc = float(trapezoid_area(budget_auc, budget_x) / (budget_x[-1] - budget_x[0]))
    else:
        area_under_budget_curve_auroc = float("nan")

    full_metrics = classification_metrics(ys, full_probs)
    tell_only_metrics = classification_metrics(ys, suff_probs)
    removed_metrics = classification_metrics(ys, removed_probs)
    intrinsic_probs = [aggregate_prob_from_tells(ex.tells) for ex in examples]
    intrinsic_metrics = classification_metrics(ys, intrinsic_probs)
    contra = contradiction_metrics(examples, high_score_threshold=high_score_threshold)
    generic = genericity_metrics(examples, high_score_threshold=high_score_threshold)

    def masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
        return float(np.mean(values[mask])) if np.any(mask) else float("nan")

    def masked_diracc(values: np.ndarray, mask: np.ndarray, eps: float = 1e-6) -> float:
        return float(np.mean(values[mask] > eps)) if np.any(mask) else float("nan")

    return {
        "n_examples": len(examples),
        "n_tells": int(sum(len(ex.tells) for ex in examples)),
        "full": full_metrics,
        "tell_only": tell_only_metrics,
        "removed": removed_metrics,
        "intrinsic_tell": intrinsic_metrics,
        "sufficiency_drop_mean": float(np.mean(suff_drop)) if len(suff_drop) else float("nan"),
        "sufficiency_drop_median": float(np.median(suff_drop)) if len(suff_drop) else float("nan"),
        "comprehensiveness_drop_mean": float(np.mean(comp_drop)) if len(comp_drop) else float("nan"),
        "comprehensiveness_drop_median": float(np.median(comp_drop)) if len(comp_drop) else float("nan"),
        "delta_auroc_removed": float(full_metrics["auroc"] - removed_metrics["auroc"]),
        "ai_positive_deletion_drop_mean": masked_mean(ai_drop, has_ai_tell),
        "ai_positive_deletion_diracc": masked_diracc(ai_drop, has_ai_tell),
        "human_positive_deletion_rise_mean": masked_mean(human_rise, has_human_tell),
        "human_positive_deletion_diracc": masked_diracc(human_rise, has_human_tell),
        "signed_deletion_score": float(
            np.nanmean(
                [
                    masked_mean(ai_drop, has_ai_tell),
                    masked_mean(human_rise, has_human_tell),
                ]
            )
        ),
        "budget_curve": budget_rows,
        "area_under_budget_curve_auroc": area_under_budget_curve_auroc,
        **contra,
        **generic,
    }


def contradiction_metrics(examples: Sequence[Example], high_score_threshold: float = 0.5) -> dict[str, float | int]:
    high_flags = []
    high_weights = []
    high_weighted_flags = []
    all_flags = []
    all_weights = []
    all_weighted_flags = []
    for ex in examples:
        for tell in ex.tells:
            is_contra = 1 if ex.y * tell.polarity < 0 else 0
            all_flags.append(is_contra)
            all_weights.append(tell.score)
            all_weighted_flags.append(tell.score * is_contra)
            if tell.score >= high_score_threshold:
                high_flags.append(is_contra)
                high_weights.append(tell.score)
                high_weighted_flags.append(tell.score * is_contra)

    def rate(flags: list[int]) -> float:
        return float(np.mean(flags)) if flags else float("nan")

    def weighted(flags: list[float], weights: list[float]) -> float:
        return float(np.sum(flags) / max(np.sum(weights), 1e-12)) if weights else float("nan")

    return {
        "contradiction_rate_all": rate(all_flags),
        "weighted_contradiction_all": weighted(all_weighted_flags, all_weights),
        "contradiction_rate_high_score": rate(high_flags),
        "weighted_contradiction_high_score": weighted(high_weighted_flags, high_weights),
        "num_high_score_tells": int(len(high_flags)),
    }


def normalize_explanation(text: str) -> str:
    lowered = text.lower().strip()
    lowered = re.sub(r"\s+", " ", lowered)
    return re.sub(r"[^a-z0-9\s-]", "", lowered)


def content_tokens(text: str) -> set[str]:
    stop = {
        "the", "a", "an", "and", "or", "it", "this", "that", "is", "are", "was", "were",
        "of", "to", "in", "for", "with", "as", "on", "by", "because", "suggests",
        "indicates", "text", "span", "phrase", "writing", "authorship", "human", "ai",
        "generated", "common", "typical",
    }
    return {tok for tok in re.findall(r"[a-zA-Z0-9]+", text.lower()) if len(tok) > 2 and tok not in stop}


def explanation_span_overlap(ex: Example, tell: Tell) -> float:
    exp_tokens = content_tokens(tell.explanation)
    span_tokens = content_tokens(ex.text[tell.start : tell.end])
    if not exp_tokens or not span_tokens:
        return 0.0
    return len(exp_tokens & span_tokens) / len(exp_tokens | span_tokens)


def genericity_metrics(
    examples: Sequence[Example],
    high_score_threshold: float = 0.5,
    repeat_threshold: int = 5,
    min_span_overlap: float = 0.05,
) -> dict[str, float | int]:
    entries: list[tuple[Example, Tell, str]] = []
    counts: dict[str, int] = {}
    for ex in examples:
        for tell in ex.tells:
            if tell.score < high_score_threshold:
                continue
            norm = normalize_explanation(tell.explanation)
            entries.append((ex, tell, norm))
            counts[norm] = counts.get(norm, 0) + 1
    if not entries:
        return {"genericity_rate": float("nan"), "weighted_genericity": float("nan")}

    flags = []
    weights = []
    for ex, tell, norm in entries:
        has_generic_phrase = any(phrase in norm for phrase in GENERIC_PHRASES)
        repeated = counts[norm] >= repeat_threshold
        low_overlap = explanation_span_overlap(ex, tell) < min_span_overlap
        generic = int((has_generic_phrase or repeated) and low_overlap)
        flags.append(generic)
        weights.append(tell.score)
    flags_arr = np.asarray(flags, dtype=float)
    weights_arr = np.asarray(weights, dtype=float)
    return {
        "genericity_rate": float(np.mean(flags_arr)),
        "weighted_genericity": float(np.sum(flags_arr * weights_arr) / max(np.sum(weights_arr), 1e-12)),
    }


