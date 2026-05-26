"""Derive document verdict score from inner annotation scores (softmax-weighted)."""

from __future__ import annotations

import math
import re
from typing import Any

from rl_detector.annotation_utils import collect_bracket_tells
from rl_detector.tell_xml import _TEXT_CLOSE_CHUNK, _TEXT_O, _VERDICT_PREF

_VERDICT_SCORE_RE = re.compile(
    r'(<verdict type="(?:AI|human)" why="(?:[^"\\]|\\.)*" score=")([0-9]*\.?[0-9]+)(")'
)


def aggregate_signed_scores(scores: list[float], beta: float) -> float:
    if not scores:
        return 0.0
    if beta <= 0.0:
        return sum(scores) / len(scores)
    weights = [math.exp(beta * abs(s)) for s in scores]
    total_w = sum(weights)
    return sum(w * s for w, s in zip(weights, scores)) / total_w


def signed_scores_from_tells(tells: list[dict[str, Any]]) -> list[float]:
    signed: list[float] = []
    for tell in tells:
        raw = tell.get("score")
        if raw is None:
            continue
        strength = max(0.0, min(1.0, float(raw)))
        typ = tell.get("type")
        if typ == "AI":
            signed.append(strength)
        elif typ == "human":
            signed.append(-strength)
    return signed


def inner_tells_from_annotation(annotation: str) -> list[dict[str, Any]]:
    probe = f'{_TEXT_O}{annotation}{_VERDICT_PREF}human" why="." score="0.50{_TEXT_CLOSE_CHUNK}'
    return collect_bracket_tells(probe) or []


def aligned_strength_from_tells(tells: list[dict[str, Any]], label: int, beta: float) -> float:
    strengths: list[float] = []
    for tell in tells:
        raw = tell.get("score")
        if raw is None:
            continue
        strength = max(0.0, min(1.0, float(raw)))
        typ = tell.get("type")
        aligned = (label == 0 and typ == "human") or (label == 1 and typ == "AI")
        if aligned:
            strengths.append(strength)
    if not strengths:
        return 0.0
    if beta <= 0.0:
        return sum(strengths) / len(strengths)
    weights = [math.exp(beta * s) for s in strengths]
    total_w = sum(weights)
    return sum(w * s for w, s in zip(weights, strengths)) / total_w


def verdict_score_from_tells(
    tells: list[dict[str, Any]],
    label: int,
    beta: float,
    scale: float,
    tau: float,
) -> float:
    signed = signed_scores_from_tells(tells=tells)
    if not signed:
        return 0.5
    agg = aggregate_signed_scores(scores=signed, beta=beta)
    compressed = math.tanh(agg / tau)
    if label == 0:
        directional = 0.5 - scale * compressed
    else:
        directional = 0.5 + scale * compressed
    strength = aligned_strength_from_tells(tells=tells, label=label, beta=beta)
    return max(0.0, min(1.0, 0.5 + (directional - 0.5) * strength))


def verdict_score_from_annotation(
    annotation: str,
    label: int,
    beta: float,
    scale: float,
    tau: float,
) -> float:
    tells = inner_tells_from_annotation(annotation=annotation)
    return verdict_score_from_tells(
        tells=tells,
        label=label,
        beta=beta,
        scale=scale,
        tau=tau,
    )


def patch_sft_text_verdict_score(sft_text: str, verdict_score: float) -> str:
    sc = f"{verdict_score:.2f}"
    matches = list(_VERDICT_SCORE_RE.finditer(sft_text))
    if not matches:
        return sft_text
    last = matches[-1]
    return sft_text[: last.start(2)] + sc + sft_text[last.end(2) :]
