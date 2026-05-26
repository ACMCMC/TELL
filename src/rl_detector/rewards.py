"""Reward functions and advantage computation for GRPO."""

import logging
import re
from functools import lru_cache

from rl_detector.tell_xml import (
    canonical_logical_document,
    full_markup_wire_escaping_ok,
    root_splits,
    strip_return_token,
    strip_text_wrapper,
)
from rl_detector.annotation_utils import (
    collect_bracket_tells,
    get_outer_bracket_metadata,
    strip_all_bracket_annotations,
    strip_score_attrs,
)
from rl_detector.config import CFG

logger = logging.getLogger(__name__)


def _tr(key: str, default):
    """Read a dotted key path from CFG.training, e.g. 'reward.cls.weight'."""
    cur = getattr(CFG, "training", None)
    for part in key.split("."):
        cur = getattr(cur, part, None)
        if cur is None:
            return default
    return cur if cur is not None else default


_VALID_TYPES = {"AI", "human"}


def parse_indicators(output: str, document: str | None = None) -> list[dict] | None:
    """Nested <span> indicators (excluding root outer annot)."""
    tells = collect_bracket_tells(output)
    if tells:
        result = []
        # Minimum span text length: exclude trivially short spans (punctuation, brackets)
        # that the model may exploit for spurious high-credibility annotations.
        # These would get positive per-tell advantages from the rubric scorer (which
        # evaluates explanation quality, not span content), creating a feedback loop
        # toward annotating empty/punctuation content. Default 0 = disabled.
        _min_span_len = int(_tr("reward.ann.min_span_text_len", 0))
        for t in tells:
            if t.get("type") not in _VALID_TYPES:
                continue
            span = t.get("span_text") or ""
            if _min_span_len > 0 and len(span.strip()) < _min_span_len:
                continue
            try:
                _ms = max(0.0, min(1.0, float(t.get("score", 0.0) or 0.0)))
            except (ValueError, TypeError):
                _ms = 0.0
            row = {
                "span_text": span,
                "explanation": t["explanation"],
                "type": t.get("type"),
                "model_score": _ms,
            }
            inner_pos = t.get("_inner_pos")
            if isinstance(inner_pos, int) and span:
                row["logical_span_start"] = inner_pos
                row["logical_span_end"] = inner_pos + len(span)
            result.append(row)
        return result if result else None
    return None


def strip_tags(tagged_text: str) -> str:
    """Remove all bracket annotations and the outer wrapper, keeping plain text."""
    return strip_all_bracket_annotations(tagged_text)


def stripped_char_diff_count(text_a: str, text_b: str) -> int:
    if text_a == text_b:
        return 0
    from difflib import SequenceMatcher
    matcher = SequenceMatcher(None, text_a, text_b, autojunk=False)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    return len(text_a) + len(text_b) - 2 * matched


def format_diagnostics(output: str, document: str) -> dict[str, int | str | bool]:
    """Structured format check; returns {ok, reason, char_diff_count}."""
    def _has_raw(s: str) -> bool:
        return ("<annotation " in s) or ("<span>" in s)

    output = strip_return_token(output)
    if not output:
        return {"ok": False, "reason": "empty_final", "char_diff_count": len(document)}

    leg = strip_text_wrapper(tx=output)
    if leg is None:
        return {
            "ok": False,
            "reason": "missing_outer_annotation",
            "char_diff_count": stripped_char_diff_count(text_a=output, text_b=document),
        }

    inn, desc, meta, ok, _aft = root_splits(tx=output)
    if not ok:
        return {
            "ok": False,
            "reason": "annotation_parse_failed",
            "char_diff_count": stripped_char_diff_count(text_a=inn, text_b=document),
        }

    if meta is None:
        diff = stripped_char_diff_count(text_a=inn, text_b=document)
        reason = "annotation_parse_failed" if (_has_raw(s=inn) or diff == 0) else "text_mismatch"
        return {"ok": False, "reason": reason, "char_diff_count": diff}

    if any(t.get("type") not in _VALID_TYPES for t in desc):
        return {
            "ok": False,
            "reason": "invalid_type",
            "char_diff_count": stripped_char_diff_count(text_a=inn, text_b=document),
        }

    if inn != canonical_logical_document(tx=document):
        diff = stripped_char_diff_count(text_a=inn, text_b=document)
        reason = "annotation_parse_failed" if _has_raw(s=inn) else "text_mismatch"
        return {"ok": False, "reason": reason, "char_diff_count": diff}

    if not full_markup_wire_escaping_ok(full=output):
        return {"ok": False, "reason": "bad_xml_escaping", "char_diff_count": 0}

    return {"ok": True, "reason": "ok", "char_diff_count": 0}


def format_exception_diag(response_text: str, document: str, exc: Exception) -> dict[str, int | str | bool]:
    try:
        diff = stripped_char_diff_count(response_text or "", document or "")
    except Exception:
        diff = len(document or "")
    return {"ok": False, "reason": "format_exception", "char_diff_count": int(diff),
            "exception_type": type(exc).__name__, "exception": str(exc)}


def safe_format_diagnostics(response_text: str, document: str) -> dict[str, int | str | bool]:
    try:
        return format_diagnostics(response_text, document)
    except Exception as exc:
        logger.exception("format_diagnostics crashed")
        return format_exception_diag(response_text, document, exc)


def format_reward(output: str, document: str) -> float:
    return 1.0 if format_diagnostics(output, document)["ok"] else 0.0


def format_status(output: str, document: str) -> tuple[bool, str]:
    diag = format_diagnostics(output, document)
    return bool(diag["ok"]), str(diag["reason"])


# ---------------------------------------------------------------------------
# Per-token reward functions (PTAD — Per-Token Annotation Decomposed)
#
# Each function returns a scalar reward for a single token type in a single
# annotation or verdict.  train.py collects these across all (annotation ×
# rollout) pairs for a document, normalises within each token-type group,
# and assigns the resulting advantages to the matching token positions.
#
# No aggregation over annotations: every span gets its own independent signal.
# Model-generated score= attribute values are intentionally excluded — they are
# gameable and carry no ground-truth information.
# ---------------------------------------------------------------------------

def annotation_type_reward(annotation_type: str, label: int) -> float:
    """Reward for an annotation type value token: label alignment only (+1 / -1)."""
    if annotation_type not in _VALID_TYPES:
        return 0.0
    type_correct = (annotation_type == "AI") == (label == 1)
    return 1.0 if type_correct else -1.0


def _ann_why_quality(why_text: str) -> float:
    """Length + repetition quality score for an annotation why string. Range [0, 1]."""
    word_low = float(_tr("reward.why.len_low", 10.0))
    word_high = float(_tr("reward.why.len_high", 40.0))
    rep_ngram = int(_tr("reward.why.rep_ngram", 3))
    words = (why_text or "").split()
    n = len(words)
    if n == 0:
        return 0.0
    if n < word_low:
        len_score = (n / word_low) ** 2
    elif n <= word_high:
        len_score = 1.0
    else:
        zero_at = word_high * 2.0
        len_score = 0.0 if n >= zero_at else ((zero_at - n) / (zero_at - word_high)) ** 2
    if n < rep_ngram:
        rep_score = 1.0
    else:
        ngrams = [tuple(words[i: i + rep_ngram]) for i in range(n - rep_ngram + 1)]
        rep_score = len(set(ngrams)) / len(ngrams)
    return len_score * rep_score


def annotation_why_reward(rubric_credibility: float, why_text: str = "") -> float:
    """
    Reward for an annotation's why value tokens.

    Multiplicative: cred * quality. Quality is a length+repetition score that
    peaks at 1.0 for why texts in [len_low, len_high] words and decays outside.
    This gates credibility on explanation quality — a confident annotation with
    an empty or too-long why gets near-zero reward.
    Range: [0, 1].
    """
    cred = max(0.0, min(1.0, float(rubric_credibility)))
    quality = _ann_why_quality(why_text)
    return cred * quality


def annotation_score_reward(model_score: float, rubric_credibility: float) -> float:
    """
    Reward for an annotation's score value token.

    Measures agreement between the model's written score= value and rubric credibility:
      reward = 1 - |model_score - rubric_credibility|

    Best case (exact match): 1.0.
    Worst case (model wrote 1.0, rubric gave 0.0, or vice versa): 0.0.
    Teaches calibration — the score should reflect the rubric's evidence strength.
    Range: [0, 1].
    """
    ms = max(0.0, min(1.0, float(model_score)))
    rc = max(0.0, min(1.0, float(rubric_credibility)))
    return 1.0 - abs(ms - rc)


def verdict_type_reward(verdict_type: str, label: int) -> float:
    """Reward for the verdict type value token: label alignment only (+1 / -1)."""
    if verdict_type not in _VALID_TYPES:
        return 0.0
    type_correct = (verdict_type == "AI") == (label == 1)
    return 1.0 if type_correct else -1.0


def _verdict_why_length_gate(verdict_why_text: str) -> float:
    """Length gate ∈ [0, 1] for the verdict why text.

    Used as a multiplicative gate: if length is off everything else is irrelevant.
    Tent shape: quadratic ramp 0→word_low, plateau at 1.0 for [word_low, word_high],
    quadratic decay to 0 at word_zero. Also penalises n-gram repetition.

    Config keys (training.reward.verdict_why.*):
      word_low  (default 50)   — start of plateau
      word_high (default 150)  — end of plateau
      word_zero (default 200)  — word count at which gate reaches 0
      rep_ngram (default 3)    — n-gram window for repetition detection
    """
    if not verdict_why_text or not verdict_why_text.strip():
        return 0.0
    word_low = float(_tr("reward.verdict_why.word_low", 50.0))
    word_high = float(_tr("reward.verdict_why.word_high", 150.0))
    word_zero = float(_tr("reward.verdict_why.word_zero", 200.0))
    rep_ngram = int(_tr("reward.verdict_why.rep_ngram", 3))
    words = verdict_why_text.split()
    n = len(words)
    if n == 0:
        return 0.0
    elif n < word_low:
        len_gate = (n / word_low) ** 2
    elif n <= word_high:
        len_gate = 1.0
    elif n >= word_zero:
        len_gate = 0.0
    else:
        len_gate = ((word_zero - n) / (word_zero - word_high)) ** 2
    if n < rep_ngram:
        rep_score = 1.0
    else:
        ngrams = [tuple(words[i:i + rep_ngram]) for i in range(n - rep_ngram + 1)]
        rep_score = len(set(ngrams)) / len(ngrams) if ngrams else 1.0
    return len_gate * rep_score


def verdict_why_quality_reward(verdict_why_text: str) -> float:
    """Legacy wrapper — returns the length gate directly. Use verdict_why_combined_reward for training."""
    return _verdict_why_length_gate(verdict_why_text)


def verdict_why_combined_reward(verdict_why_text: str, tell_scored: list[dict]) -> float:
    """Combined verdict why reward: length gate × (annotation recall + quote coverage).

    Structured as explicit layers:
      1. Length gate (0–1): if too short or too long the verdict why is useless regardless
         of content. Gates everything below — wrong length means zero reward for all else.
      2. Annotation recall: fraction of annotation explanation words present in the verdict.
         With rubric scoring disabled this is the primary signal for whether the verdict
         actually synthesises the annotated evidence rather than producing a generic summary.
      3. Quote coverage: bonus when quoted text in the verdict was also annotated as a span,
         rewarding explicit citation of the evidence the model identified.

    Result is in [0, 2]: length_gate × (recall ∈ [0,1] + quote_cov ∈ [0,1]).
    GRPO normalisation in train.py handles the scale.
    """
    gate = _verdict_why_length_gate(verdict_why_text)
    if gate == 0.0:
        return 0.0
    recall = verdict_ann_recall_reward(verdict_why_text, tell_scored)
    quote_cov = verdict_quote_coverage_reward(verdict_why_text, tell_scored)
    return gate * (recall + quote_cov)


def outer_verdict_score_reward(output: str, label: int) -> float:
    """Label-aligned outer score magnitude for training verdict score= tokens.

    AI doc (label=1): want +mag → reward = outer_document_score.
    Human doc (label=0): want −mag → reward = −outer_document_score.
    Range: [-1, 1]. Matches eval AUROC direction.
    """
    s = outer_document_score(output=output)
    return s if label == 1 else -s


def outer_document_score(output: str) -> float:
    """Signed outer prediction strength for eval/AUROC: +mag if AI, −mag if human.
    """
    try:
        meta = get_outer_bracket_metadata(output)
        if meta is None:
            return 0.0
        otyp = meta.get("type")
        sc_mag = max(0.0, min(1.0, float(meta.get("score_magnitude", 0.0))))
        if otyp == "AI":
            return sc_mag
        if otyp == "human":
            return -sc_mag
        return 0.0
    except (ValueError, TypeError):
        return 0.0


def outer_credibility(rubric_output: dict | None) -> float:
    """Extract the rubric's overall verdict credibility score. Returns 0.5 when unavailable."""
    if rubric_output is None:
        return 0.5
    return float((rubric_output.get("overall") or {}).get("credibility", 0.5))


def mean_tell_rubric_credibility(tell_scored: list[dict]) -> float:
    """Mean rubric_credibility over inner annotations (diagnostic only; PTAD uses per-tell pools)."""
    vals = [
        float(fs["rubric_credibility"])
        for fs in tell_scored
        if fs.get("rubric_credibility") is not None
    ]
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


_QUOTE_RE = re.compile("[\u201c\u201d\u2018\u2019]([^\u201c\u201d\u2018\u2019]{3,})[\u201c\u201d\u2018\u2019]")
_WORD_STRIP_RE = re.compile(r"[^\w]")


@lru_cache(maxsize=1)
def _english_stopwords() -> frozenset:
    try:
        import nltk
        nltk.download("stopwords", quiet=True)
        from nltk.corpus import stopwords
        return frozenset(stopwords.words("english"))
    except Exception:
        return frozenset()


def verdict_ann_recall_reward(verdict_why_text: str, tell_scored: list[dict]) -> float:
    """Fraction of distinctive annotation-explanation words that appear in the verdict why.

    The verdict should synthesize the evidence from individual span annotations into a
    unified argument. A verdict that ignores the annotation explanations and produces a
    generic summary provides no extra information beyond the type/score. This metric
    rewards the verdict why for incorporating vocabulary from the annotation explanations,
    ensuring the verdict is grounded in the specific evidence rather than free-floating.

    Only content words (length >= 4) are counted to avoid stopword inflation.
    Returns 0.0 when there are no annotations or no content words to match against.
    """
    stopwords = _english_stopwords()
    ann_words: set[str] = set()
    for fs in tell_scored:
        for w in (fs.get("explanation") or "").lower().split():
            w = _WORD_STRIP_RE.sub("", w)
            if len(w) >= 3 and w not in stopwords:
                ann_words.add(w)
    if not ann_words:
        return 0.0
    verdict_words: set[str] = set()
    for w in (verdict_why_text or "").lower().split():
        w = _WORD_STRIP_RE.sub("", w)
        if len(w) >= 3 and w not in stopwords:
            verdict_words.add(w)
    return len(ann_words & verdict_words) / len(ann_words)


def verdict_quote_coverage_reward(verdict_why_text: str, tell_scored: list[dict]) -> float:
    """Extra reward when verdict why= quotes text that is also annotated as a span.

    If the verdict contains no quotes: returns 0.0 (no bonus, no penalty).
    If quotes exist: returns the fraction that appear as a substring of any
    annotation span_text (or vice-versa for short spans). Range [0, 1].
    """
    quotes = _QUOTE_RE.findall(verdict_why_text or "")
    if not quotes:
        return 0.0
    span_texts = [s.get("span_text", "") or "" for s in tell_scored]
    if not span_texts:
        return 0.0
    covered = sum(
        1 for q in quotes
        if any(q.strip() in s or s in q.strip() for s in span_texts if s)
    )
    return covered / len(quotes)


# ---------------------------------------------------------------------------
# Rollout-level verdict reward (PTAD-compatible)
#
# The per-annotation token rewards (annotation_type_reward, annotation_why_reward,
# annotation_score_reward) are applied via per-token advantages in train.py — they
# do NOT appear here as aggregates.  This function covers only the verdict-level
# signals needed for the rollout reward used by struct-token and early-stopping logic.
# ---------------------------------------------------------------------------

def reward_components(
    output: str,
    document: str,
    label: int,
    tell_scored: list[dict],
    budget_ratio: float = 0.0,
    rubric_output: dict | None = None,
) -> dict[str, float]:
    """
    PTAD-compatible rollout reward components. Returns individual signals used for
    per-token advantages and logging. No total blending — each component drives its
    own gradient path via train.py. Returns zero_reward_components() when format fails.
    """
    fmt_ok = format_reward(output, document)
    if not fmt_ok:
        return zero_reward_components()

    outer_meta = None
    verdict_type_str = ""
    outer_type_ai = -1.0
    type_correct = False
    try:
        outer_meta = get_outer_bracket_metadata(output)
        if outer_meta is not None:
            verdict_type_str = outer_meta.get("type") or ""
            outer_type_ai = 1.0 if verdict_type_str == "AI" else (0.0 if verdict_type_str == "human" else -1.0)
            type_correct = verdict_type_str == ("AI" if label == 1 else "human")
    except (ValueError, TypeError):
        pass

    cred_outer = outer_credibility(rubric_output)
    cred_ann_mean = mean_tell_rubric_credibility(tell_scored=tell_scored)
    _outer_why_text = (outer_meta or {}).get("explanation", "") or ""
    vtype = verdict_type_reward(verdict_type=verdict_type_str, label=label)
    vwq = verdict_why_quality_reward(_outer_why_text)
    qcov = verdict_quote_coverage_reward(_outer_why_text, tell_scored)
    ann_recall = verdict_ann_recall_reward(_outer_why_text, tell_scored)
    vwy_combined = verdict_why_combined_reward(_outer_why_text, tell_scored)
    agg = outer_document_score(output=output)
    vscore = outer_verdict_score_reward(output=output, label=label)

    return {
        "format_ok": fmt_ok,
        "agg_score": agg,
        "verdict_score": vscore,
        "outer_correct": 1.0 if type_correct else 0.0,
        "outer_type_ai": outer_type_ai,
        "cls": vtype,
        "margin": 0.0,  # legacy key; margin band is eval-only (tri-class F1)
        "diversity": 0.0,
        "why_conciseness": 0.0,
        "tell_alignment": 0.0,
        "credibility": cred_ann_mean,
        "outer_credibility": cred_outer,
        "verdict_why_quality": vwq,
        "verdict_quote_coverage": qcov,
        "verdict_ann_recall": ann_recall,
        "verdict_why_combined": vwy_combined,
        "ann": 0.0,
        "n_tells": float(len(tell_scored)),
    }


def zero_reward_components() -> dict[str, float]:
    """All-zero reward dict for format failures and early-exit paths."""
    return {
        "format_ok": 0.0,
        "agg_score": 0.0,
        "outer_correct": 0.0,
        "outer_type_ai": -1.0,
        "cls": 0.0,
        "verdict_score": 0.0,
        "margin": 0.0,
        "diversity": 0.0,
        "why_conciseness": 0.0,
        "tell_alignment": 0.0,
        "credibility": 0.0,
        "outer_credibility": 0.0,
        "verdict_why_quality": 0.0,
        "verdict_quote_coverage": 0.0,
        "verdict_ann_recall": 0.0,
        "verdict_why_combined": 0.0,
        "ann": 0.0,
        "n_tells": 0.0,
    }


def compute_advantages(rewards: list[float], std_floor: float = 0.0, normalize: str = "mean") -> list[float]:
    """
    Normalize rewards within a group and return per-rollout scalar advantages.

    normalize="mean" (GRPO default): subtract group mean, divide by std.
    normalize="min" (positive-only): subtract group minimum — worst rollout gets 0 advantage.

    std_floor: adaptive floor from train.py EMA; prevents blowup when within-group std collapses.
    """
    n = len(rewards)
    mean = sum(rewards) / n
    baseline = min(rewards) if normalize == "min" else mean
    variance = sum((r - mean) ** 2 for r in rewards) / n
    std = variance ** 0.5
    denom = max(std, std_floor, 1e-4)
    raw = [(r - baseline) / denom for r in rewards]
    return [max(-5.0, min(5.0, a)) for a in raw]
