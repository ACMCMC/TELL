"""Unit tests for _build_per_token_advantages — per-token advantage decomposition.

Tests cover:
- Verdict type vs. why token signal separation
- Per-tell credibility assignment to inner annotation tokens
- Edge cases: empty verdict why, no rubric credibility, nested tells
- Structural token handling
- label_ctx_for_opt zeroing
"""
import sys
import types
import math

_TOL = 1e-5


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=_TOL, abs_tol=_TOL)


# ---------------------------------------------------------------------------
# Mock heavy dependencies before rl_detector imports
# ---------------------------------------------------------------------------

from test_reproducibility import (  # noqa: F401
    _make_dotenv_module,
    _make_openai_module,
    _make_tinker_module,
    _make_transformers_module,
    _make_wandb_module,
    _make_weave_module,
)

for _name, _maker in [
    ("tinker", _make_tinker_module),
    ("weave", _make_weave_module),
    ("wandb", _make_wandb_module),
    ("transformers", _make_transformers_module),
    ("openai", _make_openai_module),
    ("dotenv", _make_dotenv_module),
]:
    if _name not in sys.modules:
        sys.modules[_name] = _maker()

from rl_detector.rollouts import (  # noqa: E402
    _MASK_TEXT_OPEN,
    _MASK_TEXT_CLOSE,
    _MASK_VERDICT_OPEN,
    _MASK_SPAN_OPEN,
    _MASK_ANN_OPEN,
    _MASK_ANN_CLOSE,
    _MASK_ANN_WHY_Q,
    _MASK_ANN_SCORE_Q,
    _MASK_ANN_STRUCT,
)

# ---------------------------------------------------------------------------
# State-machine re-implementation (no tokenizer dependency)
# ---------------------------------------------------------------------------

def _simulate_ptad(
    tokens: list[int],
    cls_adv: float,
    count_adv: float,
    scaled_adv: float,
    structural_adv: float = 0.5,
    doc_copy_adv: float = 0.0,
    format_ok: bool = False,
    label_ctx_for_opt: bool = False,
    per_tell_advs: list[float] | None = None,
    verdict_why_adv: float | None = None,
) -> list[float]:
    """Mirror of _build_per_token_advantages without tokenizer (no <|return|> lookup)."""
    _STRUCT = frozenset(_MASK_ANN_STRUCT)
    in_attrs = False
    in_verdict = False
    in_verdict_why = False
    tell_idx = -1
    out: list[float] = []

    for tok in tokens:
        t = int(tok)
        if t in _STRUCT:
            if t == _MASK_SPAN_OPEN:
                out.append(count_adv)
            elif t == _MASK_ANN_OPEN:
                tell_idx += 1
                in_attrs = True
                in_verdict = False
                in_verdict_why = False
                out.append(structural_adv)
            elif t == _MASK_VERDICT_OPEN:
                in_attrs = True
                in_verdict = True
                in_verdict_why = False
                out.append(structural_adv)
            elif t == _MASK_ANN_CLOSE or t == _MASK_TEXT_CLOSE:
                in_attrs = False
                in_verdict = False
                in_verdict_why = False
                out.append(structural_adv)
            elif in_verdict and t == _MASK_ANN_WHY_Q:
                in_verdict_why = True
                out.append(structural_adv)
            else:
                out.append(structural_adv)
        elif in_attrs:
            if in_verdict:
                if label_ctx_for_opt:
                    out.append(0.0)
                elif in_verdict_why:
                    out.append(verdict_why_adv if verdict_why_adv is not None else cls_adv)
                else:
                    out.append(cls_adv)
            else:
                if per_tell_advs is not None and 0 <= tell_idx < len(per_tell_advs):
                    out.append(per_tell_advs[tell_idx])
                else:
                    out.append(scaled_adv)
        else:
            out.append(doc_copy_adv if format_ok else 0.0)

    return out


DOC = 999   # a generic non-special token
DOC2 = 998


# ---------------------------------------------------------------------------
# Tests: verdict type vs. why signal separation
# ---------------------------------------------------------------------------

def test_verdict_type_gets_cls_adv_not_why():
    """Verdict type value tokens must get cls_adv, NOT verdict_why_adv."""
    tokens = [
        _MASK_TEXT_OPEN,          # structural
        _MASK_VERDICT_OPEN,       # enters verdict, not yet why
        DOC,                      # verdict TYPE value
        _MASK_ANN_WHY_Q,          # transitions to verdict_why
        DOC,                      # verdict WHY value
        _MASK_TEXT_CLOSE,         # exits verdict
    ]
    adv = _simulate_ptad(tokens, cls_adv=1.0, count_adv=0.0, scaled_adv=0.0,
                         verdict_why_adv=0.3)

    # indices: TEXT_OPEN=0, VERDICT_OPEN=1, TYPE=2, WHY_Q=3, WHY=4, TEXT_CLOSE=5
    assert _close(adv[2], 1.0), f"type token must get cls_adv=1.0, got {adv[2]}"
    assert _close(adv[4], 0.3), f"why token must get verdict_why_adv=0.3, got {adv[4]}"


def test_verdict_why_is_independent_from_cls():
    """Changing cls_adv must not affect verdict why token advantage."""
    tokens = [
        _MASK_VERDICT_OPEN, DOC, _MASK_ANN_WHY_Q, DOC, _MASK_TEXT_CLOSE
    ]
    adv_high_cls = _simulate_ptad(tokens, cls_adv=2.0, count_adv=0.0, scaled_adv=0.0,
                                  verdict_why_adv=0.5)
    adv_low_cls = _simulate_ptad(tokens, cls_adv=-1.0, count_adv=0.0, scaled_adv=0.0,
                                 verdict_why_adv=0.5)

    # WHY value is token index 3 (VERDICT_OPEN=0, TYPE=1, WHY_Q=2, WHY=3, TEXT_CLOSE=4)
    assert _close(adv_high_cls[3], 0.5), "why must not depend on cls_adv"
    assert _close(adv_low_cls[3], 0.5), "why must not depend on cls_adv"
    # But TYPE token (index 1) does depend on cls_adv
    assert _close(adv_high_cls[1], 2.0)
    assert _close(adv_low_cls[1], -1.0)


def test_verdict_why_fallback_when_none():
    """When verdict_why_adv=None, why tokens fall back to cls_adv."""
    tokens = [_MASK_VERDICT_OPEN, DOC, _MASK_ANN_WHY_Q, DOC, _MASK_TEXT_CLOSE]
    adv = _simulate_ptad(tokens, cls_adv=0.8, count_adv=0.0, scaled_adv=0.0,
                         verdict_why_adv=None)
    # Both type and why get cls_adv when verdict_why_adv is None
    assert _close(adv[1], 0.8), f"type token fallback {adv[1]}"
    assert _close(adv[3], 0.8), f"why token fallback {adv[3]}"


def test_empty_verdict_why_no_phantom_gradient():
    """Empty verdict why (zero why tokens) → zero why-token gradient, type unaffected."""
    # Verdict with no why tokens: VERDICT_OPEN TYPE WHY_Q [empty] TEXT_CLOSE
    tokens = [_MASK_VERDICT_OPEN, DOC, _MASK_ANN_WHY_Q, _MASK_TEXT_CLOSE]
    adv = _simulate_ptad(tokens, cls_adv=1.0, count_adv=0.0, scaled_adv=0.0,
                         verdict_why_adv=0.7)
    # VERDICT_OPEN=0, TYPE=1, WHY_Q=2, TEXT_CLOSE=3
    assert len(adv) == 4
    assert _close(adv[1], 1.0), f"type must be cls_adv=1.0, got {adv[1]}"
    assert _close(adv[0], 0.5), "VERDICT_OPEN structural"
    assert _close(adv[2], 0.5), "WHY_Q structural"
    assert _close(adv[3], 0.5), "TEXT_CLOSE structural"
    # No why tokens → no gradient contamination of type from outer_credibility


def test_verdict_score_tokens_get_verdict_why_adv():
    """Score value tokens after SCORE_Q inside verdict also get verdict_why_adv."""
    tokens = [
        _MASK_VERDICT_OPEN, DOC,       # type value
        _MASK_ANN_WHY_Q, DOC,          # why value
        _MASK_ANN_SCORE_Q, DOC,        # score value — still in_verdict_why
        _MASK_TEXT_CLOSE,
    ]
    adv = _simulate_ptad(tokens, cls_adv=1.0, count_adv=0.0, scaled_adv=0.0,
                         verdict_why_adv=0.4)
    # VERDICT_OPEN=0, TYPE=1, WHY_Q=2, WHY=3, SCORE_Q=4, SCORE=5, TEXT_CLOSE=6
    assert _close(adv[1], 1.0), "type → cls_adv"
    assert _close(adv[3], 0.4), "why → verdict_why_adv"
    assert _close(adv[4], 0.5), "SCORE_Q structural"
    assert _close(adv[5], 0.4), "score value → verdict_why_adv (still in_verdict_why)"


# ---------------------------------------------------------------------------
# Tests: per-tell credibility for inner annotation tokens
# ---------------------------------------------------------------------------

def test_per_tell_advs_assigned_by_tell_index():
    """Each tell (ANN_OPEN block) gets its own per_tell_advs value."""
    tokens = [
        _MASK_SPAN_OPEN, DOC,
        _MASK_ANN_OPEN, DOC, _MASK_ANN_WHY_Q, DOC, _MASK_ANN_CLOSE,   # tell 0
        _MASK_SPAN_OPEN, DOC,
        _MASK_ANN_OPEN, DOC, _MASK_ANN_WHY_Q, DOC, _MASK_ANN_CLOSE,   # tell 1
    ]
    per_tell_advs = [0.9, 0.3]
    adv = _simulate_ptad(tokens, cls_adv=0.0, count_adv=0.1, scaled_adv=0.0,
                         per_tell_advs=per_tell_advs)

    # tell 0 value tokens: idx 3 (type), 5 (why)  — 0=SPAN, 1=DOC, 2=ANN_OPEN, 3=TYPE, 4=WHY_Q, 5=WHY, 6=ANN_CLOSE
    assert _close(adv[3], 0.9), f"tell-0 type should be 0.9, got {adv[3]}"
    assert _close(adv[5], 0.9), f"tell-0 why should be 0.9, got {adv[5]}"
    # tell 1 value tokens: idx 10 (type), 12 (why) — 7=SPAN, 8=DOC, 9=ANN_OPEN, 10=TYPE, 11=WHY_Q, 12=WHY, 13=ANN_CLOSE
    assert _close(adv[10], 0.3), f"tell-1 type should be 0.3, got {adv[10]}"
    assert _close(adv[12], 0.3), f"tell-1 why should be 0.3, got {adv[12]}"


def test_per_tell_fallback_when_none():
    """When per_tell_advs=None, inner annotation tokens get scaled_adv."""
    tokens = [
        _MASK_ANN_OPEN, DOC, _MASK_ANN_CLOSE,
    ]
    adv = _simulate_ptad(tokens, cls_adv=0.0, count_adv=0.0, scaled_adv=0.77,
                         per_tell_advs=None)
    # ANN_OPEN=0, TYPE=1, ANN_CLOSE=2
    assert _close(adv[1], 0.77), f"inner token fallback should be scaled_adv=0.77, got {adv[1]}"


def test_per_tell_advs_out_of_bounds_fallback():
    """If per_tell_advs is shorter than actual tells, extra tells fall back to scaled_adv."""
    tokens = [
        _MASK_ANN_OPEN, DOC, _MASK_ANN_CLOSE,   # tell 0 → per_tell_advs[0]
        _MASK_ANN_OPEN, DOC, _MASK_ANN_CLOSE,   # tell 1 → out of bounds → scaled_adv
    ]
    per_tell_advs = [0.9]  # only one entry
    adv = _simulate_ptad(tokens, cls_adv=0.0, count_adv=0.0, scaled_adv=0.5,
                         per_tell_advs=per_tell_advs)
    # tell 0: idx 1
    assert _close(adv[1], 0.9), f"tell-0 should be 0.9, got {adv[1]}"
    # tell 1: idx 4
    assert _close(adv[4], 0.5), f"tell-1 OOB should fall back to scaled_adv=0.5, got {adv[4]}"


# ---------------------------------------------------------------------------
# Tests: structural tokens always get structural_adv
# ---------------------------------------------------------------------------

def test_all_structural_tokens_get_structural_adv():
    """TEXT_OPEN, ANN_OPEN, ANN_CLOSE, VERDICT_OPEN, WHY_Q, SCORE_Q, TEXT_CLOSE all → structural_adv."""
    struct_adv = 0.42
    tokens = [
        _MASK_TEXT_OPEN,
        _MASK_SPAN_OPEN,   # this gets count_adv, not structural_adv
        _MASK_ANN_OPEN,
        _MASK_ANN_WHY_Q,
        _MASK_ANN_SCORE_Q,
        _MASK_ANN_CLOSE,
        _MASK_VERDICT_OPEN,
        _MASK_ANN_WHY_Q,   # inside verdict → enters in_verdict_why
        _MASK_TEXT_CLOSE,
    ]
    adv = _simulate_ptad(tokens, cls_adv=9.0, count_adv=-5.0, scaled_adv=9.0,
                         structural_adv=struct_adv, verdict_why_adv=9.0)
    # SPAN_OPEN (idx 1) → count_adv=-5.0
    assert _close(adv[1], -5.0), "SPAN_OPEN → count_adv"
    # All others → structural_adv (idx 0,2,3,4,5,6,7,8)
    for idx in [0, 2, 3, 4, 5, 6, 7, 8]:
        assert _close(adv[idx], struct_adv), f"idx {idx} should be structural_adv={struct_adv}, got {adv[idx]}"


def test_span_open_gets_count_adv_not_structural():
    """SPAN_OPEN token gets count_adv, which can be negative."""
    tokens = [_MASK_SPAN_OPEN]
    adv = _simulate_ptad(tokens, cls_adv=1.0, count_adv=-0.8, scaled_adv=1.0, structural_adv=0.5)
    assert _close(adv[0], -0.8), f"SPAN_OPEN → count_adv=-0.8, got {adv[0]}"


# ---------------------------------------------------------------------------
# Tests: label_ctx_for_opt zeroing
# ---------------------------------------------------------------------------

def test_label_ctx_for_opt_zeros_all_verdict_tokens():
    """When label_ctx_for_opt=True, all verdict value tokens (type + why) get 0.0."""
    tokens = [
        _MASK_VERDICT_OPEN, DOC,       # type
        _MASK_ANN_WHY_Q, DOC,          # why
        _MASK_TEXT_CLOSE,
    ]
    adv = _simulate_ptad(tokens, cls_adv=1.0, count_adv=0.0, scaled_adv=0.0,
                         verdict_why_adv=0.9, label_ctx_for_opt=True)
    # TYPE=1, WHY=3
    assert _close(adv[1], 0.0), f"label_ctx_for_opt must zero type token, got {adv[1]}"
    assert _close(adv[3], 0.0), f"label_ctx_for_opt must zero why token, got {adv[3]}"
    # Structural tokens unaffected
    assert _close(adv[0], 0.5), "VERDICT_OPEN still structural_adv"
    assert _close(adv[4], 0.5), "TEXT_CLOSE still structural_adv"


def test_label_ctx_for_opt_does_not_affect_inner_annotations():
    """label_ctx_for_opt should NOT zero inner annotation (non-verdict) tokens."""
    tokens = [
        _MASK_ANN_OPEN, DOC, _MASK_ANN_CLOSE,   # inner annotation
    ]
    adv = _simulate_ptad(tokens, cls_adv=1.0, count_adv=0.0, scaled_adv=0.8,
                         label_ctx_for_opt=True)
    # Inner annotation value (idx 1) should still get scaled_adv
    assert _close(adv[1], 0.8), f"inner annotation not affected by label_ctx, got {adv[1]}"


# ---------------------------------------------------------------------------
# Tests: doc-copy tokens
# ---------------------------------------------------------------------------

def test_doc_copy_tokens_get_zero_when_format_fails():
    """Non-structural, non-annotation tokens outside attrs → 0.0 when format_ok=False."""
    tokens = [DOC, DOC, _MASK_SPAN_OPEN, DOC]  # two doc tokens, span, one more doc
    adv = _simulate_ptad(tokens, cls_adv=1.0, count_adv=0.5, scaled_adv=1.0,
                         doc_copy_adv=0.2, format_ok=False)
    # doc tokens (0,1,3) → 0.0 when format_ok=False
    assert _close(adv[0], 0.0)
    assert _close(adv[1], 0.0)
    assert _close(adv[3], 0.0)


def test_doc_copy_tokens_get_doc_copy_adv_when_format_ok():
    """Non-structural tokens outside attrs → doc_copy_adv when format_ok=True."""
    tokens = [DOC, _MASK_SPAN_OPEN, DOC]
    adv = _simulate_ptad(tokens, cls_adv=1.0, count_adv=0.5, scaled_adv=1.0,
                         doc_copy_adv=0.15, format_ok=True)
    assert _close(adv[0], 0.15), f"doc_copy_adv when format_ok, got {adv[0]}"
    assert _close(adv[2], 0.15), f"doc_copy_adv when format_ok, got {adv[2]}"


# ---------------------------------------------------------------------------
# Tests: complex / edge cases
# ---------------------------------------------------------------------------

def test_verdict_after_inner_annotation_tell_idx_correct():
    """tell_idx should NOT increment on VERDICT_OPEN — only on ANN_OPEN."""
    tokens = [
        _MASK_ANN_OPEN, DOC, _MASK_ANN_CLOSE,       # tell 0
        _MASK_VERDICT_OPEN, DOC, _MASK_TEXT_CLOSE,   # verdict, NOT a tell
        _MASK_ANN_OPEN, DOC, _MASK_ANN_CLOSE,       # tell 1
    ]
    per_tell_advs = [0.9, 0.4]
    adv = _simulate_ptad(tokens, cls_adv=0.5, count_adv=0.0, scaled_adv=0.0,
                         verdict_why_adv=0.0, per_tell_advs=per_tell_advs)
    # tell 0: idx 1 → 0.9
    assert _close(adv[1], 0.9), f"tell-0 should be 0.9, got {adv[1]}"
    # verdict TYPE: idx 4 → cls_adv=0.5
    assert _close(adv[4], 0.5), f"verdict type should be cls_adv=0.5, got {adv[4]}"
    # tell 1: idx 7 → 0.4
    assert _close(adv[7], 0.4), f"tell-1 should be 0.4, got {adv[7]}"


def test_why_q_inside_inner_annotation_does_not_trigger_verdict_why():
    """WHY_Q inside an inner annotation (not verdict) must NOT set in_verdict_why."""
    tokens = [
        _MASK_ANN_OPEN, DOC, _MASK_ANN_WHY_Q, DOC, _MASK_ANN_CLOSE,
    ]
    per_tell_advs = [0.7]
    adv = _simulate_ptad(tokens, cls_adv=9.0, count_adv=0.0, scaled_adv=0.6,
                         verdict_why_adv=9.0, per_tell_advs=per_tell_advs)
    # After WHY_Q inside annotation, in_verdict is still False, so value tokens still
    # use per_tell_advs (not cls_adv or verdict_why_adv).
    # ANN_OPEN=0, TYPE=1, WHY_Q=2, WHY=3, ANN_CLOSE=4
    assert _close(adv[1], 0.7), f"inner type should use per_tell_advs=0.7, got {adv[1]}"
    assert _close(adv[3], 0.7), f"inner why should use per_tell_advs=0.7, got {adv[3]}"


def test_multiple_verdict_sections_impossible_but_state_resets():
    """After TEXT_CLOSE resets verdict state, subsequent tokens should be unaffected."""
    tokens = [
        _MASK_VERDICT_OPEN, DOC, _MASK_ANN_WHY_Q, DOC, _MASK_TEXT_CLOSE,
        DOC,   # outside verdict — doc copy
        _MASK_VERDICT_OPEN, DOC, _MASK_TEXT_CLOSE,   # second verdict (edge case)
    ]
    adv = _simulate_ptad(tokens, cls_adv=1.0, count_adv=0.0, scaled_adv=0.0,
                         verdict_why_adv=0.5)
    # First verdict: type=1→cls_adv=1.0, why=3→verdict_why_adv=0.5
    assert _close(adv[1], 1.0)
    assert _close(adv[3], 0.5)
    # After TEXT_CLOSE, state resets. DOC at idx 5 → doc copy
    assert _close(adv[5], 0.0)
    # Second verdict type (idx 7) → cls_adv=1.0 (not in_verdict_why)
    assert _close(adv[7], 1.0)


def test_count_adv_negative_span_open():
    """SPAN_OPEN gets count_adv even when negative — this penalises too many tells."""
    tokens = [_MASK_SPAN_OPEN, _MASK_SPAN_OPEN, _MASK_SPAN_OPEN]
    adv = _simulate_ptad(tokens, cls_adv=0.0, count_adv=-0.6, scaled_adv=0.0)
    assert all(_close(a, -0.6) for a in adv), f"all SPAN_OPENs → -0.6, got {adv}"


# ---------------------------------------------------------------------------
# Tests: verdict_why_quality_reward
# ---------------------------------------------------------------------------

import sys
_need_mocks = [
    ("tinker", _make_tinker_module),
    ("weave", _make_weave_module),
    ("wandb", _make_wandb_module),
    ("transformers", _make_transformers_module),
    ("openai", _make_openai_module),
    ("dotenv", _make_dotenv_module),
]
for _n, _m in _need_mocks:
    if _n not in sys.modules:
        sys.modules[_n] = _m()

from rl_detector.rewards import verdict_why_quality_reward  # noqa: E402


def test_verdict_why_empty_returns_zero():
    assert _close(verdict_why_quality_reward(""), 0.0)
    assert _close(verdict_why_quality_reward("   "), 0.0)
    assert _close(verdict_why_quality_reward(None), 0.0)


def test_verdict_why_very_short_low_score():
    """Single word → near-zero length score."""
    score = verdict_why_quality_reward("short")
    assert score < 0.1, f"very short text should score < 0.1, got {score}"


def test_verdict_why_ideal_length_scores_high():
    """75 distinct words → length=1.0, repetition≈1.0 → total near 1.0."""
    words = [f"word{i}" for i in range(75)]
    text = " ".join(words)
    score = verdict_why_quality_reward(text)
    assert score > 0.9, f"ideal-length diverse text should score > 0.9, got {score}"


def test_verdict_why_too_short_scores_low():
    """10 words → below word_low=50 → quadratic ramp → (10/50)^2 = 0.04."""
    words = [f"token{i}" for i in range(10)]
    score = verdict_why_quality_reward(" ".join(words))
    expected = (10 / 50) ** 2  # 0.04, before rep penalty
    assert score <= expected + 0.01, f"10-word text should be ≤ {expected:.3f}, got {score}"


def test_verdict_why_too_long_scores_low():
    """350 words (> 2×150) → 0.0 length score → 0.0 total."""
    words = [f"token{i}" for i in range(350)]
    score = verdict_why_quality_reward(" ".join(words))
    assert _close(score, 0.0), f"350-word text should score 0.0, got {score}"


def test_verdict_why_high_repetition_penalized():
    """60 identical words → strong repetition penalty."""
    score = verdict_why_quality_reward(" ".join(["word"] * 60))
    assert score < 0.05, f"highly repetitive text should score < 0.05, got {score}"


def test_verdict_why_moderate_repetition_partial_penalty():
    """50 diverse words + 10 repeated phrase → partial penalty."""
    diverse = [f"tok{i}" for i in range(50)]
    repeated = ["like this very much"] * 5  # rough repetition
    text = " ".join(diverse + " ".join(repeated).split())
    score_pure = verdict_why_quality_reward(" ".join(diverse))
    score_rep = verdict_why_quality_reward(text)
    # Repetitive version should score lower than pure diverse version
    assert score_rep < score_pure, f"repetitive should score lower: {score_rep} vs {score_pure}"


def test_verdict_why_150_word_boundary():
    """Exactly 150 words of distinct content → max score."""
    words = [f"uniqueword{i}" for i in range(150)]
    score = verdict_why_quality_reward(" ".join(words))
    assert score > 0.95, f"150-word ideal should score > 0.95, got {score}"


def test_verdict_why_151_word_slight_decay():
    """151 words → just past upper boundary → slight decay from 1.0."""
    words = [f"uniqueword{i}" for i in range(151)]
    score = verdict_why_quality_reward(" ".join(words))
    assert score < 1.0, f"151-word text should be < 1.0, got {score}"
    assert score > 0.9, f"151-word text should still be > 0.9, got {score}"
