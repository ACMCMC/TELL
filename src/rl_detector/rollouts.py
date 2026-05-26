"""Generate rollouts from the current policy."""

import asyncio
import logging
import math
import re

import tinker

from rl_detector.config import CFG

from rl_detector.prompt_utils import (
    ANNOTATION_TOKEN_REMAP,
    ANN_SPECIAL_ID_ANN_PREFIX,
    ANN_SPECIAL_ID_CLOSE,
    ANN_SPECIAL_ID_SCORE_Q,
    ANN_SPECIAL_ID_SPAN_OPEN,
    ANN_SPECIAL_ID_TEXT_CLOSE,
    ANN_SPECIAL_ID_TEXT_OPEN,
    ANN_SPECIAL_ID_VERDICT_PREFIX,
    ANN_SPECIAL_ID_WHY_Q,
    format_prompt_for_model,
)
from rl_detector.prompts import (
    get_focus_hint,
    label_think_continuation,
    label_think_prefix,
)
from rl_detector.tell_xml import strip_return_token
from rl_detector.rewards import (
    format_exception_diag as _format_exception_diag,
    safe_format_diagnostics as _safe_format_diagnostics_for_grpo,
)
from rl_detector.format_fix import _apply_format_fix_to_text_fields

logger = logging.getLogger(__name__)


_stub_open_tokens_cache: list[int] | None = None
_stub_close_tokens_cache: list[int] | None = None
_return_token_id_cache: int | None = None


def _get_return_token_id(tokenizer) -> int | None:
    global _return_token_id_cache
    if _return_token_id_cache is None:
        if tokenizer is None:
            return None
        ids = tokenizer.encode("<|return|>", add_special_tokens=False)
        _return_token_id_cache = int(ids[0]) if ids else -1
    return _return_token_id_cache if _return_token_id_cache != -1 else None


def _get_analysis_stub_tokens(tokenizer, think_already_open: bool) -> tuple[list[int], list[int]]:
    """Return (open_tokens, close_tokens) for the analysis channel stub, cached after first call.

    open_tokens: tokens for <|channel|>analysis<|message|>, or [] if template already opens it.
    close_tokens: tokens for <|end|><|start|>assistant<|channel|>final<|message|>.
    """
    global _stub_open_tokens_cache, _stub_close_tokens_cache
    if _stub_open_tokens_cache is None:
        open_str = "" if think_already_open else "<|channel|>analysis<|message|>"
        close_str = "<|end|><|start|>assistant<|channel|>final<|message|>"
        _stub_open_tokens_cache = tokenizer.encode(open_str, add_special_tokens=False) if open_str else []
        _stub_close_tokens_cache = tokenizer.encode(close_str, add_special_tokens=False)
        logger.debug(
            "startup | analysis stub tokens: open=%d close=%d (think_already_open=%s)",
            len(_stub_open_tokens_cache), len(_stub_close_tokens_cache), think_already_open,
        )
    return _stub_open_tokens_cache, _stub_close_tokens_cache


_CHANNEL_BLOCK_RE = re.compile(
    r"<\|channel\|>\s*([^<\s]+)\s*<\|message\|>(.*?)(?=(?:<\|channel\|>)|(?:<\|end\|>)|(?:<\|return\|>)|$)",
    re.DOTALL,
)
_THINK_BLOCK_RE = re.compile(r"<\|channel\|>\s*analysis\s*<\|message\|>.*?<\|end\|>", re.DOTALL)


# Token IDs for annotation structural tokens (guaranteed single-token by ANNOTATION_TOKEN_REMAP).
# State machine transitions:
#   IN_DOC  + SPAN_OPEN  → 1.0, stay IN_DOC   (doc text follows)
#   IN_DOC  + ANN_OPEN   → 1.0, enter IN_ATTRS
#   IN_ATTRS + any token → 1.0
#   IN_ATTRS + ANN_CLOSE → 1.0, exit → IN_DOC
#   IN_DOC  + other      → 0.0  (verbatim doc copy)
_MASK_TEXT_OPEN = ANN_SPECIAL_ID_TEXT_OPEN
_MASK_VERDICT_OPEN = ANN_SPECIAL_ID_VERDICT_PREFIX
_MASK_TEXT_CLOSE = ANN_SPECIAL_ID_TEXT_CLOSE
_MASK_SPAN_OPEN = ANN_SPECIAL_ID_SPAN_OPEN
_MASK_ANN_OPEN = ANN_SPECIAL_ID_ANN_PREFIX
_MASK_ANN_WHY_Q = ANN_SPECIAL_ID_WHY_Q
_MASK_ANN_SCORE_Q = ANN_SPECIAL_ID_SCORE_Q
_MASK_ANN_CLOSE = ANN_SPECIAL_ID_CLOSE
_MASK_ANN_STRUCT = (
    _MASK_TEXT_OPEN,
    _MASK_VERDICT_OPEN,
    _MASK_TEXT_CLOSE,
    _MASK_SPAN_OPEN,
    _MASK_ANN_OPEN,
    _MASK_ANN_WHY_Q,
    _MASK_ANN_SCORE_Q,
    _MASK_ANN_CLOSE,
)

assert frozenset(ANNOTATION_TOKEN_REMAP.keys()) == frozenset(
    _MASK_ANN_STRUCT
), "rollouts mask IDs must match prompt_utils.ANNOTATION_TOKEN_REMAP keys"


# Integer token type labels used by compute_token_type_mask.
TOKEN_TYPE_REASONING = 0
TOKEN_TYPE_STRUCTURAL = 1
TOKEN_TYPE_SPAN_OPEN = 2
TOKEN_TYPE_VERDICT_TYPE = 3
TOKEN_TYPE_VERDICT_WHY = 4
TOKEN_TYPE_ANN_TYPE = 5   # inner tell: type value tokens
TOKEN_TYPE_ANN_WHY = 6    # inner tell: why value tokens (after WHY_Q, before SCORE_Q)
TOKEN_TYPE_DOC_COPY = 7
TOKEN_TYPE_ANN_SCORE = 8  # inner tell: score value tokens (after SCORE_Q)
TOKEN_TYPE_VERDICT_SCORE = 9  # verdict score value tokens (after SCORE_Q in verdict)

# Convenience set: token types that are not doc-copy and not reasoning
TOKEN_TYPES_ANNOTATION = frozenset({
    TOKEN_TYPE_STRUCTURAL, TOKEN_TYPE_SPAN_OPEN,
    TOKEN_TYPE_VERDICT_TYPE, TOKEN_TYPE_VERDICT_WHY, TOKEN_TYPE_VERDICT_SCORE,
    TOKEN_TYPE_ANN_TYPE, TOKEN_TYPE_ANN_WHY, TOKEN_TYPE_ANN_SCORE,
})

_PTOK_LOSS_SCALE_BY_TYPE: dict[int, str] = {
    TOKEN_TYPE_SPAN_OPEN: "span_open",
    TOKEN_TYPE_STRUCTURAL: "structural",
    TOKEN_TYPE_VERDICT_TYPE: "verdict_type",
    TOKEN_TYPE_VERDICT_WHY: "verdict_why",
    TOKEN_TYPE_VERDICT_SCORE: "verdict_score",
    TOKEN_TYPE_ANN_TYPE: "ann_type",
    TOKEN_TYPE_ANN_WHY: "ann_why",
    TOKEN_TYPE_ANN_SCORE: "ann_score",
}


def compute_token_type_mask(
    tokenizer,
    completion_tokens: list[int],
    n_reasoning_tokens: int,
) -> list[int]:
    """Per-token type label for the full completion (reasoning + response).

    Returns list[int] of length len(completion_tokens) using TOKEN_TYPE_* constants.
    Reasoning tokens are classified as TOKEN_TYPE_REASONING.
    Response tokens are classified by their structural role via the special-token state machine.
    """
    R = max(0, int(n_reasoning_tokens))
    result: list[int] = [TOKEN_TYPE_REASONING] * min(R, len(completion_tokens))
    response_tokens = completion_tokens[R:]
    if not response_tokens:
        return result

    _return_id = _get_return_token_id(tokenizer)
    _STRUCT = frozenset(_MASK_ANN_STRUCT)

    in_attrs = False
    in_verdict = False
    in_verdict_why = False
    in_verdict_score = False
    in_ann_why = False    # True after WHY_Q inside a non-verdict annotation
    in_ann_score = False  # True after SCORE_Q inside a non-verdict annotation

    for tok in response_tokens:
        t = int(tok)
        if _return_id is not None and t == _return_id:
            result.append(TOKEN_TYPE_STRUCTURAL)
        elif t in _STRUCT:
            if t == _MASK_SPAN_OPEN:
                result.append(TOKEN_TYPE_SPAN_OPEN)
            elif t == _MASK_ANN_OPEN:
                in_attrs = True; in_verdict = False; in_verdict_why = False; in_verdict_score = False; in_ann_why = False; in_ann_score = False
                result.append(TOKEN_TYPE_STRUCTURAL)
            elif t == _MASK_VERDICT_OPEN:
                in_attrs = True; in_verdict = True; in_verdict_why = False; in_verdict_score = False; in_ann_why = False; in_ann_score = False
                result.append(TOKEN_TYPE_STRUCTURAL)
            elif t == _MASK_ANN_CLOSE or t == _MASK_TEXT_CLOSE:
                in_attrs = False; in_verdict = False; in_verdict_why = False; in_verdict_score = False; in_ann_why = False; in_ann_score = False
                result.append(TOKEN_TYPE_STRUCTURAL)
            elif in_verdict and t == _MASK_ANN_WHY_Q:
                in_verdict_why = True; in_verdict_score = False
                result.append(TOKEN_TYPE_STRUCTURAL)
            elif in_verdict and t == _MASK_ANN_SCORE_Q:
                in_verdict_why = False; in_verdict_score = True
                result.append(TOKEN_TYPE_STRUCTURAL)
            elif not in_verdict and in_attrs and t == _MASK_ANN_WHY_Q:
                in_ann_why = True
                result.append(TOKEN_TYPE_STRUCTURAL)
            elif not in_verdict and in_attrs and t == _MASK_ANN_SCORE_Q:
                in_ann_score = True
                result.append(TOKEN_TYPE_STRUCTURAL)
            else:  # TEXT_OPEN, WHY_Q (outside attrs), etc.
                result.append(TOKEN_TYPE_STRUCTURAL)
        elif in_attrs:
            if in_verdict:
                if in_verdict_score:
                    result.append(TOKEN_TYPE_VERDICT_SCORE)
                elif in_verdict_why:
                    result.append(TOKEN_TYPE_VERDICT_WHY)
                else:
                    result.append(TOKEN_TYPE_VERDICT_TYPE)
            elif in_ann_score:
                result.append(TOKEN_TYPE_ANN_SCORE)
            else:
                result.append(TOKEN_TYPE_ANN_WHY if in_ann_why else TOKEN_TYPE_ANN_TYPE)
        else:
            result.append(TOKEN_TYPE_DOC_COPY)

    return result


TASK_SEG_DOC = 0
TASK_SEG_VERDICT = 1
TASK_SEG_SPAN_BASE = 2


def compute_response_task_segment_ids(
    tokenizer,
    completion_tokens: list[int],
    n_reasoning_tokens: int,
) -> list[int]:
    """Per response token: doc (0), verdict (1), or span k (2+k).

    Span k starts at SPAN_OPEN and runs until the next SPAN_OPEN or the verdict block.
    """
    R = max(0, int(n_reasoning_tokens))
    response_tokens = completion_tokens[R:]
    if not response_tokens:
        return []
    span_idx = -1
    in_verdict = False
    out: list[int] = []
    for tok in response_tokens:
        t = int(tok)
        if t == _MASK_VERDICT_OPEN:
            in_verdict = True
            span_idx = -1
            out.append(TASK_SEG_VERDICT)
        elif in_verdict:
            out.append(TASK_SEG_VERDICT)
        elif t == _MASK_SPAN_OPEN:
            in_verdict = False
            span_idx += 1
            out.append(TASK_SEG_SPAN_BASE + span_idx)
        elif span_idx >= 0:
            out.append(TASK_SEG_SPAN_BASE + span_idx)
        else:
            out.append(TASK_SEG_DOC)
    return out


def _renormalize_region(weights: list[float], indices: list[int], target_sum: float) -> None:
    s = sum(weights[i] for i in indices)
    if s <= 0.0 or target_sum <= 0.0:
        return
    f = target_sum / s
    for i in indices:
        weights[i] *= f


def compute_task_loss_weights(
    tokenizer,
    completion_tokens: list[int],
    n_reasoning_tokens: int,
    span_open_loss_mass: float,
    span_ann_mass: float,
    ptok_loss_scales: dict[str, float],
) -> list[float]:
    """PPO weights: doc and verdict each sum to 1; span region sums to span_ann_mass.

    SPAN_OPEN tokens each get span_open_loss_mass (fixed, not diluted by long whies).
    Other tokens inside span segments share the remaining span_ann_mass equally.
    ptok_loss_scales multiplies per token type (ann_type, ann_why, …) then each region is
    renormalized so doc/verdict/span total mass is unchanged.
    """
    segs = compute_response_task_segment_ids(
        tokenizer=tokenizer,
        completion_tokens=completion_tokens,
        n_reasoning_tokens=n_reasoning_tokens,
    )
    if not segs:
        return []
    R = max(0, int(n_reasoning_tokens))
    types = compute_token_type_mask(
        tokenizer=tokenizer,
        completion_tokens=completion_tokens,
        n_reasoning_tokens=n_reasoning_tokens,
    )[R:]
    n_doc = sum(1 for s in segs if s == TASK_SEG_DOC)
    n_verdict = sum(1 for s in segs if s == TASK_SEG_VERDICT)
    n_span_open = sum(1 for t in types if t == TOKEN_TYPE_SPAN_OPEN)
    open_budget = min(float(span_ann_mass), n_span_open * float(span_open_loss_mass))
    rest_budget = float(span_ann_mass) - open_budget
    n_span_other = sum(
        1
        for seg, typ in zip(segs, types)
        if seg >= TASK_SEG_SPAN_BASE and typ != TOKEN_TYPE_SPAN_OPEN
    )
    weights: list[float] = []
    for seg, typ in zip(segs, types):
        if seg == TASK_SEG_DOC:
            weights.append(1.0 / n_doc if n_doc else 0.0)
        elif seg == TASK_SEG_VERDICT:
            weights.append(1.0 / n_verdict if n_verdict else 0.0)
        elif typ == TOKEN_TYPE_SPAN_OPEN:
            w_open = float(span_open_loss_mass)
            if n_span_other == 0 and rest_budget > 0.0 and n_span_open > 0:
                w_open = w_open + rest_budget / n_span_open
            weights.append(w_open)
        else:
            weights.append(rest_budget / n_span_other if n_span_other else 0.0)

    doc_idx = [i for i, seg in enumerate(segs) if seg == TASK_SEG_DOC]
    verdict_idx = [i for i, seg in enumerate(segs) if seg == TASK_SEG_VERDICT]
    span_idx = [i for i, seg in enumerate(segs) if seg >= TASK_SEG_SPAN_BASE]
    for i, typ in enumerate(types):
        key = _PTOK_LOSS_SCALE_BY_TYPE.get(typ)
        if key is not None:
            weights[i] *= float(ptok_loss_scales[key])
    _renormalize_region(weights=weights, indices=doc_idx, target_sum=1.0)
    _renormalize_region(weights=weights, indices=verdict_idx, target_sum=1.0)
    _renormalize_region(weights=weights, indices=span_idx, target_sum=float(span_ann_mass))
    return weights


def compute_annotation_token_mask(
    tokenizer,
    completion_tokens: list[int],
    n_reasoning_tokens: int,
) -> list[float]:
    """Per-token RL mask of length ``len(completion_tokens) - n_reasoning_tokens``.

    All 8 structural tokens and attribute value tokens between open/close pairs → 1.0.
    Document-verbatim copy tokens → 0.0.

    New format: <text>(200009) … <verdict type="(200010) [attrs] " /></text>(200011)
    Inner tells: <span>(200013) … <annotation type="(200014) [attrs] " /></span>(200017)
    Attr delimiters: " why="(200015)  " score="(200016)

    Document tokens get zero gradient because they should be copied verbatim. KL still
    applies to them via the separate mask field so they track the reference distribution.
    """
    R = max(0, int(n_reasoning_tokens))
    response_tokens = completion_tokens[R:]
    if not response_tokens:
        return []
    _return_id = _get_return_token_id(tokenizer)
    out: list[float] = []
    in_attrs = False
    for tok in response_tokens:
        t = int(tok)
        if t == _MASK_ANN_OPEN or t == _MASK_VERDICT_OPEN:
            in_attrs = True
            out.append(1.0)
        elif t == _MASK_ANN_CLOSE or t == _MASK_TEXT_CLOSE:
            in_attrs = False
            out.append(1.0)
        elif t == _MASK_SPAN_OPEN or t == _MASK_TEXT_OPEN or in_attrs:
            out.append(1.0)
        elif _return_id is not None and t == _return_id:
            out.append(1.0)
        else:
            out.append(0.0)
    return out


def compute_per_tell_token_weights(
    completion_tokens: list[int],
    n_reasoning_tokens: int,
    credibility_scores: list[float | None],
    alpha: float = 2.0,
) -> list[float]:
    """Per-token advantage weight multipliers based on per-tell rubric credibility.

    Tokens belonging to high-credibility annotations get amplified advantage;
    low-credibility annotations get dampened advantage. Weights are re-scaled
    so mean = 1.0, keeping expected gradient magnitude unchanged.

    Returns uniform weights [1.0, ...] when credibility scores are unavailable.
    Returns a list of length len(completion_tokens) - n_reasoning_tokens.
    """
    import math
    R = max(0, int(n_reasoning_tokens))
    response_tokens = completion_tokens[R:]
    n = len(response_tokens)
    if not response_tokens or not credibility_scores or any(c is None for c in credibility_scores):
        return [1.0] * n

    if alpha <= 0:
        return [1.0] * n
    exp_scores = [math.exp(alpha * float(c)) for c in credibility_scores]
    total = sum(exp_scores)
    if total < 1e-9:
        return [1.0] * n
    n_tells = len(credibility_scores)
    # weight_i = softmax_i * n_tells  →  mean weight = 1.0
    weights = [e / total * n_tells for e in exp_scores]

    # Walk tokens and assign tell index based on ANN_OPEN boundaries (verdict attrs reuse outer weights).
    out: list[float] = []
    tell_idx = -1
    in_attrs = False
    for tok in response_tokens:
        t = int(tok)
        if t == _MASK_ANN_OPEN:
            tell_idx += 1
            in_attrs = True
            out.append(weights[tell_idx] if tell_idx < n_tells else 1.0)
        elif t == _MASK_VERDICT_OPEN:
            in_attrs = True
            out.append(1.0)
        elif t == _MASK_ANN_CLOSE or t == _MASK_TEXT_CLOSE:
            in_attrs = False
            out.append(weights[tell_idx] if tell_idx < n_tells else 1.0)
        elif t == _MASK_SPAN_OPEN or t == _MASK_TEXT_OPEN or in_attrs:
            out.append(weights[tell_idx] if (in_attrs and tell_idx < n_tells) else 1.0)
        else:
            out.append(1.0)  # doc-copy: weight irrelevant since ann_mask = 0
    return out


def compute_structural_token_mask(
    completion_tokens: list[int],
    n_reasoning_tokens: int,
) -> list[bool]:
    """True for tokens that are mandatory structural boilerplate (span + five annotation chunks).

    These tokens must appear regardless of annotation quality. When a rollout has a
    negative advantage, clipping to max(0, adv) for these tokens prevents the model
    from learning to drop closing tags because a particular rollout was below average.
    """
    R = max(0, int(n_reasoning_tokens))
    return [int(t) in _MASK_ANN_STRUCT for t in completion_tokens[R:]]


def extract_response_text(text: str) -> str:
    """Best-effort extraction of user-facing response text from a full completion."""
    channel_blocks = list(_CHANNEL_BLOCK_RE.finditer(text))
    if channel_blocks:
        for m in channel_blocks:
            if m.group(1).strip().lower() == "final":
                return m.group(2).strip()
        return channel_blocks[-1].group(2).strip()

    without_thinking = _THINK_BLOCK_RE.sub("", text).strip()
    without_start_end_backticks = without_thinking.strip("`")
    return without_start_end_backticks if without_start_end_backticks else text.strip()


def decode_response_text(tokenizer, completion_tokens: list[int], completion_text: str, force_stub_sampling: bool) -> str:
    """Return the final XML text seen by format checks and scoring."""
    if force_stub_sampling:
        # Annotation remap uses added_token IDs marked special=True (span + split annotation wire).
        # skip_special_tokens=True would omit them and break format_diagnostics startswith(SP_OP).
        txt = tokenizer.decode(token_ids=completion_tokens, skip_special_tokens=False).strip()
        for _tail in (
            getattr(tokenizer, "eos_token", None),
            getattr(tokenizer, "pad_token", None),
        ):
            if _tail and txt.endswith(_tail):
                txt = txt[: -len(_tail)].strip()
        return strip_return_token(txt)
    return extract_response_text(completion_text)


async def _sample_training_rollout(
    sampling_client,
    tokenizer,
    document: str,
    rollout_index: int,
    main_label_hint: int,
    inject_label: bool,
    seed: int | None,
    doc_stratum: str | None = None,
    think_already_open: bool = False,
    focus_hint: str | None = None,
) -> dict:
    """Sample one training rollout.

    ``document``: LOGICAL plain source text passed only through ``format_prompt_for_model``.

    When force_stub_sampling=True:
      The analysis channel is pre-injected as a deterministic stub so the model only
      generates response tokens. Any focus_hint or label is placed inside the analysis
      channel for sampling.

    When force_stub_sampling=False (legacy):
      The model generates full reasoning and final response.
    """
    force_stub = bool(getattr(CFG.sampling, "force_stub_sampling", False))
    fix_err = getattr(CFG.training, "fix_format_errors", False)
    neutral_prompt_text, neutral_formatted = format_prompt_for_model(tokenizer=tokenizer, text=document)
    neutral_prompt_tokens = tokenizer.encode(neutral_formatted)

    rollout_seed = None if seed is None else seed + rollout_index
    sampling_params = tinker.SamplingParams(
        max_tokens=CFG.sampling.max_tokens,
        temperature=CFG.sampling.temperature,
        top_p=CFG.sampling.top_p,
        reasoning_effort=CFG.sampling.reasoning_effort,
        seed=rollout_seed,
    )

    was_text_fixed = False
    wrong_response_text = None
    sampling_prompt_tokens: list[int] = []  # set below; used for Weave/audit logging

    if force_stub:
        stub_open, stub_close = _get_analysis_stub_tokens(tokenizer, think_already_open)

        # Build sampling-only analysis content: label hint and/or focus hint.
        # With think_already_open=False, stub_open already includes <|channel|>analysis<|message|>,
        # so content tokens are just the text that follows it.
        analysis_content_tokens: list[int] = []
        if inject_label:
            # In force_stub mode, stub_open already injects '<|channel|>analysis<|message|>',
            # so the label content is just the bare continuation text regardless of think_already_open.
            label_text = label_think_continuation(main_label_hint)
            analysis_content_tokens += tokenizer.encode(label_text, add_special_tokens=False)
        if focus_hint:
            analysis_content_tokens += tokenizer.encode(" " + focus_hint, add_special_tokens=False)

        sampling_prompt_tokens = neutral_prompt_tokens + stub_open + analysis_content_tokens + stub_close
        datum_prompt_tokens = neutral_prompt_tokens + stub_open + stub_close

        sampled = await sampling_client.sample_async(
            prompt=tinker.ModelInput.from_ints(sampling_prompt_tokens),
            num_samples=1,
            sampling_params=sampling_params,
        )
        seq = sampled.sequences[0]
        completion_tokens = list(seq.tokens)
        completion_logprobs = list(seq.logprobs) if seq.logprobs is not None else [0.0] * len(completion_tokens)
        if not any(lp != 0.0 for lp in completion_logprobs):
            logger.warning("rollout %d: all completion_logprobs are 0.0", rollout_index)

        completion_text = tokenizer.decode(completion_tokens, skip_special_tokens=False)
        response_text = decode_response_text(
            tokenizer=tokenizer,
            completion_tokens=completion_tokens,
            completion_text=completion_text,
            force_stub_sampling=force_stub,
        )
        n_reasoning_tokens = 0  # stub is in datum_prompt_tokens; all sampled tokens are response
        fmt_at_sample = _safe_format_diagnostics_for_grpo(response_text, document)

        if fix_err is True:
            (
                response_text,
                completion_text,
                completion_tokens,
                completion_logprobs,
                was_text_fixed,
                wrong_response_text,
            ) = _apply_format_fix_to_text_fields(
                response_text=response_text,
                completion_text=completion_text,
                completion_tokens=completion_tokens,
                completion_logprobs=completion_logprobs,
                document=document,
                tokenizer=tokenizer,
            )
        # prompt drift: sampling may include injected label/focus text, datum keeps empty stub
        force_stub_prompt_mismatch = sampling_prompt_tokens != datum_prompt_tokens
        needs_rescore = was_text_fixed or force_stub_prompt_mismatch
        logger.debug(
            "rollout %d: force_stub | focus=%s label=%s | %d response tokens",
            rollout_index, focus_hint is not None, inject_label, len(completion_tokens),
        )


    # full_output_text shows what was actually fed to the model during sampling (includes
    # focus hint / label in the analysis channel), not the datum (which has empty stub).
    full_output_text = tokenizer.decode(sampling_prompt_tokens + completion_tokens, skip_special_tokens=False)
    fmt = _safe_format_diagnostics_for_grpo(response_text, document)
    if was_text_fixed:
        assert not bool(fmt_at_sample["ok"]), (
            "format repair implies sample-time format failed; "
            f"fmt_at_sample={fmt_at_sample!r}"
        )
        rescore_reason = "format_fix"
    elif force_stub and sampling_prompt_tokens != datum_prompt_tokens:
        rescore_reason = "force_stub_prompt_mismatch"
    else:
        rescore_reason = None

    return {
        "neutral_prompt_text": neutral_prompt_text,
        "completion_text": completion_text,
        "response_text": response_text,
        "completion_tokens": completion_tokens,
        "completion_logprobs": completion_logprobs,
        "n_reasoning_tokens": n_reasoning_tokens,
        "sampling_prompt_tokens": sampling_prompt_tokens,
        "datum_prompt_tokens": datum_prompt_tokens,
        "full_output_text": full_output_text,
        "main_label_hint": main_label_hint,
        "inject_label": inject_label,
        "was_text_fixed": was_text_fixed,
        "wrong_response_text": wrong_response_text,
        "needs_rescore": needs_rescore,
        "rescore_reason": rescore_reason,
        # format_ok / format_reason: current (post-repair) text, so reason is often "ok" when was_text_fixed.
        # format_ok_at_sample / format_reason_at_sample: raw completion before format_fix.try_fix_response.
        "format_ok": bool(fmt["ok"]),
        "format_reason": str(fmt["reason"]),
        "format_char_diff_count": int(fmt["char_diff_count"]),
        "format_ok_at_sample": bool(fmt_at_sample["ok"]),
        "format_reason_at_sample": str(fmt_at_sample["reason"]),
    }


async def generate_rollouts(
    sampling_client,
    tokenizer,
    document: str,
    main_label_hints: list[int],
    inject_label_flags: list[bool],    # whether to inject label into reasoning prefix per rollout
    K: int | None = None,
    seed: int | None = None,
    think_already_open: bool = False,  # cached from detect_assistant_generation_suffix at startup
    doc_stratum: str | None = None,
) -> list[dict]:
    """
    Generate K rollouts.

    ``document``: LOGICAL source text exactly as stored in the dataset rows (never pre‑XML‑escaped).

    The user message always comes from ``format_prompt_for_model(.., text=document)``
    which applies the single escaping rule for ``<<<>>>`` payloads. For inject_label=True rollouts, the
    label is prepended to the reasoning chain as a forced prefix, anchoring the model's
    reasoning to the known ground truth.

    Args:
        inject_label_flags: per-rollout flag; True for correct/flip noise modes.
        think_already_open: whether the chat template already emits <|channel|>analysis<|message|>
                            in the assistant generation prefix (detected once at startup).
    """
    if K is None:
        K = CFG.training.k
    if len(main_label_hints) != K:
        raise AssertionError(
            "main_label_hints length mismatch: "
            f"expected K={K}, got {len(main_label_hints)}; "
            f"hints={main_label_hints}"
        )
    if len(inject_label_flags) != K:
        raise AssertionError(
            "inject_label_flags length mismatch: "
            f"expected K={K}, got {len(inject_label_flags)}; "
            f"flags={inject_label_flags}"
        )

    tasks = [
        _sample_training_rollout(
            sampling_client, tokenizer, document, i,
            main_label_hints[i], inject_label_flags[i], seed,
            doc_stratum=doc_stratum,
            think_already_open=think_already_open,
            focus_hint=get_focus_hint(i),
        )
        for i in range(K)
    ]
    rollouts = list(await asyncio.gather(*tasks))

    fix_err = getattr(CFG.training, "fix_format_errors", False)
    if fix_err == "auto":
        pass_rate = sum(1 for r in rollouts if bool(_safe_format_diagnostics_for_grpo(r["response_text"], document)["ok"])) / len(rollouts)
        if pass_rate < float(getattr(CFG.training, "fix_format_auto_threshold", 0.9)):
            def _apply_fix(r: dict) -> dict:
                """Apply text fix and mark needs_rescore — no logprob call."""
                (
                    fixed_response,
                    fixed_completion,
                    fixed_tokens,
                    fixed_logprobs,
                    was_fixed,
                    wrong,
                ) = _apply_format_fix_to_text_fields(
                    response_text=r["response_text"],
                    completion_text=r["completion_text"],
                    completion_tokens=r["completion_tokens"],
                    completion_logprobs=r["completion_logprobs"],
                    document=document,
                    tokenizer=tokenizer,
                )
                if not was_fixed:
                    return r
                return {**r,
                    "response_text": fixed_response,
                    "completion_text": fixed_completion,
                    "completion_tokens": fixed_tokens,
                    "completion_logprobs": fixed_logprobs,
                    "was_text_fixed": True,
                    "wrong_response_text": wrong,
                    "needs_rescore": True,
                }
            rollouts = [
                _apply_fix(r) if not bool(_safe_format_diagnostics_for_grpo(r["response_text"], document)["ok"]) else r
                for r in rollouts
            ]
            # Refresh format status fields after auto-fix transformations.
            rollouts = [
                {
                    **r,
                    "format_ok": bool(_fmt["ok"]),
                    "format_reason": str(_fmt["reason"]),
                    "format_char_diff_count": int(_fmt["char_diff_count"]),
                }
                for r in rollouts
                for _fmt in [_safe_format_diagnostics_for_grpo(r["response_text"], document)]
            ]

    for _r in rollouts:
        if bool(_r.get("was_text_fixed")):
            assert not bool(_r["format_ok_at_sample"]), (
                "format repair implies sample-time format failed (stale replay row or format/dataset mismatch?)",
                _r.get("format_reason_at_sample"),
            )

    # Batch-recompute logprobs for all rollouts that need it in one asyncio.gather.
    rescore_indices = [i for i, r in enumerate(rollouts) if r.get("needs_rescore")]
    if rescore_indices:
        async def _rescore(r: dict) -> list[float]:
            prompt = r["datum_prompt_tokens"]
            full_input = tinker.ModelInput.from_ints(prompt + r["completion_tokens"])
            all_lps: list[float | None] = await sampling_client.compute_logprobs_async(full_input)
            return [lp if lp is not None else 0.0 for lp in all_lps[len(prompt):]]

        new_logprobs = await asyncio.gather(*[_rescore(rollouts[i]) for i in rescore_indices])
        for i, lps in zip(rescore_indices, new_logprobs):
            r = rollouts[i]
            is_ratio = 1.0
            if r.get("rescore_reason") == "force_stub_prompt_mismatch":
                mode = str(getattr(CFG.training, "is_ratio_prompt_mismatch_mode", "neutral"))
                if mode == "neutral":
                    # The two prompts differ by injected analysis-channel content (label/focus hint).
                    # Neutral mode disables mismatch IS correction to avoid collapsing injected rollouts.
                    is_ratio = 1.0
                elif mode == "token_mean_log_ratio":
                    # Length-normalized geometric mean ratio:
                    # exp(mean_t [log p_new(a_t) - log p_old(a_t)]).
                    old_lps = r["completion_logprobs"]
                    if len(old_lps) != len(lps):
                        raise AssertionError(
                            "token_mean_log_ratio requires equal logprob lengths, got "
                            f"{len(old_lps)} vs {len(lps)}"
                        )
                    log_ratio_mean = sum(n - o for o, n in zip(old_lps, lps)) / max(1, len(lps))
                    is_ratio = math.exp(max(min(log_ratio_mean, 20.0), -20.0))
                else:
                    raise ValueError(
                        f"unknown training.is_ratio_prompt_mismatch_mode={mode!r}; "
                        "expected 'neutral' or 'token_mean_log_ratio'"
                    )
            rollouts[i] = {**r, "completion_logprobs": lps, "needs_rescore": False, "is_ratio": is_ratio}
        logger.debug("rescored logprobs for %d/%d rollouts in batch", len(rescore_indices), len(rollouts))

    return rollouts
