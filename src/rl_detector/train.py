"""Main GRPO training loop."""

from __future__ import annotations

import asyncio
import datetime
import importlib
import json
import logging
import math
import os
import pathlib
import random
import time
import uuid
from typing import NamedTuple

import numpy as np

import tinker
import torch
import wandb
from dotenv import load_dotenv
from tinker import TensorData
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from transformers import AutoTokenizer

import hydra
from omegaconf import DictConfig, OmegaConf

from rl_detector.config import CFG, config_as_dict
from rl_detector.prompts import label_think_continuation
from rl_detector.tell_xml import escape_document_piece
from rl_detector.data import (
    StratumReplaySampler,
    StratumSampler,
    UniformReplaySampler,
    doc_ease_uid,
    doc_stratum_key,
    load_docs_preprocessed,
    pick_stratum_probe_docs,
)
from rl_detector.frozen import aggregate, get_client, self_score_from_output
from rl_detector.eval_runner import evaluate_model, select_eval_docs
from rl_detector.prompt_utils import get_think_already_open, load_tokenizer, quantile as _quantile
from rl_detector.rewards import (
    annotation_score_reward,
    annotation_type_reward,
    annotation_why_reward,
    compute_advantages,
    format_exception_diag as _format_exception_diag,
    format_reward,
    outer_credibility as _outer_credibility,
    outer_verdict_score_reward,
    parse_indicators,
    reward_components,
    safe_format_diagnostics as _safe_format_diagnostics_for_grpo,
    verdict_type_reward,
    zero_reward_components as _zero_reward_components,
)
from rl_detector.format_fix import _apply_format_fix_to_text_fields
from rl_detector.rollouts import (
    _get_analysis_stub_tokens,
    _MASK_ANN_OPEN,
    _MASK_ANN_CLOSE,
    _MASK_SPAN_OPEN,
    _MASK_TEXT_CLOSE,
    _MASK_TEXT_OPEN,
    _MASK_VERDICT_OPEN,
    TOKEN_TYPE_REASONING,
    TOKEN_TYPE_STRUCTURAL,
    TOKEN_TYPE_SPAN_OPEN,
    TOKEN_TYPE_VERDICT_TYPE,
    TOKEN_TYPE_VERDICT_WHY,
    TOKEN_TYPE_ANN_TYPE,
    TOKEN_TYPE_ANN_WHY,
    TOKEN_TYPE_ANN_SCORE,
    TOKEN_TYPE_DOC_COPY,
    TOKEN_TYPES_ANNOTATION,
    compute_token_type_mask,
    compute_task_loss_weights,
    decode_response_text,
    generate_rollouts,
)
from rl_detector.trace_payloads import (
    TrainingRolloutTracePayload,
    rubric_output_for_trace,
    trace_payload_to_weave_dict,
    training_trace_payload_to_audit_dict,
)
import weave

load_dotenv()
logger = logging.getLogger(__name__)

BASE_MODEL = CFG.model.base_model
EVAL_SAMPLE_SIZE = int(CFG.data.max_eval_docs)
EVAL_EVERY_STEPS = int(CFG.training.eval_every_steps)
EVAL_SEED = int(getattr(CFG.frozen, "seed", 2262))
GLOBAL_SEED = int(getattr(CFG.frozen, "seed", 2262))
SAVE_TTL_SECONDS = 2 * 24 * 60 * 60
_USE_FIDELITY_GATE = bool(CFG.training.use_fidelity_gate)

_EVAL_DOCS_CACHE = pathlib.Path("runs/eval_docs_fixed.json")


def _doc_hash(doc: dict) -> str:
    import hashlib
    return hashlib.sha256(doc["text"].encode()).hexdigest()[:16]


def _select_eval_docs_fixed(test_docs: list[dict], sample_size: int, seed: int) -> list[dict]:
    """Select eval docs, persisting the selection to disk so all runs compare the same pool."""
    if _EVAL_DOCS_CACHE.exists():
        cached_hashes: set[str] = set(json.loads(_EVAL_DOCS_CACHE.read_text()))
        resolved = [d for d in test_docs if _doc_hash(d) in cached_hashes]
        if resolved:
            logger.info("eval | loaded fixed eval pool from %s (%d docs)", _EVAL_DOCS_CACHE, len(resolved))
            return resolved
        logger.warning("eval | fixed eval pool cache %s had no matching docs; regenerating", _EVAL_DOCS_CACHE)
    chosen = select_eval_docs(test_docs, sample_size=sample_size, seed=seed)
    _EVAL_DOCS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    _EVAL_DOCS_CACHE.write_text(json.dumps([_doc_hash(d) for d in chosen]))
    logger.info("eval | saved fixed eval pool to %s (%d docs)", _EVAL_DOCS_CACHE, len(chosen))
    return chosen
_USE_RUBRIC_SCORER: bool = bool(getattr(CFG.training, "use_rubric_scorer", False))
_NEED_RUBRIC_CLIENT = _USE_RUBRIC_SCORER

# Running EMA of per-step reward std — used as adaptive floor in compute_advantages.
_adv_ema_std: float = 0.0
# Running EMA of per-step mean reward — used to scale learning rate down when training is going well.
_reward_ema: float = 0.0


def _training_get(path: str, default, legacy_key: str | None = None):
    cur = getattr(CFG, "training", None)
    for part in path.split("."):
        if cur is None:
            break
        cur = getattr(cur, part, None)
    if cur is not None:
        return cur
    if legacy_key:
        return getattr(CFG.training, legacy_key, default)
    return default




class _ScoreResult(NamedTuple):
    indicators: list
    tell_scored: list
    reward: float | None
    used_for_optimization: bool
    exclude_reason: str
    format_ok: bool
    format_ok_before_fix: bool
    dt_scoring: float
    format_char_diff: int
    reward_components: dict
    rubric_output: dict | None = None


_PPO_CLIP_KEYS = frozenset({"clip_low_threshold", "clip_high_threshold"})
_LOCAL_ONLY_LOSS_CFG_KEYS = frozenset({"adv_clip_scale"})


def _loss_fn_config_for_api(rl_fn: str) -> dict[str, float] | None:
    """Tinker expects Dict[str, float]; PPO ratio clip keys only for loss_fn=ppo."""
    raw = getattr(CFG.training, "rl_loss_fn_config", None)
    if raw is None:
        return None
    dc = OmegaConf.to_container(raw, resolve=True)
    if not isinstance(dc, dict) or not dc:
        return None
    out: dict[str, float] = {}
    for k, v in dc.items():
        if v is None:
            continue
        sk = str(k)
        if sk in _LOCAL_ONLY_LOSS_CFG_KEYS:
            continue
        if rl_fn != "ppo" and sk in _PPO_CLIP_KEYS:
            continue
        out[sk] = float(v)
    return out or None


def _rebind_train_globals() -> None:
    global BASE_MODEL, EVAL_SAMPLE_SIZE, EVAL_EVERY_STEPS, EVAL_SEED, GLOBAL_SEED
    global _USE_FIDELITY_GATE, _USE_RUBRIC_SCORER, _NEED_RUBRIC_CLIENT
    g = importlib.import_module("rl_detector.config").CFG
    BASE_MODEL = g.model.base_model
    EVAL_SAMPLE_SIZE = int(g.data.max_eval_docs)
    EVAL_EVERY_STEPS = int(g.training.eval_every_steps)
    EVAL_SEED = int(getattr(g.frozen, "seed", 2262))
    GLOBAL_SEED = int(getattr(g.frozen, "seed", 2262))
    _USE_FIDELITY_GATE = bool(g.training.use_fidelity_gate)
    _USE_RUBRIC_SCORER = bool(getattr(g.training, "use_rubric_scorer", False))
    _NEED_RUBRIC_CLIENT = _USE_RUBRIC_SCORER

def _assign_label_noise_modes(rng: random.Random, k: int) -> list[str]:
    """Assign exactly the configured proportion of each noise mode across K rollouts.

    Uses floor allocation with remainder distributed to the largest fractional parts,
    so proportions are as close to config as possible without probabilistic sampling.
    Shuffled so no rollout index is systematically biased.
    """
    p_correct = float(CFG.training.label_noise_correct_prob)
    p_flip = float(CFG.training.label_noise_flip_prob)
    p_unknown = float(CFG.training.label_noise_unknown_prob)
    raw_sum = p_correct + p_flip + p_unknown
    if raw_sum <= 0.0:
        # no probs set: force unknown-only so we still produce exactly k modes
        p_correct, p_flip, p_unknown = 0.0, 0.0, 1.0
    else:
        p_correct /= raw_sum
        p_flip /= raw_sum
        p_unknown /= raw_sum

    exact = {"correct": p_correct * k, "flip": p_flip * k, "unknown": p_unknown * k}
    counts = {m: int(v) for m, v in exact.items()}
    remainder = k - sum(counts.values())
    # Distribute leftover slots to the modes with the largest fractional parts.
    # If remainder > n_modes, keep cycling until all slots are assigned.
    if remainder > 0:
        ranked = sorted(exact, key=lambda m: exact[m] - counts[m], reverse=True)
        for i in range(remainder):
            counts[ranked[i % len(ranked)]] += 1

    modes = (
        ["correct"] * counts["correct"]
        + ["flip"] * counts["flip"]
        + ["unknown"] * counts["unknown"]
    )
    rng.shuffle(modes)
    return modes


async def _await_with_heartbeat(coro, step: int, phase: str, every_s: int = 20):
    # keep emitting liveness logs while waiting on remote async work
    task = asyncio.create_task(coro)
    waited_s = 0
    while True:
        done, _ = await asyncio.wait({task}, timeout=every_s)
        if done:
            return await task
        waited_s += every_s
        logger.info("step %d | still waiting on %s (%ds elapsed)", step, phase, waited_s)


async def _save_state_with_ttl(training_client, name: str) -> str:
    future = await training_client.save_state_async(name=name, ttl_seconds=SAVE_TTL_SECONDS)
    return await future.result_async()


_UCB_MAP_PATH = "runs/ucb_map.json"
_UCB_MAP_KEEP = 3


def _save_ucb_state(sampler, run_dir: str, step: int, ckpt_uri: str) -> None:
    """Save sampler UCB state to a sidecar JSON; record in manifest; keep last 3."""
    import json, os, pathlib
    state_path = os.path.join(run_dir, f"ucb_state_step-{step}.json")
    _uri_str = ckpt_uri.path if hasattr(ckpt_uri, "path") else str(ckpt_uri)
    with open(state_path, "w") as f:
        json.dump({"step": step, "ckpt_uri": _uri_str, "state": sampler.get_ucb_state()}, f)
    # Update manifest.
    pathlib.Path(_UCB_MAP_PATH).parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(_UCB_MAP_PATH) as f:
            manifest: list[dict] = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        manifest = []
    manifest = [e for e in manifest if e.get("ckpt_uri") != _uri_str]
    manifest.append({"ckpt_uri": _uri_str, "path": state_path})
    # Evict oldest beyond keep limit.
    for old in manifest[:-_UCB_MAP_KEEP]:
        try:
            os.remove(old["path"])
        except OSError:
            pass
    manifest = manifest[-_UCB_MAP_KEEP:]
    with open(_UCB_MAP_PATH, "w") as f:
        json.dump(manifest, f, indent=2)


def _load_ucb_state(resume_ckpt_uri: str) -> dict | None:
    """Load UCB state for a given checkpoint URI from the manifest, if available."""
    import json
    try:
        with open(_UCB_MAP_PATH) as f:
            manifest: list[dict] = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    for entry in manifest:
        if entry.get("ckpt_uri") == resume_ckpt_uri:
            try:
                with open(entry["path"]) as f:
                    return json.load(f).get("state")
            except (FileNotFoundError, json.JSONDecodeError):
                return None
    return None


def _repetition_score(text: str, n: int = 8) -> float:
    """Fraction of word n-grams that are repeated. 0.0 = no repetition, 1.0 = fully looping."""
    words = text.split()
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    return 1.0 - len(set(ngrams)) / len(ngrams)


@weave.op(call_display_name="training_rollout")
def _trace_training_rollout(*, trace: TrainingRolloutTracePayload) -> dict:
    rc = trace.reward_components or {}
    agg_score = float(rc.get("agg_score", 0.0))
    model_generated_label = 1 if agg_score >= 0.0 else 0
    # whether the label hint was followed (None if no hint was injected)
    hint_outer_followed: bool | None = None
    hint_agg_followed: bool | None = None
    if trace.inject_label and trace.main_label_hint is not None:
        hint_agg_followed = bool((agg_score >= 0.0) == (trace.main_label_hint == 1))
    out = trace_payload_to_weave_dict(trace)
    out.update({
        "step": int(trace.step),
        "rollout_index": int(trace.rollout_index),
        "doc_label": int(trace.doc_label),
        "noise_mode": trace.noise_mode,
        "inject_label": bool(trace.inject_label),
        "main_label_hint": trace.main_label_hint,
        "label_ctx_for_opt": bool(trace.label_ctx_for_opt),
        "response_text": trace.response_text,
        "wrong_response_text": trace.wrong_response_text,
        "output_score": agg_score,
        "model_generated_label": model_generated_label,
        "model_generated_label_str": "AI" if model_generated_label == 1 else "human",
        "hint_outer_followed": hint_outer_followed,
        "hint_agg_followed": hint_agg_followed,
        "advantage": None if trace.advantage is None else float(trace.advantage),
        "used_for_optimization": bool(trace.used_for_optimization),
        "exclude_reason": str(trace.exclude_reason),
        "format_ok": bool(trace.format_ok),
        "format_ok_before_fixing": bool(trace.format_ok_before_fixing),
        "format_reason": trace.format_reason,
        "format_reason_before_fixing": trace.format_reason_before_fixing,
        "format_char_diff_count": int(trace.format_char_diff_count),
        "is_ratio": None if trace.is_ratio is None else float(trace.is_ratio),
        "from_replay_cache": bool(trace.from_replay_cache),
        "document": trace.document,
        "doc_stratum": trace.doc_stratum,
        "neutral_prompt_text": trace.neutral_prompt_text,
        "completion_text": trace.completion_text,
        "full_output_text": trace.full_output_text,
        "raw_response_text": trace.raw_response_text,
        "was_text_fixed": bool(trace.was_text_fixed),
        "token_surprisal": trace.token_surprisal,
        "ann_token_fraction": None if trace.ann_token_fraction is None else float(trace.ann_token_fraction),
        "n_ann_tokens": None if trace.n_ann_tokens is None else float(trace.n_ann_tokens),
        "n_response_tokens": None if trace.n_response_tokens is None else float(trace.n_response_tokens),
        "indicators": trace.indicators,
        "n_optimized_tokens": sum(
            1 for t in trace.token_optimization_rows if bool(t.get("optimized", False))
        ),
    })
    out["rubric_reward_score"] = float(rc.get("credibility", 0.0))
    out["rubric_reward_diversity"] = float(rc.get("rubric_diversity", 0.0))
    out["rubric_reward_completeness"] = float(rc.get("rubric_completeness", 0.0))
    out["rubric_reward_coherence"] = float(rc.get("rubric_coherence", 0.0))
    rub = trace.rubric
    if isinstance(rub, dict):
        out["rubric_n_annotations"] = int(rub.get("n_annotations", 0))
        out["rubric_raw_response_len"] = int(rub.get("raw_response_len", 0))
    else:
        out["rubric_n_annotations"] = 0
        out["rubric_raw_response_len"] = 0
    return out


_return_token_id_cache: int | None = None


def _get_return_token_id(tokenizer) -> int | None:
    global _return_token_id_cache
    if _return_token_id_cache is None:
        ids = tokenizer.encode("<|return|>", add_special_tokens=False)
        _return_token_id_cache = int(ids[0]) if ids else -1
    return _return_token_id_cache if _return_token_id_cache != -1 else None


def _zero_outer_verdict_type_cls_adv_for_label_ctx(
    response_advantages: list[float],
    completion_tokens: list[int],
    n_reasoning_tokens: int,
    tokenizer,
    label_ctx_for_opt: bool,
) -> list[float]:
    """When the PPO forward includes the injected label hint, zero adv on <verdict type="…"> value tokens.

    Those completions were sampled under the hint; cls reward can disagree with following the hint,
    so we avoid putting the scalar cls advantage on the outer type tokens. Stub datums (no hint in
    ``_datum_prompt_tokens``) keep full advantages there.
    """
    if not label_ctx_for_opt:
        return response_advantages
    R = max(0, int(n_reasoning_tokens))
    response_tokens = completion_tokens[R:]
    if not response_tokens:
        return response_advantages
    from rl_detector.rollouts import (
        _MASK_ANN_CLOSE,
        _MASK_ANN_OPEN,
        _MASK_ANN_STRUCT,
        _MASK_SPAN_OPEN,
        _MASK_TEXT_CLOSE,
        _MASK_VERDICT_OPEN,
    )

    _STRUCT = frozenset(_MASK_ANN_STRUCT)
    _return_id = _get_return_token_id(tokenizer)
    in_attrs = False
    in_verdict = False
    out = list(response_advantages[: len(response_tokens)])
    out.extend([0.0] * max(0, len(response_tokens) - len(out)))
    for i, tok in enumerate(response_tokens):
        t = int(tok)
        if _return_id is not None and t == _return_id:
            pass
        elif t in _STRUCT:
            if t == _MASK_SPAN_OPEN:
                pass
            elif t == _MASK_ANN_OPEN:
                in_attrs = True
                in_verdict = False
            elif t == _MASK_VERDICT_OPEN:
                in_attrs = True
                in_verdict = True
            elif t == _MASK_ANN_CLOSE or t == _MASK_TEXT_CLOSE:
                in_attrs = False
                in_verdict = False
        elif in_attrs and in_verdict:
            out[i] = 0.0
    return out


def _build_aux_ce_datum(
    prompt_tokens: list[int],
    completion_tokens: list[int],
    n_reasoning_tokens: int,
    token_type_mask: list[int] | None = None,
) -> tinker.Datum | None:
    """CE datum for format-broken (but fixable) rollouts.

    Supervises doc-copy tokens (verbatim document text) and SPAN_OPEN tokens (where
    tells start). Both come from the fixer-aligned completion; no separate span-open datum.
    """
    full_tokens = prompt_tokens + list(completion_tokens)
    if len(full_tokens) < 2:
        return None
    n_completion = len(completion_tokens)
    R = min(max(0, int(n_reasoning_tokens)), n_completion)
    model_input = full_tokens[:-1]
    target_tokens = full_tokens[1:]
    n_prefix_in_target = max(0, len(prompt_tokens) - 1 + R)
    n_response = len(target_tokens) - n_prefix_in_target
    _doc_copy_w = float(getattr(CFG.training, "doc_copy_ce_weight", 1.0))
    if token_type_mask is not None and len(token_type_mask) == n_completion:
        response_weights = [
            _doc_copy_w if token_type_mask[R + j] == TOKEN_TYPE_DOC_COPY else 1.0
            for j in range(n_response)
        ]
    else:
        response_tokens = list(completion_tokens)[R:]
        response_weights = [_doc_copy_w if int(t) != _MASK_SPAN_OPEN else 1.0 for t in response_tokens]
    weights = [0.0] * n_prefix_in_target + response_weights
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(model_input),
        loss_fn_inputs={
            "target_tokens": torch.tensor(target_tokens, dtype=torch.long),
            "weights": torch.tensor(weights, dtype=torch.float32),
        },
    )


def _build_contrastive_neg_datum(
    prompt_tokens: list[int],
    fixed_completion_tokens: list[int],
    pre_fix_completion_tokens: list[int],
    n_reasoning_tokens: int,
) -> tinker.Datum | None:
    """Negative CE datum at the first divergence between pre-fix and fixed completions.

    Finds the first position d (in completion space, after reasoning tokens) where
    fixed_completion_tokens[d] != pre_fix_completion_tokens[d]. Builds a datum whose
    model_input is the full shared context (prompt + pre_fix[:d]), target_tokens are the
    usual next-token shift, and weight = -contrastive_ce_neg_weight only at the single
    bad token position (pre_fix[d]). All other weights are 0.

    This pushes P(bad_token | shared_prefix) DOWN at the exact decision point where the
    model made a format-breaking choice (e.g. the extra <span> that starts span-spam).
    Only the divergence token gets a gradient; the shared prefix is weight-0 for efficiency.
    """
    _neg_w = float(getattr(CFG.training, "contrastive_ce_neg_weight", 1.0))
    if _neg_w == 0.0:
        return None
    n_fixed = len(fixed_completion_tokens)
    n_pre = len(pre_fix_completion_tokens)
    R = min(max(0, int(n_reasoning_tokens)), min(n_fixed, n_pre))
    # Find first diverging position in response space (skip reasoning prefix R).
    d = None
    for j in range(min(n_fixed - R, n_pre - R)):
        if fixed_completion_tokens[R + j] != pre_fix_completion_tokens[R + j]:
            d = R + j
            break
    if d is None:
        return None  # completions are identical in response space — no contrastive signal
    # Build full sequence: prompt + pre_fix[:d+1] (context + bad token).
    full_neg = prompt_tokens + list(pre_fix_completion_tokens[: d + 1])
    if len(full_neg) < 2:
        return None
    model_input = full_neg[:-1]   # context before the bad token
    target_tokens = full_neg[1:]  # next-token targets; last = pre_fix[d] (the bad token)
    # Only the last position (the bad token) gets non-zero weight.
    weights = [0.0] * (len(target_tokens) - 1) + [-_neg_w]
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(model_input),
        loss_fn_inputs={
            "target_tokens": torch.tensor(target_tokens, dtype=torch.long),
            "weights": torch.tensor(weights, dtype=torch.float32),
        },
    )


def _norm_pool_ann_type_advs(
    type_rows: list[list[float | None]],
    type_correct_rows: list[list[bool | None]],
    *,
    adv_std_floor: float,
    adv_norm: str,
    adv_clip_low: float,
    adv_clip_high: float,
) -> list[list[float] | None]:
    """GRPO on correct-type annotations only; wrong type never gets positive adv.

    Pool is all correct tells across rollouts in one doc. Best credible+correct rises
    above mean; wrong types use signed reward capped at 0 (no GRPO lift).
    """
    correct_indexed: list[tuple[int, int, float]] = []
    wrong_indexed: list[tuple[int, int, float]] = []
    for ri, row in enumerate(type_rows):
        tcr = type_correct_rows[ri]
        for ci, v in enumerate(row):
            if v is None:
                continue
            ok = bool(tcr[ci]) if ci < len(tcr) and tcr[ci] is not None else False
            if ok:
                correct_indexed.append((ri, ci, float(v)))
            else:
                wrong_indexed.append((ri, ci, float(v)))
    adv_map: dict[tuple[int, int], float] = {}
    if len(correct_indexed) >= 2:
        rewards = [x[2] for x in correct_indexed]
        advs = compute_advantages(rewards, std_floor=adv_std_floor, normalize=adv_norm)
        for (ri, ci, _), a in zip(correct_indexed, advs):
            adv_map[(ri, ci)] = max(adv_clip_low, min(adv_clip_high, float(a)))
    elif len(correct_indexed) == 1:
        ri, ci, r = correct_indexed[0]
        adv_map[(ri, ci)] = max(adv_clip_low, min(adv_clip_high, r))
    for ri, ci, r in wrong_indexed:
        adv_map[(ri, ci)] = max(adv_clip_low, min(0.0, r))
    out: list[list[float] | None] = []
    for ri, row in enumerate(type_rows):
        if all(v is None for v in row):
            out.append(None)
        else:
            out.append([
                adv_map.get((ri, ci), 0.0) if v is not None else 0.0
                for ci, v in enumerate(row)
            ])
    return out


def _build_per_token_advantages(
    completion_tokens: list[int],
    n_reasoning_tokens: int,
    cls_adv: float,
    tell_scored: list[dict],
    tokenizer,
    label_ctx_for_opt: bool = False,
    per_tell_advs: list[float] | None = None,
    per_tell_type_advs: list[float] | None = None,
    per_tell_why_advs: list[float] | None = None,
    per_tell_score_advs: list[float] | None = None,
    verdict_why_adv: float | None = None,
    verdict_why_quality_adv: float | None = None,
    verdict_ann_recall_adv: float | None = None,
    verdict_score_adv: float | None = None,
    per_span_open_advs: list[float] | None = None,
    struct_token_adv: float = 0.3,
    span_open_fail_penalty: float = 0.0,
    per_token_is: list[float] | None = None,
) -> list[float]:
    """Per-token advantage by structural role (pure special-token-ID state machine).

    Each token type gets its own independently normalized PTAD signal:
      TEXT_OPEN / ANN_OPEN / ANN_CLOSE / TEXT_CLOSE → struct_token_adv (structural)
      WHY_Q / SCORE_Q / VERDICT_PREFIX              → struct_token_adv (structural)
      SPAN_OPEN (any inner tell)     → per_span_open_advs[span_idx]
                                       Falls back to -span_open_fail_penalty when unavailable.
      Verdict type value tokens      → cls_adv (GRPO on label alignment; 0 when label_ctx_for_opt)
      Verdict why value tokens       → verdict_why_adv (outer_credibility pool) + verdict_why_quality_adv
                                       + verdict_ann_recall_adv (annotation recall pool)
                                       Falls back to cls_adv when verdict_why_adv is None.
      Verdict score value tokens     → verdict_score_adv (outer_verdict_score_reward pool)
                                       Falls back to cls_adv when verdict_score_adv is None.
      Inner ann type value tokens    → per_tell_type_advs[tell_idx] (GRPO on correct-type pool)
                                       Falls back to per_tell_advs[tell_idx] then 0.0.
      Inner ann why value tokens     → per_tell_why_advs[tell_idx] (annotation_why_reward pool)
                                       Falls back to per_tell_advs[tell_idx] then 0.0.
      Inner ann score value tokens   → max(0, per_tell_score_advs[tell_idx]) (calibration, clamped ≥0)
                                       Falls back to max(0, per_tell_advs[tell_idx]) then 0.0.
      Doc-copy tokens                → 0.0 (zero-entropy; trained via aux CE)

    per_tell_type/why/score_advs take precedence over the legacy per_tell_advs fallback.
    All 8 special IDs are guaranteed single-token (ANNOTATION_TOKEN_REMAP).
    """
    R = max(0, int(n_reasoning_tokens))
    response_tokens = completion_tokens[R:]
    if not response_tokens:
        return []

    # Per-token IS correction: each token gets π_new(t)/π_old(t) rather than a
    # single per-rollout geometric mean. per_token_is covers all completion tokens;
    # slice off reasoning prefix so indexing aligns with response_tokens.
    _response_is: list[float] | None = per_token_is[R:] if per_token_is is not None else None

    from rl_detector.rollouts import _MASK_ANN_STRUCT, _MASK_ANN_WHY_Q, _MASK_ANN_SCORE_Q
    _STRUCT = frozenset(_MASK_ANN_STRUCT)
    _return_id = _get_return_token_id(tokenizer)

    in_attrs = False
    in_verdict = False
    in_verdict_why = False   # True after WHY_Q inside a verdict (why explanation sub-section)
    in_verdict_score = False  # True after SCORE_Q inside a verdict (score magnitude sub-section)
    in_ann_why = False      # True after WHY_Q inside a non-verdict inner tell
    in_ann_score = False    # True after SCORE_Q inside a non-verdict inner tell
    tell_idx = -1  # incremented on each ANN_OPEN (inner annotation open)
    span_idx = -1  # incremented on each SPAN_OPEN (for pacing indexing)
    out: list[float] = []

    def _append(adv: float) -> None:
        """Append advantage scaled by this token's per-token IS ratio (if available)."""
        is_scale = _response_is[len(out)] if _response_is is not None and len(out) < len(_response_is) else 1.0
        out.append(adv * is_scale)

    for tok in response_tokens:
        t = int(tok)
        if _return_id is not None and t == _return_id:
            _append(0.0)
        elif t in _STRUCT:
            if t == _MASK_SPAN_OPEN:
                span_idx += 1
                soa = per_span_open_advs[span_idx] if (per_span_open_advs is not None and span_idx < len(per_span_open_advs)) else -abs(span_open_fail_penalty)
                # SPAN_OPEN fallback: -span_open_fail_penalty for format-failed rollouts.
                # Bug 3 fix: changed from +struct_token_adv (gave span-spam +0.1 per token) to
                # -span_open_fail_penalty. This matches the above-ceiling penalty so format-failed
                # and above-ceiling paths both get the same negative signal. Without this, the
                # model prefers format-failed (0.0) over above-ceiling valid (-0.3), causing
                # span spam to re-emerge after the ceiling penalty is introduced (Bug 5).
                _append(soa)
            elif t == _MASK_ANN_OPEN:
                tell_idx += 1
                in_attrs = True
                in_verdict = False
                in_verdict_why = False
                in_verdict_score = False
                in_ann_why = False
                in_ann_score = False
                _append(struct_token_adv)
            elif t == _MASK_VERDICT_OPEN:
                in_attrs = True
                in_verdict = True
                in_verdict_why = False
                in_verdict_score = False
                in_ann_why = False
                in_ann_score = False
                _append(struct_token_adv)
            elif t == _MASK_ANN_CLOSE or t == _MASK_TEXT_CLOSE:
                in_attrs = False
                in_verdict = False
                in_verdict_why = False
                in_verdict_score = False
                in_ann_why = False
                in_ann_score = False
                _append(struct_token_adv)
            elif in_verdict and t == _MASK_ANN_WHY_Q:
                in_verdict_why = True
                in_verdict_score = False
                _append(struct_token_adv)
            elif in_verdict and t == _MASK_ANN_SCORE_Q:
                in_verdict_why = False
                in_verdict_score = True
                _append(struct_token_adv)
            elif not in_verdict and in_attrs and t == _MASK_ANN_WHY_Q:
                in_ann_why = True
                _append(struct_token_adv)
            elif not in_verdict and in_attrs and t == _MASK_ANN_SCORE_Q:
                in_ann_score = True
                _append(struct_token_adv)
            else:  # TEXT_OPEN, WHY_Q (outside attrs), etc.
                _append(struct_token_adv)
        elif in_attrs:
            if in_verdict:
                if label_ctx_for_opt:
                    _append(0.0)
                elif in_verdict_score:
                    # IS-corrected advantage computed; gradient blocked via ptok_loss_scale_verdict_score=0.
                    # Re-enable optimization by setting that config value non-zero — no code change needed.
                    _append(verdict_score_adv if verdict_score_adv is not None else cls_adv)
                elif in_verdict_why:
                    # Verdict why tokens: outer_credibility + length/repetition quality
                    # + annotation recall. verdict_ann_recall_adv rewards the verdict for
                    # incorporating vocabulary from annotation explanations, ensuring the
                    # verdict synthesizes the annotated evidence rather than producing a
                    # generic summary disconnected from the spans.
                    _vwhy = verdict_why_adv if verdict_why_adv is not None else cls_adv
                    if verdict_why_quality_adv is not None:
                        _vwhy = _vwhy + verdict_why_quality_adv
                    if verdict_ann_recall_adv is not None:
                        _vwhy = _vwhy + verdict_ann_recall_adv
                    _append(_vwhy)
                else:
                    # Type value tokens: classification correctness only.
                    _append(cls_adv)
            elif in_ann_score:
                # IS-corrected advantage computed; gradient blocked via ptok_loss_scale_ann_score=0.
                # Re-enable optimization by setting that config value non-zero — no code change needed.
                if per_tell_score_advs is not None and 0 <= tell_idx < len(per_tell_score_advs):
                    _append(per_tell_score_advs[tell_idx])
                else:
                    _append(per_tell_advs[tell_idx] if (per_tell_advs is not None and 0 <= tell_idx < len(per_tell_advs)) else 0.0)
            elif in_ann_why:
                # Why value tokens: annotation_why_reward pool (pure rubric credibility).
                if per_tell_why_advs is not None and 0 <= tell_idx < len(per_tell_why_advs):
                    _append(per_tell_why_advs[tell_idx])
                else:
                    _append(per_tell_advs[tell_idx] if (per_tell_advs is not None and 0 <= tell_idx < len(per_tell_advs)) else 0.0)
            else:
                # Type value tokens: GRPO among correct-type tells (+1 / -1 label alignment); wrong ≤0.
                if label_ctx_for_opt:
                    _append(0.0)
                elif per_tell_type_advs is not None and 0 <= tell_idx < len(per_tell_type_advs):
                    _append(per_tell_type_advs[tell_idx])
                else:
                    _append(per_tell_advs[tell_idx] if (per_tell_advs is not None and 0 <= tell_idx < len(per_tell_advs)) else 0.0)
        else:
            _append(0.0)

    return out


def _token_optimization_rows(
    tokenizer,
    prompt_tokens: list[int],
    completion_tokens: list[int],
    completion_logprobs: list[float],
    n_reasoning_tokens: int,
    response_advantages: list[float],
) -> list[dict]:
    """Full token trace covering prompt → reasoning → response.

    Each row: token_id, decoded_token, token_type, logprob, surprisal, advantage, optimized.
    Prompt and reasoning tokens have logprob/advantage = 0 and optimized = False.
    token_type is one of: prompt, reasoning, structural, span_open, verdict_type, verdict_why,
                          ann_type, ann_why, doc_copy.
    """
    from rl_detector.rollouts import _MASK_ANN_STRUCT, _MASK_ANN_WHY_Q, _MASK_ANN_SCORE_Q
    _STRUCT = frozenset(_MASK_ANN_STRUCT)
    _return_id = _get_return_token_id(tokenizer)

    def _decode(tok_id: int) -> str:
        return tokenizer.decode([tok_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)

    rows: list[dict] = []

    for tok in prompt_tokens:
        rows.append({"token_id": int(tok), "decoded_token": _decode(int(tok)), "token_type": "prompt",
                     "logprob": 0.0, "surprisal": 0.0, "advantage": 0.0, "optimized": False})

    R = min(max(0, int(n_reasoning_tokens)), len(completion_tokens))
    for tok, lp in zip(completion_tokens[:R], completion_logprobs[:R]):
        rows.append({"token_id": int(tok), "decoded_token": _decode(int(tok)), "token_type": "reasoning",
                     "logprob": float(lp), "surprisal": -float(lp), "advantage": 0.0, "optimized": False})

    response_tokens = completion_tokens[R:]
    response_lps = completion_logprobs[R:]
    advs = list(response_advantages[:len(response_tokens)]) + [0.0] * max(0, len(response_tokens) - len(response_advantages))

    in_attrs = False
    in_verdict = False
    in_verdict_why = False
    in_verdict_score = False
    in_ann_why = False
    in_ann_score = False
    for tok, lp, adv in zip(response_tokens, response_lps, advs):
        t = int(tok)
        if _return_id is not None and t == _return_id:
            token_type = "structural"
        elif t in _STRUCT:
            if t == _MASK_SPAN_OPEN:
                token_type = "span_open"
            elif t == _MASK_ANN_OPEN:
                in_attrs = True; in_verdict = False; in_verdict_why = False; in_verdict_score = False; in_ann_why = False; in_ann_score = False
                token_type = "structural"
            elif t == _MASK_VERDICT_OPEN:
                in_attrs = True; in_verdict = True; in_verdict_why = False; in_verdict_score = False; in_ann_why = False; in_ann_score = False
                token_type = "structural"
            elif t == _MASK_ANN_CLOSE or t == _MASK_TEXT_CLOSE:
                in_attrs = False; in_verdict = False; in_verdict_why = False; in_verdict_score = False; in_ann_why = False; in_ann_score = False
                token_type = "structural"
            elif in_verdict and t == _MASK_ANN_WHY_Q:
                in_verdict_why = True; in_verdict_score = False
                token_type = "structural"
            elif in_verdict and t == _MASK_ANN_SCORE_Q:
                in_verdict_why = False; in_verdict_score = True
                token_type = "structural"
            elif not in_verdict and in_attrs and t == _MASK_ANN_WHY_Q:
                in_ann_why = True
                token_type = "structural"
            elif not in_verdict and in_attrs and t == _MASK_ANN_SCORE_Q:
                in_ann_score = True
                token_type = "structural"
            else:
                token_type = "structural"
        elif in_attrs:
            if in_verdict:
                token_type = "verdict_score" if in_verdict_score else ("verdict_why" if in_verdict_why else "verdict_type")
            elif in_ann_score:
                token_type = "ann_score"
            else:
                token_type = "ann_why" if in_ann_why else "ann_type"
        else:
            token_type = "doc_copy"
        rows.append({"token_id": t, "decoded_token": _decode(t), "token_type": token_type,
                     "logprob": float(lp), "surprisal": -float(lp), "advantage": float(adv),
                     "optimized": abs(float(adv)) > 0.0})

    return rows


def _ptok_loss_scales_from_cfg() -> dict[str, float]:
    t = CFG.training
    return {
        "ann_type": float(t.ptok_loss_scale_ann_type),
        "ann_why": float(t.ptok_loss_scale_ann_why),
        "ann_score": float(t.ptok_loss_scale_ann_score),
        "verdict_type": float(t.ptok_loss_scale_verdict_type),
        "verdict_why": float(t.ptok_loss_scale_verdict_why),
        "verdict_score": float(t.ptok_loss_scale_verdict_score),
        "span_open": float(t.ptok_loss_scale_span_open),
        "structural": float(t.ptok_loss_scale_structural),
    }


def build_datum(
    prompt_tokens: list[int],
    completion_tokens: list[int],
    completion_logprobs: list[float],
    response_advantages: list[float],
    response_task_weights: list[float],
    n_reasoning_tokens: int,
) -> tinker.Datum:
    """Build a GRPO/PPO datum. Task balance via adv *= response_task_weights (Tinker ppo has no weights input).

    PPO loss is -sum_t objective_t; scaling advantages by w_t matches per-task weighting
    (doc/verdict O(1) mass; SPAN_OPEN fixed mass per open).
    """
    N = len(prompt_tokens)
    M = len(completion_tokens)
    R = min(n_reasoning_tokens, M)
    S = M - R
    full_seq = prompt_tokens + completion_tokens
    input_tokens = full_seq[:-1]
    target_tokens = full_seq[1:]
    logprobs = [0.0] * (N - 1) + [0.0] * R + completion_logprobs[R:]
    if len(response_advantages) != S or len(response_task_weights) != S:
        raise AssertionError(
            "build_datum: response len mismatch advantages=%d weights=%d S=%d"
            % (len(response_advantages), len(response_task_weights), S)
        )
    response_weighted_adv = [
        float(a) * float(w) for a, w in zip(response_advantages, response_task_weights)
    ]
    advantages = [0.0] * (N - 1) + [0.0] * R + response_weighted_adv
    mask = [0.0] * (N - 1) + [0.0] * R + [1.0] * S
    assert len(input_tokens) == len(target_tokens) == len(logprobs) == len(advantages) == len(mask)
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(input_tokens),
        loss_fn_inputs={
            "target_tokens": TensorData.from_torch(torch.tensor(target_tokens, dtype=torch.long)),
            "logprobs": TensorData.from_torch(torch.tensor(logprobs, dtype=torch.float32)),
            "advantages": TensorData.from_torch(torch.tensor(advantages, dtype=torch.float32)),
            "mask": TensorData.from_torch(torch.tensor(mask, dtype=torch.float32)),
        },
    )


def _flip_hint_in_prompt_tokens(
    sampling_prompt_tokens: list[int],
    hint: int,
    tokenizer,
) -> list[int] | None:
    """Return a copy of sampling_prompt_tokens with the hint text swapped to the opposite label.

    Finds "Text origin is AI." / "Text origin is human." as a token sublist and
    replaces it in-place.  Returns None if the current hint text is not found.
    """
    current_text = label_think_continuation(hint)
    flipped_text = label_think_continuation(1 - hint)
    current_toks = tokenizer.encode(current_text, add_special_tokens=False)
    flipped_toks = tokenizer.encode(flipped_text, add_special_tokens=False)
    n = len(current_toks)
    toks = list(sampling_prompt_tokens)
    for i in range(len(toks) - n + 1):
        if toks[i : i + n] == current_toks:
            return toks[:i] + flipped_toks + toks[i + n :]
    return None


def _build_outer_type_ce_datum(
    prompt_tokens: list[int],
    completion_tokens: list[int],
    n_reasoning_tokens: int,
    target_type: int,
    tokenizer,
) -> tinker.Datum | None:
    """CE datum with weight=1 only on the type value token(s) in the outer annotation.

    prompt_tokens should be sampling_prompt_tokens (includes the hint text) so the
    model is conditioned on the hint when learning to produce the outer type.
    target_type: 0=human, 1=AI — the value that should appear in type="…".

    Critically: we do NOT decode-then-re-encode the attribute token sequence.  Instead
    we scan the ORIGINAL token IDs for the type-value span and splice in the target
    token IDs directly.  This preserves all surrounding tokenization boundaries exactly
    as the model produced them, so the CE datum context is always in-distribution.
    """
    R = min(max(0, int(n_reasoning_tokens)), len(completion_tokens))
    response_tokens = completion_tokens[R:]

    # Find the verdict block: <verdict type="…" /></text>
    # New format uses VERDICT_OPEN + TEXT_CLOSE; old inner annotations use ANN_OPEN + ANN_CLOSE.
    # We want ONLY the outer verdict, so scan for VERDICT_OPEN/TEXT_CLOSE pair.
    last_open = -1
    last_close = -1
    close_tok = _MASK_TEXT_CLOSE
    for idx, t in enumerate(response_tokens):
        if int(t) == _MASK_VERDICT_OPEN:
            last_open = idx
        elif int(t) == _MASK_TEXT_CLOSE and last_open >= 0:
            last_close = idx
    if last_open < 0 or last_close <= last_open:
        return None

    attr_tokens = list(response_tokens[last_open + 1 : last_close])

    # Target token IDs for the correct type label (1-3 tokens, never re-encoded).
    correct_type = "AI" if target_type == 1 else "human"
    _expected_type_toks: list[int] = tokenizer.encode(correct_type, add_special_tokens=False)
    _n_val = len(_expected_type_toks)
    assert 1 <= _n_val <= 3, f"unexpected token count for {correct_type!r}: {_n_val}"

    # Find the type-value span in the ORIGINAL attr_tokens by scanning for EITHER
    # "AI" OR "human" token IDs anchored with type=" pre-context and " post-context.
    # Scan backward: type= appears before why= so the last valid match is correct.
    _ai_toks: list[int] = tokenizer.encode("AI", add_special_tokens=False)
    _human_toks: list[int] = tokenizer.encode("human", add_special_tokens=False)
    _candidates = [_ai_toks, _human_toks]

    orig_type_tok_start = -1
    orig_type_tok_end = -1
    for _cand in _candidates:
        _nc = len(_cand)
        for _s in range(len(attr_tokens) - _nc, -1, -1):
            if attr_tokens[_s : _s + _nc] == _cand:
                _pre = tokenizer.decode(
                    response_tokens[last_open : last_open + 1 + _s],
                    skip_special_tokens=False,
                )
                _post = tokenizer.decode(attr_tokens[_s + _nc : _s + _nc + 2], skip_special_tokens=False)
                if _pre.endswith('type="') and _post.startswith('"'):
                    orig_type_tok_start = _s
                    orig_type_tok_end = _s + _nc
                    break
        if orig_type_tok_start >= 0:
            break

    if orig_type_tok_start < 0:
        return None

    # Splice: keep original tokens on both sides, replace only the type-value span.
    corrected_attr_tokens: list[int] = (
        attr_tokens[:orig_type_tok_start]
        + _expected_type_toks
        + attr_tokens[orig_type_tok_end:]
    )
    # Position of the target type-value tokens in corrected_attr_tokens.
    type_tok_start = orig_type_tok_start
    type_tok_end = orig_type_tok_start + _n_val

    # Hard assertions.
    assert corrected_attr_tokens[type_tok_start:type_tok_end] == _expected_type_toks, (
        f"_build_outer_type_ce_datum: spliced IDs {corrected_attr_tokens[type_tok_start:type_tok_end]}"
        f" != expected {_expected_type_toks}"
    )
    _pre_ctx_final = tokenizer.decode(
        response_tokens[last_open : last_open + 1 + type_tok_start],
        skip_special_tokens=False,
    )
    _post_ctx_final = tokenizer.decode(corrected_attr_tokens[type_tok_end : type_tok_end + 2], skip_special_tokens=False)
    assert _pre_ctx_final.endswith('type="'), (
        f"_build_outer_type_ce_datum: pre-context {_pre_ctx_final!r} does not end with type=\""
    )
    assert _post_ctx_final.startswith('"'), (
        f"_build_outer_type_ce_datum: post-context {_post_ctx_final!r} does not start with \""
    )

    # Build corrected completion (all tokens outside the type-value span are unchanged).
    pre_block = list(completion_tokens[:R]) + list(response_tokens[: last_open + 1])
    post_block = [close_tok] + list(response_tokens[last_close + 1 :])
    corrected_completion = pre_block + corrected_attr_tokens + post_block

    full_tokens = list(prompt_tokens) + corrected_completion
    if len(full_tokens) < 2:
        return None
    model_input = full_tokens[:-1]
    target_tokens = full_tokens[1:]

    # Map type value positions into target_tokens index space.
    attr_base_in_target = len(prompt_tokens) - 1 + R + last_open + 1
    val_start = attr_base_in_target + type_tok_start
    val_end = attr_base_in_target + type_tok_end

    weights = [0.0] * len(target_tokens)
    for idx in range(val_start, min(val_end, len(weights))):
        weights[idx] = 1.0

    # Final assertion: weighted positions contain exactly _expected_type_toks.
    _weighted_ids = [target_tokens[i] for i in range(val_start, val_end) if i < len(target_tokens)]
    assert _weighted_ids == _expected_type_toks, (
        f"_build_outer_type_ce_datum: weighted target IDs {_weighted_ids} != expected {_expected_type_toks}"
    )
    assert sum(1 for w in weights if w > 0.0) == _n_val, (
        f"_build_outer_type_ce_datum: nonzero weight count != {_n_val}"
    )

    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(model_input),
        loss_fn_inputs={
            "target_tokens": torch.tensor(target_tokens, dtype=torch.long),
            "weights": torch.tensor(weights, dtype=torch.float32),
        },
    )


def _token_surprisal_stats(
    completion_logprobs: list[float],
    n_reasoning_tokens: int,
    token_type_mask: list[int] | None,
) -> dict[str, float]:
    R = min(n_reasoning_tokens, len(completion_logprobs))
    response_lps = completion_logprobs[R:]
    surprisals = [-float(lp) for lp in response_lps]
    if token_type_mask is not None:
        response_types = token_type_mask[R:R + len(surprisals)]
    else:
        response_types = None
    if response_types is not None:
        ann_vals = [s for s, t in zip(surprisals, response_types) if t != TOKEN_TYPE_DOC_COPY]
        doc_vals = [s for s, t in zip(surprisals, response_types) if t == TOKEN_TYPE_DOC_COPY]
    else:
        ann_vals = surprisals
        doc_vals = []
    return {
        "mean": sum(surprisals) / len(surprisals) if surprisals else 0.0,
        "p10": _quantile(surprisals, 0.10),
        "p50": _quantile(surprisals, 0.50),
        "p90": _quantile(surprisals, 0.90),
        "ann_mean": sum(ann_vals) / len(ann_vals) if ann_vals else 0.0,
        "doc_mean": sum(doc_vals) / len(doc_vals) if doc_vals else 0.0,
        "n_response_tokens": float(len(surprisals)),
        "n_ann_tokens": float(len(ann_vals)),
        "n_doc_tokens": float(len(doc_vals)),
    }


async def _process_doc(sampling_client, tokenizer, rubric_client, doc, rng: random.Random, rollout_seed: int | None = None, k: int | None = None, adv_std_floor: float = 0.0, step: int = 0):
    """Process a single doc: generate rollouts, score, compute rewards/advantages, build datums.

    Label injection: for correct/flip noise modes the label is prepended to the start of
    the reasoning chain as a forced prefix.  The user message always comes from
    ``format_prompt_for_model(.., text=document)`` (same escaping as ``build_prompt`` fences),
    so sampling logprobs are already correct reference logprobs, no IS correction needed.
    """
    document = doc["text"]
    escaped_document = escape_document_piece(document)
    label = doc["label"]
    label_str = "AI" if label == 1 else "human"
    snippet = document[:60].replace("\n", " ")

    if k is None:
        k = int(getattr(CFG.training, "k_final"))
    # assign noise modes with exact proportions across the group (no probabilistic sampling)
    noise_modes = _assign_label_noise_modes(rng, k)
    main_label_hints: list[int] = []
    inject_label_flags: list[bool] = []
    for mode in noise_modes:
        if mode == "correct":
            main_label_hints.append(label)
            inject_label_flags.append(True)
        elif mode == "flip":
            main_label_hints.append(1 - label)
            inject_label_flags.append(True)
        else:
            # unknown: keep label for reward computation, but don't inject into reasoning
            main_label_hints.append(label)
            inject_label_flags.append(False)

    logger.debug("rollouts | generating K=%d for %s doc: %r... (seed=%s)", k, label_str, snippet, rollout_seed)
    logger.debug(
        "rollouts | noisy modes: correct=%d flip=%d unknown=%d",
        sum(1 for m in noise_modes if m == "correct"),
        sum(1 for m in noise_modes if m == "flip"),
        sum(1 for m in noise_modes if m == "unknown"),
    )
    t0_rollouts = time.perf_counter()
    rollouts = await generate_rollouts(
        sampling_client,
        tokenizer,
        document,
        main_label_hints=main_label_hints,
        inject_label_flags=inject_label_flags,
        K=k,
        seed=rollout_seed,
        think_already_open=get_think_already_open(tokenizer),
        doc_stratum="|".join(str(x) for x in doc_stratum_key(doc)),
    )
    # Attach per-rollout noise mode for downstream auditing (especially format_fail_audit).
    # Without this, failure rows cannot be split by correct/flip/unknown regime.
    for i, r in enumerate(rollouts):
        r["noise_mode"] = noise_modes[i]
    dt_rollouts = time.perf_counter() - t0_rollouts
    n_tells_per_rollout = []
    for r in rollouts:
        try:
            n_tells_per_rollout.append(len(parse_indicators(r["response_text"]) or []))
        except Exception:
            logger.exception("indicator parsing crashed while logging rollout tell counts")
            n_tells_per_rollout.append(0)
    n_reasoning_tokens_per_rollout = [r.get("n_reasoning_tokens", 0) for r in rollouts]
    logger.debug("rollouts | done in %.1fs — tells per rollout: %s", dt_rollouts, n_tells_per_rollout)
    logger.debug("rollouts | reasoning tokens masked (per rollout): %s", n_reasoning_tokens_per_rollout)

    is_ratios = [r.get("is_ratio", 1.0) for r in rollouts]
    is_ratio_mean = sum(is_ratios) / len(is_ratios) if is_ratios else 1.0
    is_ratio_min = min(is_ratios) if is_ratios else 1.0
    is_ratio_max = max(is_ratios) if is_ratios else 1.0

    _lco = getattr(CFG.training, "label_context_opt", {})
    _lco_enabled = bool(getattr(_lco, "enabled", False))
    _lco_keep_prob = float(getattr(_lco, "keep_label_prob", 0.0))
    _lco_keep_prob = max(0.0, min(1.0, _lco_keep_prob))
    for r in rollouts:
        _base_prompt = r.get("datum_prompt_tokens", r["sampling_prompt_tokens"])
        _use_label_ctx = (
            _lco_enabled
            and bool(r.get("inject_label", False))
            and (rng.random() < _lco_keep_prob)
        )
        r["_datum_prompt_tokens"] = r["sampling_prompt_tokens"] if _use_label_ctx else _base_prompt
        r["_label_ctx_for_opt"] = bool(_use_label_ctx)

    AUDIT_FORMAT_FAIL_PATH = getattr(CFG.training, "format_fail_audit_path", "format_fail_audit.jsonl")
    async def score_and_reward(i, r):
        response_text = r["response_text"]
        before_fix_text = r.get("wrong_response_text") or response_text
        parse_exc: Exception | None = None
        try:
            indicators = parse_indicators(response_text) or []
        except Exception as exc:
            logger.exception("indicator parsing crashed during GRPO scoring")
            indicators = []
            parse_exc = exc
        fmt = _safe_format_diagnostics_for_grpo(response_text, document)
        if parse_exc is not None:
            fmt = {**fmt, "parse_exception_type": type(parse_exc).__name__, "parse_exception": str(parse_exc)}
            if bool(fmt.get("ok")):
                fmt = _format_exception_diag(response_text, document, parse_exc)
        format_ok = bool(fmt["ok"])
        fmt_before_fix = _safe_format_diagnostics_for_grpo(before_fix_text, document)
        format_ok_before_fix = bool(fmt_before_fix["ok"])
        format_reason = str(fmt["reason"])
        format_char_diff = int(fmt["char_diff_count"])
        _k_total = len(rollouts)
        logger.debug("scoring  | rollout %d/%d: %d tells", i + 1, _k_total, len(indicators))
        if not format_ok:
            suffix = ", reward=0 and skip scoring" if _USE_FIDELITY_GATE else ", continuing (no fidelity gate)"
            logger.debug(
                "scoring  | rollout %d/%d format invalid (%s, char_diff=%d)%s",
                i + 1, _k_total, format_reason, format_char_diff, suffix,
            )
            # Always write format failures to the audit log for diagnostics.
            fail_obj = {
                "doc_id": doc.get("id", None),
                "doc_label": label,
                "rollout_index": i,
                "noise_mode": r.get("noise_mode"),
                "main_label_hint": r.get("main_label_hint"),
                "format_reason": format_reason,
                "format_char_diff": format_char_diff,
                "n_completion_tokens": len(r.get("completion_tokens", [])),
                "budget_hit": len(r.get("completion_tokens", [])) >= int(CFG.sampling.max_tokens),
                "repetition_score": _repetition_score(r.get("completion_text", "")),
                "input_text": document,
                "input_text_escaped": escaped_document,
                "response_text": response_text,
                "was_text_fixed": r.get("was_text_fixed"),
                "wrong_response_text": r.get("wrong_response_text"),
                "format_ok_before_fix": format_ok_before_fix,
                "format_diag_before_fix": fmt_before_fix,
                "indicators": indicators,
                "format_diag": fmt,
            }
            pathlib.Path(AUDIT_FORMAT_FAIL_PATH).parent.mkdir(parents=True, exist_ok=True)
            with open(AUDIT_FORMAT_FAIL_PATH, "a") as f:
                f.write(json.dumps(fail_obj, ensure_ascii=False) + "\n")
            if _USE_FIDELITY_GATE:
                wf = bool(r.get("was_text_fixed", False))
                # format failed → exclude from optimization entirely; applying uniform negative
                # advantage to all tokens is wrong when only 1 token caused the failure.
                # Exception: un-repairable failures (wf=False) when format_fail_penalty is
                # configured — include in PPO with reward=-format_fail_penalty so ALL tokens
                # receive uniform strongly-negative gradient. Using a negative reward (not 0)
                # prevents the model from switching between format-failure modes (e.g. from
                # span-spam to mock-copy) to exploit whichever is least penalized token-wise.
                _ffp = float(getattr(CFG.training, "format_fail_penalty", 0.0))
                _opt_format_fail = not wf and _ffp > 0.0
                _fail_reward = -_ffp if _opt_format_fail else 0.0
                return _ScoreResult(indicators, [], _fail_reward, _opt_format_fail, "format_fix_failed" if wf else f"format:{format_reason}", False, format_ok_before_fix, 0.0, format_char_diff, _zero_reward_components())
            # No fidelity gate: fall through to scoring even with bad format.
        t0_scoring = time.perf_counter()
        rubric_output = None
        _rubric_parse_failed = False
        try:
            if _USE_RUBRIC_SCORER:
                from rl_detector.frozen import rubric_evaluate, rubric_to_tell_scored
                from rl_detector.tell_xml import strip_score_attrs
                rubric_output = await rubric_evaluate(rubric_client, strip_score_attrs(response_text), indicators)
                if rubric_output is None:
                    # Rubric parse failed — fall back to self-scores from the model's own score=
                    # attributes. rubric_output stays None so has_rubric=False in reward_components
                    # and credibility term is dropped, but alignment/geo proxies still work.
                    logger.warning("scoring  | rollout %d/%d: rubric parse failed, falling back to self-scores", i + 1, _k_total)
                    tell_scored = self_score_from_output(response_text, indicators) or []
                    _rubric_parse_failed = True
                else:
                    tell_scored = rubric_to_tell_scored(rubric_output, indicators) if rubric_output else []
                    _rubric_parse_failed = False
            else:
                tell_scored = self_score_from_output(response_text, indicators) or []
        except Exception:
            logger.exception("scorer extraction crashed during GRPO scoring")
            if not format_ok:
                wf = bool(r.get("was_text_fixed", False))
                return _ScoreResult(indicators, [], 0.0, False, "format_fix_failed" if wf else f"format:{format_reason}", False, format_ok_before_fix, 0.0, format_char_diff, _zero_reward_components())
            return _ScoreResult(indicators, [], None, False, "scorer_exception", format_ok, format_ok_before_fix, 0.0, 0, _zero_reward_components())
        dt_scoring = time.perf_counter() - t0_scoring
        try:
            n_completion = len(r.get("completion_tokens", []))
            bud_ratio = n_completion / max(1, int(CFG.sampling.max_tokens))
            components = reward_components(response_text, document, label, tell_scored, budget_ratio=bud_ratio, rubric_output=rubric_output)
        except Exception as exc:
            logger.exception("reward component computation crashed during GRPO scoring")
            fmt = _format_exception_diag(response_text, document, exc)
            format_reason = str(fmt["reason"])
            format_ok = False
            format_char_diff = int(fmt["char_diff_count"])
            components = _zero_reward_components()
        components["rubric_parse_failed"] = 1.0 if _rubric_parse_failed else 0.0
        components["rubric_zero_annotations"] = 1.0 if (not indicators and not _rubric_parse_failed) else 0.0
        reward = components["cls"]
        agg = components["agg_score"]
        logger.debug("scoring  | rollout %d/%d done in %.1fs — agg=%.3f reward=%.3f", i + 1, _k_total, dt_scoring, agg, reward)
        wf = bool(r.get("was_text_fixed", False))
        # Optimization gate for format-fixed rollouts.
        # Default (format_fix_opt_char_ratio=0): strict_format_opt=True means only pre-fixed
        # rollouts are used for RL — avoids counterfactual gradient when fixer substantially
        # rewrote the output.
        # When format_fix_opt_char_ratio > 0 (e.g. 0.05): allow fixed rollouts for RL when
        # the fixer changed ≤ ratio * len(response) characters. Small diffs usually fix
        # doc-copy token alignment or minor structural issues, leaving annotation content
        # (type/why/score values) intact — safe for PTAD per-token advantages.
        _fix_opt_ratio = float(getattr(CFG.training, "format_fix_opt_char_ratio", 0.0))
        if format_ok and not format_ok_before_fix and _fix_opt_ratio > 0.0:
            _response_len = max(1, len(response_text))
            use_for_opt = (format_char_diff / _response_len) <= _fix_opt_ratio
        else:
            _strict_opt = bool(getattr(CFG.training, "strict_format_opt", True))
            use_for_opt = format_ok and (format_ok_before_fix if _strict_opt else True)
        # Hard exclusion for zero-annotation rollouts (distinct from reward=0 penalisation).
        # When excluded (used_for_opt=False), these rollouts are removed from GRPO group
        # normalization entirely — they contribute NO advantage to any token, including
        # no negative advantage flowing to annotation tokens in other rollouts.
        # Within the restricted group (annotating rollouts only), more annotations always
        # wins via count_factor → monotonic gradient toward target annotation count.
        # This differs from require_annotations (rewards.py) which returns reward=0 and
        # keeps used_for_opt=True, causing negative advantage on all tokens in those rollouts.
        _exclude_zero_ann = bool(_training_get("reward.ann.exclude_zero_annotation_rollouts", False))
        _excluded_zero_ann = _exclude_zero_ann and format_ok and not tell_scored
        if _excluded_zero_ann:
            use_for_opt = False
        if _excluded_zero_ann:
            exclude_reason = "zero_annotations_excluded"
        else:
            exclude_reason = "ok" if use_for_opt else f"format:{format_reason}"
        return _ScoreResult(indicators, tell_scored, reward, use_for_opt, exclude_reason, format_ok, format_ok_before_fix, dt_scoring, format_char_diff, components, rubric_output)

    logger.debug(
        "scoring  | scoring %d rollouts via %s",
        len(rollouts),
        "rubric + self-score fallback" if _USE_RUBRIC_SCORER else "policy self-scores",
    )
    t0_scoring = time.perf_counter()
    raw_results = await asyncio.gather(*[score_and_reward(i, r) for i, r in enumerate(rollouts)], return_exceptions=True)
    results: list[_ScoreResult] = []
    for i, res in enumerate(raw_results):
        if isinstance(res, Exception):
            logger.error("scoring task crashed; converting rollout to local format failure", exc_info=(type(res), res, res.__traceback__))
            response_text = rollouts[i].get("response_text", "")
            fmt = _format_exception_diag(response_text, document, res)
            results.append(_ScoreResult([], [], 0.0, False, "format:score_exception", False, False, 0.0, int(fmt["char_diff_count"]), _zero_reward_components()))
            continue
        results.append(res)
    dt_scoring = time.perf_counter() - t0_scoring
    all_indicators = [r.indicators for r in results]
    all_tell_scored = [r.tell_scored for r in results]
    rewards = [r.reward for r in results]
    used_for_optimization = [r.used_for_optimization for r in results]
    exclude_reasons = [r.exclude_reason for r in results]
    format_ok_flags = [r.format_ok for r in results]
    format_ok_before_fix_flags = [r.format_ok_before_fix for r in results]
    scoring_times = [r.dt_scoring for r in results]
    format_char_diffs = [r.format_char_diff for r in results]
    all_reward_components = [r.reward_components for r in results]
    all_rubric_traces = [rubric_output_for_trace(rubric_output=r.rubric_output) for r in results]
    for _i, _r in enumerate(rollouts):
        if bool(_r.get("was_text_fixed", False)):
            assert not bool(format_ok_before_fix_flags[_i]), (
                "was_text_fixed but pre-fix text now passes format_diagnostics; "
                "stale replay cache or validator/document migration vs cached wrong_response_text "
                f"(format_reason_at_sample={_r.get('format_reason_at_sample')!r})"
            )
    dt_scoring_mean = sum(scoring_times) / len(scoring_times) if scoring_times else 0.0
    logger.debug(
        "timing   | doc=%r rollouts=%.1fs scoring=%.1fs (scorer mean/rollout=%.1fs)",
        snippet, dt_rollouts, dt_scoring, dt_scoring_mean,
    )

    # Span excess reward penalty: subtract per-excess-span penalty from rewards used for GRPO
    # advantage computation. This makes above-target rollouts have lower (potentially negative)
    # group advantage, not just reduced per-token span-open advantages. Without this, cls/credibility
    # rewards keep above-target rollouts positive, and GRPO still reinforces the span-inflation behavior
    # even though per-token span-open advantages are negative.
    # Cutoff is target_n (matching the lottery hard cutoff), not a separate ceiling factor.
    _n_words_doc = len(document.split())
    _words_per_ann_r = float(CFG.training.reward.count.words_per_ann)
    _target_n_tells_r = max(1, int(_n_words_doc / max(1.0, _words_per_ann_r)))
    _excess_pen_r = float(getattr(CFG.training, "span_excess_penalty", 0.3))
    rewards_for_optimization = []
    for i in range(len(rollouts)):
        if used_for_optimization[i] and rewards[i] is not None:
            _n_tells_i = len(all_tell_scored[i])
            _n_excess_i = max(0, _n_tells_i - _target_n_tells_r)
            rewards_for_optimization.append(float(rewards[i]) - _n_excess_i * _excess_pen_r)

    _adv_norm = str(getattr(CFG.training, "advantage_normalization", "mean"))
    _struct_token_adv = float(getattr(CFG.training, "struct_token_adv", 0.3))
    _span_open_loss_mass = float(getattr(CFG.training, "span_open_loss_mass", 0.15))
    _span_ann_mass = float(getattr(CFG.training, "span_ann_mass", 1.0))
    _span_count_adv_weight = float(getattr(CFG.training, "span_count_adv_weight", 0.0))
    _excess_penalty = float(getattr(CFG.training, "span_excess_penalty", 0.3))
    _words_per_ann = float(CFG.training.reward.count.words_per_ann)
    # Sentinel zeros only for truly-excluded format failures (not those now in PPO with negative reward).
    # Per-component GRPO advantages → per-token roles; task loss weights balance doc/verdict/spans.
    cls_advs_sep: list[float] = []
    verdict_score_advs_sep: list[float] = []
    outer_cred_advs: list[float] = []
    vwy_combined_advs: list[float] = []
    _outer_cred_w = 0.0
    _vwy_combined_adv_w = 0.0
    _per_tell_type_advs_map: dict[int, list[float] | None] = {}
    _per_tell_why_advs_map: dict[int, list[float] | None] = {}
    _per_tell_score_advs_map: dict[int, list[float] | None] = {}
    _per_span_open_advs_map: dict[int, list[float] | None] = {}
    if rewards_for_optimization:
        _valid_idx = [i for i in range(len(rollouts)) if used_for_optimization[i] and rewards[i] is not None]
        _rlcfg = getattr(CFG.training, "rl_loss_fn_config", None)
        _clip_low = float(getattr(_rlcfg, "clip_low_threshold", 0.8)) if _rlcfg else 0.8
        _clip_high = float(getattr(_rlcfg, "clip_high_threshold", 1.28)) if _rlcfg else 1.28
        _adv_clip_scale = float(getattr(_rlcfg, "adv_clip_scale", 10.0)) if _rlcfg else 10.0
        _adv_clip_low = -(1.0 - _clip_low) * _adv_clip_scale
        _adv_clip_high = (_clip_high - 1.0) * _adv_clip_scale
        # Verdict type: GRPO on label-alignment rewards (+1 / -1); no advantage clipping.
        _verdict_type_vals = [
            float(all_reward_components[i].get("cls", 0.0))
            for i in _valid_idx
        ]
        _outer_cred_vals = [float(all_reward_components[i].get("outer_credibility", 0.0)) for i in _valid_idx]
        _vwy_combined_vals = [float(all_reward_components[i].get("verdict_why_combined", 0.0)) for i in _valid_idx]
        _verdict_score_vals = [
            float(outer_verdict_score_reward(output=rollouts[i]["response_text"], label=label))
            for i in _valid_idx
        ]
        verdict_score_advs_sep = compute_advantages(_verdict_score_vals, std_floor=adv_std_floor, normalize=_adv_norm)
        # Verdict-type token advantage: format-gated, mode selected by training.cls_adv_mode.
        # "score_grpo" (default): GRPO on outer_verdict_score_reward — continuous variance prevents
        #   GRPO collapse; quality-weighted; bad-format rollouts contribute 0.
        # "binary": ground-truth label sign (±1 for AI/human) — stronger signal when model collapses
        #   to one direction; bad-format rollouts contribute 0.
        _cls_adv_mode = getattr(CFG.training, "cls_adv_mode", "score_grpo")
        if _cls_adv_mode == "binary":
            # Use per-rollout correctness (+1 correct, -1 wrong) from cls reward component.
            # _verdict_type_vals already has this; _label_sign (constant per doc) was wrong —
            # it gave all human-doc rollouts -1 regardless of prediction, killing the gradient.
            _cls_raw_vals = [
                _verdict_type_vals[k] if format_ok_flags[_valid_idx[k]] else 0.0
                for k in range(len(_valid_idx))
            ]
        else:
            _cls_raw_vals = [
                v if format_ok_flags[_valid_idx[k]] else 0.0
                for k, v in enumerate(_verdict_score_vals)
            ]
        # Binary mode: raw ±W directly — skipping compute_advantages avoids GRPO asymmetry bug
        # where format-gate zeros create perverse incentives (bad-format human rollout gets +3.0 adv).
        # cls_adv_weight scales the binary signal to compete with annotation PTAD advantages.
        _cls_adv_w = float(getattr(CFG.training, "cls_adv_weight", 1.0))
        if _cls_adv_mode == "binary":
            cls_advs_sep = [v * _cls_adv_w for v in _cls_raw_vals]
        else:
            cls_advs_sep = compute_advantages(_cls_raw_vals, std_floor=adv_std_floor, normalize=_adv_norm)
        outer_cred_advs = compute_advantages(_outer_cred_vals, std_floor=adv_std_floor, normalize=_adv_norm)
        vwy_combined_advs = compute_advantages(_vwy_combined_vals, std_floor=adv_std_floor, normalize=_adv_norm)
        _outer_cred_w = float(getattr(CFG.training, "outer_credibility_adv_weight", 0.0))
        _vwy_combined_adv_w = float(getattr(CFG.training, "verdict_why_adv_weight", 5.0))

        # Per-annotation per-token-type rewards (PTAD).
        #   type: binary ±ann_type_adv_weight — label alignment only, same principle as
        #         binary cls_adv_mode for the outer verdict. Wrong-type gets -weight,
        #         correct-type gets +weight, unconditionally, no GRPO normalization.
        #   why:  GRPO within each rollout's annotations, type-agnostic. Better explanations
        #         get positive advantage, worse ones negative — this is the intended learning
        #         signal. Type alignment must NOT factor in: the type token already owns that
        #         signal, and the model writes the why after deciding the type.
        _ann_type_adv_w = float(getattr(CFG.training, "ann_type_adv_weight", 1.0))
        _per_tell_type_advs_by_vi: list[list[float] | None] = []
        _why_rows: list[list[float | None]] = []
        _score_rows: list[list[float | None]] = []
        for vi in _valid_idx:
            type_adv_row: list[float] = []
            why_row: list[float | None] = []
            score_row: list[float | None] = []
            for fs in all_tell_scored[vi]:
                rc = fs.get("rubric_credibility")
                if rc is not None:
                    ann_type = fs.get("type") or ""
                    tr = annotation_type_reward(annotation_type=ann_type, label=label)
                    wr = annotation_why_reward(float(rc), why_text=fs.get("explanation", ""))
                    type_adv_row.append(tr * _ann_type_adv_w)
                    why_row.append(wr)
                    score_row.append(None)
                else:
                    type_adv_row.append(0.0)
                    why_row.append(None)
                    score_row.append(None)
            _per_tell_type_advs_by_vi.append(type_adv_row if type_adv_row else None)
            _why_rows.append(why_row)
            _score_rows.append(score_row)

        def _norm_pool_per_rollout(rows: list[list[float | None]]) -> list[list[float] | None]:
            """GRPO within each rollout's annotations only (not across rollouts or docs)."""
            out: list[list[float] | None] = []
            for row in rows:
                vals = [float(v) for v in row if v is not None]
                if len(vals) < 2:
                    out.append(None)
                    continue
                advs = compute_advantages(vals, std_floor=adv_std_floor, normalize=_adv_norm)
                it = iter(advs)
                out.append([next(it) if v is not None else 0.0 for v in row])
            return out

        def _norm_pool_cross_rollout(rows: list[list[float | None]]) -> list[list[float] | None]:
            """GRPO across all annotations from all rollouts of this document.

            Annotations from correct-verdict rollouts compete against those from
            wrong-verdict rollouts in the same pool, so better explanations on
            better rollouts get positive advantages. Falls back to per-rollout
            normalization for rollouts whose annotations are all None.
            """
            positions: list[tuple[int, int]] = []
            vals: list[float] = []
            for row_idx, row in enumerate(rows):
                for ann_idx, v in enumerate(row):
                    if v is not None:
                        vals.append(float(v))
                        positions.append((row_idx, ann_idx))
            if len(vals) < 2:
                return [None] * len(rows)
            flat_advs = compute_advantages(vals, std_floor=adv_std_floor, normalize=_adv_norm)
            adv_map: dict[tuple[int, int], float] = {pos: adv for pos, adv in zip(positions, flat_advs)}
            out: list[list[float] | None] = []
            for row_idx, row in enumerate(rows):
                if not any(v is not None for v in row):
                    out.append(None)
                else:
                    out.append([adv_map.get((row_idx, ann_idx), 0.0) for ann_idx in range(len(row))])
            return out

        _per_tell_why_advs_by_vi = _norm_pool_cross_rollout(_why_rows)
        _per_tell_score_advs_by_vi = _norm_pool_per_rollout(_score_rows)

        _per_tell_type_advs_map: dict[int, list[float] | None] = {vi: _per_tell_type_advs_by_vi[k] for k, vi in enumerate(_valid_idx)}
        _per_tell_why_advs_map: dict[int, list[float] | None] = {vi: _per_tell_why_advs_by_vi[k] for k, vi in enumerate(_valid_idx)}
        _per_tell_score_advs_map: dict[int, list[float] | None] = {vi: _per_tell_score_advs_by_vi[k] for k, vi in enumerate(_valid_idx)}
        # Per-span-open credibility: unsigned rubric_credibility, cross-rollout normalized.
        # Nested tells (_nested=True) are forced to 0.0 credibility so their SPAN_OPEN tokens
        # get a negative per-span-open advantage after normalization (penalising the decision
        # to open a nested span). Their content tokens (ann_why, ann_type) are unaffected —
        # the rubric still scores those normally via per_tell_advs above.
        _all_soc: list[float] = []
        _soc_by_vi: list[list[float | None]] = []
        for vi in _valid_idx:
            row: list[float | None] = []
            for fs in all_tell_scored[vi]:
                if fs.get("_nested"):
                    row.append(0.0)
                    _all_soc.append(0.0)
                else:
                    rc = fs.get("rubric_credibility")
                    if rc is not None:
                        row.append(float(rc))
                        _all_soc.append(float(rc))
                    else:
                        row.append(None)
            _soc_by_vi.append(row)
        if len(_all_soc) >= 2:
            _soc_mean = sum(_all_soc) / len(_all_soc)
            _soc_std = max(float(np.std(_all_soc)), 0.1)
            _soa_by_vi: list[list[float] | None] = []
            for row in _soc_by_vi:
                if all(rc is None for rc in row):
                    _soa_by_vi.append(None)
                else:
                    raw = [(float(rc) - _soc_mean) / _soc_std if rc is not None else 0.0 for rc in row]
                    _soa_by_vi.append([max(_adv_clip_low, min(_adv_clip_high, a)) for a in raw])
        else:
            _soa_by_vi = [None] * len(_valid_idx)
        # Base per-span-open map from credibility only.
        _cred_soa_map: dict[int, list[float] | None] = {
            vi: _soa_by_vi[k] for k, vi in enumerate(_valid_idx)
        }
        # Per-span-open: K-of-N lottery with budget-aware ticket scaling.
        # Tickets for each span are scaled by how much token budget remains when that
        # span opens: spans opening near the budget limit receive fewer tickets,
        # discouraging late span-opening that crowds out the verdict.
        # Under target: all spans win, but advantage = ticket_scale (0–1) rather than
        # flat +1.0 — spans near budget limit get neutral (0) rather than a push.
        # Over target: credibility-weighted lottery among budget-scaled tickets; losers
        # get -excess_penalty. Always fires regardless of rubric availability.
        _target_n_tells = _target_n_tells_r
        _per_span_open_advs_map: dict[int, list[float] | None] = {}
        _doc_id_str = str(doc.get("id", ""))
        _base_tickets = 0.5
        _max_tokens_f = float(CFG.sampling.max_tokens)
        _bpen_start = float(getattr(CFG.training, "budget_penalty_start", 0.40))
        _bpen_zero = float(getattr(CFG.training, "budget_penalty_zero", 0.60))
        for vi in _valid_idx:
            _n_tells_vi = len(all_tell_scored[vi])
            if _n_tells_vi == 0:
                _per_span_open_advs_map[vi] = []
                continue
            # Compute budget ticket scale for each span based on its token position.
            _completion_vi = rollouts[vi]["completion_tokens"]
            _n_reas_vi = int(rollouts[vi].get("n_reasoning_tokens", 0))
            _span_tok_positions: list[int] = []
            for _pos, _tok in enumerate(_completion_vi[_n_reas_vi:]):
                if int(_tok) == _MASK_SPAN_OPEN:
                    _span_tok_positions.append(_n_reas_vi + _pos)
            _budget_scales: list[float] = []
            for _spos in _span_tok_positions:
                _bfrac = _spos / _max_tokens_f
                if _bfrac <= _bpen_start:
                    _budget_scales.append(1.0)
                elif _bfrac >= _bpen_zero:
                    _budget_scales.append(0.0)
                else:
                    _budget_scales.append(1.0 - (_bfrac - _bpen_start) / (_bpen_zero - _bpen_start))
            # Pad/trim to match n_tells (should always match; guard against edge cases)
            while len(_budget_scales) < _n_tells_vi:
                _budget_scales.append(1.0)
            _budget_scales = _budget_scales[:_n_tells_vi]
            if _n_tells_vi <= _target_n_tells:
                _per_span_open_advs_map[vi] = [_budget_scales[j] for j in range(_n_tells_vi)]
            else:
                _lottery_rng = random.Random(hash((_doc_id_str, vi, step)) & 0xFFFFFFFF)
                _ann_weights = [
                    (max(0.0, float(fs.get("rubric_credibility") or 0.0)) + _base_tickets) * _budget_scales[j]
                    for j, fs in enumerate(all_tell_scored[vi])
                ]
                _pool_idx = list(range(_n_tells_vi))
                _pool_w = list(_ann_weights)
                _selected: set[int] = set()
                for _ in range(_target_n_tells):
                    _total = sum(_pool_w)
                    if _total <= 0.0:
                        break
                    _r = _lottery_rng.random() * _total
                    _cum = 0.0
                    for _pi, _pw in enumerate(_pool_w):
                        _cum += _pw
                        if _cum >= _r:
                            _selected.add(_pool_idx[_pi])
                            _pool_idx.pop(_pi)
                            _pool_w.pop(_pi)
                            break
                _per_span_open_advs_map[vi] = [
                    1.0 if j in _selected else -_excess_penalty
                    for j in range(_n_tells_vi)
                ]

    is_ratio_max_cfg = float(getattr(CFG.training, "is_ratio_max", float("inf")))
    token_type_masks = [
        compute_token_type_mask(tokenizer, r["completion_tokens"], r.get("n_reasoning_tokens", 0))
        for r in rollouts
    ]
    token_surprisal_stats = [
        _token_surprisal_stats(
            r["completion_logprobs"],
            int(r.get("n_reasoning_tokens", 0)),
            token_type_masks[i],
        )
        for i, r in enumerate(rollouts)
    ]
    indicator_rows_by_rollout = [
        [
            {
                "span_text": ind.get("span_text", ""),
                "explanation": ind.get("explanation", ""),
                "type": ind.get("type"),
                "model_score": float(ind.get("model_score", 0.0) or 0.0),
                "rubric_reasoning": fs.get("rubric_reasoning", ""),
                "rubric_credibility": fs.get("rubric_credibility"),
                "_nested": fs.get("_nested", False),
            }
            for ind, fs in zip(all_indicators[i], all_tell_scored[i])
        ]
        for i in range(len(rollouts))
    ]
    why_mean_len_by_rollout = [
        (sum(len(ind.get("explanation", "")) for ind in all_indicators[i]) / len(all_indicators[i])) if all_indicators[i] else 0.0
        for i in range(len(rollouts))
    ]
    why_max_len_by_rollout = [
        max((len(ind.get("explanation", "")) for ind in all_indicators[i]), default=0)
        for i in range(len(rollouts))
    ]
    why_repetition_score_by_rollout = [
        _repetition_score(" ".join(ind.get("explanation", "") for ind in all_indicators[i]))
        for i in range(len(rollouts))
    ]
    training_traces = [
        TrainingRolloutTracePayload(
            step=int(step),
            rollout_index=i,
            doc_label=int(label),
            noise_mode=r.get("noise_mode"),
            inject_label=bool(r.get("inject_label", False)),
            main_label_hint=r.get("main_label_hint"),
            label_ctx_for_opt=bool(r.get("_label_ctx_for_opt", False)),
            response_text=r.get("response_text", ""),
            wrong_response_text=r.get("wrong_response_text"),
            reward=rewards[i],
            reward_components=all_reward_components[i],
            advantage=None,  # populated below after per-rollout adv is computed
            component_advantages=None,
            used_for_optimization=bool(used_for_optimization[i]),
            exclude_reason=str(exclude_reasons[i]),
            format_ok=bool(format_ok_flags[i]),
            format_ok_before_fixing=bool(format_ok_before_fix_flags[i]),
            format_reason=r.get("format_reason"),
            format_reason_before_fixing=r.get("format_reason_at_sample"),
            format_char_diff_count=int(format_char_diffs[i]),
            is_ratio=r.get("is_ratio"),
            from_replay_cache=False,
            document=document,
            doc_stratum="|".join(str(x) for x in doc_stratum_key(doc)),
            neutral_prompt_text=r.get("neutral_prompt_text", ""),
            completion_text=r.get("completion_text", ""),
            full_output_text=r.get("full_output_text", ""),
            raw_response_text=(r.get("wrong_response_text") or r.get("response_text", "")),
            was_text_fixed=bool(r.get("was_text_fixed", False)),
            token_surprisal=token_surprisal_stats[i],
            ann_token_fraction=None,
            n_ann_tokens=None,
            n_response_tokens=None,
            indicators=indicator_rows_by_rollout[i],
            token_optimization_rows=[],  # populated below after per-token advantages are computed
            completion_tokens_len=len(r["completion_tokens"]),
            completion_logprobs_len=len(r["completion_logprobs"]),
            n_reasoning_tokens=int(r.get("n_reasoning_tokens", 0)),
            budget_hit=len(r["completion_tokens"]) >= int(CFG.sampling.max_tokens),
            repetition_score=_repetition_score(r.get("completion_text", "")),
            why_count=len(all_indicators[i]),
            why_mean_len=why_mean_len_by_rollout[i],
            why_max_len=why_max_len_by_rollout[i],
            why_repetition_score=why_repetition_score_by_rollout[i],
            rubric=all_rubric_traces[i],
        )
        for i, r in enumerate(rollouts)
    ]
    datums = []
    cacheable_rollouts: list[dict] = []  # populated below; consumed by replay sampler
    format_repair_ce_datums: list[tinker.Datum] = []
    hint_ce_datums: list[tinker.Datum] = []
    adv_idx = 0
    for i, r in enumerate(rollouts):
        if not used_for_optimization[i] or rewards[i] is None:
            continue
        if not r["completion_tokens"]:
            adv_idx += 1  # rollout is in _valid_idx; must consume its slot to keep adv_idx aligned
            continue
        clipped_is = min(is_ratios[i], is_ratio_max_cfg)
        _cls_adv = cls_advs_sep[adv_idx] * clipped_is if cls_advs_sep else None
        _verdict_score_adv = verdict_score_advs_sep[adv_idx] * clipped_is if verdict_score_advs_sep else None
        _verdict_why_adv = outer_cred_advs[adv_idx] * _outer_cred_w * clipped_is if outer_cred_advs else None
        _vwy_combined_adv = vwy_combined_advs[adv_idx] * _vwy_combined_adv_w * clipped_is if vwy_combined_advs else None
        adv_idx += 1
        n_reason = r.get("n_reasoning_tokens", 0)
        ttm = token_type_masks[i]
        # Diag stats per rollout: how many response tokens actually receive gradient.
        # If this collapses to ~0 or jumps to 100%, the format gate / mask helper is
        # broken and we want to see it in W&B before training silently fails.
        R_i = min(int(n_reason), len(r["completion_tokens"]))
        if ttm is not None:
            response_types_i = ttm[R_i:]
            n_resp = len(response_types_i)
            n_ann = int(sum(1 for t in response_types_i if t != TOKEN_TYPE_DOC_COPY))
        else:
            n_resp = max(0, len(r["completion_tokens"]) - R_i)
            n_ann = n_resp
        ann_token_fraction = (n_ann / n_resp) if n_resp else 0.0
        response_advantages = _build_per_token_advantages(
            completion_tokens=r["completion_tokens"],
            n_reasoning_tokens=int(n_reason),
            cls_adv=_cls_adv,
            tell_scored=all_tell_scored[i],
            tokenizer=tokenizer,
            label_ctx_for_opt=bool(r.get("_label_ctx_for_opt", False)),
            per_tell_type_advs=_per_tell_type_advs_map.get(i),
            per_tell_why_advs=_per_tell_why_advs_map.get(i),
            per_tell_score_advs=_per_tell_score_advs_map.get(i),
            verdict_why_adv=_verdict_why_adv,
            verdict_ann_recall_adv=_vwy_combined_adv,
            verdict_score_adv=_verdict_score_adv,
            per_span_open_advs=[a * clipped_is for a in _per_span_open_advs_map[i]] if _per_span_open_advs_map.get(i) is not None else None,
            struct_token_adv=_struct_token_adv,
            span_open_fail_penalty=float(getattr(CFG.training, "span_open_fail_penalty", _excess_penalty)),
        )
        response_advantages = _zero_outer_verdict_type_cls_adv_for_label_ctx(
            response_advantages=response_advantages,
            completion_tokens=r["completion_tokens"],
            n_reasoning_tokens=int(n_reason),
            tokenizer=tokenizer,
            label_ctx_for_opt=bool(r.get("_label_ctx_for_opt", False)),
        )
        training_traces[i].advantage = None
        _soa_list = _per_span_open_advs_map.get(i)
        _soa_mean = (sum(_soa_list) / len(_soa_list)) if _soa_list else None
        training_traces[i].component_advantages = {
            "span_open_cred": _soa_mean,
            "cls": float(_cls_adv),
            "outer_cred": float(_verdict_why_adv) if _verdict_why_adv is not None else None,
        }
        training_traces[i].ann_token_fraction = ann_token_fraction
        training_traces[i].n_ann_tokens = float(n_ann)
        training_traces[i].n_response_tokens = float(n_resp)
        training_traces[i].token_optimization_rows = _token_optimization_rows(
            tokenizer=tokenizer,
            prompt_tokens=r["_datum_prompt_tokens"],
            completion_tokens=r["completion_tokens"],
            completion_logprobs=r["completion_logprobs"],
            n_reasoning_tokens=n_reason,
            response_advantages=response_advantages,
        )
        _rtn_id = _get_return_token_id(tokenizer)
        _ctokens = list(r["completion_tokens"])
        _clogprobs = list(r["completion_logprobs"])
        _radv = list(response_advantages)
        if format_ok_flags[i] and _rtn_id is not None and (not _ctokens or _ctokens[-1] != _rtn_id):
            _ctokens.append(_rtn_id)
            _clogprobs.append(0.0)
            _radv.append(0.0)
        _rweights = compute_task_loss_weights(
            tokenizer=tokenizer,
            completion_tokens=_ctokens,
            n_reasoning_tokens=int(n_reason),
            span_open_loss_mass=_span_open_loss_mass,
            span_ann_mass=_span_ann_mass,
            ptok_loss_scales=_ptok_loss_scales_from_cfg(),
        )
        if len(_rweights) != len(_radv):
            raise AssertionError(
                "task weights len %d != advantages len %d" % (len(_rweights), len(_radv))
            )
        datums.append(
            build_datum(
                prompt_tokens=r["_datum_prompt_tokens"],
                completion_tokens=_ctokens,
                completion_logprobs=_clogprobs,
                response_advantages=_radv,
                response_task_weights=_rweights,
                n_reasoning_tokens=int(n_reason),
            )
        )
        # Format-repair CE: only for rollouts that were broken but got fixed.
        if bool(format_ok_flags[i]) and not bool(format_ok_before_fix_flags[i]):
            ce_datum = _build_aux_ce_datum(
                prompt_tokens=r["_datum_prompt_tokens"],
                completion_tokens=r["completion_tokens"],
                n_reasoning_tokens=int(n_reason),
                token_type_mask=ttm,
            )
            if ce_datum is not None:
                format_repair_ce_datums.append(ce_datum)
            # Contrastive negative CE: push down the bad token at the divergence point.
            _wrong_text = r.get("wrong_response_text")
            if _wrong_text:
                _pre_fix_tokens = tokenizer.encode(_wrong_text, add_special_tokens=False)
                neg_datum = _build_contrastive_neg_datum(
                    prompt_tokens=r["_datum_prompt_tokens"],
                    fixed_completion_tokens=r["completion_tokens"],
                    pre_fix_completion_tokens=_pre_fix_tokens,
                    n_reasoning_tokens=int(n_reason),
                )
                if neg_datum is not None:
                    format_repair_ce_datums.append(neg_datum)
        # Wrong-verdict CE: for unknown rollouts where the model predicted the wrong
        # outer verdict type, add a CE datum correcting it toward the true label.
        # This provides a direct, absolute gradient for type correction — GRPO only
        # discriminates relatively within a group, so it's weak when the model is
        # consistently wrong on specific docs. CE pushes unconditionally toward the
        # correct type using the no-hint stub prompt (same distribution as eval).
        _wrong_verdict_ce = bool(getattr(CFG.training, "wrong_verdict_ce_enabled", True))
        if _wrong_verdict_ce and not bool(r.get("inject_label", False)) and bool(format_ok_before_fix_flags[i]):
            _outer_type_ai_wv = float(all_reward_components[i].get("outer_type_ai", -1.0))
            if _outer_type_ai_wv >= 0.0:
                _predicted_ai = _outer_type_ai_wv >= 0.5
                _true_ai = label == 1
                if _predicted_ai != _true_ai:
                    _wv_datum = _build_outer_type_ce_datum(
                        prompt_tokens=r["_datum_prompt_tokens"],  # stub prompt, no hint
                        completion_tokens=r["completion_tokens"],
                        n_reasoning_tokens=int(n_reason),
                        target_type=int(_true_ai),
                        tokenizer=tokenizer,
                    )
                    if _wv_datum is not None:
                        hint_ce_datums.append(_wv_datum)
        _hint_outer_ce = bool(getattr(CFG.training, "hint_outer_ce_enabled", False))
        if _hint_outer_ce and bool(r.get("inject_label", False)) and bool(format_ok_before_fix_flags[i]):
            _hint = r.get("main_label_hint")
            _outer_type_ai = float(all_reward_components[i].get("outer_type_ai", -1.0))
            _sampling_prompt = r.get("sampling_prompt_tokens") or r["_datum_prompt_tokens"]
            _stub_prompt = r["_datum_prompt_tokens"]
            _noise_mode = r.get("noise_mode", "unknown")
            if _hint is not None and _outer_type_ai >= 0.0:
                _hint_followed = (_outer_type_ai >= 0.5) == (_hint == 1)
                if not _hint_followed:
                    # Model didn't follow the hint. CE corrects toward hint.
                    # Always use sampling_prompt (contains the hint text) so the CE is
                    # conditioned on the trigger.  Using stub here would optimise the type
                    # token without the hint present — teaching the model to output a label
                    # unconditionally rather than in response to the hint.
                    # 1. Forward: hint → correct outer type.
                    fwd_datum = _build_outer_type_ce_datum(
                        prompt_tokens=_sampling_prompt,
                        completion_tokens=r["completion_tokens"],
                        n_reasoning_tokens=int(n_reason),
                        target_type=int(_hint),
                        tokenizer=tokenizer,
                    )
                    if fwd_datum is not None:
                        hint_ce_datums.append(fwd_datum)
                    # 2. Mirror: flipped hint → opposite type (contrastive counterpart).
                    _flipped_prompt = _flip_hint_in_prompt_tokens(_sampling_prompt, int(_hint), tokenizer)
                    if _flipped_prompt is not None:
                        mirror_datum = _build_outer_type_ce_datum(
                            prompt_tokens=_flipped_prompt,
                            completion_tokens=r["completion_tokens"],
                            n_reasoning_tokens=int(n_reason),
                            target_type=1 - int(_hint),
                            tokenizer=tokenizer,
                        )
                        if mirror_datum is not None:
                            hint_ce_datums.append(mirror_datum)
        if format_ok_flags[i]:
            cacheable_rollouts.append({
                "prompt_tokens": list(r["_datum_prompt_tokens"]),
                "completion_tokens": list(r["completion_tokens"]),
                "completion_logprobs": list(r["completion_logprobs"]),
                "n_reasoning_tokens": int(n_reason),
                "token_type_mask": list(ttm),
                "reward": float(rewards[i]),
                "format_ok": True,
                "format_ok_before_fixing": bool(format_ok_before_fix_flags[i]),
                "format_reason": r.get("format_reason"),
                "format_reason_at_sample": r.get("format_reason_at_sample"),
                "format_char_diff_count": int(format_char_diffs[i]),
                "noise_mode": r.get("noise_mode"),
                "inject_label": bool(r.get("inject_label", False)),
                "main_label_hint": r.get("main_label_hint"),
                "_label_ctx_for_opt": bool(r.get("_label_ctx_for_opt", False)),
                "response_text": r.get("response_text", ""),
                "wrong_response_text": r.get("wrong_response_text"),
                "neutral_prompt_text": r.get("neutral_prompt_text", ""),
                "completion_text": r.get("completion_text", ""),
                "full_output_text": r.get("full_output_text", ""),
                "was_text_fixed": bool(r.get("was_text_fixed", False)),
                "reward_components": all_reward_components[i],
                "indicators": indicator_rows_by_rollout[i],
                "rubric_trace": all_rubric_traces[i],
            })

    _pri = [rw for rw, use in zip(rewards, used_for_optimization) if use and rw is not None]
    reward_mean = sum(_pri) / len(_pri) if _pri else 0.0
    _cms = str(_training_get("curriculum.signal", "cls", legacy_key="curriculum_signal")).strip().lower()
    if _cms == "cls":
        _cls_cur = [
            float(all_reward_components[i].get("cls", 0.0))
            for i in range(len(rollouts))
            if format_ok_flags[i]
        ]
        curriculum_mean_doc = (sum(_cls_cur) / len(_cls_cur)) if _cls_cur else None
    else:
        curriculum_mean_doc = reward_mean
    # Rates are computed over ALL rollouts for this doc (not just optimized ones) so the
    # raw sampling pass-rate stays comparable across steps regardless of how many rollouts
    # got format-repaired or excluded.
    format_rate = (sum(1 for fmt in format_ok_flags if fmt) / len(format_ok_flags)) if format_ok_flags else 0.0
    format_rate_before_fixing = (
        sum(1 for fmt in format_ok_before_fix_flags if fmt) / len(format_ok_before_fix_flags)
        if format_ok_before_fix_flags else 0.0
    )

    # Intra-doc why quality: how diverse/unique are why texts across K rollouts for this doc.
    # why_unique_rate=1.0 → all explanations distinct across rollouts (good).
    # why_unique_rate→1/K → every rollout copies the same explanations (collapsed).
    # why_intra_collapse_rate = fraction of rollout pairs with identical explanation sets.
    _why_texts_by_rollout = [
        [ind.get("explanation", "") for ind in all_indicators[i] if ind.get("explanation", "")]
        for i in range(len(rollouts))
    ]
    _all_why_flat = [w for ws in _why_texts_by_rollout for w in ws]
    _why_unique_rate = len(set(_all_why_flat)) / max(len(_all_why_flat), 1)
    _why_sets = [frozenset(ws) for ws in _why_texts_by_rollout if ws]
    _n_pairs = len(_why_sets) * (len(_why_sets) - 1) // 2
    _identical_pairs = sum(
        1 for a in range(len(_why_sets)) for b in range(a + 1, len(_why_sets))
        if _why_sets[a] == _why_sets[b]
    )
    _why_intra_collapse_rate = (_identical_pairs / _n_pairs) if _n_pairs > 0 else 0.0
    _why_rep_scores = [_repetition_score(" ".join(ws)) for ws in _why_texts_by_rollout]

    doc_audit = {
        "ease_uid": doc_ease_uid(doc),
        "stratum_key": "|".join(str(x) for x in doc_stratum_key(doc)),
        "document": document,
        "label": label,
        "noise_modes": noise_modes,
        "main_label_hints": main_label_hints,
        "inject_label_flags": inject_label_flags,
        "reward_mean": reward_mean,
        "curriculum_mean": curriculum_mean_doc,
        "format_rate": format_rate,
        "format_rate_before_fixing": format_rate_before_fixing,
        "n_excluded_rollouts": sum(1 for use in used_for_optimization if not use),
        "budget_hit_rate": sum(1 for r in rollouts if len(r["completion_tokens"]) >= int(CFG.sampling.max_tokens)) / max(len(rollouts), 1),
        "repetition_score_mean": sum(_repetition_score(r.get("completion_text", "")) for r in rollouts) / max(len(rollouts), 1),
        "why_unique_rate": _why_unique_rate,
        "why_intra_collapse_rate": _why_intra_collapse_rate,
        "why_repetition_score_mean": sum(_why_rep_scores) / max(len(_why_rep_scores), 1),
        "is_ratios": is_ratios,
        "is_ratio_mean": is_ratio_mean,
        "is_ratio_min": is_ratio_min,
        "is_ratio_max": is_ratio_max,
        "rollouts": [training_trace_payload_to_audit_dict(trace) for trace in training_traces],
    }
    for trace in training_traces:
        _trace_training_rollout(trace=trace)

    doc_audit["cacheable_rollouts"] = cacheable_rollouts

    doc_timing = {
        "dt_rollouts": dt_rollouts,
        "dt_scoring": dt_scoring,
        "dt_scoring_mean": dt_scoring_mean,
    }
    return datums, doc_audit, doc_timing, format_repair_ce_datums, hint_ce_datums


def _stratum_mean_rewards(docs: list[dict], docs_audit: list[dict]) -> dict[tuple[str, str, int], float]:
    """Per-stratum mean for StratumSampler.update: curriculum_signal cls (format_ok only) or total (optimized rollouts)."""
    sig = str(_training_get("curriculum.signal", "cls", legacy_key="curriculum_signal")).strip().lower()
    buckets: dict[tuple[str, str, int], list[float]] = {}
    for doc, da in zip(docs, docs_audit):
        key = (str(doc.get("dataset_id", "unknown")), str(doc.get("domain", "unknown")), int(doc.get("label", 0)))
        if sig == "cls":
            vs = [
                float((ro.get("reward_components") or {}).get("cls", 0.0))
                for ro in da["rollouts"]
                if ro.get("format_ok")
            ]
            if vs:
                buckets.setdefault(key, []).extend(vs)
        else:
            rewards = [ro["reward"] for ro in da["rollouts"] if ro.get("used_for_optimization") and ro["reward"] is not None]
            if rewards:
                buckets.setdefault(key, []).extend(rewards)
    return {k: sum(v) / len(v) for k, v in buckets.items()}


def _stratum_format_rates(docs: list[dict], docs_audit: list[dict]) -> dict[tuple[str, str, int], float]:
    """Compute pre-fix format pass rate per stratum — used to identify strata causing format collapse."""
    sums: dict[tuple[str, str, int], float] = {}
    cnts: dict[tuple[str, str, int], int] = {}
    for doc, da in zip(docs, docs_audit):
        key = (str(doc.get("dataset_id", "unknown")), str(doc.get("domain", "unknown")), int(doc.get("label", 0)))
        for ro in da["rollouts"]:
            # prefer the canonical name written earlier: format_ok_before_fixing
            fok = ro.get("format_ok_before_fixing", ro.get("format_ok_before_fix", None))
            if fok is None:
                fok = ro.get("format_ok", False)
            sums[key] = sums.get(key, 0.0) + (1.0 if fok else 0.0)
            cnts[key] = cnts.get(key, 0) + 1
    return {k: sums[k] / cnts[k] for k in sums if cnts[k] > 0}


def _probe_rollout_scalar(ro: dict) -> float | None:
    sig = str(_training_get("curriculum.signal", "cls", legacy_key="curriculum_signal")).strip().lower()
    if sig == "cls":
        if not bool(ro.get("format_ok", False)):
            return None
        return float((ro.get("reward_components") or {}).get("cls", 0.0))
    rw = ro.get("reward")
    if rw is None:
        return None
    if not bool(ro.get("format_ok", False)):
        return 0.0
    return float(rw)


def _aggregate_probe_stratum_stats(
    docs: list[dict], audits: list[dict]
) -> tuple[dict[tuple[str, str, int], float], dict[tuple[str, str, int], int]]:
    sums: dict[tuple[str, str, int], float] = {}
    cnts: dict[tuple[str, str, int], int] = {}
    for doc, da in zip(docs, audits):
        k = doc_stratum_key(doc)
        for ro in da["rollouts"]:
            v = _probe_rollout_scalar(ro)
            if v is None:
                continue
            sums[k] = sums.get(k, 0.0) + v
            cnts[k] = cnts.get(k, 0) + 1
    means = {k: sums[k] / cnts[k] for k in sums if cnts.get(k, 0) > 0}
    return means, cnts


async def run_stratum_bootstrap_probe(
    training_client,
    tokenizer,
    rubric_client,
    all_docs: list[dict],
    seed: int,
) -> tuple[dict[tuple[str, str, int], float], dict[tuple[str, str, int], int], dict[str, float]]:
    per_stratum = int(
        _training_get("curriculum.probe.samples_per_stratum", 5, legacy_key="stratum_probe_samples_per_stratum")
    )
    pk = int(_training_get("curriculum.probe.k", 1, legacy_key="stratum_probe_k"))
    vmax = int(_training_get("curriculum.probe.visit_boost_max", 8, legacy_key="stratum_probe_visit_boost_max"))
    probe_docs = pick_stratum_probe_docs(all_docs, per_stratum, seed)
    if not probe_docs:
        return {}, {}, {}
    logger.debug(
        "stratum probe | %d train docs (%d/stratum, no val leakage), k=%d rollouts each",
        len(probe_docs),
        per_stratum,
        pk,
    )
    t0 = time.perf_counter()
    sampling_client = await training_client.save_weights_and_get_sampling_client_async()
    # measure raw model outputs at probe time; no repair pass here
    _old_fix = getattr(CFG.training, "fix_format_errors", True)
    _old_correct = float(getattr(CFG.training, "label_noise_correct_prob", 0.5))
    _old_flip = float(getattr(CFG.training, "label_noise_flip_prob", 0.0))
    _old_unknown = float(getattr(CFG.training, "label_noise_unknown_prob", 0.5))
    # probe must match eval: no label injection so difficulty reflects true model capability
    CFG.training.fix_format_errors = False
    CFG.training.label_noise_correct_prob = 0.0
    CFG.training.label_noise_flip_prob = 0.0
    CFG.training.label_noise_unknown_prob = 1.0
    try:
        raw = await asyncio.gather(
            *[
                _process_doc(
                    sampling_client,
                    tokenizer,
                    rubric_client,
                    doc,
                    rng=random.Random(seed + 313 + i),
                    rollout_seed=seed + 17_000 + i,
                    k=pk,
                    adv_std_floor=0.0,
                    step=0,
                )
                for i, doc in enumerate(probe_docs)
            ],
            return_exceptions=True,
        )
    finally:
        CFG.training.fix_format_errors = _old_fix
        CFG.training.label_noise_correct_prob = _old_correct
        CFG.training.label_noise_flip_prob = _old_flip
        CFG.training.label_noise_unknown_prob = _old_unknown
    ok_docs: list[dict] = []
    ok_audits: list[dict] = []
    for doc, res in zip(probe_docs, raw):
        if isinstance(res, Exception):
            logger.error("stratum probe | doc failed: %s", res)
            continue
        _datums, audit, _timing, _fmt_ce, _hint_ce = res
        ok_docs.append(doc)
        ok_audits.append(audit)
    per_mean, per_cnt = _aggregate_probe_stratum_stats(ok_docs, ok_audits)
    init_vis = {k: min(vmax, per_cnt[k]) for k in per_cnt}
    n_ro = 0
    n_fmt = 0
    rw_sum = 0.0
    for da in ok_audits:
        for ro in da["rollouts"]:
            n_ro += 1
            if ro.get("format_ok"):
                n_fmt += 1
            pr = _probe_rollout_scalar(ro)
            if pr is not None:
                rw_sum += pr
    rw_vals = list(per_mean.values())
    meta = {
        "probe_docs_ok": float(len(ok_docs)),
        "probe_rollouts": float(n_ro),
        "probe_docs_per_stratum": float(per_stratum),
        "probe_format_rate": (n_fmt / n_ro) if n_ro else 0.0,
        "probe_mean_scalar_reward": (rw_sum / n_ro) if n_ro else 0.0,
        "probe_wall_s": time.perf_counter() - t0,
        "probe_n_strata_touched": float(len(per_mean)),
        "probe_stratum_reward_min": float(min(rw_vals)) if rw_vals else 0.0,
        "probe_stratum_reward_max": float(max(rw_vals)) if rw_vals else 0.0,
    }
    logger.debug(
        "stratum probe | done in %.1fs: strata=%d per-stratum reward min=%.3f max=%.3f format=%.2f mean_rw=%.3f",
        meta["probe_wall_s"],
        len(per_mean),
        meta["probe_stratum_reward_min"],
        meta["probe_stratum_reward_max"],
        meta["probe_format_rate"],
        meta["probe_mean_scalar_reward"],
    )
    return per_mean, init_vis, meta


async def _process_cached_rollouts(doc: dict, cached_rollouts: list[dict], tokenizer, sampling_client, step: int) -> tuple[list[tinker.Datum], dict, dict, list[tinker.Datum], list[tinker.Datum], list[tinker.Datum]]:
    """Build datums + audit from cached rollouts only — no fresh sampling, no scoring,
    no rubric-client calls. IS ratio is computed via a forward pass with the current
    policy to correct for policy drift since the rollout was cached.
    """
    document = doc["text"]
    label = int(doc["label"])

    # Compute IS ratios for all cached rollouts via a single batched forward pass.
    is_ratio_max_cfg = float(getattr(CFG.training, "is_ratio_max", float("inf")))
    async def _compute_is_ratio(c: dict) -> tuple[float, list[float]]:
        """Return (geometric_mean_scalar, per_token_clipped_IS_list).

        Geometric mean is used only for monitoring metrics.
        Per-token IS ratios are passed into _build_per_token_advantages so each
        token's gradient is corrected by its own π_new/π_old, rather than a
        single scalar dominated by the low-variance doc-copy bulk.
        """
        prompt = c["prompt_tokens"]
        full_input = tinker.ModelInput.from_ints(prompt + c["completion_tokens"])
        all_lps: list[float | None] = await sampling_client.compute_logprobs_async(full_input)
        new_lps = [lp if lp is not None else 0.0 for lp in all_lps[len(prompt):]]
        old_lps = c["completion_logprobs"]
        if len(new_lps) != len(old_lps) or not new_lps:
            return 1.0, [1.0] * len(c["completion_tokens"])
        per_tok_is = [
            min(math.exp(max(min(n - o, 20.0), -20.0)), is_ratio_max_cfg)
            for n, o in zip(new_lps, old_lps)
        ]
        log_ratio_mean = sum(n - o for n, o in zip(new_lps, old_lps)) / len(new_lps)
        scalar = math.exp(max(min(log_ratio_mean, 20.0), -20.0))
        return scalar, per_tok_is

    valid_cached = [c for c in cached_rollouts if c.get("completion_tokens")]
    computed_is = await asyncio.gather(*[_compute_is_ratio(c) for c in valid_cached])
    # Map back to original index (includes empty-completion entries that get is=1.0)
    _is_iter = iter(computed_is)
    _n_toks_fallback = [len(c.get("completion_tokens") or []) for c in cached_rollouts]
    cached_is_ratios: list[float] = []        # geometric mean scalar — monitoring only
    cached_is_pertok: list[list[float]] = []  # per-token IS — used in _build_per_token_advantages
    for _idx, c in enumerate(cached_rollouts):
        if c.get("completion_tokens"):
            _scalar, _pertok = next(_is_iter)
        else:
            _scalar, _pertok = 1.0, [1.0] * _n_toks_fallback[_idx]
        cached_is_ratios.append(_scalar)
        cached_is_pertok.append(_pertok)
    datums: list[tinker.Datum] = []
    format_repair_ce_datums: list[tinker.Datum] = []
    rollout_audits: list[dict] = []
    _adv_norm_c = str(getattr(CFG.training, "advantage_normalization", "mean"))
    _struct_token_adv_c = float(getattr(CFG.training, "struct_token_adv", 0.3))
    _adv_std_floor_c = float(_adv_ema_std) * float(getattr(CFG.training, "advantage_std_floor_frac", 0.5))
    _valid_c = [c for c in cached_rollouts if c.get("completion_tokens")]
    _rlcfg_c = getattr(CFG.training, "rl_loss_fn_config", None)
    _clip_low_c = float(getattr(_rlcfg_c, "clip_low_threshold", 0.8)) if _rlcfg_c else 0.8
    _clip_high_c = float(getattr(_rlcfg_c, "clip_high_threshold", 1.28)) if _rlcfg_c else 1.28
    _adv_clip_scale_c = float(getattr(_rlcfg_c, "adv_clip_scale", 10.0)) if _rlcfg_c else 10.0
    _adv_clip_low_c = -(1.0 - _clip_low_c) * _adv_clip_scale_c
    _adv_clip_high_c = (_clip_high_c - 1.0) * _adv_clip_scale_c
    _outer_cred_vals_c = [float(c["reward_components"].get("outer_credibility", 0.0)) for c in _valid_c]
    _outer_cred_advs_c = compute_advantages(_outer_cred_vals_c, std_floor=_adv_std_floor_c, normalize=_adv_norm_c) if _outer_cred_vals_c else []
    _verdict_score_vals_c = [
        float(outer_verdict_score_reward(output=c["response_text"], label=label))
        for c in _valid_c
    ]
    _verdict_score_advs_c = compute_advantages(_verdict_score_vals_c, std_floor=_adv_std_floor_c, normalize=_adv_norm_c) if _verdict_score_vals_c else []
    # Verdict-type token advantage for cached rollouts: format-gated, mirrors fresh-rollout cls_adv_mode.
    _cls_adv_mode_c = getattr(CFG.training, "cls_adv_mode", "score_grpo")
    if _cls_adv_mode_c == "binary":
        # Per-rollout correctness from cached reward_components['cls'] (+1 correct, -1 wrong).
        _cls_raw_vals_c = [
            float((c.get("reward_components") or {}).get("cls", 0.0)) if c.get("format_ok", True) else 0.0
            for c in _valid_c
        ]
    else:
        _cls_raw_vals_c = [
            v if c.get("format_ok", True) else 0.0
            for v, c in zip(_verdict_score_vals_c, _valid_c)
        ]
    # Binary mode: raw ±W directly — mirrors fresh-rollout fix; cls_adv_weight scales the signal.
    _cls_adv_w_c = float(getattr(CFG.training, "cls_adv_weight", 1.0))
    if _cls_adv_mode_c == "binary":
        _cls_advs_sep_c = [v * _cls_adv_w_c for v in _cls_raw_vals_c]
    else:
        _cls_advs_sep_c = compute_advantages(_cls_raw_vals_c, std_floor=_adv_std_floor_c, normalize=_adv_norm_c) if _cls_raw_vals_c else []
    _outer_cred_w_c = float(getattr(CFG.training, "outer_credibility_adv_weight", 0.5))
    # Per-annotation PTAD pools for cached rollouts.
    _ctype_by_c: list[list[float | None]] = []
    _ctype_ok_by_c: list[list[bool | None]] = []
    _cwhy_by_c: list[list[float | None]] = []
    _cscore_by_c: list[list[float | None]] = []
    for c in _valid_c:
        tr_row: list[float | None] = []
        tok_row: list[bool | None] = []
        wr_row: list[float | None] = []
        sr_row: list[float | None] = []
        for ind in (c.get("indicators") or []):
            rc = ind.get("rubric_credibility")
            if rc is not None:
                ann_type = ind.get("type") or ""
                model_sc_c = float(ind.get("model_score", 0.0) or 0.0)
                _tok = (ann_type == "AI") == (label == 1)
                tr = annotation_type_reward(annotation_type=ann_type, label=label)
                wr = annotation_why_reward(float(rc), why_text=ind.get("explanation", ""))
                tr_row.append(tr)
                tok_row.append(_tok)
                wr_row.append(wr)
                sr_row.append(None)
            else:
                tr_row.append(None)
                tok_row.append(None)
                wr_row.append(None)
                sr_row.append(None)
        _ctype_by_c.append(tr_row)
        _ctype_ok_by_c.append(tok_row)
        _cwhy_by_c.append(wr_row)
        _cscore_by_c.append(sr_row)

    def _norm_c_per_rollout(rows: list[list[float | None]]) -> list[list[float] | None]:
        out: list[list[float] | None] = []
        for row in rows:
            vals = [float(v) for v in row if v is not None]
            if len(vals) < 2:
                out.append(None)
                continue
            advs = compute_advantages(vals, std_floor=_adv_std_floor_c, normalize=_adv_norm_c)
            it = iter(advs)
            out.append([next(it) if v is not None else 0.0 for v in row])
        return out

    _per_tell_type_c = _norm_pool_ann_type_advs(
        type_rows=_ctype_by_c,
        type_correct_rows=_ctype_ok_by_c,
        adv_std_floor=_adv_std_floor_c,
        adv_norm=_adv_norm_c,
        adv_clip_low=_adv_clip_low_c,
        adv_clip_high=_adv_clip_high_c,
    )
    _per_tell_why_c = _norm_c_per_rollout(_cwhy_by_c)
    _per_tell_score_c = _norm_c_per_rollout(_cscore_by_c)
    cadv_idx = 0
    for i, c in enumerate(cached_rollouts):
        if not c.get("completion_tokens"):
            continue
        n_reason = int(c.get("n_reasoning_tokens", 0))
        _inds = c.get("indicators") or []
        # IS correction is now applied per-token inside _build_per_token_advantages.
        # Scalar advantages are NOT pre-multiplied by IS here; the per-token IS list
        # handles each token's drift independently (annotation tokens drift more than
        # doc-copy bulk, so the geometric mean scalar would under-correct them).
        _c_cls_adv = _cls_advs_sep_c[cadv_idx]
        _c_vscore_adv = _verdict_score_advs_c[cadv_idx] if _verdict_score_advs_c else None
        _c_vwhy_adv = _outer_cred_advs_c[cadv_idx] * _outer_cred_w_c if _outer_cred_advs_c else None
        response_advantages = _build_per_token_advantages(
            completion_tokens=c["completion_tokens"],
            n_reasoning_tokens=n_reason,
            cls_adv=_c_cls_adv,
            tell_scored=_inds,
            tokenizer=tokenizer,
            label_ctx_for_opt=bool(c.get("_label_ctx_for_opt", False)),
            per_tell_type_advs=_per_tell_type_c[cadv_idx] if _per_tell_type_c else None,
            per_tell_why_advs=_per_tell_why_c[cadv_idx] if _per_tell_why_c else None,
            per_tell_score_advs=_per_tell_score_c[cadv_idx] if _per_tell_score_c else None,
            verdict_score_adv=_c_vscore_adv,
            verdict_why_adv=_c_vwhy_adv,
            per_span_open_advs=None,
            struct_token_adv=_struct_token_adv_c,
            per_token_is=cached_is_pertok[i],
        )
        response_advantages = _zero_outer_verdict_type_cls_adv_for_label_ctx(
            response_advantages=response_advantages,
            completion_tokens=c["completion_tokens"],
            n_reasoning_tokens=int(n_reason),
            tokenizer=tokenizer,
            label_ctx_for_opt=bool(c.get("_label_ctx_for_opt", False)),
        )
        cadv_idx += 1
        token_surprisal = _token_surprisal_stats(
            c["completion_logprobs"],
            n_reason,
            c.get("token_type_mask"),
        )
        _rtn_id = _get_return_token_id(tokenizer)
        _ctokens_c = list(c["completion_tokens"])
        _clogprobs_c = list(c["completion_logprobs"])
        _radv_c = list(response_advantages)
        if _rtn_id is not None and (not _ctokens_c or _ctokens_c[-1] != _rtn_id):
            _ctokens_c.append(_rtn_id)
            _clogprobs_c.append(0.0)
            _radv_c.append(0.0)
        _span_open_loss_mass_c = float(getattr(CFG.training, "span_open_loss_mass", 0.15))
        _span_ann_mass_c = float(getattr(CFG.training, "span_ann_mass", 1.0))
        _rweights_c = compute_task_loss_weights(
            tokenizer=tokenizer,
            completion_tokens=_ctokens_c,
            n_reasoning_tokens=int(n_reason),
            span_open_loss_mass=_span_open_loss_mass_c,
            span_ann_mass=_span_ann_mass_c,
            ptok_loss_scales=_ptok_loss_scales_from_cfg(),
        )
        datums.append(
            build_datum(
                prompt_tokens=c["prompt_tokens"],
                completion_tokens=_ctokens_c,
                completion_logprobs=_clogprobs_c,
                response_advantages=_radv_c,
                response_task_weights=_rweights_c,
                n_reasoning_tokens=int(n_reason),
            )
        )
        trace = TrainingRolloutTracePayload(
            step=int(step),
            rollout_index=i,
            doc_label=int(label),
            noise_mode=c["noise_mode"],
            inject_label=bool(c["inject_label"]),
            main_label_hint=c["main_label_hint"],
            label_ctx_for_opt=bool(c["_label_ctx_for_opt"]),
            response_text=c["response_text"],
            wrong_response_text=c["wrong_response_text"],
            reward=float(c["reward"]),
            reward_components=c["reward_components"],
            advantage=None,
            used_for_optimization=True,
            exclude_reason="cached",
            format_ok=bool(c["format_ok"]),
            format_ok_before_fixing=bool(c["format_ok_before_fixing"]),
            format_reason=c["format_reason"],
            format_reason_before_fixing=c["format_reason_at_sample"],
            format_char_diff_count=int(c["format_char_diff_count"]),
            is_ratio=cached_is_ratios[i],
            from_replay_cache=True,
            document=document,
            doc_stratum="|".join(str(x) for x in doc_stratum_key(doc)),
            neutral_prompt_text=c["neutral_prompt_text"],
            completion_text=c["completion_text"],
            full_output_text=c["full_output_text"],
            raw_response_text=c["wrong_response_text"] or c["response_text"],
            was_text_fixed=bool(c["was_text_fixed"]),
            token_surprisal=token_surprisal,
            ann_token_fraction=token_surprisal["n_ann_tokens"] / token_surprisal["n_response_tokens"] if token_surprisal["n_response_tokens"] > 0 else 0.0,
            n_ann_tokens=float(token_surprisal["n_ann_tokens"]),
            n_response_tokens=float(token_surprisal["n_response_tokens"]),
            indicators=c["indicators"],
            token_optimization_rows=_token_optimization_rows(
                tokenizer=tokenizer,
                prompt_tokens=c["prompt_tokens"],
                completion_tokens=c["completion_tokens"],
                completion_logprobs=c["completion_logprobs"],
                n_reasoning_tokens=n_reason,
                response_advantages=response_advantages,
            ),
            completion_tokens_len=len(c["completion_tokens"]),
            completion_logprobs_len=len(c["completion_logprobs"]),
            n_reasoning_tokens=n_reason,
            budget_hit=len(c["completion_tokens"]) >= int(CFG.sampling.max_tokens),
            repetition_score=_repetition_score(c["completion_text"]),
            why_count=len(c["indicators"]),
            why_mean_len=(sum(len(ind["explanation"]) for ind in c["indicators"]) / len(c["indicators"])) if c["indicators"] else 0.0,
            why_max_len=max((len(ind["explanation"]) for ind in c["indicators"]), default=0),
            why_repetition_score=_repetition_score(" ".join(ind["explanation"] for ind in c["indicators"])),
            rubric=c.get("rubric_trace"),
            component_advantages=None,
        )
        _trace_training_rollout(trace=trace)
        rollout_audits.append(training_trace_payload_to_audit_dict(trace))
    rewards = [float(c["reward"]) for c in cached_rollouts]
    reward_mean = sum(rewards) / len(rewards) if rewards else 0.0
    n = len(cached_rollouts)
    doc_audit = {
        "ease_uid": doc_ease_uid(doc),
        "stratum_key": "|".join(str(x) for x in doc_stratum_key(doc)),
        "document": document,
        "label": label,
        "noise_modes": [None] * n,
        "main_label_hints": [None] * n,
        "inject_label_flags": [False] * n,
        "reward_mean": reward_mean,
        "curriculum_mean": reward_mean,
        "format_rate": 1.0,
        "format_rate_before_fixing": 1.0,
        "n_excluded_rollouts": 0,
        "is_ratios": cached_is_ratios,
        "is_ratio_mean": sum(cached_is_ratios) / len(cached_is_ratios) if cached_is_ratios else 1.0,
        "is_ratio_min": min(cached_is_ratios) if cached_is_ratios else 1.0,
        "is_ratio_max": max(cached_is_ratios) if cached_is_ratios else 1.0,
        "rollouts": rollout_audits,
        "from_replay_cache": True,
        # Don't re-cache replays: keeps the cache as snapshots from FRESH sampling steps,
        # avoids advantage staleness compounding across re-uses.
        "cacheable_rollouts": [],
    }
    doc_timing = {"dt_rollouts": 0.0, "dt_scoring": 0.0, "dt_scoring_mean": 0.0}
    return datums, doc_audit, doc_timing, format_repair_ce_datums, []


def _zero_train_metrics() -> dict:
    """Zero-valued metrics dict returned when a step produces no usable datums."""
    return {
        "train/reward_mean": 0.0, "train/format_rate": 0.0, "train/format_rate_before_fixing": 0.0,
        "train/n_positive_rollouts": 0, "train/n_negative_rollouts": 0, "train/n_zero_rollouts": 0,
        "train/pct_zero_reward": 0.0, "train/n_zero_format_fail": 0, "train/n_zero_anomaly_format_ok": 0,
        "train/n_excluded_rollouts": 0, "train/n_cached_docs": 0, "train/n_fresh_docs": 0,
        "train/effective_lr": 0.0, "train/reward_ema": 0.0, "train/lr_scale": 0.0, "train/effective_k": 0,
        "train/rubric_parse_fail_rate": 0.0, "train/rubric_zero_ann_rate": 0.0,
        "train_diag/completion_tokens_p95": 0, "train_diag/completion_tokens_max": 0,
        "train_reward/credibility_std": 0.0, "train_reward/outer_credibility_std": 0.0,
        "why_quality/step_unique_rate": 1.0, "why_quality/step_unique_rate_mean_per_doc": 1.0,
        "why_quality/step_p95_len": 0,
        "aux_ce/fmt_repair_ce_n": 0, "aux_ce/fmt_repair_ce_loss": 0.0, "aux_ce/hint_ce_loss": 0.0,
        "_train_rubric_ann_scores": [],
        "_train_is_ratios": [], "stratum_mean_rewards": {}, "stratum_format_rates": {},
        "curriculum_reward_rows": [], "replay_rows": [],
    }


async def train_step(
    training_client,
    tokenizer,
    rubric_client,
    docs: list[dict],
    step: int,
    audit_log,
    kl_ref_client: tinker.SamplingClient | None = None,
    cached_per_doc: list[list[dict] | None] | None = None,
) -> dict:
    """One GRPO update for a batch of docs. Returns aggregate metrics.

    cached_per_doc: parallel to `docs`. Where non-empty, the doc's slot is filled by
    replaying the cached rollouts (no fresh sampling / scoring). PPO clip handles drift.
    """
    global _adv_ema_std, _reward_ema
    t0_step_wall = time.perf_counter()

    # Adaptive K: linear decay from k_init to k_final over k_decay_steps.
    k_init = int(getattr(CFG.training, "k_init"))
    k_final = int(getattr(CFG.training, "k_final"))
    k_decay_steps = int(getattr(CFG.training, "k_decay_steps", CFG.training.max_steps))
    progress = min(1.0, step / max(1, k_decay_steps))
    effective_k = max(k_final, round(k_init + (k_final - k_init) * progress))

    # Adaptive advantage normalization: use running EMA std as a floor on the denominator.
    adv_std_floor_frac = float(getattr(CFG.training, "advantage_std_floor_frac", 0.0))
    adv_std_floor = _adv_ema_std * adv_std_floor_frac

    t0_save = time.perf_counter()
    sampling_client = await training_client.save_weights_and_get_sampling_client_async()
    dt_save = time.perf_counter() - t0_save
    logger.info("step %d | sampling client ready in %.1fs", step, dt_save)

    if cached_per_doc is None:
        cached_per_doc = [None] * len(docs)

    fresh_idx = [i for i, c in enumerate(cached_per_doc) if not c]
    cached_idx = [i for i, c in enumerate(cached_per_doc) if c]
    logger.info(
        "step %d | processing %d docs (fresh=%d cached=%d, effective_k=%d adv_std_floor=%.4f)",
        step, len(docs), len(fresh_idx), len(cached_idx), effective_k, adv_std_floor,
    )

    fresh_results_coros = [
        _process_doc(
            sampling_client, tokenizer, rubric_client, docs[i],
            rng=random.Random(GLOBAL_SEED + step * 1000 + i),
            rollout_seed=GLOBAL_SEED + step * 10_000 + i * 100,
            k=effective_k,
            adv_std_floor=adv_std_floor,
            step=step,
        )
        for i in fresh_idx
    ]
    t0_doc_gather = time.perf_counter()
    fresh_raw = await asyncio.gather(*fresh_results_coros, return_exceptions=True) if fresh_results_coros else []
    dt_doc_gather = time.perf_counter() - t0_doc_gather

    docs_used = []
    doc_results = []
    docs_used_from_cache: list[bool] = []
    fresh_lookup = dict(zip(fresh_idx, fresh_raw))
    for i, doc in enumerate(docs):
        if cached_per_doc[i]:
            res = await _process_cached_rollouts(
                doc=doc,
                cached_rollouts=cached_per_doc[i],
                tokenizer=tokenizer,
                sampling_client=sampling_client,
                step=step,
            )
            docs_used.append(doc)
            doc_results.append(res)
            docs_used_from_cache.append(True)
            continue
        res = fresh_lookup[i]
        if isinstance(res, Exception):
            logger.error("doc processing crashed; skipping doc for this step", exc_info=(type(res), res, res.__traceback__))
            fail_path = pathlib.Path(getattr(CFG.training, "format_fail_audit_path", "format_fail_audit.jsonl"))
            fail_path.parent.mkdir(parents=True, exist_ok=True)
            with open(fail_path, "a") as f:
                f.write(json.dumps({
                    "doc_id": doc.get("id", None),
                    "doc_label": doc.get("label"),
                    "format_reason": "process_doc_exception",
                    "input_text": doc.get("text", ""),
                    "exception_type": type(res).__name__,
                    "exception": str(res),
                }, ensure_ascii=False) + "\n")
            continue
        docs_used.append(doc)
        doc_results.append(res)
        docs_used_from_cache.append(False)

    # Top-p group variance filtering: drop groups where reward variance is low,
    # since the model is already consistent there and contributes little gradient.
    top_p = getattr(CFG.training, "group_variance_top_p", 1.0)
    if top_p < 1.0 and doc_results:
        def _group_variance(doc_result):
            _, doc_audit, _dt, _fmt_ce, _hint_ce = doc_result
            rewards = [ro["reward"] for ro in doc_audit["rollouts"] if ro.get("used_for_optimization") and ro["reward"] is not None]
            if len(rewards) < 2:
                return 0.0
            mean = sum(rewards) / len(rewards)
            return sum((r - mean) ** 2 for r in rewards) / len(rewards)

        variances = [_group_variance(dr) for dr in doc_results]
        n_keep = max(1, round(top_p * len(doc_results)))
        sorted_indices = sorted(range(len(doc_results)), key=lambda i: variances[i], reverse=True)
        keep_set = set(sorted_indices[:n_keep])
        n_dropped = len(doc_results) - len(keep_set)
        if n_dropped > 0:
            logger.info(
                "step %d | group_variance_top_p=%.2f: dropping %d/%d low-variance docs (variances: kept min=%.4f, dropped max=%.4f)",
                step, top_p, n_dropped, len(doc_results),
                min(variances[i] for i in keep_set),
                max((variances[i] for i in range(len(doc_results)) if i not in keep_set), default=0.0),
            )
        doc_results = [dr for i, dr in enumerate(doc_results) if i in keep_set]

    all_datums = []
    format_repair_ce_datums_all: list[tinker.Datum] = []
    hint_ce_datums_all: list[tinker.Datum] = []
    docs_audit = []
    doc_timings = []
    for datums, doc_audit, doc_timing, fmt_ce_datums, hce_datums in doc_results:
        all_datums.extend(datums)
        format_repair_ce_datums_all.extend(fmt_ce_datums)
        hint_ce_datums_all.extend(hce_datums)
        docs_audit.append(doc_audit)
        doc_timings.append(doc_timing)

    if not all_datums:
        logger.warning("step %d | no valid datums, skipping update", step)
        return _zero_train_metrics()

    completion_lens = [ro["completion_tokens_len"] for da in docs_audit for ro in da["rollouts"]]
    completion_total = sum(completion_lens)
    completion_mean = (completion_total / len(completion_lens)) if completion_lens else 0.0
    completion_max = max(completion_lens) if completion_lens else 0
    completion_p95 = int(_quantile(completion_lens, 0.95))
    budget_hit_rate = (sum(1 for l in completion_lens if l >= CFG.sampling.max_tokens) / len(completion_lens)) if completion_lens else 0.0
    logger.info(
        "step %d | datum stats: n=%d completion_total=%d completion_mean=%.1f completion_p95=%d completion_max=%d budget_hit_rate=%.3f",
        step,
        len(all_datums),
        completion_total,
        completion_mean,
        completion_p95,
        completion_max,
        budget_hit_rate,
    )

    # Step-level why annotation quality metrics (collapse / diversity early-warning).
    # why_step_unique_rate: unique explanation strings / total across the whole step.
    #   → 1.0 = perfectly diverse, approaching 0 = every rollout says the same thing.
    # why_intra_collapse_rate_mean: mean per-doc fraction of rollout pairs with identical explanation sets.
    #   → 0.0 = all rollouts for a doc differ, 1.0 = all rollouts copy each other.
    _step_all_why: list[str] = []
    _step_all_why_lens: list[int] = []
    _step_why_unique_rates: list[float] = []
    _step_why_collapse_rates: list[float] = []
    _step_why_rep_scores: list[float] = []
    for da in docs_audit:
        _step_why_unique_rates.append(float(da.get("why_unique_rate", 1.0)))
        _step_why_collapse_rates.append(float(da.get("why_intra_collapse_rate", 0.0)))
        _step_why_rep_scores.append(float(da.get("why_repetition_score_mean", 0.0)))
        for ro in da.get("rollouts", []):
            for ind in ro.get("indicators", []):
                w = ind.get("explanation", "")
                if w:
                    _step_all_why.append(w)
                    _step_all_why_lens.append(len(w))
    _why_step_n = max(len(_step_all_why), 1)
    why_step_unique_rate = len(set(_step_all_why)) / _why_step_n
    why_step_mean_len = sum(_step_all_why_lens) / max(len(_step_all_why_lens), 1)
    why_step_p95_len = int(_quantile(_step_all_why_lens, 0.95)) if _step_all_why_lens else 0
    why_intra_collapse_rate_mean = sum(_step_why_collapse_rates) / max(len(_step_why_collapse_rates), 1)
    why_step_unique_rate_mean = sum(_step_why_unique_rates) / max(len(_step_why_unique_rates), 1)
    why_step_repetition_score_mean = sum(_step_why_rep_scores) / max(len(_step_why_rep_scores), 1)

    _token_surprisal_by_rollout = [
        ro.get("token_surprisal", {})
        for da in docs_audit
        for ro in da.get("rollouts", [])
        if ro.get("token_surprisal")
    ]
    def _ts_mean(key: str) -> float:
        vals = [float(s[key]) for s in _token_surprisal_by_rollout if key in s]
        return sum(vals) / len(vals) if vals else 0.0
    token_surprisal_mean = _ts_mean("mean")
    token_surprisal_p10 = _ts_mean("p10")
    token_surprisal_p50 = _ts_mean("p50")
    token_surprisal_p90 = _ts_mean("p90")
    token_surprisal_ann_mean = _ts_mean("ann_mean")
    token_surprisal_doc_mean = _ts_mean("doc_mean")
    token_surprisal_ann_minus_doc = token_surprisal_ann_mean - token_surprisal_doc_mean
    token_surprisal_n_ann = sum(float(s.get("n_ann_tokens", 0.0)) for s in _token_surprisal_by_rollout)
    token_surprisal_n_doc = sum(float(s.get("n_doc_tokens", 0.0)) for s in _token_surprisal_by_rollout)
    token_surprisal_ann_frac = (
        token_surprisal_n_ann / (token_surprisal_n_ann + token_surprisal_n_doc)
        if (token_surprisal_n_ann + token_surprisal_n_doc) > 0.0
        else 0.0
    )

    train_kl_policy_base = 0.0
    dt_kl_compute = 0.0
    _kl_coef = float(getattr(CFG.training, "kl_penalty_coef", 0.0))
    _kl_disc = float(getattr(CFG.training, "kl_discount_factor", 0.0))
    if _kl_coef != 0.0 and kl_ref_client is not None:
        from tinker_cookbook.rl.metrics import incorporate_kl_penalty as _incorporate_kl_penalty

        _t0_kl = time.perf_counter()
        _klm = await _incorporate_kl_penalty(all_datums, kl_ref_client, _kl_coef, _kl_disc)
        dt_kl_compute = time.perf_counter() - _t0_kl
        train_kl_policy_base = float(_klm.get("kl_policy_base", 0.0))

    _rl_fn = "ppo"
    _loss_cfg = _loss_fn_config_for_api(_rl_fn)

    logger.info(
        "step %d | forward/backward on %d datums loss_fn=%s kl_base=%.5f loss_fn_config=%r",
        step,
        len(all_datums),
        _rl_fn,
        train_kl_policy_base,
        _loss_cfg,
    )
    # KL code above needs mask; tinker_cookbook.rl.train.forward_backward drops mask before API call, else 400 array record
    _fb_datums = (
        [
            tinker.Datum(
                model_input=_d.model_input,
                loss_fn_inputs={k: v for k, v in _d.loss_fn_inputs.items() if k != "mask"},
            )
            for _d in all_datums
        ]
        if _rl_fn in ("ppo", "importance_sampling", "cispo", "dro")
        else all_datums
    )
    _base_lr = float(CFG.training.learning_rate)
    _alr_low = float(getattr(CFG.training, "adaptive_lr_reward_low", 1.1))
    _alr_high = float(getattr(CFG.training, "adaptive_lr_reward_high", 1.1))
    _alr_min = float(getattr(CFG.training, "adaptive_lr_min_scale", 0.1))
    if _alr_low < _alr_high and _reward_ema > _alr_low:
        _t = min(1.0, (_reward_ema - _alr_low) / (_alr_high - _alr_low))
        _lr_scale = 1.0 - (1.0 - _alr_min) * _t
    else:
        _lr_scale = 1.0
    _warmup_steps = int(getattr(CFG.training, "lr_warmup_steps", 0))
    if _warmup_steps > 0 and step < _warmup_steps:
        _lr_scale *= step / _warmup_steps
    learning_rate = _base_lr * _lr_scale
    _adam = tinker.AdamParams(learning_rate=learning_rate)

    # Multi-epoch PPO: repeat fwd/bwd+optim N times on the same rollout batch.
    # PPO IS-ratio clip (0.8, 1.28) naturally limits overstepping as policy drifts.
    # n_gradient_steps_per_batch=1 preserves original single-step behavior.
    _n_grad_steps = max(1, int(getattr(CFG.training, "n_gradient_steps_per_batch", 1)))

    async def _fwd_bwd():
        future = await training_client.forward_backward_async(
            data=_fb_datums, loss_fn=_rl_fn, loss_fn_config=_loss_cfg
        )
        return await future.result_async()

    async def _optim():
        future = await training_client.optim_step_async(_adam)
        return await future.result_async()

    fb_t0 = time.perf_counter()
    fb_result = None
    _opt_result = None
    for _grad_i in range(_n_grad_steps):
        _suffix = f" ({_grad_i+1}/{_n_grad_steps})" if _n_grad_steps > 1 else ""
        fb_result = await _await_with_heartbeat(
            _fwd_bwd(),
            step,
            f"forward/backward result ({len(all_datums)} datums){_suffix}",
        )
        _opt_result = await _await_with_heartbeat(_optim(), step, f"optimizer result{_suffix}")
    fb_dt = time.perf_counter() - fb_t0
    if hasattr(fb_result, "loss"):
        logger.info("step %d | forward/backward done in %.1fs loss=%s (%d grad steps)", step, fb_dt, fb_result.loss, _n_grad_steps)
    else:
        logger.info("step %d | forward/backward done in %.1fs (%d grad steps)", step, fb_dt, _n_grad_steps)

    opt_dt = fb_dt  # folded into fb_dt above since they're interleaved
    _opt_grad_norm = ((_opt_result.metrics or {}).get("grad_norm") if _opt_result is not None else None)
    logger.info("step %d | optimizer done in %.1fs grad_norm=%s", step, opt_dt, f"{_opt_grad_norm:.4f}" if _opt_grad_norm is not None else "n/a")

    # Unified CE pass: collect all enabled CE pools, shuffle together, one forward/backward loop,
    # one optimizer step. This prevents multiple sequential weight updates (one per CE source)
    # that would make CE collectively stronger than PPO within a single training step.
    _ce_lr_scale = float(getattr(CFG.training, "sft_ce_lr_scale", 1.0))
    _hint_ce_enabled = bool(getattr(CFG.training, "hint_outer_ce_enabled", False))
    _ce_bs = int(getattr(CFG.training, "sft_ce_batch_size", 8))
    _ce_lr = float(CFG.training.learning_rate) * _ce_lr_scale

    # CE pool: format-repair (fixed rollouts only) + hint (wrong-label rollouts only).
    # Both are ad hoc — if format is perfect and hints are always followed, pool is empty
    # and no CE forward/backward or optimizer step runs.
    _ce_pool: list = []
    _fmt_repair_ce_n = 0
    _hint_ce_n = 0
    if format_repair_ce_datums_all:
        _fmt_repair_ce_n = len(format_repair_ce_datums_all)
        _ce_pool.extend(format_repair_ce_datums_all)
    if _hint_ce_enabled and hint_ce_datums_all:
        _hint_ce_n = len(hint_ce_datums_all)
        _ce_pool.extend(hint_ce_datums_all)

    fmt_repair_ce_loss = 0.0
    hint_ce_loss = 0.0
    dt_ce_pass = 0.0
    if _ce_pool:
        from rl_detector.sft.train_tinker_sft import _scalar_loss as _sft_scalar_loss
        random.Random(GLOBAL_SEED + step * 3571 + 13).shuffle(_ce_pool)
        logger.info(
            "step %d | format-repair-ce n=%d (fmt_repair=%d hint=%d) lr=%.2e",
            step, len(_ce_pool), _fmt_repair_ce_n, _hint_ce_n, _ce_lr,
        )
        _t0_ce = time.perf_counter()
        _ce_scalar_bad = False
        _ce_fb_future = await training_client.forward_backward_async(data=_ce_pool, loss_fn="cross_entropy")
        _ce_fb_out = await _ce_fb_future.result_async()
        try:
            _ce_loss_val = float(_sft_scalar_loss(_ce_pool, _ce_fb_out))
        except Exception:
            _ce_scalar_bad = True
            _ce_loss_val = 0.0
        _ce_opt_future = await training_client.optim_step_async(tinker.AdamParams(learning_rate=_ce_lr))
        await _ce_opt_future.result_async()
        dt_ce_pass = time.perf_counter() - _t0_ce
        _ce_loss = float("nan") if _ce_scalar_bad else _ce_loss_val
        fmt_repair_ce_loss = _ce_loss if _fmt_repair_ce_n > 0 else 0.0
        hint_ce_loss = _ce_loss if _hint_ce_n > 0 else 0.0

    _opt_rollouts = [
        ro
        for da in docs_audit
        for ro in da["rollouts"]
        if ro.get("used_for_optimization") and ro["reward"] is not None
    ]
    all_rewards = [ro["reward"] for ro in _opt_rollouts]
    reward_mean = (sum(all_rewards) / len(all_rewards)) if all_rewards else 0.0

    # Reward component means across optimized rollouts (format-failed rollouts have all-zero components).
    _opt_components = [ro["reward_components"] for da in docs_audit for ro in da["rollouts"] if ro.get("used_for_optimization") and ro.get("reward_components")]
    def _comp_mean(key: str) -> float:
        vals = [c[key] for c in _opt_components if key in c]
        return sum(vals) / len(vals) if vals else 0.0

    def _comp_std(key: str) -> float:
        vals = [c[key] for c in _opt_components if key in c]
        if len(vals) < 2:
            return 0.0
        m = sum(vals) / len(vals)
        return (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5

    def _cadv_mean(key: str) -> float | None:
        """Mean of per-rollout component advantages (ptad mode only)."""
        vals = [
            ro["component_advantages"][key]
            for da in docs_audit for ro in da["rollouts"]
            if ro.get("component_advantages") and ro["component_advantages"].get(key) is not None
        ]
        return sum(vals) / len(vals) if vals else None

    # Per-token-type mean advantage: mean over all optimized tokens of that type this step.
    # Logged as train_ptok_adv_mean/* (signed) and train_ptok_adv_abs/* (magnitude); not autograd gradients.
    # train_ptok_grad/* = mean |adv * ptok_loss_scale| — effective PPO loss contribution per token.
    # train_ptok_grad_total/* = sum |adv * ptok_loss_scale| — total gradient mass per type this step.
    # These reveal which components dominate the parameter update (e.g. verdict_type at 50x scale
    # vs verdict_why at 1x; ann_type at 25x vs ann_why at 1x).
    _ptok_scales = _ptok_loss_scales_from_cfg()
    _type_adv_buckets: dict[str, list[float]] = {}
    _type_grad_buckets: dict[str, list[float]] = {}
    for da in docs_audit:
        for ro in da["rollouts"]:
            for row in ro.get("token_optimization_rows", []):
                if row.get("optimized") and row.get("token_type") not in (None, "prompt", "reasoning", "doc_copy"):
                    _tt = row["token_type"]
                    _adv = float(row["advantage"])
                    _type_adv_buckets.setdefault(_tt, []).append(_adv)
                    _scale = _ptok_scales.get(_tt, 1.0)
                    _type_grad_buckets.setdefault(_tt, []).append(abs(_adv * _scale))
    def _type_adv_mean(typ: str) -> float | None:
        vals = _type_adv_buckets.get(typ)
        return sum(vals) / len(vals) if vals else None

    def _type_adv_abs_mean(typ: str) -> float | None:
        vals = _type_adv_buckets.get(typ)
        return sum(abs(v) for v in vals) / len(vals) if vals else None

    def _type_adv_count(typ: str) -> int:
        return len(_type_adv_buckets.get(typ) or [])

    def _type_grad_mean(typ: str) -> float | None:
        vals = _type_grad_buckets.get(typ)
        return sum(vals) / len(vals) if vals else None

    def _type_grad_total(typ: str) -> float:
        return sum(_type_grad_buckets.get(typ) or [])

    # Update running EMAs of reward mean and std for next step's adaptive LR and advantage floor.
    if all_rewards:
        _step_reward_mean = reward_mean
        _step_reward_std = (sum((r - _step_reward_mean) ** 2 for r in all_rewards) / len(all_rewards)) ** 0.5
        _ema_alpha = float(getattr(CFG.training, "advantage_ema_alpha", 0.05))
        _adv_ema_std = _ema_alpha * _step_reward_std + (1.0 - _ema_alpha) * _adv_ema_std
        _reward_ema_alpha = float(getattr(CFG.training, "adaptive_lr_ema_alpha", _ema_alpha))
        _reward_ema = _reward_ema_alpha * _step_reward_mean + (1.0 - _reward_ema_alpha) * _reward_ema

    all_format_ok = [ro.get("format_ok", False) for da in docs_audit for ro in da["rollouts"]]
    format_rate = (sum(1 for ok in all_format_ok if ok) / len(all_format_ok)) if all_format_ok else 0.0
    all_format_ok_before_fixing = [
        ro.get("format_ok_before_fixing", ro.get("format_ok", False))
        for da in docs_audit
        for ro in da["rollouts"]
    ]
    format_rate_before_fixing = (
        sum(1 for ok in all_format_ok_before_fixing if ok) / len(all_format_ok_before_fixing)
        if all_format_ok_before_fixing else 0.0
    )
    n_excluded_rollouts = sum(1 for da in docs_audit for ro in da["rollouts"] if not ro.get("used_for_optimization"))
    _injected_opt_rollouts = [
        ro
        for da in docs_audit
        for ro in da["rollouts"]
        if ro.get("used_for_optimization") and bool(ro.get("inject_label", False))
    ]
    train_label_ctx_for_opt_rate = (
        sum(1 for ro in _injected_opt_rollouts if bool(ro.get("label_ctx_for_opt", False))) / len(_injected_opt_rollouts)
        if _injected_opt_rollouts else 0.0
    )

    # hint-follow rates: among injected+format_ok rollouts only (format failures have undefined outer type / zeroed agg)
    _injected_fmt_ok_rollouts = [
        ro
        for da in docs_audit
        for ro in da["rollouts"]
        if bool(ro.get("inject_label", False)) and ro.get("format_ok", False) and ro.get("main_label_hint") is not None
    ]
    _outer_follow_vals = []
    _outer_follow_correct_vals: list[float] = []
    _outer_follow_flip_vals: list[float] = []
    _agg_follow_vals = []
    for ro in _injected_fmt_ok_rollouts:
        hint = ro["main_label_hint"]
        outer_type_ai = float(ro.get("reward_components", {}).get("outer_type_ai", -1.0))
        if outer_type_ai >= 0.0:
            followed = 1.0 if (outer_type_ai >= 0.5) == (hint == 1) else 0.0
            _outer_follow_vals.append(followed)
            nm = ro.get("noise_mode", "unknown")
            if nm == "correct":
                _outer_follow_correct_vals.append(followed)
            elif nm == "flip":
                _outer_follow_flip_vals.append(followed)
        agg = float(ro.get("reward_components", {}).get("agg_score", 0.0))
        _agg_follow_vals.append(1.0 if (agg >= 0.0) == (hint == 1) else 0.0)
    label_hint_outer_follow_rate = sum(_outer_follow_vals) / len(_outer_follow_vals) if _outer_follow_vals else 0.0
    label_hint_outer_follow_rate_correct = sum(_outer_follow_correct_vals) / len(_outer_follow_correct_vals) if _outer_follow_correct_vals else 0.0
    label_hint_outer_follow_rate_flip = sum(_outer_follow_flip_vals) / len(_outer_follow_flip_vals) if _outer_follow_flip_vals else 0.0
    label_hint_agg_follow_rate = sum(_agg_follow_vals) / len(_agg_follow_vals) if _agg_follow_vals else 0.0

    # Binary F1 (positive=AI) from verdict type labels, format-ok no-hint rollouts only.
    # Excludes label_ctx_for_opt rollouts (hint injected → model trivially writes correct type,
    # inflating F1 to ~0.9 regardless of learning progress).
    _bf1_tp = _bf1_fp = _bf1_fn = 0
    for da in docs_audit:
        true_ai = da["label"] == 1
        for ro in da["rollouts"]:
            if not ro.get("format_ok", False):
                continue
            if ro.get("_label_ctx_for_opt", False):
                continue
            ota = float(ro.get("reward_components", {}).get("outer_type_ai", -1.0))
            if ota < 0.0:
                continue
            pred_ai = ota >= 0.5
            if pred_ai and true_ai:
                _bf1_tp += 1
            elif pred_ai and not true_ai:
                _bf1_fp += 1
            elif not pred_ai and true_ai:
                _bf1_fn += 1
    _bf1_prec = _bf1_tp / (_bf1_tp + _bf1_fp) if (_bf1_tp + _bf1_fp) > 0 else 0.0
    _bf1_rec = _bf1_tp / (_bf1_tp + _bf1_fn) if (_bf1_tp + _bf1_fn) > 0 else 0.0
    train_binary_f1 = (2 * _bf1_prec * _bf1_rec / (_bf1_prec + _bf1_rec)) if (_bf1_prec + _bf1_rec) > 0 else 0.0

    all_is_ratios = [r for da in docs_audit for r in da.get("is_ratios", [])]
    step_is_ratio_mean = sum(all_is_ratios) / len(all_is_ratios) if all_is_ratios else 1.0
    step_is_ratio_min = min(all_is_ratios) if all_is_ratios else 1.0
    step_is_ratio_max = max(all_is_ratios) if all_is_ratios else 1.0
    step_is_ratio_p10 = _quantile(all_is_ratios, 0.10)
    step_is_ratio_p90 = _quantile(all_is_ratios, 0.90)

    all_rubric_creds = [
        float(ind["rubric_credibility"])
        for da in docs_audit
        for ro in da["rollouts"]
        for ind in ro.get("indicators", [])
        if ind.get("rubric_credibility") is not None
    ]
    rubric_cred_mean = (sum(all_rubric_creds) / len(all_rubric_creds)) if all_rubric_creds else 0.0
    rubric_cred_std = (
        (sum((s - rubric_cred_mean) ** 2 for s in all_rubric_creds) / len(all_rubric_creds)) ** 0.5
        if all_rubric_creds else 0.0
    )
    # per-label reward breakdown: AI docs (label=1) vs human docs (label=0)
    ai_rewards = [ro["reward"] for da in docs_audit if da["label"] == 1 for ro in da["rollouts"] if ro.get("used_for_optimization") and ro["reward"] is not None]
    human_rewards = [ro["reward"] for da in docs_audit if da["label"] == 0 for ro in da["rollouts"] if ro.get("used_for_optimization") and ro["reward"] is not None]
    ai_reward_mean = (sum(ai_rewards) / len(ai_rewards)) if ai_rewards else 0.0
    human_reward_mean = (sum(human_rewards) / len(human_rewards)) if human_rewards else 0.0

    all_exclude_reasons = [ro.get("exclude_reason") for da in docs_audit for ro in da["rollouts"]]
    train_format_invalid_type = sum(1 for r in all_exclude_reasons if r == "format:invalid_type")
    train_format_text_mismatch = sum(1 for r in all_exclude_reasons if r == "format:text_mismatch")
    train_scorer_exception_rate = (
        sum(1 for r in all_exclude_reasons if r == "scorer_exception") / max(1, len(all_exclude_reasons))
    )
    train_text_mismatch_char_diffs = [ro.get("format_char_diff_count", 0) for da in docs_audit for ro in da["rollouts"] if ro.get("exclude_reason") == "format:text_mismatch"]
    train_all_format_char_diffs = [ro.get("format_char_diff_count", 0) for da in docs_audit for ro in da["rollouts"]]

    train_rubric_ann_scores = list(all_rubric_creds)

    # tell coverage and count stats across valid (format_ok) rollouts
    _valid_ro_docs = [
        (da["document"], ro)
        for da in docs_audit
        for ro in da["rollouts"]
        if ro.get("format_ok") and ro.get("indicators")
    ]
    if _valid_ro_docs:
        _per_ro_n_tells: list[float] = []
        _per_ro_tells_per_100w: list[float] = []
        _per_ro_coverage: list[float] = []
        _per_tell_coverages: list[float] = []
        for _doc, _ro in _valid_ro_docs:
            _doc_len = max(1, len(_doc))
            _doc_words = max(1, len(_doc.split()))
            _inds = _ro["indicators"]
            _per_ro_n_tells.append(len(_inds))
            _per_ro_tells_per_100w.append(len(_inds) / _doc_words * 100)
            _covered: set[int] = set()
            for _ind in _inds:
                _span = _ind.get("span_text", "")
                if _span.strip() == _doc.strip() or "".join(_span.split()) == "".join(_doc.split()):
                    continue
                _idx = _doc.find(_span)
                if _idx >= 0:
                    _covered.update(range(_idx, _idx + len(_span)))
                _per_tell_coverages.append(len(_span) / _doc_len)
            _per_ro_coverage.append(len(_covered) / _doc_len)
        train_mean_n_tells = sum(_per_ro_n_tells) / len(_per_ro_n_tells)
        train_tells_per_100w = sum(_per_ro_tells_per_100w) / len(_per_ro_tells_per_100w)
        train_mean_coverage = sum(_per_ro_coverage) / len(_per_ro_coverage)
        train_mean_per_tell_coverage = sum(_per_tell_coverages) / len(_per_tell_coverages) if _per_tell_coverages else 0.0
    else:
        train_mean_n_tells = 0.0
        train_tells_per_100w = 0.0
        train_mean_coverage = 0.0
        train_mean_per_tell_coverage = 0.0

    logger.info("step %d | mean_n_tells=%.2f tells_per_100w=%.2f", step, train_mean_n_tells, train_tells_per_100w)
    # per-rollout reward breakdown; n_zero is exact float zero (hard gate or scorer-kill), not residual failures
    n_positive = sum(1 for rw in all_rewards if rw > 0)
    n_negative = sum(1 for rw in all_rewards if rw < 0)
    n_zero = sum(1 for rw in all_rewards if rw == 0.0)
    n_opt = len(all_rewards)
    pct_zero = (n_zero / n_opt) if n_opt else 0.0
    n_zero_fmt_fail = sum(
        1
        for ro in _opt_rollouts
        if ro["reward"] == 0.0 and (not ro.get("format_ok", False))
    )
    n_zero_anomaly = sum(
        1
        for ro in _opt_rollouts
        if ro["reward"] == 0.0 and ro.get("format_ok", False)
    )

    audit_entry = {
        "step": step,
        "reward_mean": reward_mean,
        "format_rate": format_rate,
        "format_rate_before_fixing": format_rate_before_fixing,
        "docs": docs_audit,
    }
    audit_log.write(json.dumps(audit_entry) + "\n")
    audit_log.flush()

    def _mean(vals): return sum(vals) / len(vals) if vals else 0.0
    dt_rollouts_mean = _mean([t["dt_rollouts"] for t in doc_timings])
    dt_scoring_mean = _mean([t["dt_scoring"] for t in doc_timings])
    dt_scoring_mean_mean = _mean([t["dt_scoring_mean"] for t in doc_timings])

    dt_step_wall = time.perf_counter() - t0_step_wall
    step_total_dt = dt_save + fb_dt + opt_dt  # excludes doc processing (runs in parallel)
    logger.debug(
        "timing   | step %d: save_weights=%.1fs fwd_bwd=%.1fs optim=%.1fs ce_pass=%.1fs kl=%.1fs | step_wall=%.1fs",
        step, dt_save, fb_dt, opt_dt, dt_ce_pass, dt_kl_compute, dt_step_wall,
    )
    logger.debug(
        "timing   | step %d (per-doc means): doc_gather=%.1fs rollouts=%.1fs scoring=%.1fs scorer/rollout=%.1fs",
        step, dt_doc_gather, dt_rollouts_mean, dt_scoring_mean, dt_scoring_mean_mean,
    )

    replay_rows = []
    for d, da in zip(docs_used, docs_audit):
        _opt_rw = [ro["reward"] for ro in da["rollouts"] if ro.get("used_for_optimization") and ro["reward"] is not None]
        if len(_opt_rw) >= 2:
            _rw_mean = sum(_opt_rw) / len(_opt_rw)
            _reward_var = sum((r - _rw_mean) ** 2 for r in _opt_rw) / len(_opt_rw)
        else:
            _reward_var = 0.0
        replay_rows.append(
            {
                "uid": da["ease_uid"],
                "reward_mean": float(da.get("reward_mean", 0.0)),
                "reward_var": _reward_var,
                "format_rate_before_fixing": float(da.get("format_rate_before_fixing", 0.0)),
                "doc": d,
                # Only fresh-sampled docs surface a non-empty cacheable_rollouts; cached
                # replays return [] from _process_cached_rollouts so the cache stays anchored
                # to fresh-policy snapshots and we don't compound advantage staleness.
                "cached_rollouts": da.get("cacheable_rollouts") or [],
            }
        )

    n_cached_docs = sum(1 for fc in docs_used_from_cache if fc)
    n_fresh_docs = len(docs_used) - n_cached_docs
    _ann_tok_vals = [ro["ann_token_fraction"] for da in docs_audit for ro in da["rollouts"] if ro.get("ann_token_fraction") is not None]
    return {
        # --- core train metrics (logged to train/ in W&B) ---
        "train/reward_mean": reward_mean,
        "train/format_rate": format_rate,
        "train/format_rate_before_fixing": format_rate_before_fixing,
        "train/n_positive_rollouts": n_positive,
        "train/n_negative_rollouts": n_negative,
        "train/n_zero_rollouts": n_zero,
        "train/pct_zero_reward": pct_zero,
        "train/n_zero_format_fail": n_zero_fmt_fail,
        "train/n_zero_anomaly_format_ok": n_zero_anomaly,
        "train/n_excluded_rollouts": n_excluded_rollouts,
        "train/label_ctx_for_opt_rate": train_label_ctx_for_opt_rate,
        "train/label_hint_outer_follow_rate": label_hint_outer_follow_rate,
        "train/label_hint_outer_follow_rate_correct": label_hint_outer_follow_rate_correct,
        "train/label_hint_outer_follow_rate_flip": label_hint_outer_follow_rate_flip,
        "train/label_hint_agg_follow_rate": label_hint_agg_follow_rate,
        "train/n_cached_docs": n_cached_docs,
        "train/n_fresh_docs": n_fresh_docs,
        "train/ai_reward_mean": ai_reward_mean,
        "train/human_reward_mean": human_reward_mean,
        "train/effective_lr": learning_rate,
        "train/reward_ema": _reward_ema,
        "train/lr_scale": _lr_scale,
        "train/effective_k": effective_k,
        "train/rubric_parse_fail_rate": _comp_mean("rubric_parse_failed"),
        "train/rubric_zero_ann_rate": _comp_mean("rubric_zero_annotations"),
        "train/scorer_exception_rate": train_scorer_exception_rate,
        "train/grad_norm": _opt_grad_norm,
        # --- diagnostic metrics (train_diag/) ---
        "train_diag/binary_f1": train_binary_f1,
        "train_diag/rubric_cred_mean": rubric_cred_mean,
        "train_diag/rubric_cred_std": rubric_cred_std,
        "train_diag/mean_n_tells": train_mean_n_tells,
        "train_diag/tells_per_100w": train_tells_per_100w,
        "train_diag/mean_coverage": train_mean_coverage,
        "train_diag/mean_per_tell_coverage": train_mean_per_tell_coverage,
        "train_diag/format_invalid_type": train_format_invalid_type,
        "train_diag/format_text_mismatch": train_format_text_mismatch,
        "train_diag/format_char_diff_mean": (sum(train_all_format_char_diffs) / len(train_all_format_char_diffs)) if train_all_format_char_diffs else 0.0,
        "train_diag/text_mismatch_char_diff_mean": (sum(train_text_mismatch_char_diffs) / len(train_text_mismatch_char_diffs)) if train_text_mismatch_char_diffs else 0.0,
        "train_diag/text_mismatch_char_diff_p95": _quantile(train_text_mismatch_char_diffs, 0.95),
        "train_diag/text_mismatch_char_diff_max": max(train_text_mismatch_char_diffs) if train_text_mismatch_char_diffs else 0,
        "train_diag/adv_ema_std": _adv_ema_std,
        "train_diag/adv_std_floor": adv_std_floor,
        "train_diag/is_ratio_mean": step_is_ratio_mean,
        "train_diag/is_ratio_min": step_is_ratio_min,
        "train_diag/is_ratio_max": step_is_ratio_max,
        "train_diag/is_ratio_p10": step_is_ratio_p10,
        "train_diag/is_ratio_p90": step_is_ratio_p90,
        # Annotation-mask diagnostics: expect ~5–25% in steady state.
        # 0% = mask broken (training nothing); 100% = no doc text emitted.
        "train_diag/ann_token_fraction": (sum(_ann_tok_vals) / max(1, len(_ann_tok_vals))),
        "train_diag/ann_token_fraction_min": min(_ann_tok_vals, default=0.0),
        "train_diag/ann_token_fraction_max": max(_ann_tok_vals, default=0.0),
        "train_diag/completion_tokens_mean": completion_mean,
        "train_diag/completion_tokens_p95": completion_p95,
        "train_diag/completion_tokens_max": completion_max,
        "train_diag/completion_budget_hit_rate": budget_hit_rate,
        "train_diag/kl_policy_base": train_kl_policy_base,
        # --- per-component reward means (train_reward/) ---
        "train_reward/agg_score": _comp_mean("agg_score"),
        "train_reward/cls": _comp_mean("cls"),
        "train_reward/margin": _comp_mean("margin"),
        "train_reward/tell_alignment": _comp_mean("tell_alignment"),
        "train_reward/credibility": _comp_mean("credibility"),
        "train_reward/credibility_std": _comp_std("credibility"),
        "train_reward/outer_credibility": _comp_mean("outer_credibility"),
        "train_reward/outer_credibility_std": _comp_std("outer_credibility"),
        "train_reward/verdict_why_combined": _comp_mean("verdict_why_combined"),
        "train_reward/verdict_quote_coverage": _comp_mean("verdict_quote_coverage"),
        # --- per-token PPO advantages (not autograd gradients) ---
        "train_ptok_adv_mean/ann_score": _type_adv_mean("ann_score"),
        "train_ptok_adv_mean/ann_why": _type_adv_mean("ann_why"),
        "train_ptok_adv_mean/ann_type": _type_adv_mean("ann_type"),
        "train_ptok_adv_mean/verdict_score": _type_adv_mean("verdict_score"),
        "train_ptok_adv_mean/verdict_why": _type_adv_mean("verdict_why"),
        "train_ptok_adv_mean/verdict_type": _type_adv_mean("verdict_type"),
        "train_ptok_adv_mean/span_open": _type_adv_mean("span_open"),
        "train_ptok_adv_mean/structural": _type_adv_mean("structural"),
        "train_ptok_adv_abs/ann_score": _type_adv_abs_mean("ann_score"),
        "train_ptok_adv_abs/ann_why": _type_adv_abs_mean("ann_why"),
        "train_ptok_adv_abs/ann_type": _type_adv_abs_mean("ann_type"),
        "train_ptok_adv_abs/verdict_score": _type_adv_abs_mean("verdict_score"),
        "train_ptok_adv_abs/verdict_why": _type_adv_abs_mean("verdict_why"),
        "train_ptok_adv_abs/verdict_type": _type_adv_abs_mean("verdict_type"),
        "train_ptok_adv_abs/span_open": _type_adv_abs_mean("span_open"),
        "train_ptok_adv_abs/structural": _type_adv_abs_mean("structural"),
        "train_ptok_adv_count/ann_score": _type_adv_count("ann_score"),
        "train_ptok_adv_count/ann_why": _type_adv_count("ann_why"),
        "train_ptok_adv_count/ann_type": _type_adv_count("ann_type"),
        "train_ptok_adv_count/verdict_score": _type_adv_count("verdict_score"),
        "train_ptok_adv_count/verdict_why": _type_adv_count("verdict_why"),
        "train_ptok_adv_count/verdict_type": _type_adv_count("verdict_type"),
        "train_ptok_adv_count/span_open": _type_adv_count("span_open"),
        "train_ptok_adv_count/structural": _type_adv_count("structural"),
        # Effective gradient mass per token type: |adv * ptok_loss_scale|.
        # _mean = per-token pressure; _total = total mass this step (count * mean).
        # These show which components actually dominate the parameter update.
        "train_ptok_grad/ann_score": _type_grad_mean("ann_score"),
        "train_ptok_grad/ann_why": _type_grad_mean("ann_why"),
        "train_ptok_grad/ann_type": _type_grad_mean("ann_type"),
        "train_ptok_grad/verdict_score": _type_grad_mean("verdict_score"),
        "train_ptok_grad/verdict_why": _type_grad_mean("verdict_why"),
        "train_ptok_grad/verdict_type": _type_grad_mean("verdict_type"),
        "train_ptok_grad/span_open": _type_grad_mean("span_open"),
        "train_ptok_grad/structural": _type_grad_mean("structural"),
        "train_ptok_grad_total/ann_score": _type_grad_total("ann_score"),
        "train_ptok_grad_total/ann_why": _type_grad_total("ann_why"),
        "train_ptok_grad_total/ann_type": _type_grad_total("ann_type"),
        "train_ptok_grad_total/verdict_score": _type_grad_total("verdict_score"),
        "train_ptok_grad_total/verdict_why": _type_grad_total("verdict_why"),
        "train_ptok_grad_total/verdict_type": _type_grad_total("verdict_type"),
        "train_ptok_grad_total/span_open": _type_grad_total("span_open"),
        "train_ptok_grad_total/structural": _type_grad_total("structural"),
        # --- rollout-level component advantages (GRPO-normalized scalars per rollout) ---
        "train_roll_adv/span_open_cred": _cadv_mean("span_open_cred"),
        "train_roll_adv/ann": _cadv_mean("ann"),
        "train_roll_adv/cls": _cadv_mean("cls"),
        "train_roll_adv/outer_cred": _cadv_mean("outer_cred"),
        # --- timing ---
        "timing/step_wall_s": dt_step_wall,
        "timing/doc_gather_wall_s": dt_doc_gather,
        "timing/save_weights_s": dt_save,
        "timing/fwd_bwd_s": fb_dt,
        "timing/fwd_bwd_per_step_s": fb_dt / max(1, _n_grad_steps),
        "timing/optim_s": opt_dt,
        "timing/ce_pass_s": dt_ce_pass,
        "timing/kl_compute_s": dt_kl_compute,
        "timing/rollouts_mean_s": dt_rollouts_mean,
        "timing/scoring_mean_s": dt_scoring_mean,
        "timing/scoring_per_rollout_mean_s": dt_scoring_mean_mean,
        # --- why quality ---
        "why_quality/step_unique_rate": why_step_unique_rate,
        "why_quality/step_unique_rate_mean_per_doc": why_step_unique_rate_mean,
        "why_quality/step_mean_len": why_step_mean_len,
        "why_quality/step_p95_len": why_step_p95_len,
        "why_quality/intra_collapse_rate_mean": why_intra_collapse_rate_mean,
        "why_quality/repetition_score_mean": why_step_repetition_score_mean,
        # --- entropy ---
        "train_entropy/token_surprisal_mean": token_surprisal_mean,
        "train_entropy/token_surprisal_p10": token_surprisal_p10,
        "train_entropy/token_surprisal_p50": token_surprisal_p50,
        "train_entropy/token_surprisal_p90": token_surprisal_p90,
        "train_entropy/annotation_token_surprisal_mean": token_surprisal_ann_mean,
        "train_entropy/document_token_surprisal_mean": token_surprisal_doc_mean,
        "train_entropy/annotation_minus_document_surprisal": token_surprisal_ann_minus_doc,
        "train_entropy/annotation_token_fraction": token_surprisal_ann_frac,
        # --- auxiliary CE losses ---
        "aux_ce/fmt_repair_ce_n": _fmt_repair_ce_n,
        "aux_ce/fmt_repair_ce_loss": fmt_repair_ce_loss,
        "aux_ce/hint_ce_loss": hint_ce_loss,
        # --- internal keys (not logged to W&B, no "/" in key) ---
        "_train_rubric_ann_scores": train_rubric_ann_scores,
        "_train_is_ratios": all_is_ratios,
        "stratum_mean_rewards": _stratum_mean_rewards(docs_used, docs_audit),
        "stratum_format_rates": _stratum_format_rates(docs_used, docs_audit),
        "curriculum_reward_rows": [
            {"u": da["stratum_key"], "rw": min(1.0, max(0.0, float(da["curriculum_mean"])))}
            for da in docs_audit
            if da.get("curriculum_mean") is not None
        ],
        "replay_rows": replay_rows,
    }


_EVAL_CORE_KEYS = {
    "eval_reward_mean",
    "eval_format_rate",
    "eval_n_excluded_rollouts",
    "eval_auroc",
    "eval_tpr_at_fpr_001",
}


async def _do_sft_replay_step(
    training_client: tinker.TrainingClient,
    pool: list,
    ptr: int,
    batch_size: int,
    lr: float,
) -> tuple[dict, int]:
    """Cross-entropy forward-backward + optimizer step on a batch of pre-built SFT datums."""
    from rl_detector.sft.train_tinker_sft import _scalar_loss
    n = len(pool)
    end = ptr + batch_size
    batch = pool[ptr:end] + pool[: max(0, end - n)]
    new_ptr = end % n
    fwd_future = await training_client.forward_backward_async(data=batch, loss_fn="cross_entropy")
    fwd_out = await fwd_future.result_async()
    opt_future = await training_client.optim_step_async(tinker.AdamParams(learning_rate=lr))
    await opt_future.result_async()
    try:
        loss = _scalar_loss(batch, fwd_out)
    except AttributeError:
        loss = float("nan")
    return {"sft_replay_loss": loss}, new_ptr


async def _sampling_client_for_kl_anchor(
    service_client: tinker.ServiceClient,
    training_client: tinker.TrainingClient,
    resume_uri: str | None,
    anchor_weights_uri: str,
    persist_ttl_seconds: int | None,
) -> tinker.SamplingClient:
    """
    Tinker: SamplingClient for logprobs needs a sampler session or sampler_weights path, not raw /weights/.
    Default: save_weights_and_get (ephemeral, no durable sampler blob to bill).
    If persist_ttl_seconds set: named save_weights_for_sampler + create_sampling_client so storage expires.
    """
    anchor_weights_uri = str(anchor_weights_uri)
    a = anchor_weights_uri.rstrip("/")
    r = (resume_uri or "").rstrip("/")
    _ttl = int(persist_ttl_seconds) if persist_ttl_seconds is not None else None
    if _ttl is not None and _ttl <= 0:
        _ttl = None
    if a == r:
        _tc = training_client
    else:
        _tc = await service_client.create_training_client_from_state_with_optimizer_async(path=anchor_weights_uri)
    if _ttl is None:
        logger.info("KL anchor | ephemeral sampler export (save_weights_and_get), no named sampler_weights")
        return await _tc.save_weights_and_get_sampling_client_async()
    # rare escape hatch if something about ephemeral sessions breaks long KL runs
    _nm = "kl-ref-" + uuid.uuid4().hex[:16]
    logger.info("KL anchor | named sampler_weights %s ttl_s=%d (auto purge)", _nm, _ttl)
    _fut = await _tc.save_weights_for_sampler_async(name=_nm, ttl_seconds=_ttl)
    _saved = await _fut
    return await service_client.create_sampling_client_async(model_path=_saved.path)


class RunLogger:
    """Single point of truth for all metric writes: W&B + local JSONL mirror.

    Usage:
        rl = RunLogger(path)
        rl.log({"train/reward": 0.5}, step=3)   # → wandb + jsonl
        rl.log_eval(eval_metrics, step=3)        # → translates eval_ keys + histograms
        rl.close()
    """

    def __init__(self, jsonl_path: str) -> None:
        self._fh = open(jsonl_path, "a", buffering=1)  # line-buffered
        self._buf: dict = {}
        self._buf_step: int | None = None

    def _flush(self) -> None:
        if self._buf_step is None or not self._buf:
            return
        self._fh.write(json.dumps({"step": self._buf_step, **self._buf}) + "\n")
        self._buf = {}
        self._buf_step = None

    def log(self, data: dict, step: int) -> None:
        wandb.log(data, step=step)
        if step != self._buf_step:
            self._flush()
            self._buf_step = step
        for k, v in data.items():
            if isinstance(v, wandb.Histogram):
                continue
            try:
                json.dumps(v)
                self._buf[k] = v
            except (TypeError, ValueError):
                self._buf[k] = repr(v)

    def log_eval(self, eval_metrics: dict, step: int) -> None:
        log_data: dict = {}
        for k, v in eval_metrics.items():
            if k.startswith("_") or not k.startswith("eval_"):
                continue
            prefix = "eval" if k in _EVAL_CORE_KEYS else "eval_diag"
            log_data[f"{prefix}/{k[len('eval_'):]}"] = v
        if eval_metrics.get("_eval_ai_scores"):
            log_data["eval_diag/hist_ai_scores"] = wandb.Histogram(eval_metrics["_eval_ai_scores"])
        if eval_metrics.get("_eval_human_scores"):
            log_data["eval_diag/hist_human_scores"] = wandb.Histogram(eval_metrics["_eval_human_scores"])
        self.log(log_data, step=step)

    def close(self) -> None:
        self._flush()
        self._fh.close()


async def main(resume: str | None = None, resume_step: int = 0, eval_only: bool = False, checkpoint: str | None = None, run_name: str | None = None):
    _rebind_train_globals()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if run_name:
        CFG.wandb.name = run_name

    _dt = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    _slug = (getattr(CFG.wandb, "name", None) or "unnamed").replace("/", "-").replace(" ", "_")
    run_dir = pathlib.Path("runs") / f"{_dt}_{_slug}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.debug("run directory: %s", run_dir.resolve())

    CFG.training.audit_log_path = str(run_dir / "audit_log.jsonl")
    CFG.training.eval_audit_log_path = str(run_dir / "eval_audit_log.jsonl")
    CFG.training.format_fail_audit_path = str(run_dir / "format_fail_audit.jsonl")

    with open(run_dir / "config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(CFG))

    # Deterministic seeding — must cover every RNG used in the Python process.
    # NOTE: PYTHONHASHSEED only takes effect for child processes spawned after
    # this point; for the current interpreter it must be set in the environment
    # *before* Python starts (e.g. PYTHONHASHSEED=2242 python -m rl_detector.train).
    # All critical random choices already use explicit random.Random(seed) instances,
    # so hash-randomisation does not affect numerical results.
    os.environ["PYTHONHASHSEED"] = str(GLOBAL_SEED)
    random.seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    torch.manual_seed(GLOBAL_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(GLOBAL_SEED)

    _wandb_init = dict(
        project=CFG.wandb.project,
        entity=CFG.wandb.entity,
        config=config_as_dict(CFG),
        resume="allow",
    )
    _wandb_name = getattr(CFG.wandb, "name", None)
    if _wandb_name is not None:
        _wandb_init["name"] = _wandb_name
    _exp_description = os.environ.get("WANDB_EXP_DESCRIPTION")
    if _exp_description:
        _wandb_init["notes"] = _exp_description
    wandb.init(**_wandb_init)
    run_logger = RunLogger(getattr(CFG.training, "metrics_log_path", str(run_dir / "metrics_log.jsonl")))

    _weave_enabled = bool(getattr(CFG.wandb, "weave_trace", True))
    if _weave_enabled:
        _entity = getattr(CFG.wandb, "entity", None) or ""
        _proj = f"{_entity}/{CFG.wandb.project}" if _entity else CFG.wandb.project
        weave.init(project_name=_proj, settings={"print_call_link": False, "log_level": "WARNING", "implicitly_patch_integrations": False})

    service_client = tinker.ServiceClient()
    loop = asyncio.get_event_loop()

    def _load_tokenizer():
        logger.info("startup | loading tokenizer...")
        tok = load_tokenizer()
        logger.info("startup | tokenizer ready")
        return tok

    _max_train_docs = int(getattr(CFG.data, "max_train_docs", 2000))
    _max_eval_docs = int(getattr(CFG.data, "max_eval_docs", 200))

    if eval_only:
        if not checkpoint:
            raise SystemExit("--checkpoint is required with --eval-only")
        logger.info("eval-only | loading checkpoint %s + tokenizer + eval dataset in parallel...", checkpoint)
        training_client, tokenizer = await asyncio.gather(
            service_client.create_training_client_from_state_with_optimizer_async(path=checkpoint),
            loop.run_in_executor(None, _load_tokenizer),
        )
        _mdt = int(CFG.data.max_doc_tokens)
        _min_dt = int(getattr(CFG.data, "min_doc_tokens", 0))
        test_docs, _eval_cache_meta = await loop.run_in_executor(
            None,
            lambda: load_docs_preprocessed(
                tokenizer=tokenizer,
                use_eval_split=True,
                max_docs=_max_eval_docs,
                max_doc_tokens=_mdt,
                min_doc_tokens=_min_dt,
            ),
        )
        logger.info(
            "eval-only | eval cache %s (%s) docs=%d",
            "HIT" if _eval_cache_meta.get("cache_hit") else "MISS",
            _eval_cache_meta.get("cache_path", ""),
            len(test_docs),
        )
        eval_docs = _select_eval_docs_fixed(test_docs, sample_size=EVAL_SAMPLE_SIZE, seed=EVAL_SEED)
        rubric_client = get_client() if _NEED_RUBRIC_CLIENT else None
        logger.debug(
            "eval-only | use_rubric_scorer=%s",
            _USE_RUBRIC_SCORER,
        )
        eval_metrics = await evaluate_model(
            training_client, tokenizer, eval_docs, "eval-only", eval_seed=EVAL_SEED,
            eval_audit_path=CFG.training.eval_audit_log_path,
        )
        run_logger.log_eval(eval_metrics, step=0)
        run_logger.close()
        wandb.finish()
        if _weave_enabled:
            weave.finish()
        logger.info("eval-only | AUROC=%.4f TPR@1%%FPR=%.4f format_rate=%.3f", eval_metrics["eval_auroc"], eval_metrics["eval_tpr_at_fpr_001"], eval_metrics["eval_format_rate"])
        return

    if resume:
        logger.debug("startup | resuming from checkpoint %s (step offset=%d), rank=%d + loading tokenizer and dataset in parallel...", resume, resume_step, CFG.model.lora_rank)
        training_client_coro = service_client.create_training_client_from_state_with_optimizer_async(path=resume)
    else:
        logger.debug("startup | creating LoRA training client (base_model=%s, rank=%d) + loading tokenizer and dataset in parallel...", CFG.model.base_model, CFG.model.lora_rank)
        training_client_coro = service_client.create_lora_training_client_async(
            base_model=CFG.model.base_model,
            rank=CFG.model.lora_rank,
            seed=GLOBAL_SEED,
        )
    training_client, tokenizer = await asyncio.gather(
        training_client_coro,
        loop.run_in_executor(None, _load_tokenizer),
    )
    _mdt = int(CFG.data.max_doc_tokens)
    _min_dt = int(getattr(CFG.data, "min_doc_tokens", 0))
    all_docs, test_docs = await asyncio.gather(
        loop.run_in_executor(
            None,
            lambda: load_docs_preprocessed(
                tokenizer=tokenizer,
                use_eval_split=False,
                max_docs=_max_train_docs,
                max_doc_tokens=_mdt,
                min_doc_tokens=_min_dt,
                balance_eval_pool=False,
            ),
        ),
        loop.run_in_executor(
            None,
            lambda: load_docs_preprocessed(
                tokenizer=tokenizer,
                use_eval_split=True,
                max_docs=_max_eval_docs,
                max_doc_tokens=_mdt,
                min_doc_tokens=_min_dt,
            ),
        ),
    )
    all_docs, _train_cache_meta = all_docs
    test_docs, _eval_cache_meta = test_docs
    logger.info(
        "startup | train cache %s (%s) docs=%d",
        "HIT" if _train_cache_meta.get("cache_hit") else "MISS",
        _train_cache_meta.get("cache_path", ""),
        len(all_docs),
    )
    logger.info(
        "startup | eval cache %s (%s) docs=%d",
        "HIT" if _eval_cache_meta.get("cache_hit") else "MISS",
        _eval_cache_meta.get("cache_path", ""),
        len(test_docs),
    )
    if not all_docs:
        raise SystemExit(
            "train split loaded 0 documents after filters. Check data.train_dataset_ids_filter vs parquet dataset_id "
            "(Hydra ListConfig must expand to real ids, not a single repr string)."
        )
    logger.debug("startup | all ready")

    eval_docs = _select_eval_docs_fixed(test_docs, sample_size=EVAL_SAMPLE_SIZE, seed=EVAL_SEED)

    _kl_coef = float(getattr(CFG.training, "kl_penalty_coef", 0.0))
    kl_ref_client: tinker.SamplingClient | None = None
    if _kl_coef != 0.0:
        _kl_ref = getattr(CFG.model, "checkpoint", None)
        _cfg_kl = getattr(CFG.training, "kl_reference_checkpoint", None)
        if _cfg_kl not in (None, "", "null") and str(_cfg_kl) != str(_kl_ref):
            logger.warning(
                "startup | ignoring training.kl_reference_checkpoint=%s; KL anchor=model.checkpoint=%s",
                _cfg_kl,
                _kl_ref,
            )
        _kl_persist_ttl = getattr(CFG.training, "kl_sampler_persist_ttl_seconds", None)
        if _kl_persist_ttl is not None:
            _kl_persist_ttl = int(_kl_persist_ttl)
        if not _kl_ref:
            # Fresh run with no reference checkpoint: snapshot initial weights once and
            # use that as the anchor. This is the "anchor to where we started" semantic
            # and matches what the Tinker cookbook recommends as a safe default.
            logger.info("startup | KL anchor: no resume / kl_reference_checkpoint set, snapshotting initial weights")
            _kl_ref = await _save_state_with_ttl(training_client, name="kl-init")
        logger.info("startup | KL penalty coef=%s anchor_weights=%s persist_ttl_s=%s", _kl_coef, _kl_ref, _kl_persist_ttl)
        kl_ref_client = await _sampling_client_for_kl_anchor(
            service_client, training_client, resume, _kl_ref, _kl_persist_ttl
        )
    logger.info("startup | rl_loss_fn=ppo (tinker built-in)")

    _sft_replay_enabled = bool(getattr(CFG.training, "sft_replay_enabled", False))
    sft_replay_pool: list = []
    sft_replay_ptr: int = 0
    if _sft_replay_enabled:
        try:
            from rl_detector.sft.train_tinker_sft import _build_sft_datum, _load_tell_split
            _sft_rows = _load_tell_split(CFG.sft.dataset_path, "train")
            _sft_rng = random.Random(GLOBAL_SEED + 7)
            _sft_rng.shuffle(_sft_rows)
            _sft_inject = bool(getattr(CFG.sft, "label_injection_enabled", True))
            _sft_mix = float(getattr(CFG.sft, "label_injection_mix_ratio", 0.5))
            for _row in _sft_rows:
                _inject = _sft_inject and (_sft_rng.random() < _sft_mix)
                _built = _build_sft_datum(
                    tokenizer=tokenizer,
                    row=_row,
                    inject_label_instruction=_inject,
                    annotation_xml=_row["annotation"],
                )
                if _built is not None:
                    sft_replay_pool.append(_built[0])
            _sft_rng.shuffle(sft_replay_pool)
            logger.info("startup | SFT replay pool: %d datums from %s", len(sft_replay_pool), CFG.sft.dataset_path)
            if not sft_replay_pool:
                logger.warning("startup | SFT replay pool empty after filtering — disabling replay")
                _sft_replay_enabled = False
        except Exception:
            logger.exception("startup | SFT replay pool load failed — disabling replay")
            _sft_replay_enabled = False

    rubric_client = get_client() if _NEED_RUBRIC_CLIENT else None
    logger.debug(
        "startup | use_rubric_scorer=%s — %s",
        _USE_RUBRIC_SCORER,
        ("rubric evaluator: external LLM rates annotation credibility/quality per rollout"
         if _USE_RUBRIC_SCORER
         else "self-score only, rubric client not created"),
    )
    step = resume_step
    best_eval_auroc = float("-inf")
    best_eval_tpr = float("-inf")
    best_eval_reward = float("-inf")
    best_eval_path = None
    stratum_sampler = None
    _early_stop_patience_evals = int(getattr(CFG.training, "early_stop_patience_evals", 10))
    _eval_no_improve_streak = 0
    with logging_redirect_tqdm(), open(CFG.training.audit_log_path, "w") as audit_log:
        if eval_docs:
            eval_metrics = await evaluate_model(
                training_client,
                tokenizer,
                eval_docs,
                step,
                eval_seed=EVAL_SEED,
                eval_audit_path=CFG.training.eval_audit_log_path,
            )
            run_logger.log_eval(eval_metrics, step=step)
            _eval_auroc_improved = eval_metrics["eval_auroc"] > best_eval_auroc
            _eval_tpr_improved = eval_metrics["eval_tpr_at_fpr_001"] > best_eval_tpr
            _eval_reward_improved = eval_metrics["eval_reward_mean"] > best_eval_reward
            _best_log: dict = {}
            if _eval_auroc_improved:
                best_eval_auroc = eval_metrics["eval_auroc"]
                best_eval_path = await _save_state_with_ttl(training_client, name=f"best-step-{step}")
                logger.info("Saved new best eval checkpoint at step %d: %s", step, best_eval_path)
                if stratum_sampler is not None: _save_ucb_state(stratum_sampler, run_dir, step, best_eval_path)
                _best_log["eval/best_auroc"] = best_eval_auroc
            if _eval_tpr_improved:
                best_eval_tpr = eval_metrics["eval_tpr_at_fpr_001"]
                _best_log["eval/best_tpr_at_fpr_001"] = best_eval_tpr
            if _eval_reward_improved:
                best_eval_reward = eval_metrics["eval_reward_mean"]
                _best_log["eval/best_reward_mean"] = best_eval_reward
            if _best_log:
                run_logger.log(_best_log, step=step)
            if _eval_auroc_improved or _eval_tpr_improved or _eval_reward_improved:
                _eval_no_improve_streak = 0
            else:
                _eval_no_improve_streak += 1
                logger.info(
                    "early-stop monitor | no eval improvement (auroc/tpr/reward) at step %d: streak %d/%d",
                    step,
                    _eval_no_improve_streak,
                    _early_stop_patience_evals,
                )
            run_logger.log({"eval/no_improve_streak": _eval_no_improve_streak}, step=step)

        _sampler_mode = str(getattr(getattr(CFG.training, "sampler", {}), "mode", "stratum_ucb"))
        _ssa = float(_training_get("curriculum.stratum_sampler.alpha", 0.05, legacy_key="stratum_sampler_alpha"))
        _ssb = float(_training_get("curriculum.stratum_sampler.beta", 6.0, legacy_key="stratum_sampler_beta"))
        _ssc = float(_training_get("curriculum.stratum_sampler.ucb_c", 0.2, legacy_key="stratum_sampler_ucb_c"))
        _ssf = float(_training_get("curriculum.stratum_sampler.floor", 0.1, legacy_key="stratum_sampler_floor"))
        _ssh = float(_training_get("curriculum.stratum_sampler.hardness_weight", 0.0))
        _ssm = float(_training_get("curriculum.stratum_sampler.max_stratum_weight", 1.0))
        _blend = int(_training_get("curriculum.probe.blend_ramp_steps", 0, legacy_key="stratum_probe_blend_ramp_steps"))
        if _blend <= 0:
            _blend = int(getattr(CFG.training, "stratum_curriculum_ramp_steps", 0))
        _ppb = float(_training_get("curriculum.probe.prior_beta", 6.0, legacy_key="stratum_probe_prior_beta"))
        _probe_on = bool(_training_get("curriculum.probe.enabled", True, legacy_key="stratum_probe_enabled")) and _sampler_mode not in ("uniform_plus_replay",)
        init_ema = None
        init_vis = None
        probe_reward_for_sampler: dict[tuple[str, str, int], float] | None = None
        if _probe_on:
            init_ema, init_vis, probe_meta = await run_stratum_bootstrap_probe(
                training_client, tokenizer, rubric_client, all_docs, GLOBAL_SEED
            )
            if probe_meta:
                run_logger.log({f"probe/{k}": float(v) for k, v in probe_meta.items()}, step=max(0, int(resume_step)))
            probe_reward_for_sampler = init_ema if init_ema else None
        if _sampler_mode == "stratum_plus_replay":
            _spr = getattr(CFG.training.sampler, "stratum_plus_replay")
            stratum_sampler = StratumReplaySampler(
                all_docs,
                docs_per_step=int(CFG.training.docs_per_step),
                seed=GLOBAL_SEED,
                alpha=_ssa,
                beta=_ssb,
                ucb_c=_ssc,
                floor=_ssf,
                hardness_weight=_ssh,
                max_stratum_weight=_ssm,
                global_batch_offset=int(resume_step),
                initial_ema=init_ema,
                initial_n_visits=init_vis,
                probe_stratum_reward=probe_reward_for_sampler,
                probe_blend_ramp_steps=_blend,
                probe_prior_beta=_ppb,
                curriculum_gaussian_gamma=float(
                    _training_get("curriculum.gaussian_gamma", 0.0, legacy_key="curriculum_gaussian_gamma")
                ),
                curriculum_reward_ema_alpha=float(
                    _training_get("curriculum.reward_ema_alpha", 0.12, legacy_key="curriculum_reward_ema_alpha")
                ),
                curriculum_tau_start=float(_training_get("curriculum.tau_start", 0.35, legacy_key="curriculum_tau_start")),
                curriculum_tau_end=float(_training_get("curriculum.tau_end", 0.55, legacy_key="curriculum_tau_end")),
                curriculum_ramp_steps=int(
                    _training_get("curriculum.ramp_steps", None, legacy_key="curriculum_ramp_steps")
                    or int(getattr(CFG.training, "max_steps", 1000))
                ),
                replay_fraction_start=float(_spr.replay_fraction_start),
                replay_fraction_end=float(_spr.replay_fraction_end),
                replay_fraction_ramp_steps=int(_spr.replay_fraction_ramp_steps),
                replay_min_count=int(_spr.replay_min_count),
                replay_pool_max_size=int(_spr.replay_pool_max_size),
                priority_var_weight=float(_spr.priority_var_weight),
                priority_hard_weight=float(_spr.priority_hard_weight),
                priority_format_weight=float(_spr.priority_format_weight),
                reward_ema_alpha=float(_spr.reward_ema_alpha),
                monitor_top_k=10,
                cache_rollouts=bool(getattr(_spr, "cache_rollouts", False)),
                max_rollout_reuses=int(getattr(_spr, "max_rollout_reuses", 3)),
                cache_pool_max_size=int(getattr(_spr, "cache_pool_max_size", 800)),
            )
        elif _sampler_mode == "uniform_plus_replay":
            _spr = getattr(CFG.training.sampler, "uniform_plus_replay")
            stratum_sampler = UniformReplaySampler(
                all_docs,
                docs_per_step=int(CFG.training.docs_per_step),
                seed=GLOBAL_SEED,
                replay_fraction_start=float(_spr.replay_fraction_start),
                replay_fraction_end=float(_spr.replay_fraction_end),
                replay_fraction_ramp_steps=int(_spr.replay_fraction_ramp_steps),
                replay_min_count=int(_spr.replay_min_count),
                replay_pool_max_size=int(_spr.replay_pool_max_size),
                priority_var_weight=float(_spr.priority_var_weight),
                priority_hard_weight=float(_spr.priority_hard_weight),
                priority_format_weight=float(_spr.priority_format_weight),
                reward_ema_alpha=float(_spr.reward_ema_alpha),
                monitor_top_k=10,
                cache_rollouts=bool(getattr(_spr, "cache_rollouts", False)),
                max_rollout_reuses=int(getattr(_spr, "max_rollout_reuses", 3)),
                cache_pool_max_size=int(getattr(_spr, "cache_pool_max_size", 800)),
            )
        else:
            stratum_sampler = StratumSampler(
                all_docs,
                docs_per_step=int(CFG.training.docs_per_step),
                seed=GLOBAL_SEED,
                alpha=_ssa,
                beta=_ssb,
                ucb_c=_ssc,
                floor=_ssf,
                hardness_weight=_ssh,
                max_stratum_weight=_ssm,
                global_batch_offset=int(resume_step),
                initial_ema=init_ema,
                initial_n_visits=init_vis,
                probe_stratum_reward=probe_reward_for_sampler,
                probe_blend_ramp_steps=_blend,
                probe_prior_beta=_ppb,
                curriculum_gaussian_gamma=float(
                    _training_get("curriculum.gaussian_gamma", 0.0, legacy_key="curriculum_gaussian_gamma")
                ),
                curriculum_reward_ema_alpha=float(
                    _training_get("curriculum.reward_ema_alpha", 0.12, legacy_key="curriculum_reward_ema_alpha")
                ),
                curriculum_tau_start=float(_training_get("curriculum.tau_start", 0.25, legacy_key="curriculum_tau_start")),
                curriculum_tau_end=float(_training_get("curriculum.tau_end", 0.85, legacy_key="curriculum_tau_end")),
                curriculum_ramp_steps=int(
                    _training_get("curriculum.ramp_steps", None, legacy_key="curriculum_ramp_steps")
                    or int(getattr(CFG.training, "max_steps", 1000))
                ),
            )
        _ucb_state = _load_ucb_state(resume or "")
        if _ucb_state is not None:
            stratum_sampler.set_ucb_state(_ucb_state)
            logger.info("UCB state restored from checkpoint %s", resume)
        if _sampler_mode not in ("uniform_plus_replay",) and _blend > 0 and probe_reward_for_sampler:
            logger.debug(
                "startup | stratum probe prior: blend_ramp=%d steps, prior_beta=%.2f, probed_strata=%d",
                _blend,
                _ppb,
                len(probe_reward_for_sampler),
            )
        if _sampler_mode not in ("uniform_plus_replay",) and _blend > 0 and not probe_reward_for_sampler:
            logger.warning(
                "stratum_probe_blend_ramp_steps=%d but no probe rewards (probe off or failed); blend disabled in sampler",
                _blend,
            )
        logger.debug("startup | stratum sampler active (%d strata)", len(stratum_sampler.stratum_emas()))
        pbar = tqdm(total=CFG.training.max_steps, desc="training", unit="step")
        _fmt_health_bad_streak = 0
        _FMT_HEALTH_THRESHOLD = float(getattr(CFG.training, "format_health_threshold", 0.4))
        _FMT_HEALTH_STREAK = int(getattr(CFG.training, "format_health_streak", 3))
        while True:
            docs = stratum_sampler.sample_batch()
            cached_per_doc = (
                stratum_sampler.last_cached_rollouts()
                if hasattr(stratum_sampler, "last_cached_rollouts")
                else None
            )
            metrics = await train_step(
                training_client, tokenizer, rubric_client, docs, step=step, audit_log=audit_log,
                kl_ref_client=kl_ref_client,
                cached_per_doc=cached_per_doc,
            )
            if _sampler_mode not in ("uniform_plus_replay",):
                stratum_sampler.ingest_curriculum_reward_rows(metrics.pop("curriculum_reward_rows", []))
                stratum_sampler.update(metrics.get("stratum_mean_rewards", {}))
            else:
                metrics.pop("curriculum_reward_rows", None)
            _replay_rows = metrics.pop("replay_rows", [])
            if hasattr(stratum_sampler, "observe_docs"):
                stratum_sampler.observe_docs(_replay_rows)
            if _sampler_mode in ("uniform_plus_replay", "stratum_plus_replay") and hasattr(stratum_sampler, "replay_snapshot"):
                _m_path = "replay_monitor.jsonl"
                pathlib.Path(_m_path).parent.mkdir(parents=True, exist_ok=True)
                _snap = stratum_sampler.replay_snapshot()
                with open(_m_path, "a", encoding="utf-8") as _mf:
                    _mf.write(json.dumps({"step": int(step), **_snap}, ensure_ascii=False) + "\n")
            pbar.set_postfix(
                train_reward=f"{metrics['train/reward_mean']:.3f}",
                train_format=f"{metrics['train/format_rate']:.2f}",
                train_format_pre=f"{metrics['train/format_rate_before_fixing']:.2f}",
            )
            pbar.update(1)

            # Format health early-stop: if format_rate_before_fixing stays below threshold
            # for _FMT_HEALTH_STREAK consecutive steps, save a checkpoint and halt cleanly
            # so the run can be inspected rather than continuing to degrade.
            _fmt_pre = metrics.get("train/format_rate_before_fixing", 1.0)
            if _fmt_pre < _FMT_HEALTH_THRESHOLD:
                _fmt_health_bad_streak += 1
                logger.warning(
                    "FORMAT HEALTH: step %d fmt_pre=%.3f < threshold=%.2f  (streak %d/%d)",
                    step, _fmt_pre, _FMT_HEALTH_THRESHOLD, _fmt_health_bad_streak, _FMT_HEALTH_STREAK,
                )
                if _fmt_health_bad_streak >= _FMT_HEALTH_STREAK:
                    ckpt_name = f"format-health-stop-step-{step}"
                    logger.error(
                        "FORMAT HEALTH: format_rate_before_fixing < %.2f for %d consecutive steps — "
                        "saving checkpoint '%s' and halting to prevent further collapse.",
                        _FMT_HEALTH_THRESHOLD, _FMT_HEALTH_STREAK, ckpt_name,
                    )
                    await _save_state_with_ttl(training_client, ckpt_name)
                    break
            else:
                _fmt_health_bad_streak = 0

            train_log_data = {k: v for k, v in metrics.items() if "/" in k and v is not None}
            if metrics.get("_train_is_ratios"):
                train_log_data["train_diag/hist_is_ratios"] = wandb.Histogram(metrics["_train_is_ratios"])
            if metrics.get("_train_rubric_ann_scores"):
                train_log_data["train_diag/hist_rubric_ann_score"] = wandb.Histogram(metrics["_train_rubric_ann_scores"])
            lbl_name = {0: "human", 1: "ai"}
            for (src, dom, lbl), ema_val in stratum_sampler.stratum_emas().items():
                train_log_data[f"stratum_ema/{src}/{dom}/{lbl_name.get(lbl, lbl)}"] = ema_val
            for (src, dom, lbl), w in stratum_sampler.stratum_weights().items():
                train_log_data[f"stratum_weight/{src}/{dom}/{lbl_name.get(lbl, lbl)}"] = w
            for (src, dom, lbl), c in stratum_sampler.last_batch_counts().items():
                train_log_data[f"stratum_sample_step/{src}/{dom}/{lbl_name.get(lbl, lbl)}"] = c
            for (src, dom, lbl), c in stratum_sampler.cumulative_sample_counts().items():
                train_log_data[f"stratum_sample_total/{src}/{dom}/{lbl_name.get(lbl, lbl)}"] = c
            for (src, dom, lbl), fr in metrics.get("stratum_format_rates", {}).items():
                train_log_data[f"stratum_format/{src}/{dom}/{lbl_name.get(lbl, lbl)}"] = fr
            for _sk, _sv in stratum_sampler.last_stratum_diag().items():
                train_log_data[f"train_diag/{_sk}"] = _sv
            for _ek, _ev in stratum_sampler.last_curriculum_diag().items():
                train_log_data[f"sampler_curriculum/{_ek}"] = _ev
            run_logger.log(train_log_data, step=step)

            if _sft_replay_enabled and sft_replay_pool:
                _replay_every = int(getattr(CFG.training, "sft_replay_every_n_steps", 10))
                _replay_fmt_threshold = float(getattr(CFG.training, "sft_replay_low_format_threshold", 0.0))
                _should_sft_replay = (
                    _fmt_pre < _replay_fmt_threshold if _replay_fmt_threshold > 0.0
                    else step % _replay_every == 0
                )
                if _should_sft_replay:
                    _replay_bs = int(getattr(CFG.training, "sft_replay_batch_size", 4))
                    _replay_lr = float(getattr(CFG.training, "sft_replay_lr_scale", 0.3)) * float(CFG.training.learning_rate)
                    _replay_metrics, sft_replay_ptr = await _do_sft_replay_step(
                        training_client, sft_replay_pool, sft_replay_ptr, _replay_bs, _replay_lr
                    )
                    logger.info("step %d | sft_replay loss=%.4f lr=%.2e", step, _replay_metrics["sft_replay_loss"], _replay_lr)
                    run_logger.log({"sft_replay/loss": _replay_metrics["sft_replay_loss"]}, step=step)

            step += 1

            if step % EVAL_EVERY_STEPS == 0 and eval_docs:
                eval_metrics = await evaluate_model(
                    training_client,
                    tokenizer,
                    eval_docs,
                    step,
                    eval_seed=EVAL_SEED,
                    eval_audit_path=CFG.training.eval_audit_log_path,
                )
                run_logger.log_eval(eval_metrics, step=step)
                _eval_auroc_improved = eval_metrics["eval_auroc"] > best_eval_auroc
                _eval_tpr_improved = eval_metrics["eval_tpr_at_fpr_001"] > best_eval_tpr
                _eval_reward_improved = eval_metrics["eval_reward_mean"] > best_eval_reward
                _best_log = {}
                if _eval_auroc_improved:
                    best_eval_auroc = eval_metrics["eval_auroc"]
                    best_eval_path = await _save_state_with_ttl(training_client, name=f"best-step-{step}")
                    logger.info("Saved new best eval checkpoint at step %d: %s", step, best_eval_path)
                    if stratum_sampler is not None: _save_ucb_state(stratum_sampler, run_dir, step, best_eval_path)
                    _best_log["eval/best_auroc"] = best_eval_auroc
                if _eval_tpr_improved:
                    best_eval_tpr = eval_metrics["eval_tpr_at_fpr_001"]
                    _best_log["eval/best_tpr_at_fpr_001"] = best_eval_tpr
                if _eval_reward_improved:
                    best_eval_reward = eval_metrics["eval_reward_mean"]
                    _best_log["eval/best_reward_mean"] = best_eval_reward
                if _best_log:
                    run_logger.log(_best_log, step=step)
                if _eval_auroc_improved or _eval_tpr_improved or _eval_reward_improved:
                    _eval_no_improve_streak = 0
                else:
                    _eval_no_improve_streak += 1
                    logger.info(
                        "early-stop monitor | no eval improvement (auroc/tpr/reward) at step %d: streak %d/%d",
                        step,
                        _eval_no_improve_streak,
                        _early_stop_patience_evals,
                    )
                run_logger.log({"eval/no_improve_streak": _eval_no_improve_streak}, step=step)
                if _eval_no_improve_streak >= _early_stop_patience_evals:
                    logger.info(
                        "early stopping | no eval AUROC/reward improvement for %d consecutive evals, stopping at step %d",
                        _early_stop_patience_evals,
                        step,
                    )
                    break

            if step >= CFG.training.max_steps:
                logger.info("Reached max_steps, stopping.")
                break

            if CFG.training.checkpoint_every > 0 and step % CFG.training.checkpoint_every == 0:
                step_ckpt_path = await _save_state_with_ttl(training_client, name=f"step-{step}")
                logger.info("Saved checkpoint at step %d: %s", step, step_ckpt_path)
                if stratum_sampler is not None: _save_ucb_state(stratum_sampler, run_dir, step, step_ckpt_path)
                run_logger.log({"checkpoint": step}, step=step)

        pbar.close()

    run_logger.close()
    logger.info("training complete | saving final checkpoint")
    final_path = await _save_state_with_ttl(training_client, name="final")
    logger.info(f"final checkpoint saved: {final_path}")
    if best_eval_path is not None:
        logger.info("best eval checkpoint kept at: %s", best_eval_path)

    _wn = getattr(CFG.wandb, "name", None)
    _mdt_run = int(CFG.data.max_doc_tokens)
    if best_eval_auroc > float("-inf"):
        logger.debug(
            "RUN_SUMMARY best_eval_auroc=%.6f max_doc_tokens=%d use_rubric_scorer=%s wandb_name=%r",
            best_eval_auroc,
            _mdt_run,
            _USE_RUBRIC_SCORER,
            _wn,
        )
    else:
        logger.debug(
            "RUN_SUMMARY best_eval_auroc=nan max_doc_tokens=%d use_rubric_scorer=%s wandb_name=%r (no eval docs)",
            _mdt_run,
            _USE_RUBRIC_SCORER,
            _wn,
        )

    if _weave_enabled:
        weave.finish()
    wandb.finish()


@hydra.main(
    version_base=None,
    config_path="../../conf",
    config_name="config",
)
def _hydra_run(cfg: DictConfig) -> None:
    import rl_detector.config as c
    import rl_detector.data as data_mod
    import rl_detector.eval_runner as eval_runner_mod
    import rl_detector.frozen as frozen_mod
    import rl_detector.prompt_utils as prompt_utils_mod
    import rl_detector.prompts as prompts_mod
    import rl_detector.rewards as rewards_mod
    import rl_detector.rollouts as rollouts_mod

    global CFG
    c.CFG = cfg
    CFG = cfg
    # Hydra swaps config after imports; keep modules that cached CFG in sync.
    for mod in (
        data_mod,
        eval_runner_mod,
        frozen_mod,
        prompt_utils_mod,
        prompts_mod,
        rewards_mod,
        rollouts_mod,
    ):
        mod.CFG = cfg
    resume = None if cfg.run.resume in (None, "null") else str(cfg.run.resume)
    if cfg.run.resume_step is not None:
        resume_step = int(cfg.run.resume_step)
    elif resume and (m := __import__("re").search(r"step-(\d+)", resume)):
        resume_step = int(m.group(1))
    else:
        resume_step = 0
    eval_only = bool(cfg.run.eval_only)
    checkpoint = None if cfg.run.checkpoint in (None, "null") else str(cfg.run.checkpoint)
    run_name = None if cfg.run.run_name in (None, "null") else str(cfg.run.run_name)
    asyncio.run(
        main(
            resume=resume,
            resume_step=resume_step,
            eval_only=eval_only,
            checkpoint=checkpoint,
            run_name=run_name,
        )
    )


if __name__ == "__main__":
    _hydra_run()
