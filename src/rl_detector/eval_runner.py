"""Shared evaluation helpers for train-time and manual checkpoint eval."""

import asyncio
import json
import logging
import pathlib
import random

import tinker
from sklearn.metrics import f1_score, roc_auc_score, roc_curve

from rl_detector.config import CFG
from rl_detector.frozen import self_score_from_output
from rl_detector.prompt_utils import format_prompt_for_model, get_think_already_open, quantile as _quantile
from rl_detector.annotation_utils import get_outer_bracket_metadata
from rl_detector.rewards import (
    format_diagnostics,
    reward_components as _reward_components,
    outer_document_score,
    parse_indicators,
    safe_format_diagnostics as _safe_format_diagnostics_for_eval,
)
from rl_detector.format_fix import _apply_format_fix_to_text_fields
from rl_detector.rollouts import _get_analysis_stub_tokens, decode_response_text
from rl_detector.trace_payloads import EvalRolloutTracePayload, trace_payload_to_weave_dict

logger = logging.getLogger(__name__)

# eval subsampling and per doc rollout seeds; CFG.frozen.seed overrides but we pin 2262 if missing
_DEFAULT_REPRO_SEED: int = 2262


def _tri_class_from_score(score: float, margin: float) -> str:
    """Map signed aggregate score to tri-class label using an abstain band.

    Classes:
      - "human" when score <= -margin
      - "AI" when score >= +margin
      - "unknown" when -margin < score < +margin

    Why this exists:
      Some documents can contain mixed evidence. Treating the margin band as an explicit
      "unknown" class lets eval measure whether the detector abstains on ambiguous cases
      instead of forcing over-confident binary decisions. This becomes more important when
      we train/evaluate on mixed-origin texts.
    """
    if score >= margin:
        return "AI"
    if score <= -margin:
        return "human"
    return "unknown"


def select_eval_docs(docs: list[dict], sample_size: int, seed: int | None = None) -> list[dict]:
    s = int(seed) if seed is not None else int(getattr(CFG.frozen, "seed", _DEFAULT_REPRO_SEED))
    rng = random.Random(s)
    ai_docs = [d for d in docs if d["label"] == 1]
    human_docs = [d for d in docs if d["label"] == 0]
    rng.shuffle(ai_docs)
    rng.shuffle(human_docs)
    # prefer equal pos/neg so AUROC tracks separation not base rate (pool may still be short one class)
    half = sample_size // 2
    m = min(half, len(ai_docs), len(human_docs))
    chosen = ai_docs[:m] + human_docs[:m]
    if len(chosen) < sample_size:
        remaining = [d for d in docs if d not in chosen]
        rng.shuffle(remaining)
        chosen.extend(remaining[: sample_size - len(chosen)])
    rng.shuffle(chosen)
    return chosen[:sample_size]


async def _sample_standard_rollout(sampling_client, tokenizer, document: str, eval_seed: int, apply_format_fix: bool = False) -> dict:
    prompt_text, prompt_text_formatted = format_prompt_for_model(tokenizer=tokenizer, text=document)
    neutral_prompt_tokens = tokenizer.encode(prompt_text_formatted)

    force_stub = bool(CFG.sampling.force_stub_sampling)
    if force_stub:
        think_already_open = get_think_already_open(tokenizer)
        stub_open, stub_close = _get_analysis_stub_tokens(tokenizer, think_already_open)
        prompt_tokens = neutral_prompt_tokens + stub_open + stub_close
    else:
        prompt_tokens = neutral_prompt_tokens

    # greedy eval: train sampling temps would make AUROC noisy run to run
    _sp = tinker.SamplingParams(
        max_tokens=CFG.sampling.max_tokens,
        seed=eval_seed,
        temperature=0.0,
        top_p=1.0,
        reasoning_effort=CFG.sampling.reasoning_effort,
    )
    sampled = await sampling_client.sample_async(
        prompt=tinker.ModelInput.from_ints(prompt_tokens),
        num_samples=1,
        sampling_params=_sp,
    )
    seq = sampled.sequences[0]
    completion_tokens = list(seq.tokens)
    completion_logprobs = list(seq.logprobs) if seq.logprobs is not None else [0.0] * len(completion_tokens)
    if not any(lp != 0.0 for lp in completion_logprobs):
        logger.warning("eval rollout: all completion_logprobs are 0.0 (logprobs not returned by API)")
    completion_text = tokenizer.decode(completion_tokens, skip_special_tokens=False)
    full_output_text = tokenizer.decode(prompt_tokens + completion_tokens, skip_special_tokens=False)
    response_text = decode_response_text(
        tokenizer=tokenizer,
        completion_tokens=completion_tokens,
        completion_text=completion_text,
        force_stub_sampling=force_stub,
    )
    was_text_fixed = False
    wrong_response_text = None
    if apply_format_fix:
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
        full_output_text = tokenizer.decode(prompt_tokens + completion_tokens, skip_special_tokens=False)
    return {
        "neutral_prompt_text": prompt_text,
        "completion_text": completion_text,
        "full_output_text": full_output_text,
        "response_text": response_text,
        "completion_tokens": completion_tokens,
        "completion_logprobs": completion_logprobs,
        "n_reasoning_tokens": 0,
        "was_text_fixed": was_text_fixed,
        "wrong_response_text": wrong_response_text,
        "main_label_hint": None,
    }


def _trace_sample_standard_rollout(*, trace: EvalRolloutTracePayload) -> dict:
    out = trace_payload_to_weave_dict(trace)
    out.update({
        "eval_seed": int(trace.eval_seed),
        "was_text_fixed": bool(trace.was_text_fixed),
        "format_char_diff": int(trace.format_char_diff),
        "reward": float(trace.reward),
        "agg_score": None if trace.agg_score is None else float(trace.agg_score),
        "advantage": None,
        "indicators": [
            {
                "span_text": ind.get("span_text", ""),
                "explanation": ind.get("explanation", ""),
                "type": ind.get("type"),
                "model_score": float(ind.get("model_score", fs.get("model_score", 0.0)) or 0.0),
                "rubric_credibility": fs.get("rubric_credibility"),
                "rubric_reasoning": fs.get("rubric_reasoning", ""),
            }
            for ind, fs in zip(trace.indicators, trace.tell_scored)
        ],
    })
    return out


async def evaluate_model(
    training_client,
    tokenizer,
    eval_docs: list[dict],
    step: int | str,
    eval_seed: int | None = None,
    eval_audit_path: str | None = None,
) -> dict:
    seed_base = int(eval_seed) if eval_seed is not None else int(getattr(CFG.frozen, "seed", _DEFAULT_REPRO_SEED))
    use_fidelity_gate = bool(CFG.training.use_fidelity_gate)
    fix_err = getattr(CFG.training, "fix_format_errors", False)
    apply_format_fix = bool(fix_err is True or fix_err == "auto")

    logger.info("eval | step %s | evaluating %d test docs with neutral prompt", step, len(eval_docs))
    sampling_client = await training_client.save_weights_and_get_sampling_client_async()

    async def score_eval_doc(doc, doc_idx: int):
        rollout = await _sample_standard_rollout(
            sampling_client, tokenizer, doc["text"], seed_base + doc_idx, apply_format_fix=apply_format_fix
        )
        neutral_prompt_text = rollout["neutral_prompt_text"]
        completion_text = rollout["completion_text"]
        full_output_text = rollout["full_output_text"]
        response_text = rollout["response_text"]
        trace = EvalRolloutTracePayload(
            step=step,
            doc_id=doc.get("id"),
            dataset_id=doc.get("dataset_id", "unknown"),
            domain=doc.get("domain", "unknown"),
            label=doc["label"],
            document=doc["text"],
            eval_seed=seed_base + doc_idx,
            neutral_prompt_text=neutral_prompt_text,
            completion_text=completion_text,
            full_output_text=full_output_text,
            response_text=response_text,
            was_text_fixed=rollout["was_text_fixed"],
            wrong_response_text=rollout["wrong_response_text"],
            format_reason="",
            format_char_diff=0,
            reward=0.0,
            agg_score=None,
            indicators=[],
            tell_scored=[],
        )
        try:
            indicators = parse_indicators(response_text) or []
        except Exception:
            logger.exception("indicator parsing crashed during eval")
            indicators = []
        trace.indicators = indicators
        fmt = _safe_format_diagnostics_for_eval(response_text, doc["text"])
        format_ok = bool(fmt["ok"])
        format_reason = str(fmt["reason"])
        format_char_diff = int(fmt["char_diff_count"])
        trace.format_char_diff = format_char_diff
        if not format_ok and use_fidelity_gate:
            reason = f"format:{format_reason}"
            trace.format_reason = reason
            _trace_sample_standard_rollout(trace=trace)
            return 0.0, True, False, reason, None, doc["label"], format_char_diff, completion_text, response_text, indicators, [], neutral_prompt_text, full_output_text, rollout
        try:
            tell_scored = self_score_from_output(response_text, indicators) or []
        except Exception:
            logger.exception("scorer extraction crashed during eval")
            reason = "scorer_exception"
            trace.format_reason = reason
            _trace_sample_standard_rollout(trace=trace)
            return 0.0, True, False, reason, None, doc["label"], format_char_diff, completion_text, response_text, indicators, [], neutral_prompt_text, full_output_text, rollout
        trace.tell_scored = tell_scored
        try:
            reward = _reward_components(response_text, doc["text"], doc["label"], tell_scored)["cls"]
        except Exception:
            logger.exception("reward computation crashed during eval")
            reward = 0.0
            format_ok = False
            reason = "format:format_exception"
            trace.format_reason = reason
            trace.reward = reward
            _trace_sample_standard_rollout(trace=trace)
            return reward, True, format_ok, reason, None, doc["label"], format_char_diff, completion_text, response_text, indicators, tell_scored, neutral_prompt_text, full_output_text, rollout
        try:
            agg_score = outer_document_score(output=response_text)
        except Exception:
            logger.exception("outer_document_score crashed during eval")
            agg_score = 0.0
            format_ok = False
            reason = "format:outer_score_exception"
            trace.format_reason = reason
            trace.reward = reward
            trace.agg_score = None
            _trace_sample_standard_rollout(trace=trace)
            return reward, True, format_ok, reason, None, doc["label"], format_char_diff, completion_text, response_text, indicators, tell_scored, neutral_prompt_text, full_output_text, rollout
        reason = f"format:{format_reason}" if not format_ok else "ok"
        trace.format_reason = reason
        trace.reward = reward
        trace.agg_score = agg_score
        _trace_sample_standard_rollout(trace=trace)
        return reward, True, format_ok, reason, agg_score, doc["label"], format_char_diff, completion_text, response_text, indicators, tell_scored, neutral_prompt_text, full_output_text, rollout

    _eval_conc = int(getattr(CFG.training, "eval_max_concurrent_rollouts", 64))
    _sem = asyncio.Semaphore(_eval_conc)

    async def _bounded_score_eval_doc(doc, doc_idx: int):
        async with _sem:
            return await score_eval_doc(doc=doc, doc_idx=doc_idx)

    raw_results = await asyncio.gather(*[_bounded_score_eval_doc(doc=doc, doc_idx=i) for i, doc in enumerate(eval_docs)], return_exceptions=True)
    results = []
    for doc, res in zip(eval_docs, raw_results):
        if isinstance(res, Exception):
            logger.error("eval doc scoring crashed; converting to local format failure", exc_info=(type(res), res, res.__traceback__))
            empty_rollout = {"was_text_fixed": False, "wrong_response_text": None}
            results.append((
                0.0, True, False, "format:eval_exception", None, doc["label"], len(doc.get("text", "")),
                "", "", [], [], "", "", empty_rollout,
            ))
            continue
        results.append(res)

    if eval_audit_path:
        pathlib.Path(eval_audit_path).parent.mkdir(parents=True, exist_ok=True)
        with open(eval_audit_path, "a") as f:
            doc_traces = []
            for doc, res in zip(eval_docs, results):
                reward, include, format_ok, reason, agg_score, label, char_diff, completion_text, response_text, indicators, tell_scored, neutral_prompt_text, full_output_text, rollout = res
                doc_traces.append({
                    "doc_id": doc.get("id"),
                    "dataset_id": doc.get("dataset_id", "unknown"),
                    "domain": doc.get("domain", "unknown"),
                    "label": label,
                    "document": doc["text"],
                    "reward": reward,
                    "agg_score": agg_score,
                    "format_reason": reason,
                    "format_char_diff": char_diff,
                    "neutral_prompt_text": neutral_prompt_text,
                    "completion_text": completion_text,
                    "full_output_text": full_output_text,
                    "response_text": response_text,
                    "was_text_fixed": rollout.get("was_text_fixed", False),
                    "wrong_response_text": rollout.get("wrong_response_text"),
                    "indicators": [
                        {
                            "span_text": ind.get("span_text", ""),
                            "explanation": ind.get("explanation", ""),
                            "type": ind.get("type"),
                            "model_score": float(ind.get("model_score", fs.get("model_score", 0.0)) or 0.0),
                            "rubric_credibility": fs.get("rubric_credibility"),
                            "rubric_reasoning": fs.get("rubric_reasoning", ""),
                        }
                        for ind, fs in zip(indicators, tell_scored)
                    ],
                })
            f.write(json.dumps({"step": step, "docs": doc_traces}, ensure_ascii=False) + "\n")

    rewards = [0.0 if res[0] is None else res[0] for res in results if res[1]]
    format_ok_flags = [res[2] for res in results if res[1]]
    n_excluded = sum(1 for res in results if not res[1])
    format_reasons = [res[3] for res in results if res[3] is not None]
    format_char_diffs = [res[6] for res in results]
    eval_format_invalid_type = sum(1 for r in format_reasons if r == "format:invalid_type")
    eval_format_text_mismatch = sum(1 for r in format_reasons if r == "format:text_mismatch")
    eval_text_mismatch_char_diffs = [res[6] for res in results if res[3] == "format:text_mismatch"]
    # Format failures return agg_score=None; substitute 0.0 so every doc always
    # contributes to AUROC. Dropping them would inflate the metric if format failures
    # are correlated with label (e.g. model consistently fails on one class).
    agg_scores = [res[4] if res[4] is not None else 0.0 for res in results if res[1]]
    true_labels = [res[5] for res in results if res[1]]

    eval_reward_mean = (sum(rewards) / len(rewards)) if rewards else 0.0
    eval_format_rate = (sum(1 for ok in format_ok_flags if ok) / len(format_ok_flags)) if format_ok_flags else 0.0

    eval_auroc = roc_auc_score(true_labels, agg_scores) if len(agg_scores) >= 2 else 0.0
    eval_tpr_at_fpr_001 = 0.0
    if len(agg_scores) >= 2:
        fpr, tpr, _ = roc_curve(true_labels, agg_scores)
        tpr_at_or_below = [tpr_i for fpr_i, tpr_i in zip(fpr, tpr) if fpr_i <= 0.01]
        eval_tpr_at_fpr_001 = max(tpr_at_or_below) if tpr_at_or_below else 0.0

    eval_ai_scores = [s for s, y in zip(agg_scores, true_labels) if y == 1]
    eval_human_scores = [s for s, y in zip(agg_scores, true_labels) if y == 0]
    eval_ai_score_mean = (sum(eval_ai_scores) / len(eval_ai_scores)) if eval_ai_scores else 0.0
    eval_human_score_mean = (sum(eval_human_scores) / len(eval_human_scores)) if eval_human_scores else 0.0
    eval_score_gap_ai_minus_human = eval_ai_score_mean - eval_human_score_mean
    eval_ai_positive_rate = (sum(1 for s in eval_ai_scores if s > 0.0) / len(eval_ai_scores)) if eval_ai_scores else 0.0
    eval_human_negative_rate = (sum(1 for s in eval_human_scores if s < 0.0) / len(eval_human_scores)) if eval_human_scores else 0.0
    eval_ambiguous_rate = (sum(1 for s in agg_scores if abs(s) < 0.2) / len(agg_scores)) if agg_scores else 0.0

    # Tri-class F1 using margin_target_positive as an explicit abstain ("unknown") band.
    # Current binary datasets typically have only AI/human labels, but we keep unknown in
    # the label space so this metric is compatible with future mixed-origin datasets.
    margin = float(getattr(CFG.training, "margin_target_positive", getattr(CFG.training, "margin_target", 0.1)))
    pred_tri = [_tri_class_from_score(s, margin) for s in agg_scores]
    true_tri = [
        "AI" if y == 1 else "human" if y == 0 else "unknown"
        for y in true_labels
    ]
    eval_trinary_f1 = (
        f1_score(true_tri, pred_tri, labels=["human", "unknown", "AI"], average="macro", zero_division=0)
        if pred_tri and true_tri else 0.0
    )

    # Binary F1 using verdict type token only (AI/human), not score magnitude.
    _verdict_types = []
    for res in results:
        if not res[1]:
            continue
        response_text = res[8]
        try:
            meta = get_outer_bracket_metadata(response_text)
            vtype = (meta.get("type") or "") if meta else ""
        except Exception:
            vtype = ""
        _verdict_types.append(vtype)
    pred_binary = ["AI" if t == "AI" else "human" for t in _verdict_types]
    true_binary = ["AI" if y == 1 else "human" for y in true_labels]
    eval_binary_f1 = (
        f1_score(true_binary, pred_binary, pos_label="AI", average="binary", zero_division=0)
        if pred_binary else 0.0
    )

    logger.info(
        "eval | step %s | reward=%.3f format=%.2f auroc=%.3f tpr@fpr01=%.3f excluded=%d",
        step,
        eval_reward_mean,
        eval_format_rate,
        eval_auroc,
        eval_tpr_at_fpr_001,
        n_excluded,
    )

    # per-stratum breakdown keyed by (dataset_id, domain) so each stratum contains
    # both AI and human examples, making AUROC meaningful
    stratum_rows: dict[tuple[str, str], dict] = {}
    for doc, res in zip(eval_docs, results):
        reward, _fu, format_ok, reason, agg_score, label, *_ = res
        key = (str(doc.get("dataset_id", "unknown")), str(doc.get("domain", "unknown")))
        if key not in stratum_rows:
            stratum_rows[key] = {"rewards": [], "agg_scores": [], "labels": [], "format_ok": []}
        stratum_rows[key]["rewards"].append(0.0 if reward is None else reward)
        stratum_rows[key]["agg_scores"].append(0.0 if agg_score is None else agg_score)
        stratum_rows[key]["labels"].append(int(label))
        stratum_rows[key]["format_ok"].append(bool(format_ok))
    stratum_stats: dict[str, dict] = {}
    for (ds, dom), s in stratum_rows.items():
        n = len(s["rewards"])
        fmt_rate = sum(s["format_ok"]) / n if n else 0.0
        rw_mean = sum(s["rewards"]) / n if n else 0.0
        agg_mean = sum(s["agg_scores"]) / n if n else 0.0
        try:
            auroc = roc_auc_score(s["labels"], s["agg_scores"]) if len(set(s["labels"])) >= 2 else None
        except Exception:
            auroc = None
        key_str = f"{ds}|{dom}"
        stratum_stats[key_str] = {
            "n": n, "reward_mean": rw_mean, "agg_score_mean": agg_mean,
            "format_rate": fmt_rate, "auroc": auroc,
        }
    if stratum_stats:
        header = f"{'stratum':<50} {'n':>4} {'reward':>8} {'agg_score':>10} {'format':>8} {'auroc':>7}"
        logger.info("eval | step %s | per-stratum breakdown:\n%s\n%s", step, header,
            "\n".join(
                f"  {k:<50} {v['n']:>4} {v['reward_mean']:>8.4f} {v['agg_score_mean']:>10.4f} {v['format_rate']:>8.3f} {v['auroc'] if v['auroc'] is not None else '   n/a':>7}"
                for k, v in sorted(stratum_stats.items(), key=lambda x: x[1].get("reward_mean", 0.0))
            ))

    return {
        "eval_reward_mean": eval_reward_mean,
        "eval_format_rate": eval_format_rate,
        "eval_n_excluded_rollouts": n_excluded,
        "eval_auroc": eval_auroc,
        "eval_tpr_at_fpr_001": eval_tpr_at_fpr_001,
        "eval_ai_score_mean": eval_ai_score_mean,
        "eval_human_score_mean": eval_human_score_mean,
        "eval_score_gap_ai_minus_human": eval_score_gap_ai_minus_human,
        "eval_ai_positive_rate": eval_ai_positive_rate,
        "eval_human_negative_rate": eval_human_negative_rate,
        "eval_ambiguous_rate_abs_lt_02": eval_ambiguous_rate,
        "eval_trinary_f1_macro": eval_trinary_f1,
        "eval_binary_f1": eval_binary_f1,
        "eval_ai_score_p10": _quantile(eval_ai_scores, 0.10),
        "eval_ai_score_p50": _quantile(eval_ai_scores, 0.50),
        "eval_ai_score_p90": _quantile(eval_ai_scores, 0.90),
        "eval_human_score_p10": _quantile(eval_human_scores, 0.10),
        "eval_human_score_p50": _quantile(eval_human_scores, 0.50),
        "eval_human_score_p90": _quantile(eval_human_scores, 0.90),
        "eval_format_invalid_type": eval_format_invalid_type,
        "eval_format_text_mismatch": eval_format_text_mismatch,
        "eval_format_char_diff_mean": (sum(format_char_diffs) / len(format_char_diffs)) if format_char_diffs else 0.0,
        "eval_text_mismatch_char_diff_mean": (sum(eval_text_mismatch_char_diffs) / len(eval_text_mismatch_char_diffs)) if eval_text_mismatch_char_diffs else 0.0,
        "eval_text_mismatch_char_diff_p95": _quantile(eval_text_mismatch_char_diffs, 0.95),
        "eval_text_mismatch_char_diff_max": max(eval_text_mismatch_char_diffs) if eval_text_mismatch_char_diffs else 0,
        "_eval_ai_scores": eval_ai_scores,
        "_eval_human_scores": eval_human_scores,
        "stratum_stats": stratum_stats,
    }
