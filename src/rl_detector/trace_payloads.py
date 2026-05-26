from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

_RUBRIC_TRACE_RAW_MAX = 24000


def rubric_output_for_trace(rubric_output: dict[str, Any] | None) -> dict[str, Any] | None:
    """JSON-safe rubric blob for audit + Weave (truncates huge rubric raw strings)."""
    if rubric_output is None:
        return None
    raw = rubric_output.get("_raw_response") or ""
    if not isinstance(raw, str):
        raw = str(raw)
    anns = rubric_output.get("annotations")
    overall = rubric_output.get("overall")
    reasoning = rubric_output.get("_reasoning") or ""
    if not isinstance(reasoning, str):
        reasoning = str(reasoning)
    out: dict[str, Any] = {
        "n_annotations": len(anns) if isinstance(anns, list) else 0,
        "annotations": list(anns) if isinstance(anns, list) else [],
        "overall": dict(overall) if isinstance(overall, dict) else {},
        "raw_response_len": len(raw),
        "reasoning": reasoning[:_RUBRIC_TRACE_RAW_MAX] if len(reasoning) > _RUBRIC_TRACE_RAW_MAX else reasoning,
    }
    if len(raw) <= _RUBRIC_TRACE_RAW_MAX:
        out["raw_response"] = raw
    else:
        out["raw_response"] = raw[:_RUBRIC_TRACE_RAW_MAX] + "\n...[truncated]"
    return out


@dataclass
class TrainingRolloutTracePayload:
    step: int
    rollout_index: int
    doc_label: int
    noise_mode: str | None
    inject_label: bool
    main_label_hint: int | None
    label_ctx_for_opt: bool
    response_text: str
    wrong_response_text: str | None
    reward: float | None
    reward_components: dict[str, Any]
    advantage: float | None          # None — see component_advantages
    component_advantages: dict[str, float] | None  # per-component GRPO-normalized advantages
    used_for_optimization: bool
    exclude_reason: str
    format_ok: bool
    format_ok_before_fixing: bool
    format_reason: str | None
    format_reason_before_fixing: str | None
    format_char_diff_count: int
    is_ratio: float | None
    from_replay_cache: bool
    document: str
    doc_stratum: str
    neutral_prompt_text: str
    completion_text: str
    full_output_text: str
    raw_response_text: str
    was_text_fixed: bool
    token_surprisal: dict[str, Any]
    ann_token_fraction: float | None
    n_ann_tokens: float | None
    n_response_tokens: float | None
    indicators: list[dict[str, Any]]
    token_optimization_rows: list[dict[str, Any]]  # full sequence: prompt → reasoning → response, each with token_id / decoded_token / token_type / logprob / advantage / optimized
    completion_tokens_len: int
    completion_logprobs_len: int
    n_reasoning_tokens: int
    budget_hit: bool
    repetition_score: float
    why_count: int
    why_mean_len: float
    why_max_len: int
    why_repetition_score: float
    rubric: dict[str, Any] | None


@dataclass
class EvalRolloutTracePayload:
    step: int | str
    doc_id: str | None
    dataset_id: str
    domain: str
    label: int
    document: str
    eval_seed: int
    neutral_prompt_text: str
    completion_text: str
    full_output_text: str
    response_text: str
    was_text_fixed: bool
    wrong_response_text: str | None
    format_reason: str
    format_char_diff: int
    reward: float
    agg_score: float | None
    indicators: list[dict[str, Any]]
    tell_scored: list[dict[str, Any]]


def trace_payload_to_weave_dict(payload: TrainingRolloutTracePayload | EvalRolloutTracePayload) -> dict[str, Any]:
    return asdict(payload)


def training_trace_payload_to_audit_dict(payload: TrainingRolloutTracePayload) -> dict[str, Any]:
    out = asdict(payload)
    out["index"] = out.pop("rollout_index")
    return out
