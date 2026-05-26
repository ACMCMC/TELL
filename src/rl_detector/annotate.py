"""Inference: annotate a document with bracket annotations using a trained checkpoint."""

import asyncio
import datetime
import json
import inspect
import logging

import tinker
from dotenv import load_dotenv
from transformers import AutoTokenizer

from rl_detector.config import CFG
from rl_detector.frozen import aggregate, self_score_from_output
from rl_detector.prompt_utils import format_prompt_for_model, get_think_already_open, load_tokenizer
from rl_detector.annotation_utils import get_outer_bracket_metadata
from rl_detector.tell_xml import wrap_logical_leaf_span
from rl_detector.rewards import parse_indicators
from rl_detector.rollouts import _get_analysis_stub_tokens, decode_response_text
from rl_detector.format_fix import try_fix_response

logger = logging.getLogger(__name__)

load_dotenv()


def render_annotation(document: str, indicators: list[dict], scores: list[float]) -> str:
    """Rebuild with nested <span> leaves (flatten order matches indicator sort)."""
    scored = sorted(
        zip(indicators, scores),
        key=lambda x: document.find(x[0]["span_text"]),
    )
    result = document
    offset = 0
    for ind, score in scored:
        span = ind["span_text"]
        type_val = ind.get("type", "AI")
        piece = wrap_logical_leaf_span(
            span,
            {"type": type_val, "why": ind["explanation"].strip(), "score": f"{score:.2f}"},
        )
        pos = result.find(span, offset)
        if pos == -1:
            continue
        L = len(span)
        result = result[:pos] + piece + result[pos + L :]
        offset = pos + len(piece)
    return result


async def _emit_progress(progress_cb, pct: int, stage: str) -> None:
    """Emit progress event if callback is provided."""
    if progress_cb is None:
        return
    maybe = progress_cb(pct, stage)
    if inspect.isawaitable(maybe):
        await maybe


async def create_runtime(checkpoint_path: str | None = None, base_model: str | None = None) -> dict:
    """Create reusable inference runtime objects for a checkpoint."""
    logger.debug("runtime | initializing")
    service_client = tinker.ServiceClient()
    tokenizer = load_tokenizer()
    
    # load sampling client
    if checkpoint_path:
        weights_path = checkpoint_path.replace("/sampler_weights/", "/weights/")
        sampler_path = weights_path.replace("/weights/", "/sampler_weights/")
        try:
            sampling_client = await service_client.create_sampling_client_async(model_path=sampler_path)
        except Exception:
            training_client = await service_client.create_training_client_from_state_with_optimizer_async(
                path=weights_path
            )
            save_name = f"webui-derived-{datetime.datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
            save_future = await training_client.save_weights_for_sampler_async(name=save_name)
            save_out = await save_future.result_async()
            sampler_path = getattr(save_out, "path", "") or sampler_path
            sampling_client = await service_client.create_sampling_client_async(model_path=sampler_path)
            checkpoint_path = sampler_path
            logger.info(
                "runtime | derived sampler from training weights %s -> %s; "
                "set this in web.checkpoint_path for faster future startup",
                weights_path,
                sampler_path,
            )
    else:
        resolved_base_model = base_model if base_model is not None else CFG.model.base_model
        sampling_client = await service_client.create_sampling_client_async(base_model=resolved_base_model)
    
    # same detect as train / rollouts, web prompt must match sampler weights
    _think_opn = get_think_already_open(tokenizer)
    logger.info("runtime | ready, think_alrdy_open=%s", _think_opn)
    return {
        "checkpoint_path": checkpoint_path,
        "tokenizer": tokenizer,
        "sampling_client": sampling_client,
        "think_already_open": _think_opn,
    }


async def annotate_with_runtime(document: str, runtime: dict, progress_cb=None) -> dict:
    """Run one-rollout annotation pipeline with a preloaded runtime."""
    await _emit_progress(progress_cb, 15, "Preparing input...")
    tokenizer = runtime["tokenizer"]
    sampling_client = runtime["sampling_client"]

    _neutral_txt, _formatted = format_prompt_for_model(tokenizer=tokenizer, text=document)
    _neutral_toks = tokenizer.encode(_formatted)
    _think = runtime.get("think_already_open")
    if _think is None:
        _think = get_think_already_open(tokenizer)
    _force_stub = bool(getattr(CFG.sampling, "force_stub_sampling", False))
    if _force_stub:
        _o, _c = _get_analysis_stub_tokens(tokenizer, _think)
        # web has no label injeciton or focus hint, so empty mid segment like rollouts
        prompt_tokens = _neutral_toks + _o + _c
    else:
        prompt_tokens = _neutral_toks
    logger.debug("annotate | sampling (stub=%s) model output", _force_stub)
    await _emit_progress(progress_cb, 15, "Analyzing text...")
    model_input = tinker.ModelInput.from_ints(prompt_tokens)
    sampled = await sampling_client.sample_async(
        prompt=model_input,
        num_samples=1,
        sampling_params=tinker.SamplingParams(
            max_tokens=CFG.sampling.max_tokens,
            temperature=1.0,
            top_k=2,
            seed=CFG.frozen.seed,
            reasoning_effort=CFG.sampling.reasoning_effort,
        ),
    )
    _comp_toks = list(sampled.sequences[0].tokens)
    _comp_txt = tokenizer.decode(_comp_toks, skip_special_tokens=False)
    response_text = decode_response_text(
        tokenizer, _comp_toks, _comp_txt, _force_stub,
    )
    output = _comp_txt

    max_fix_ratio = float(getattr(CFG.training, "format_fix_max_ratio", 0.50))
    fixed = try_fix_response(response_text=response_text, document=document, max_fix_ratio=max_fix_ratio)
    if fixed is not None:
        logger.info("annotate | format fix applied to malformed response")
        response_text = fixed

    indicators = parse_indicators(response_text, document) or []
    logger.debug("annotate | parsed %d indicators", len(indicators))
    await _emit_progress(progress_cb, 85, "Processing indicators...")
    await _emit_progress(progress_cb, 95, "Finishing up...")
    logger.debug("annotate | scoring indicators (self-score)")
    tell_scored = self_score_from_output(response_text, indicators) or []
    agg = aggregate(tell_scored)
    scores = [s["score"] for s in tell_scored]
    annotated = render_annotation(document, indicators, scores)
    logger.debug("annotate | complete, aggregate_score=%.3f", agg)
    await _emit_progress(progress_cb, 100, "Complete")

    out_meta = get_outer_bracket_metadata(response_text)
    outer_for_ui = None
    if out_meta:
        t = out_meta.get("type")
        mag = float(out_meta.get("score_magnitude", 0.0))
        sgn = mag if t == "AI" else (-mag if t == "human" else 0.0)
        outer_for_ui = {
            "explanation": out_meta.get("explanation", out_meta.get("why", "")),
            "type": t,
            "score_magnitude": mag,
            "signed_score": sgn,
        }

    verdict_outer = None
    score_outer = None
    if outer_for_ui:
        tc = str(outer_for_ui.get("type") or "").strip().lower()
        if tc == "ai":
            verdict_outer = "AI"
            score_outer = float(outer_for_ui["signed_score"])
        elif tc == "human":
            verdict_outer = "Human"
            score_outer = float(outer_for_ui["signed_score"])

    return {
        "aggregate_score": agg,
        "verdict": "AI" if agg > 0 else "Human",
        "verdict_outer": verdict_outer,
        "score_outer": score_outer,
        "model_response": response_text,
        "outer_comment": outer_for_ui,
        "indicators": [
            {
                **ind,
                "model_score": float(ind.get("model_score", fs.get("model_score", 0.0)) or 0.0),
                "signed_score": float(fs.get("score", 0.0) or 0.0),
                "rubric_credibility": fs.get("rubric_credibility"),
            }
            for ind, fs in zip(indicators, tell_scored)
        ],
        "annotated_text": annotated,
    }


async def annotate(
    document: str,
    checkpoint_path: str | None = None,
    progress_cb=None,
) -> dict:
    logger.debug("annotate | starting annotation run")
    await _emit_progress(progress_cb, 5, "Starting pipeline...")
    if checkpoint_path:
        logger.debug("annotate | loading checkpoint: %s", checkpoint_path)
        await _emit_progress(progress_cb, 15, "Loading checkpoint")
    else:
        logger.debug("annotate | no checkpoint provided, using base LoRA client")
        await _emit_progress(progress_cb, 15, "Creating base client")

    runtime = await create_runtime(checkpoint_path=checkpoint_path)
    return await annotate_with_runtime(document, runtime, progress_cb=progress_cb)


if __name__ == "__main__":
    import sys
    text = sys.stdin.read()
    checkpoint = sys.argv[1] if len(sys.argv) > 1 else None
    result = asyncio.run(annotate(text, checkpoint))
    print(json.dumps(result, indent=2))
