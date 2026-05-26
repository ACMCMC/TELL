"""Frozen model scoring via xAI (Grok), OpenAI, or DeepInfra."""

import asyncio
import json
import logging
import math
import os
import re
import warnings

warnings.filterwarnings("ignore", message="Pydantic serializer warnings", category=UserWarning)

from openai import APIConnectionError, AsyncOpenAI, InternalServerError, LengthFinishReasonError, PermissionDeniedError, RateLimitError
from pydantic import BaseModel, Field, ValidationError

from rl_detector.annotation_utils import collect_bracket_tells
from rl_detector.config import CFG

logger = logging.getLogger(__name__)

# global semaphore shared across all docs and rollouts
_SEMAPHORE: asyncio.Semaphore | None = None

def _extract_scored_tells(text: str) -> list[dict]:
    """Parse bracket annotations and extract explanation/span/score.

    Used by self_score_from_output.
    """
    tells = collect_bracket_tells(text)
    if tells:
        return [
            {
                "span_text": t["span_text"],
                "explanation": t["explanation"],
                "type": t.get("type"),
                "score_raw": t.get("score"),
                "score_present": bool(t.get("score_present", t.get("score") is not None)),
            }
            for t in tells
        ]
    return []


def _semaphore() -> asyncio.Semaphore:
    global _SEMAPHORE
    if _SEMAPHORE is None:
        _SEMAPHORE = asyncio.Semaphore(CFG.frozen.max_concurrent)
    return _SEMAPHORE


def get_client() -> AsyncOpenAI:
    provider = getattr(CFG.frozen, "provider", "xai").lower()
    if provider == "xai":
        return AsyncOpenAI(
            api_key=os.environ["XAI_API_KEY"],
            base_url="https://api.x.ai/v1",
            timeout=3600.0,
        )
    if provider == "openai":
        return AsyncOpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            timeout=3600.0,
        )
    if provider == "deepinfra":
        return AsyncOpenAI(
            api_key=os.environ["DEEPINFRA_API_KEY"],
            base_url="https://api.deepinfra.com/v1/openai",
        )
    raise ValueError(f"Unknown rubric provider: {provider!r}. Use 'xai', 'openai', or 'deepinfra'.")


def self_score_from_output(tagged_text: str, indicators: list[dict]) -> list[dict] | None:
    """
    Extract self-assigned scores from the policy model's own output.

    The model is prompted to add score="FLOAT" (0.0–1.0) to each tell; those values
    flow through aggregate() → reward unchanged.
    Falls back to a neutral signed score (0.0) for any tell that omits the attribute.
    """
    if not indicators:
        return []

    scored_tells = _extract_scored_tells(tagged_text)

    score_pool: dict[tuple[str, str, str], list[float]] = {}
    strength_pool: dict[tuple[str, str, str], list[float]] = {}
    n_missing = 0
    for tell in scored_tells:
        raw = tell.get("score_raw") if tell.get("score_present", True) else None
        try:
            strength = max(0.0, min(1.0, float(raw))) if raw is not None else None
        except (ValueError, TypeError):
            strength = None
        if strength is None:
            n_missing += 1
            strength = 0.0  # neutral fallback when model omits the score attribute
        tell_type = tell.get("type")
        signed = strength if tell_type == "AI" else (-strength if tell_type == "human" else 0.0)
        if tell.get("id"):
            key = ("id", str(tell.get("id")), tell["explanation"])
        else:
            key = ("span", tell["span_text"], tell["explanation"])
        score_pool.setdefault(key, []).append(signed)
        strength_pool.setdefault(key, []).append(strength)

    if n_missing:
        logger.debug("self_score | %d/%d tells missing score attribute; defaulting to neutral 0.0", n_missing, len(scored_tells))

    scores: list[float] = []
    strengths: list[float] = []
    for ind in indicators:
        if ind.get("id"):
            key = ("id", str(ind.get("id")), ind.get("explanation", ""))
        else:
            key = ("span", ind["span_text"], ind.get("explanation", ""))
        bucket = score_pool.get(key)
        scores.append(bucket.pop(0) if bucket else 0.0)
        sbucket = strength_pool.get(key)
        strengths.append(sbucket.pop(0) if sbucket else 0.0)

    return [
        {
            "score": s,
            "rubric_credibility": strength,
            "model_score": strength,
            "type": ind.get("type"),
            "id": ind.get("id"),
            "span_text": ind.get("span_text", ""),
            "span_start": ind.get("span_start"),
            "span_end": ind.get("span_end"),
            "explanation": ind.get("explanation", ""),
            "rubric_reasoning": "",
            "rubric_response": "",
        }
        for s, strength, ind in zip(scores, strengths, indicators)
    ]


def aggregate(scored: list[dict]) -> float:
    """
    Softmax-weighted mean of scores, where weights are exp(beta * |score|).
    Strong tells (scores near ±1) dominate; weak tells (near 0) contribute little.
    With beta=0 this reduces to a plain mean. Beta is set by agg_softmax_beta in config.
    """
    if not scored:
        return 0.0
    beta = float(getattr(CFG.training, "agg_softmax_beta", 3.0))
    scores = [s["score"] for s in scored]
    if beta <= 0:
        return sum(scores) / len(scores)
    weights = [math.exp(beta * abs(s)) for s in scores]
    total_w = sum(weights)
    return sum(w * s for w, s in zip(weights, scores)) / total_w


# ---------------------------------------------------------------------------
# Rubric-based evaluation (external LLM; use_rubric_scorer=True)
# ---------------------------------------------------------------------------

_ANN_DIMS = ("credibility",)
_OV_DIMS = ("credibility",)
_JSON_ANN_START_RE = re.compile(r'\{\s*"ann"\s*:', re.IGNORECASE)


class _RubricOutput(BaseModel):
    ann: list[float] = Field(description="Credibility score 0.0-1.0 for each annotation in input order; empty list if no annotations")
    overall: float = Field(description="Overall verdict credibility 0.0-1.0")


def _parse_rubric_json(text: str, n: int) -> dict | None:
    """Extract and validate rubric JSON from rubric model response.

    Expected compact format: {"ann":[c1,...,cN],"overall":c}
    Handles reasoning model output (preamble before the JSON) by scanning from
    the last occurrence of {"ann": forward and counting braces.
    Returns the parsed dict with all floats clamped to [0, 1], or None on failure.
    """
    starts = [m.start() for m in _JSON_ANN_START_RE.finditer(text)]
    for start in reversed(starts):
        depth = 0
        end = start
        for i, ch in enumerate(text[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        else:
            continue
        try:
            parsed = json.loads(text[start:end])
        except json.JSONDecodeError:
            continue
        anns = parsed.get("ann")
        overall = parsed.get("overall")
        if not isinstance(anns, list) or len(anns) != n:
            logger.debug("rubric | annotation count mismatch: got %d, expected %d", len(anns) if isinstance(anns, list) else -1, n)
            continue
        if not isinstance(overall, (int, float)):
            logger.debug("rubric | missing or invalid 'overall' key in rubric response")
            continue
        ann_creds = [max(0.0, min(1.0, float(v))) for v in anns]
        overall_cred = max(0.0, min(1.0, float(overall)))
        return {
            "annotations": [{"credibility": c} for c in ann_creds],
            "overall": {"credibility": overall_cred},
        }
    return None


async def rubric_evaluate(
    client: AsyncOpenAI,
    tagged_text: str,
    indicators: list[dict],
) -> dict | None:
    """Evaluate all annotations with a rubric in one external model call.

    Returns {"annotations": [{credibility}, ...],
             "overall": {credibility}}
    or None if the response cannot be parsed (rollout will be excluded).
    When indicators is empty, still calls the rubric to score the outer verdict's why=
    explanation. The prompt shows no <annotation id="N"> elements so the model returns
    ann=[] naturally. We use the overall credibility and ignore ann for n=0.
    """
    from rl_detector.prompts import build_rubric_prompt

    n = len(indicators)
    prompt = build_rubric_prompt(tagged_text, n=n)

    sem = _semaphore()
    in_use_before = CFG.frozen.max_concurrent - sem._value
    logger.debug("rubric | waiting semaphore slot (%d tells, in_use=%d/%d)", n, in_use_before, CFG.frozen.max_concurrent)
    async with sem:
        in_use_after = CFG.frozen.max_concurrent - sem._value
        logger.debug("rubric | acquired semaphore slot (%d tells, in_use=%d/%d)", n, in_use_after, CFG.frozen.max_concurrent)
        _MAX_RETRIES = 10
        _BASE_DELAY = 2.0
        response = None
        for attempt in range(_MAX_RETRIES):
            try:
                provider = getattr(CFG.frozen, "provider", "xai").lower()
                extra = {}
                if provider == "deepinfra":
                    extra["reasoning_effort"] = CFG.frozen.reasoning_effort
                if provider == "openai":
                    extra["max_completion_tokens"] = CFG.frozen.max_tokens
                else:
                    extra["max_tokens"] = CFG.frozen.max_tokens
                _temp = float(getattr(CFG.frozen, "temperature", 0.7))
                _top_p = float(getattr(CFG.frozen, "top_p", 0.95))
                response = await client.beta.chat.completions.parse(
                    model=CFG.frozen.model,
                    messages=[{"role": "user", "content": prompt}],
                    seed=CFG.frozen.seed,
                    temperature=_temp,
                    top_p=_top_p,
                    response_format=_RubricOutput,
                    **extra,
                )
                break
            except PermissionDeniedError as e:
                logger.warning("rubric | permission denied (non-retryable); skipping rubric: %s", e)
                return None
            except (LengthFinishReasonError, ValidationError, json.JSONDecodeError) as e:
                # Truncated response: max_tokens too low, JSON cut off mid-stream.
                # Return None so the caller counts this as rubric_parse_failed (rollout
                # still trains without rubric signal) instead of scorer_exception (excluded).
                raw_content = None
                try:
                    # LengthFinishReasonError attaches the completion object
                    raw_content = e.completion.choices[0].message.content  # type: ignore[union-attr]
                except Exception:
                    pass
                if raw_content is None:
                    try:
                        # ValidationError: the raw text may be in args
                        raw_content = str(e.args[0])[:600] if e.args else None
                    except Exception:
                        pass
                logger.warning("rubric | response parse error (%d tells) — %s\n  raw: %r", n, type(e).__name__, raw_content)
            except (RateLimitError, InternalServerError, APIConnectionError) as e:
                delay = _BASE_DELAY * (2 ** attempt)
                logger.warning("rubric | transient error (attempt %d/%d), retrying in %.1fs: %s", attempt + 1, _MAX_RETRIES, delay, e)
                if attempt < _MAX_RETRIES - 1:
                    await asyncio.sleep(delay)
        if response is None:
            logger.error("rubric | all %d retries exhausted; returning None (rubric skipped)", _MAX_RETRIES)
            return None
    logger.debug("rubric | released semaphore slot (%d tells)", n)
    msg = response.choices[0].message
    parsed: _RubricOutput | None = getattr(msg, "parsed", None)
    content = msg.content or ""
    reasoning_content = getattr(msg, "reasoning_content", None) or ""
    if parsed is None:
        # Structured output refusal or provider doesn't support .parsed — fall back to manual parse
        logger.debug("rubric model raw response (fallback parse): %r", content)
        result = _parse_rubric_json(content, n)
        if result is None:
            logger.warning("rubric | parse failed (expected %d annotations); raw response: %r", n, content[:600])
            return None
    else:
        if n == 0:
            # Zero-annotation rollout: ignore parsed.ann (model may return [] or hallucinate some).
            # Only the overall verdict score matters here.
            overall_cred = max(0.0, min(1.0, parsed.overall))
            result = {
                "annotations": [],
                "overall": {"credibility": overall_cred},
                "_zero_ann_skip": True,
            }
        elif len(parsed.ann) == n + 1:
            # Model stuffed the overall verdict score into ann as the last element.
            # Recover: use ann[:n] for annotations, keep parsed.overall for the verdict.
            logger.debug("rubric | off-by-one recovery: got %d ann scores, expected %d; dropping last", len(parsed.ann), n)
            ann_creds = [max(0.0, min(1.0, float(a))) for a in parsed.ann[:n]]
            overall_cred = max(0.0, min(1.0, parsed.overall))
            result = {
                "annotations": [{"credibility": c} for c in ann_creds],
                "overall": {"credibility": overall_cred},
            }
        elif len(parsed.ann) != n:
            logger.warning("rubric | annotation count mismatch: got %d, expected %d; ann=%r; raw: %r", len(parsed.ann), n, parsed.ann, content[:400])
            return None
        else:
            ann_creds = [max(0.0, min(1.0, float(a))) for a in parsed.ann]
            overall_cred = max(0.0, min(1.0, parsed.overall))
            result = {
                "annotations": [{"credibility": c} for c in ann_creds],
                "overall": {"credibility": overall_cred},
            }
    result["_raw_response"] = content
    result["_reasoning"] = reasoning_content
    return result


def rubric_aggregate(rubric_output: dict) -> float:
    """Compute a single 0-1 quality score from rubric output.

    Blends the mean per-annotation score with the rollout-level overall score.
    Weights are configurable via rubric_per_ann_weight / rubric_rollout_weight in training config.
    """
    per_ann_w = float(getattr(CFG.training, "rubric_per_ann_weight", 0.7))
    rollout_w = float(getattr(CFG.training, "rubric_rollout_weight", 0.3))
    anns = rubric_output.get("annotations", [])
    if anns:
        ann_mean = sum(ann.get("credibility", 0.0) for ann in anns) / len(anns)
    else:
        ann_mean = 0.0
    overall = rubric_output.get("overall", {})
    rollout_score = float(overall.get("credibility", 0.0))
    total_w = per_ann_w + rollout_w
    return (per_ann_w * ann_mean + rollout_w * rollout_score) / max(1e-8, total_w)


def rubric_to_tell_scored(rubric_output: dict, indicators: list[dict]) -> list[dict]:
    """Convert rubric annotation scores to the tell_scored list format.

    Preserves the signed score= convention (AI→positive, human→negative) so that
    existing per_tell_reward and aggregate() callers work unchanged.
    """
    anns = rubric_output.get("annotations", [])
    raw_response = rubric_output.get("_raw_response", "")
    reasoning = rubric_output.get("_reasoning", "")
    result = []
    for i, ind in enumerate(indicators):
        ann = anns[i] if i < len(anns) else None
        # rubric_credibility is None when rubric returned fewer annotations than indicators.
        # None excludes the tell from per-tell advantage normalization (treated as missing signal)
        # rather than polluting the pool with an arbitrary 0.5 neutral value.
        # Empty span_text or empty explanation → credibility 0: nothing to evaluate.
        explanation = ind.get("explanation", "") or ""
        span_text = ind.get("span_text", "") or ""
        if ann is not None and (not explanation.strip() or not span_text.strip()):
            credibility = 0.0
        else:
            credibility = float(ann.get("credibility", 0.5)) if ann is not None else None
        score_for_gradient = float(credibility) if credibility is not None else 0.5
        typ = ind.get("type")
        signed = score_for_gradient if typ == "AI" else (-score_for_gradient if typ == "human" else 0.0)
        result.append({
            "score": signed,
            "rubric_credibility": credibility,
            "model_score": float(ind.get("model_score", 0.0) or 0.0),
            "type": typ,
            "span_text": ind.get("span_text", ""),
            "explanation": ind.get("explanation", ""),
            "rubric_reasoning": reasoning,
            "rubric_response": raw_response,
        })
    return result
