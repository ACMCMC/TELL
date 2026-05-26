"""Generate paired SFT audit data from pangram/editlens_iclr.

Design constraints:
- one generation prompt per document (annotation + global comment together)
- strict formatting acceptance (must strip back to exact source text)
- progress tracking via tqdm
- length filtering by token-length Q3 (drop longest quartile)
- reproducibility artifacts (manifest + per-doc audit log)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import hydra
from tqdm.auto import tqdm

from datasets import load_dataset
from openai import AsyncOpenAI
from omegaconf import DictConfig, OmegaConf
from dotenv import load_dotenv
import rl_detector.config as config_module
from rl_detector.sft.annotated_sft_prompt_examples import PROMPT_EXAMPLES, style_example_for_label
from rl_detector.config import CFG
from rl_detector.prompt_utils import load_tokenizer
from rl_detector.data import clean_document_text, truncate_document_text
from rl_detector.rewards import format_diagnostics
from rl_detector.tell_xml import escape_document_piece, root_splits

logger = logging.getLogger(__name__)


def _json_safe(obj: Any) -> Any:
    return json.loads(json.dumps(obj, default=str))


def _response_model_dump_json(response: Any) -> dict[str, Any]:
    if hasattr(response, "model_dump"):
        try:
            dumped = response.model_dump(mode="json")
        except TypeError:
            dumped = response.model_dump()
        return json.loads(json.dumps(dumped, default=str))
    return json.loads(json.dumps(response, default=str))


def _responses_extract_output_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if isinstance(text, str):
        return text
    d = _response_model_dump_json(response)
    output = d.get("output", [])
    joined_parts: list[str] = []
    for item in output if isinstance(output, list) else []:
        if not isinstance(item, dict):
            continue
        for content_item in item.get("content", []) if isinstance(item.get("content"), list) else []:
            if isinstance(content_item, dict) and content_item.get("type") in {"output_text", "text"}:
                text_part = content_item.get("text")
                if isinstance(text_part, str):
                    joined_parts.append(text_part)
    return "".join(joined_parts)


def _responses_reasoning_trace_text(api_response: dict[str, Any]) -> str | None:
    parts: list[str] = []
    for item in api_response.get("output") or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") not in {"reasoning", "reasoning_item"}:
            continue
        for c in item.get("content") or []:
            if isinstance(c, str) and c:
                parts.append(c)
                continue
            if not isinstance(c, dict):
                continue
            for k in ("text", "reasoning", "summary"):
                v = c.get(k)
                if isinstance(v, str) and v:
                    parts.append(v)
    joined = "".join(parts).strip()
    return joined if joined else None


_DATASET_ID = "pangram/editlens_iclr"
_SEED = 2242
_PREFETCH_FACTOR = 2.5


@dataclass
class PairDoc:
    source_id: str
    source: str
    ai_edited: dict[str, Any]
    source_text: str


def _build_client_for_model(model: str) -> AsyncOpenAI:
    load_dotenv()
    if model.startswith("gpt-"):
        return AsyncOpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            timeout=3600.0,
            max_retries=0,
        )
    return AsyncOpenAI(
        api_key=os.environ["XAI_API_KEY"],
        base_url="https://api.x.ai/v1",
        timeout=3600.0,
        max_retries=0,
    )


def _truncate(tokenizer, text: str, max_doc_tokens: int) -> str:
    cleaned = clean_document_text(text)
    return truncate_document_text(tokenizer=tokenizer, text=cleaned, max_doc_tokens=max_doc_tokens)


def _collect_pairs(split: str, scan_limit: int, max_pairs: int | None = None) -> tuple[list[PairDoc], int]:
    ds = load_dataset(_DATASET_ID, split=split, streaming=True)
    pairs: list[PairDoc] = []
    rows_scanned = 0

    for i, row in enumerate(tqdm(ds, total=scan_limit, desc="scan_rows", unit="row")):
        if i >= scan_limit:
            break
        rows_scanned += 1
        sid = str(row.get("source_id") or "").strip()
        text_type = str(row.get("text_type") or "").strip()
        source_text = clean_document_text(str(row.get("source_text") or ""))
        if not sid or text_type != "ai_edited" or not source_text:
            continue

        pairs.append(
            PairDoc(
                source_id=sid,
                source=str(row.get("source") or "unknown"),
                ai_edited=row,
                source_text=source_text,
            )
        )
        if max_pairs is not None and len(pairs) >= max_pairs:
            break
    pairs.sort(key=lambda x: (x.source, x.source_id))
    return pairs, rows_scanned


def _q3(values: list[int]) -> int:
    if not values:
        return 0
    xs = sorted(values)
    idx = int(0.75 * (len(xs) - 1))
    return int(xs[idx])


def _token_len(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text=text, add_special_tokens=False))


def _filter_pairs_by_q3(tokenizer, pairs: list[PairDoc]) -> tuple[list[PairDoc], dict[str, int]]:
    lengths: list[int] = []
    pair_len: dict[str, int] = {}
    for pair in pairs:
        h = clean_document_text(pair.source_text)
        a = clean_document_text(str(pair.ai_edited.get("text") or ""))
        if not h or not a:
            continue
        m = max(_token_len(tokenizer, h), _token_len(tokenizer, a))
        pair_len[pair.source_id] = m
        lengths.append(m)

    q3 = _q3(lengths)
    kept = [p for p in pairs if pair_len.get(p.source_id, 10**9) <= q3]
    stats = {
        "q3_max_pair_tokens": q3,
        "num_source_pairs_before": len(pairs),
        "num_source_pairs_after": len(kept),
    }
    return kept, stats


def _load_selected_example_ids(path_str: str | None) -> list[str] | None:
    if not path_str:
        return None
    path = Path(path_str)
    data = json.loads(path.read_text(encoding="utf-8"))
    picked = data.get("selected_example_ids")
    if not isinstance(picked, list) or not picked:
        raise ValueError(f"manifest missing selected_example_ids: {path}")
    out: list[str] = []
    for item in picked:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"bad selected_example_id in manifest: {path}")
        out.append(item.strip())
    return out


def _paired_annotation_prompt(human_text: str, ai_text: str, target_label: str, style_example_index: int) -> str:
    return f"""You are an annotator of AI or human tells. You have a target text in front of you, to annotate it with tells to say why it looks like it was written by either AI or a human. 

Use this exact compact format:
<span>ANNOTATED_TEXT<annotation type="LABEL" why="EXPLANATION" score="FLOAT" /></span>

Important:
1. Copy the target text exactly in ANNOTATED_TEXT after XML decoding. Do not fix typos, spacing, punctuation, Unicode, casing, or grammar. In the XML output, text runs inside spans must use the same XML escaping as the target text
2. label must be exactly "AI" or "human"
3. score must be 0.0 to 1.0 and indicate how much that exact tell should move the document decision. Use the full range: 0.0-0.25 for weak hints, 0.35-0.65 for moderate evidence, and 0.75-1.0 only for undeniable evidence. Try to have a varied range of scores. For the outer annotation, pick a score that makes sense based on the tells you found in the text.
4. Wrap the whole target text in one outer annotation too. The output must start with <span> and end with </span>, with the outer <annotation ... /> immediately before the final </span>
5. Try to be as granular as possible; it’s better to keep spans small, e.g., annotate a specific character instead of a whole word or phrase
6. The explanations must be detailed and explicitly explain why the span is a tell for the given label, by explaining the mechanism that leads to the tell, you should teach the reader your reasoning process
7. Use the reference text to help spot differences and clues, but you mustn’t directly compare the target text to the reference text in your annotations, you CAN’T MENTION IT EXISTS but you can quote things from the reference text as “a human/AI might say e.g. …”, because the annotations should be valid even if you ONLY saw the target text alone
8. Think like a detective: consider the writer’s intention and context, look for subtle clues in style, content, formatting, semantics, grammar, and vocabulary, flow and inconsistencies
9. Pay close attention to the writing style of the why="EXPLANATION" in the examples. YOU SHOULD USE THE SAME WRITING STYLE as the explanations, thinking out loud and from your perspective ("I guess", "maybe", "this doesn’t make sense", "I think", …), honest, simple English, with a 80-90 Flesch score. However, do not copy the content, exact clues, or topic since that will be different for each input. Try to be creative.
10. Keep annotations balanced. All texts contain both AI and human tells. Make sure the majority of the tells support the known label, but include 20-40% of the opposite label tells as well. This helps to keep your annotation nuanced and credible, and prevents it from being too one-sided

{style_example_for_label(target_label=target_label, style_example_index=style_example_index)}

Here is the real pair to annotate.

Human:
<<<
{escape_document_piece(human_text)}
>>>

AI:
<<<
{escape_document_piece(ai_text)}
>>>

Annotate only the {target_label} text. The other text is secret context to help you notice differences and possible tells.

Now output exactly this structure:
<span>ANNOTATED TARGET TEXT<annotation type="LABEL" why="ONE SHORT GLOBAL COMMENT" score="FLOAT" /></span>"""


def _extract_blocks(raw: str) -> tuple[str | None, str | None]:
    annotation = raw.strip()
    _inner, _tells, meta, ok, end_pos = root_splits(annotation)
    if not ok or meta is None or end_pos != len(annotation):
        return None, None
    return meta.get("why", "").strip(), annotation


def _xml_inplace_format_diagnostics(output: str, document: str) -> dict[str, int | str | bool]:
    diag = format_diagnostics(output, document)
    if not bool(diag.get("ok", False)):
        return diag
    _inner, tells, _meta, _ok, _end_pos = root_splits(output)
    if len(tells) == 0:
        return {"ok": False, "reason": "no_local_tells", "char_diff_count": 0}
    return diag


def _comment_is_standalone(text: str) -> bool:
    low = text.lower()
    banned = [
        "compared to",
        "compared with",
        "original version",
        "source version",
        "human version",
        "ai version",
        "reference text",
    ]
    return all(x not in low for x in banned)


def _record_failure(
    *,
    example_id: str,
    source_id: str,
    text_type: str,
    label: int,
    raw: str,
    prompt: str,
    reason: str,
    ok: bool,
    diag: dict[str, Any] | None,
    api_request: dict[str, Any] | None = None,
    api_response: dict[str, Any] | None = None,
    reasoning_trace_text: str | None = None,
) -> dict[str, Any]:
    payload = {
        "example_id": example_id,
        "source_id": source_id,
        "text_type": text_type,
        "label": label,
        "ok": ok,
        "reason": reason,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "response_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        "input_prompt": prompt,
        "model_output_text": raw,
        "raw_head": raw[:500],
    }
    if api_request is not None:
        payload["api_request"] = api_request
    if api_response is not None:
        payload["api_response"] = api_response
    rt = reasoning_trace_text
    if rt is None and api_response is not None:
        rt = _responses_reasoning_trace_text(api_response=api_response)
    if rt:
        payload["reasoning_trace_text"] = rt
    if diag is not None:
        payload.update({
            "format_ok": bool(diag.get("ok", False)),
            "format_reason": str(diag.get("reason", "unknown")),
            "char_diff_count": int(diag.get("char_diff_count", -1)),
        })
    return payload


def _responses_model_supports_reasoning_effort(model: str) -> bool:
    name = model.lower().split(":")[0]
    return name.startswith("gpt-5") or name.startswith("o1-") or name.startswith("o3-") or name.startswith("o4-")


async def _chat_once(
    client: AsyncOpenAI,
    *,
    model: str,
    prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int | None,
    service_tier: str | None,
    reasoning_effort: str,
) -> tuple[str, dict[str, Any]]:
    """Return stripped text for parsing, plus metadata including full API request/response dumps."""
    meta: dict[str, Any] = {"generation_model": model, "provider": None}
    if model.startswith("gpt-"):
        meta["provider"] = "openai.responses"
        req: dict[str, Any] = {
            "model": model,
            "instructions": "Return plain text only; do not call tools; do not include markdown fences unless explicitly requested.",
            "input": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "store": False,
        }
        if _responses_model_supports_reasoning_effort(model=model):
            req["reasoning"] = {"effort": reasoning_effort}
        if max_tokens is not None and int(max_tokens) > 0:
            req["max_output_tokens"] = int(max_tokens)
        if service_tier:
            req["service_tier"] = service_tier
        meta["api_request"] = _json_safe(req)
        response = await _call_with_rate_limit_backoff(
            create_call=lambda: client.responses.create(
                **req,
            ),
            model=model,
            provider="openai.responses",
        )
        api_response = _response_model_dump_json(response)
        meta["api_response"] = api_response
        exact = _responses_extract_output_text(response=response)
        meta["model_output_text_exact"] = exact
        refusal = api_response.get("refusal")
        if isinstance(refusal, str) and refusal.strip():
            raise _AbortGeneration(f"model_refusal:{refusal.strip()[:200]}")
        stripped = exact.strip()
        if not stripped:
            raise _AbortGeneration("empty_model_output:responses_output_empty")
        rt = _responses_reasoning_trace_text(api_response=api_response)
        if rt is not None:
            meta["reasoning_trace_text"] = rt
        return stripped, meta

    meta["provider"] = "chat.completions"
    req_chat: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "top_p": top_p,
    }
    if max_tokens is not None and int(max_tokens) > 0:
        req_chat["max_tokens"] = int(max_tokens)
    meta["api_request"] = _json_safe(req_chat)
    response = await _call_with_rate_limit_backoff(
        create_call=lambda: client.chat.completions.create(**req_chat),
        model=model,
        provider="chat.completions",
    )
    meta["api_response"] = _response_model_dump_json(response=response)
    d = meta["api_response"]
    msg = None
    try:
        msg = d.get("choices", [])[0].get("message")
    except Exception:
        msg = None

    message = response.choices[0].message
    raw_content = (msg or {}).get("content") if isinstance(msg, dict) else getattr(message, "content", None)
    exact = _extract_content_text(content=raw_content)
    meta["model_output_text_exact"] = exact
    content = exact.strip()
    if content:
        return content, meta
    refusal = (msg or {}).get("refusal") if isinstance(msg, dict) else getattr(message, "refusal", None)
    if isinstance(refusal, str) and refusal.strip():
        raise _AbortGeneration(f"model_refusal:{refusal.strip()[:200]}")
    rc = (msg or {}).get("reasoning_content") if isinstance(msg, dict) else getattr(message, "reasoning_content", None)
    if isinstance(rc, str):
        if rc.strip():
            meta["model_output_text_exact"] = rc
            return rc.strip(), meta
        raise _AbortGeneration("empty_model_output:reasoning_content_empty")
    if isinstance(rc, list):
        parts: list[str] = []
        for item in rc:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
                continue
            t = getattr(item, "text", None)
            if isinstance(t, str):
                parts.append(t)
                continue
        joined = "".join(parts)
        meta["model_output_text_exact"] = joined
        joined_stripped = joined.strip()
        if joined_stripped:
            return joined_stripped, meta
        raise _AbortGeneration("empty_model_output:reasoning_content_list_empty")
    raise _AbortGeneration("empty_model_output:missing_content")


def _is_api_client_call_error(exc: BaseException) -> bool:
    # Abort on transport / auth / rate-limit / provider-side API errors.
    # Keep this heuristic broad so we don't accidentally keep hammering on transient provider errors.
    if isinstance(exc, _AbortGeneration):
        return True
    mod = getattr(exc.__class__, "__module__", "") or ""
    name = getattr(exc.__class__, "__name__", "") or ""
    if mod.startswith("openai"):
        return True
    if mod.startswith("httpx") or mod.startswith("httpcore"):
        return True
    if "RateLimit" in name or "APIError" in name or "Timeout" in name or "Connection" in name:
        return True
    return False


def _exception_http_status(exc: BaseException) -> int | None:
    sc = getattr(exc, "status_code", None)
    if isinstance(sc, int):
        return sc
    response = getattr(exc, "response", None)
    if response is not None:
        rsc = getattr(response, "status_code", None)
        if isinstance(rsc, int):
            return rsc
    return None


def _retry_after_seconds(exc: BaseException) -> float | None:
    response = getattr(exc, "response", None)
    if response is not None:
        headers = getattr(response, "headers", None)
        if headers is not None:
            ra = headers.get("retry-after") or headers.get("Retry-After")
            if ra is not None:
                try:
                    return float(ra)
                except ValueError:
                    pass
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        raw = body.get("retry_after")
        if isinstance(raw, (int, float)):
            return float(raw)
        if isinstance(raw, str):
            try:
                return float(raw)
            except ValueError:
                return None
    return None


# Transient overloads / gateways: same backoff policy as flex 429 bursts.
_RETRYABLE_HTTP_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


def _is_rate_limit_error(exc: BaseException) -> bool:
    status_code = _exception_http_status(exc=exc)
    if status_code == 429:
        return True
    name = getattr(exc.__class__, "__name__", "") or ""
    if "RateLimit" in name:
        return True
    text = str(exc).lower()
    if "429" in text and "rate" in text and "limit" in text:
        return True
    return False


def _is_transient_retryable_http_error(exc: BaseException) -> bool:
    if _is_rate_limit_error(exc=exc):
        return True
    code = _exception_http_status(exc=exc)
    if code is not None and code in _RETRYABLE_HTTP_STATUS_CODES:
        return True
    return False


async def _call_with_rate_limit_backoff(create_call, *, model: str, provider: str):
    delay_s = 1.0
    attempt = 0
    while True:
        try:
            return await create_call()
        except Exception as exc:
            if not _is_transient_retryable_http_error(exc=exc):
                raise
            attempt += 1
            status = _exception_http_status(exc=exc)
            ra = _retry_after_seconds(exc=exc)
            sleep_s = delay_s
            if ra is not None:
                sleep_s = max(sleep_s, min(float(ra), 600.0))
            logger.warning(
                "Transient API error from %s model=%s status=%s attempt=%d, backing off %.1fs",
                provider,
                model,
                status,
                attempt,
                sleep_s,
            )
            await asyncio.sleep(sleep_s)
            delay_s = min(delay_s * 2.0, 120.0)


def _is_invalid_prompt_policy_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    if "invalid_prompt" in text:
        return True
    if "flagged as potentially violating our usage policy" in text:
        return True
    return False


class _AbortGeneration(RuntimeError):
    pass


def _extract_message_text(message: Any) -> str:
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                t = item.get("text")
                if isinstance(t, str):
                    parts.append(t)
                    continue
                inner = item.get("content")
                if isinstance(inner, str):
                    parts.append(inner)
                    continue
            t = getattr(item, "text", None)
            if isinstance(t, str):
                parts.append(t)
                continue
            inner = getattr(item, "content", None)
            if isinstance(inner, str):
                parts.append(inner)
                continue
        return "".join(parts)
    return ""


def _extract_content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                t = item.get("text")
                if isinstance(t, str):
                    parts.append(t)
                    continue
                inner = item.get("content")
                if isinstance(inner, str):
                    parts.append(inner)
                    continue
            t = getattr(item, "text", None)
            if isinstance(t, str):
                parts.append(t)
                continue
            inner = getattr(item, "content", None)
            if isinstance(inner, str):
                parts.append(inner)
                continue
        return "".join(parts)
    return ""


async def _generate_best_annotation(
    gen_client: AsyncOpenAI,
    *,
    sem: asyncio.Semaphore,
    model: str,
    target_text: str,
    reference_text: str,
    label: int,
    source_id: str,
    example_id: str,
    text_type: str,
    temperature: float,
    top_p: float,
    max_tokens: int | None,
    samples_per_doc: int,
    service_tier: str | None,
    reasoning_effort: str,
    style_example_index: int,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    target_label = "AI text" if label == 1 else "human text"
    human_text = reference_text if label == 1 else target_text
    ai_text = target_text if label == 1 else reference_text
    prompt = _paired_annotation_prompt(
        human_text=human_text,
        ai_text=ai_text,
        target_label=target_label,
        style_example_index=style_example_index,
    )
    audit_records: list[dict[str, Any]] = []
    for sample_index in range(samples_per_doc):
        try:
            async with sem:
                stripped, meta = await _chat_once(
                    gen_client,
                    model=model,
                    prompt=prompt,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    service_tier=service_tier,
                    reasoning_effort=reasoning_effort,
                )
            exact = meta.get("model_output_text_exact", stripped)
        except Exception as exc:
            logger.exception("generation failed example_id=%s text_type=%s", example_id, text_type)
            if _is_invalid_prompt_policy_error(exc):
                # provider-side prompt policy block on this one candidate; skip and continue run
                audit_records.append(
                    _record_failure(
                        example_id=example_id,
                        source_id=source_id,
                        text_type=text_type,
                        label=label,
                        raw=f"provider_invalid_prompt:{exc}",
                        prompt=prompt,
                        reason="provider_invalid_prompt",
                        ok=False,
                        diag=None,
                    )
                )
                return None, audit_records
            raise

        global_comment, annotation = _extract_blocks(raw=stripped)
        diag = _xml_inplace_format_diagnostics(output=annotation or "", document=target_text)
        comment_ok = _comment_is_standalone(global_comment or "")
        ok = bool(diag.get("ok", False)) and bool(global_comment) and comment_ok

        record = _record_failure(
            example_id=example_id,
            source_id=source_id,
            text_type=text_type,
            label=label,
            raw=exact,
            prompt=prompt,
            reason="format_or_comment_failed" if not ok else "ok",
            ok=ok,
            diag=diag,
            api_request=meta.get("api_request"),
            api_response=meta.get("api_response"),
            reasoning_trace_text=meta.get("reasoning_trace_text"),
        )
        record["sample_index"] = sample_index
        record["style_example_id"] = PROMPT_EXAMPLES[style_example_index % len(PROMPT_EXAMPLES)]["id"]
        record["standalone_comment_ok"] = comment_ok
        record["format_ok"] = bool(diag.get("ok", False))
        audit_records.append(record)
        if not ok:
            logger.warning(
                "invalid generation example_id=%s text_type=%s sample=%d reason=%s format=%s comment=%s raw_head=%r",
                example_id,
                text_type,
                sample_index,
                diag.get("reason", "unknown"),
                diag.get("ok", False),
                comment_ok,
                exact[:180],
            )
            continue

        return {
            "annotation": annotation,
            "global_comment": global_comment,
            "tell_score": None,
            "format_diag": diag,
            "raw_response": exact,
            "model_output_text_exact": exact,
            "input_prompt": prompt,
            "api_request": meta.get("api_request"),
            "api_response": meta.get("api_response"),
            "reasoning_trace_text": meta.get("reasoning_trace_text"),
            "provider": meta.get("provider"),
            "sample_index": sample_index,
        }, audit_records

    return None, audit_records


async def generate_audit_examples(
    args: SimpleNamespace,
    *,
    existing_flat_rows: list[dict[str, Any]] | None = None,
    existing_example_rows: list[dict[str, Any]] | None = None,
    existing_audit_rows: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    tokenizer = load_tokenizer()
    target_total_examples = args.num_ai_examples + args.num_human_examples
    target_sources = max(1, target_total_examples // 2)
    desired_pairs_before_q3 = min(
        args.scan_limit,
        max(args.max_candidate_sources, int(target_sources * _PREFETCH_FACTOR)),
    )
    pairs, rows_scanned = _collect_pairs(
        split=args.split,
        scan_limit=args.scan_limit,
        max_pairs=desired_pairs_before_q3,
    )
    if not pairs:
        return [], [], [], {"rows_scanned": rows_scanned, "num_source_pairs_before": 0, "num_source_pairs_after": 0}

    pairs, q_stats = _filter_pairs_by_q3(tokenizer, pairs)
    pairs = pairs[: args.max_candidate_sources]

    gen_client = _build_client_for_model(args.generation_model)
    sem = asyncio.Semaphore(args.concurrency)

    flat_rows: list[dict[str, Any]] = list(existing_flat_rows or [])
    example_rows: list[dict[str, Any]] = list(existing_example_rows or [])
    audit_rows: list[dict[str, Any]] = list(existing_audit_rows or [])
    candidate_examples: list[dict[str, Any]] = []
    for pair in pairs:
        human_text = _truncate(tokenizer, pair.source_text, args.max_doc_tokens)
        ai_text = _truncate(tokenizer, str(pair.ai_edited.get("text") or ""), args.max_doc_tokens)
        if human_text and ai_text:
            candidate_examples.append(
                {
                    "example_id": f"{pair.source_id}:ai_edited",
                    "source_id": pair.source_id,
                    "text_id": pair.ai_edited.get("text_id"),
                    "source": pair.source,
                    "text_type": "ai_edited",
                    "label": 1,
                    "target_text": ai_text,
                    "reference_text": human_text,
                    "human_text": human_text,
                    "ai_text": ai_text,
                }
            )
            candidate_examples.append(
                {
                    "example_id": f"{pair.source_id}:human_written",
                    "source_id": pair.source_id,
                    "text_id": None,
                    "source": pair.source,
                    "text_type": "human_written",
                    "label": 0,
                    "target_text": human_text,
                    "reference_text": ai_text,
                    "human_text": human_text,
                    "ai_text": ai_text,
                }
            )

    locked_example_ids = _load_selected_example_ids(args.selected_example_ids_manifest)
    if locked_example_ids:
        by_id = {c["example_id"]: c for c in candidate_examples}
        missing_ids = [example_id for example_id in locked_example_ids if example_id not in by_id]
        if missing_ids:
            raise ValueError(f"selected_example_ids not available after filtering: {missing_ids[:3]}")
        # keep the exact same examples, same order, for cross-model comparsons
        candidate_examples = [by_id[example_id] for example_id in locked_example_ids]
    else:
        rng = random.Random(_SEED)
        rng.shuffle(candidate_examples)

    existing_ids = {str(r.get("example_id", "")) for r in example_rows if str(r.get("example_id", "")).strip()}
    if existing_ids:
        candidate_examples = [c for c in candidate_examples if c["example_id"] not in existing_ids]
        logger.info("Resuming run: found %d already-generated examples; skipping those", len(existing_ids))

    async def _run_example(candidate: dict[str, Any]) -> dict[str, Any]:
        logger.info("Generating example example_id=%s", candidate["example_id"])
        style_rng = random.Random(f"{_SEED}:{candidate['example_id']}")
        style_example_index = style_rng.randrange(len(PROMPT_EXAMPLES))
        best, audit_records = await _generate_best_annotation(
            gen_client,
            sem=sem,
            model=args.generation_model,
            target_text=candidate["target_text"],
            reference_text=candidate["reference_text"],
            label=candidate["label"],
            source_id=candidate["source_id"],
            example_id=candidate["example_id"],
            text_type=candidate["text_type"],
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=getattr(args, "max_tokens", None),
            samples_per_doc=args.samples_per_doc,
            service_tier=args.service_tier,
            reasoning_effort=str(getattr(args, "reasoning_effort", "medium")),
            style_example_index=style_example_index,
        )
        return {
            "candidate": candidate,
            "best": best,
            "audit_records": audit_records,
            "style_example_id": PROMPT_EXAMPLES[style_example_index]["id"],
        }

    def _build_manifest(current_ai: int, current_human: int) -> dict[str, Any]:
        return {
            "dataset_id": _DATASET_ID,
            "split": args.split,
            "seed": _SEED,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "rows_scanned": rows_scanned,
            "prefetch_factor": _PREFETCH_FACTOR,
            "q3_filter": q_stats,
            "num_requested_examples": target_ai_examples + target_human_examples,
            "num_requested_ai_examples": target_ai_examples,
            "num_requested_human_examples": target_human_examples,
            "num_valid_examples": len(example_rows),
            "num_valid_ai_examples": current_ai,
            "num_valid_human_examples": current_human,
            "max_doc_tokens": args.max_doc_tokens,
            "generation_model": args.generation_model,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": getattr(args, "max_tokens", None),
            "service_tier": args.service_tier,
            "samples_per_doc": args.samples_per_doc,
            "selected_example_ids_manifest": args.selected_example_ids_manifest,
            "selected_example_ids": [r["example_id"] for r in example_rows],
        }

    def _flush_checkpoint(current_ai: int, current_human: int) -> None:
        _write_jsonl(Path(args.output_flat), flat_rows)
        _write_jsonl(Path(args.output_examples), example_rows)
        _write_jsonl(Path(args.output_audit), audit_rows)
        manifest = _build_manifest(current_ai=current_ai, current_human=current_human)
        Path(args.output_manifest).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_manifest).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def _process_example_result(result: dict[str, Any]) -> None:
        candidate = result["candidate"]
        audit_rows.extend(
            {**record, "example_id": candidate["example_id"], "source_id": candidate["source_id"], "label": candidate["label"], "text_type": candidate["text_type"]}
            for record in result["audit_records"]
        )

        best = result["best"]
        if best is None:
            logger.warning(
                "skipping example_id=%s due to missing valid candidate",
                candidate["example_id"],
            )
            return

        row = {
            "example_id": candidate["example_id"],
            "source_id": candidate["source_id"],
            "text_id": candidate["text_id"],
            "source": candidate["source"],
            "source_dataset": _DATASET_ID,
            "split": args.split,
            "text_type": candidate["text_type"],
            "label": candidate["label"],
            "text": candidate["target_text"],
            "human_text": candidate["human_text"],
            "ai_text": candidate["ai_text"],
            "annotation": best["annotation"],
            "global_comment": best["global_comment"],
            "tell_score": best["tell_score"],
            "format_diag": best["format_diag"],
            "generation": {
                "provider": best.get("provider"),
                "input_prompt": best["input_prompt"],
                "api_request": best.get("api_request"),
                "api_response": best.get("api_response"),
                "reasoning_trace_text": best.get("reasoning_trace_text"),
                "model_output_text_exact": best["model_output_text_exact"],
            },
            "generation_model": args.generation_model,
            "style_example_id": result["style_example_id"],
            "scorer_model": None,
            "seed": _SEED,
        }
        flat_rows.append(row)
        example_rows.append(row)
        pbar.update(1)

    ai_candidates = [c for c in candidate_examples if c["label"] == 1]
    human_candidates = [c for c in candidate_examples if c["label"] == 0]
    target_ai_examples = args.num_ai_examples
    target_human_examples = args.num_human_examples
    valid_ai_examples = sum(1 for r in example_rows if int(r.get("label", -1)) == 1)
    valid_human_examples = sum(1 for r in example_rows if int(r.get("label", -1)) == 0)
    pbar = tqdm(total=target_ai_examples + target_human_examples, desc="valid_examples", unit="example", initial=min(len(example_rows), target_ai_examples + target_human_examples))
    example_concurrency = max(1, args.concurrency)
    next_ai_index = 0
    next_human_index = 0
    pending_tasks: dict[asyncio.Task[dict[str, Any]], dict[str, Any]] = {}

    def _fill_pending_examples() -> None:
        nonlocal next_ai_index, next_human_index
        remaining_slots = (target_ai_examples - valid_ai_examples) + (target_human_examples - valid_human_examples)
        target_pending = min(example_concurrency, remaining_slots)
        while len(pending_tasks) < target_pending:
            candidate: dict[str, Any] | None = None
            need_ai = valid_ai_examples < target_ai_examples and next_ai_index < len(ai_candidates)
            need_human = valid_human_examples < target_human_examples and next_human_index < len(human_candidates)
            if need_ai and need_human:
                if next_ai_index <= next_human_index:
                    candidate = ai_candidates[next_ai_index]
                    next_ai_index += 1
                else:
                    candidate = human_candidates[next_human_index]
                    next_human_index += 1
            elif need_ai:
                candidate = ai_candidates[next_ai_index]
                next_ai_index += 1
            elif need_human:
                candidate = human_candidates[next_human_index]
                next_human_index += 1
            else:
                break
            task = asyncio.create_task(_run_example(candidate=candidate))
            pending_tasks[task] = candidate

    _fill_pending_examples()

    while pending_tasks and (valid_ai_examples < target_ai_examples or valid_human_examples < target_human_examples):
        done_tasks, _ = await asyncio.wait(set(pending_tasks.keys()), return_when=asyncio.FIRST_COMPLETED)
        for task in done_tasks:
            candidate = pending_tasks.pop(task, None)
            try:
                result = task.result()
            except Exception as exc:
                logger.exception("task failed example_id=%s", candidate["example_id"] if candidate else "unknown")
                for t in pending_tasks.keys():
                    t.cancel()
                if pending_tasks:
                    await asyncio.gather(*pending_tasks.keys(), return_exceptions=True)
                raise
            before_count = len(example_rows)
            _process_example_result(result)
            if len(example_rows) > before_count:
                if result["candidate"]["label"] == 1:
                    valid_ai_examples += 1
                else:
                    valid_human_examples += 1
                if args.checkpoint_every > 0 and len(example_rows) % args.checkpoint_every == 0:
                    _flush_checkpoint(current_ai=valid_ai_examples, current_human=valid_human_examples)
            _fill_pending_examples()

    for task in pending_tasks.keys():
        task.cancel()
    if pending_tasks:
        await asyncio.gather(*pending_tasks.keys(), return_exceptions=True)

    pbar.close()

    _flush_checkpoint(current_ai=valid_ai_examples, current_human=valid_human_examples)
    manifest = _build_manifest(current_ai=valid_ai_examples, current_human=valid_human_examples)

    return flat_rows, example_rows, audit_rows, manifest


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _run_cfg_to_args(cfg: DictConfig) -> SimpleNamespace:
    data = OmegaConf.to_container(cfg.sft_editlens, resolve=True)
    if not isinstance(data, dict):
        raise TypeError("cfg.sft_editlens must be a mapping")
    return SimpleNamespace(**data)


async def _amain(args: SimpleNamespace) -> int:
    if args.num_ai_examples is None and args.num_human_examples is None:
        args.num_ai_examples = args.num_examples // 2
        args.num_human_examples = args.num_examples - args.num_ai_examples
    elif args.num_ai_examples is None or args.num_human_examples is None:
        raise ValueError("set both sft_editlens.num_ai_examples and sft_editlens.num_human_examples, or set neither")
    args.num_examples = args.num_ai_examples + args.num_human_examples

    existing_flat_rows = _read_jsonl(Path(args.output_flat)) if args.resume else []
    existing_example_rows = _read_jsonl(Path(args.output_examples)) if args.resume else []
    existing_audit_rows = _read_jsonl(Path(args.output_audit)) if args.resume else []
    flat_rows, example_rows, audit_rows, manifest = await generate_audit_examples(
        args,
        existing_flat_rows=existing_flat_rows,
        existing_example_rows=existing_example_rows,
        existing_audit_rows=existing_audit_rows,
    )
    _write_jsonl(Path(args.output_flat), flat_rows)
    _write_jsonl(Path(args.output_examples), example_rows)
    _write_jsonl(Path(args.output_audit), audit_rows)
    Path(args.output_manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_manifest).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info("Wrote %d flat rows to %s", len(flat_rows), args.output_flat)
    logger.info("Wrote %d example rows to %s", len(example_rows), args.output_examples)
    logger.info("Wrote %d audit rows to %s", len(audit_rows), args.output_audit)
    logger.info("Wrote manifest to %s", args.output_manifest)
    if len(example_rows) < args.num_examples:
        logger.warning("requested %d examples but only produced %d valid examples", args.num_examples, len(example_rows))
    return 0 if example_rows else 1


@hydra.main(
    version_base=None,
    config_path="../../../conf",
    config_name="config",
)
def _hydra_run(cfg: DictConfig) -> None:
    global CFG

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    config_module.CFG = cfg
    CFG = cfg
    args = _run_cfg_to_args(cfg=cfg)
    raise SystemExit(asyncio.run(_amain(args)))


if __name__ == "__main__":
    _hydra_run()
