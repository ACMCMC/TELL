"""Generate SFT annotation data from acmc/expert-annotated-TELL.

One generation per (document, annotator) pair on the validation split only
(hard requirement: never test/train). Each annotator comment guides span targets.

After generation, use --push-to-hub / --hub-repo to upload a HF dataset that
_load_expert_annot_split() in train_tinker_sft.py can consume directly.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import weave
from datasets import Dataset, DatasetDict, load_dataset
from dotenv import load_dotenv
from tqdm.auto import tqdm

from rl_detector.data import clean_document_text, truncate_document_text
from rl_detector.prompt_utils import load_tokenizer
from rl_detector.sft.generate_editlens import (
    _AbortGeneration,
    _build_client_for_model,
    _call_with_rate_limit_backoff,
    _chat_once,
    _is_invalid_prompt_policy_error,
    _record_failure,
    _write_jsonl,
    _read_jsonl,
)
from rl_detector.format_fix import try_fix_response
from rl_detector.rewards import format_diagnostics
from rl_detector.sft.verdict_score_agg import verdict_score_from_annotation
from rl_detector.tell_xml import (
    _TEXT_O,
    _VERDICT_PREF,
    _TEXT_CLOSE_CHUNK,
    escape_attr_piece,
    escape_document_piece,
    root_splits,
    strip_text_wrapper,
)

logger = logging.getLogger(__name__)

_DATASET_ID = "acmc/expert-annotated-TELL"
# Hard requirement: only the HF validation split (100 docs); do not use test.
_SOURCE_SPLIT = "validation"
_SEED = 2242
_ANNOTATOR_INDICES = (1, 2, 3, 4, 5)


def _ground_truth_to_label(ground_truth: str) -> int:
    return 0 if str(ground_truth).strip().lower() == "human-written" else 1


def _to_curly_quotes(text: str) -> str:
    """Convert straight quotes to curly typographic quotes (no XML escaping)."""
    text = re.sub(r'(?:(?<=^)|(?<=\s)|(?<=[({\[]))"', '“', text)
    text = text.replace('"', '”')
    text = re.sub(r"(\w)'(\w)", r'\1’\2', text)
    text = re.sub(r"(?:(?<=^)|(?<=\s)|(?<=[({\[]))'", '‘', text)
    text = text.replace("'", '’')
    return text


def _comment_word_count(comment: str) -> int:
    """Count words in comment, excluding text inside quoted sections."""
    stripped = re.sub(r'“[^”]*”|"[^"]*"', '', comment)
    return len(stripped.split())


_PROMPT_EXAMPLE_REVIEWER_HINT = """Some of the author’s assertions are so garbled that only a human who doesn’t quite understand the process must have written it. For example, referring to a patch of Escherichia coli (which I’m guessing is E. coli) as “a tasty snack” is a funny contradiction, and definitely not something I would expect from machine-generated text. Or maybe it’s an L2 English speaker, when one considers that the author wants to “put the agent loose” upon those poor worms. The purpose and methodology of the study are also quite detailed and well-explained, whereas AI seems to be vague around these subjects as it feels like it lacks understanding and would rather say less than be incorrect. The Latin names are not italicized and ‘one’ should be capitalized as it's at the beginning of a spoken sentence. Though something that throws me off is that all names are referred to as ‘Dr.’, even the engineer. It also doesn’t follow the formulaic structure that AI likes to use, e.g. there’s no bland conclusion at the end."""

_PROMPT_EXAMPLE_TEXT = """Scientists have given artificial intelligence a direct line into the nervous systems of millimeter-long worms, letting it guide the creatures to a tasty target—and demonstrating intriguing brain-AI collaboration. They trained the AI with a methodology called deep-reinforcement learning; the same is used to help AI players learn to master games such as Go. An artificial neural network, software roughly modeled on biological brains, analyzes strings of actions and outcomes, extracting strategies for an AI “agent” to interact with its environment and achieve a goal.

In the study, published in Nature Machine Intelligence, researchers trained an AI agent to direct one-millimeter-long Caenorhabditis elegans worms toward tasty patches of Escherichia coli in a four-centimeter dish. A nearby camera recorded the location and orientation of every worm’s head and body; three times per second the agent received this information for the previous 15 frames, giving it a sense of the past and present at each moment. The agent could also turn on or off a light aimed at the dish. The worms were optogenetically engineered so certain neurons would become active or inactive in response to the light, sometimes prompting movement.

The research team tested six genetic lines in which the number of light-sensitive neurons ranged from one to all 302 the worms possessed. Stimulation had a different effect in each line, making the worm turn, for instance, or preventing it from turning. The scientists first collected training data by flashing lights randomly at the worms for five hours, then fed the data to the AI agent to find patterns before putting the agent loose.

With five of the six lines, including the line where all neurons responded to light, the agent learned to direct the worm to the target faster than if the worm had been left alone or the light had flashed randomly. What’s more, the agent and the worm cooperated: if the agent steered the worm straight toward a target but there were small obstacles in the path, the worm would crawl around them.

Dr. Thang, an engineer at the University of Queensland in Australia, who has independently worked on cyborg insects, praised the work for its simple setup—reinforcement learning is flexible, and AI based on it can figure out how to perform complex tasks. According to Harvard University biophysicist Dr. Li, the paper’s lead author, “one can easily see how it might be extended to harder problems.” Her team is now exploring whether their method can improve electrical deep-brain stimulation to treat Parkinson’s disease in humans by adjusting the voltage used and its timing. One day reinforcement learning plus implants might even give us new skills, Li says—artificial and real neural nets united."""

_PROMPT_EXAMPLE_ANNOTATED = """Scientists have given artificial intelligence a direct line into the nervous systems of millimeter-long worms, letting it guide the creatures to a tasty target—and demonstrating intriguing brain-AI collaboration. They trained the AI with a methodology called deep-reinforcement learning; the same is used to help AI players learn to master games such as Go. An artificial neural network, software roughly modeled on biological brains, analyzes strings of actions and outcomes, extracting strategies for an AI “agent” to interact with its environment and achieve a goal.

In the study, published in Nature Machine Intelligence, researchers trained an AI agent to direct <span>one-millimeter-long<annotation type="human" why="The purpose and methodology of the study are quite detailed and well-explained, whereas AI seems to be vague around these subjects as it feels like it lacks understanding and would rather say less than be incorrect" score="0.58" /></span> <span>Caenorhabditis elegans<annotation type="human" why="Latin name is not italicized; should be capitalized as it’s at the beginning of a spoken sentence" score="0.43" /></span> worms toward <span>tasty<annotation type="human" why="this is a funny contradiction, definitely not something I would expect from machine-generated text because AI lacks creativity" score="0.74" /></span> patches of <span>Escherichia coli<annotation type="human" why="Latin name is not italicized" score="" /></span> in a four-centimeter dish. A nearby camera recorded the location and orientation of every worm’s head and body; <span>three times per second<annotation type="human" why="again, very specific; only someone who actually ran the experiment can know that" score="0.61" /></span> the agent received this information for <span>the previous 15 frames<annotation type="human" why="another specific detail" score="0.59" /></span>, giving it a sense of the past and present at each moment. The agent could also turn on or off a light aimed at the dish. The worms were optogenetically engineered so certain neurons would become active or inactive in response to the light, sometimes prompting movement.

The research team tested six genetic lines in which the number of light-sensitive neurons ranged from one to <span>all 302<annotation type="human" why="exact count; AI might have approximated (“about 300”)" score="0.60" /></span> the worms possessed. Stimulation had a different effect in each line, making the worm turn, for instance, or preventing it from turning. The scientists first collected training data by flashing lights randomly at the worms for five hours, then fed the data to the AI agent to find patterns before <span>putting<annotation type="human" why="odd word; maybe the author is an L2 English speaker" score="0.62" /></span> the agent loose.

With five of the six lines, including the line where all neurons responded to light, the agent learned to direct the worm to the target faster than if the worm had been left alone or the light had flashed randomly. What’s more, the agent and the worm cooperated: if the agent steered the worm straight toward a target but there were small obstacles in the path, the worm would crawl around them.

<span>Dr. Thang<annotation type="AI" why="Dr. or engineer? It throws me off that all names are referred to as ‘Dr.’, AI might assume that since it’s common to have it in universities (to be checked: is Dr. Thang actually a doctor?)" score="0.51" /></span>, an engineer at the University of Queensland in Australia, who has independently worked on cyborg insects, praised the work for its simple setup—reinforcement learning is flexible, and AI based on it can figure out how to perform complex tasks. According to Harvard University biophysicist Dr. Li, the paper’s lead author, <span>“one<annotation type="human" why="lowercase at the start of a quote; that’s wrong, AI would use the standard format, but a human might not notice" score="0.46" /></span> can easily see how it might be extended to harder problems.” Her team is now exploring whether their method can improve electrical deep-brain stimulation to treat Parkinson’s disease in humans by adjusting the voltage used and its timing. One day reinforcement learning plus implants might even give us new skills, Li says—<span>artificial and real neural nets united.<annotation type="human" why="This doesn’t follow the formulaic structure that AI likes to use; e.g. a bland conclusion at the end to make the text feel complete" score="0.69" /></span>"""


def _single_annotation_prompt(
    target_text: str,
    target_label: str,
    annotator_comment: str,
) -> str:
    return f"""Annotate a text with AI-or-human tells. Wrap individual spans like this:
<span>ANNOTATED_TEXT<annotation type="LABEL" why="EXPLANATION" score="FLOAT" /></span>

Rules:
- Copy the text exactly: no typo fixes, no reformatting. Use the same XML escaping as the input.
- type is "AI" or "human". score is 0.0–1.0 (0–0.25 weak, 0.35–0.65 moderate, 0.75–1.0 strong).
- Keep spans small and granular: annotate a word or phrase, not a whole sentence.
- Write the why in first person, YOU are the annotator. Mirror the exact writing style and voice of the hint; same vocabulary, same rhythm. Keep it casual and direct; no academic language, no formal analysis, simple English. Never write "the reviewer said/pointed out/noted", you ARE the one observing this.
- Do not directly reference the hint in your explanations. You should write the annotations as if you are analyzing the text without knowing the hint beforehand (since that's what you write at the end). That means you can’t refer to it, but you should write as if you have the same knowledge as the hint (e.g. if the hint points out a specific detail, you should also mention that detail in your explanation, since you know it from reading the text).
- You should annotate ALL the items in the hint, be comprehensive in your annotations. DO NOT annotate items that are not in the hint.
- The why="..." explanations can be concise if you already explained why a pattern is a tell, i.e., don't repeat the mechanism, a short callback ("again, XXX", "another XXX") is enough. Try to use the same words and phrasing as the hint in your explanations when possible, since that’s the voice we want to capture.
- Output the full text with inline spans inserted, do NOT add any outer wrapper.
- The explanations should explain the mechanism (the underlying cause that would make an AI or human produce that exact text) that is explicit or implicit in the hint. For example, instead of "this is a funny contradiction, and it feels very human" (feeling human is not a mechanism), the hint said "definitely not something I would expect from machine-generated text", so we can write "this is a funny contradiction, definitely not something I would expect from machine-generated text because AI lacks creativity" (the mechanism is that AI lacks creativity, and would be unlikely to pick that word).
- Avoid unspecific, generic mechanisms: "feels like something a person would choose", "it doesn’t feel like AI", "this is a common human pattern"... all of these are NOT mechanisms that can be checked and verified. Think about what is the underlying reason. We need specific, checkable mechanisms about how AI works or is trained, or about the limitations and reality of the world, or anything that an external observer could verify. This should be grounded on the specific text and the specific hint. You want the reader to learn how to identify the mechanisms that distinguish AI-generated from human-generated text, so you should explain your reasoning clearly and specifically.
- You can also add notes for the human reader to check things we can’t verify but an external observer could, e.g. "(to be checked: is Dr. Thang actually a doctor?)"

Example:

Reviewer hint:
<<<
{_PROMPT_EXAMPLE_REVIEWER_HINT}
>>>

human text:
<<<
{escape_document_piece(_PROMPT_EXAMPLE_TEXT)}
>>>

Annotated:
<<<
{_PROMPT_EXAMPLE_ANNOTATED}
>>>

---

Reviewer hint:
<<<
{annotator_comment}
>>>

Find ALL the exact spans in the text that correspond to these clues and annotate them. Be comprehensive — cover every clue in the hint, don't skip any. Do not add tells that aren't in the hint.
If you cannot locate any of the clues as specific spans, output exactly: SKIP

{target_label}:
<<<
{escape_document_piece(target_text)}
>>>

Annotated:"""


def _is_skip_response(raw: str) -> bool:
    return raw.strip().upper() == "SKIP"


def _has_reviewer_leak(inner_annotated: str) -> bool:
    """Return True if any why attribute mentions 'reviewer' (model broke first-person POV)."""
    return bool(re.search(r'\breviewer\b', inner_annotated, re.IGNORECASE))


@weave.op()
async def _generate_annotation(
    client,
    *,
    sem: asyncio.Semaphore,
    model: str,
    target_text: str,
    target_label: str,
    annotator_comment: str,
    annotator_confidence: float,
    label: int,
    source_id: Any,
    example_id: str,
    annotator_idx: int,
    temperature: float,
    top_p: float,
    max_tokens: int | None,
    samples_per_doc: int,
    service_tier: str | None,
    reasoning_effort: str,
    verdict_agg_beta: float,
    verdict_agg_scale: float,
    verdict_agg_tau: float,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Return (result, audit_records).

    result is one of:
      {"skip": True, ...}            — model explicitly said SKIP; store as skip=True
      {"annotation": ..., "sft_text": ..., ...}  — valid annotation
      None                           — format validation failed
    """
    prompt = _single_annotation_prompt(
        target_text=target_text,
        target_label=target_label,
        annotator_comment=annotator_comment,
    )
    text_type = f"annotator_{annotator_idx}"
    audit_records: list[dict[str, Any]] = []

    for sample_index in range(samples_per_doc):
        try:
            async with sem:
                stripped, meta = await _chat_once(
                    client,
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
            logger.exception("generation failed example_id=%s", example_id)
            if _is_invalid_prompt_policy_error(exc):
                audit_records.append(
                    _record_failure(
                        example_id=example_id,
                        source_id=str(source_id),
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

        # SKIP response: model says it can't annotate — accept immediately, no retries.
        if _is_skip_response(stripped):
            logger.info("model skipped example_id=%s", example_id)
            audit_records.append({
                "example_id": example_id,
                "source_id": str(source_id),
                "text_type": text_type,
                "label": label,
                "ok": True,
                "reason": "skip",
                "sample_index": sample_index,
                "model_output_text_exact": exact,
                "input_prompt": prompt,
            })
            return {"skip": True, "input_prompt": prompt, "model_output_text_exact": exact}, audit_records

        # Reject if any why attribute leaks third-person "reviewer" phrasing.
        if _has_reviewer_leak(stripped):
            logger.warning("reviewer leak in why attributes example_id=%s — rejecting", example_id)
            audit_records.append(
                _record_failure(
                    example_id=example_id,
                    source_id=str(source_id),
                    text_type=text_type,
                    label=label,
                    raw=exact,
                    prompt=prompt,
                    reason="reviewer_leak",
                    ok=False,
                    diag=None,
                )
            )
            return None, audit_records

        # The model outputs inner annotated content (spans only, no outer wrapper).
        # Assemble the full SFT-ready output with the real verdict before validation.
        verdict_type_str = "AI" if label == 1 else "human"
        verdict_score = verdict_score_from_annotation(
            annotation=stripped,
            label=label,
            beta=verdict_agg_beta,
            scale=verdict_agg_scale,
            tau=verdict_agg_tau,
        )
        t = escape_attr_piece(verdict_type_str)
        w = escape_attr_piece(annotator_comment)
        sc = escape_attr_piece(f"{verdict_score:.2f}")
        sft_text = f'{_TEXT_O}{stripped}{_VERDICT_PREF}{t}" why="{w}" score="{sc}{_TEXT_CLOSE_CHUNK}'

        # Try format-fixing before discarding near-valid outputs.
        fixed = try_fix_response(sft_text, target_text, max_fix_ratio=0.5)
        if fixed is not None:
            logger.debug("format_fix repaired annotation example_id=%s", example_id)
            sft_text = fixed

        diag = format_diagnostics(sft_text, target_text)
        ok = bool(diag.get("ok", False))

        record = _record_failure(
            example_id=example_id,
            source_id=str(source_id),
            text_type=text_type,
            label=label,
            raw=exact,
            prompt=prompt,
            reason="format_failed" if not ok else "ok",
            ok=ok,
            diag=diag,
            api_request=meta.get("api_request"),
            api_response=meta.get("api_response"),
            reasoning_trace_text=meta.get("reasoning_trace_text"),
        )
        record["sample_index"] = sample_index
        record["format_ok"] = ok
        audit_records.append(record)

        if not ok:
            logger.warning(
                "invalid generation example_id=%s reason=%s",
                example_id,
                diag.get("reason", "unknown"),
            )
            return None, audit_records

        # Extract inner annotated content (spans + text) from the validated sft_text.
        annotation = strip_text_wrapper(sft_text)  # inner content with spans, before <verdict>

        return {
            "skip": False,
            "annotation": annotation or stripped,
            "sft_text": sft_text,
            "verdict_type": verdict_type_str,
            "verdict_score": verdict_score,
            "verdict_why": annotator_comment,
            "format_diag": diag,
            "model_output_text_exact": exact,
            "input_prompt": prompt,
            "api_request": meta.get("api_request"),
            "api_response": meta.get("api_response"),
            "reasoning_trace_text": meta.get("reasoning_trace_text"),
            "provider": meta.get("provider"),
            "sample_index": sample_index,
        }, audit_records

    return None, audit_records


def _is_good_example_row(row: dict[str, Any]) -> bool:
    return bool(row.get("example_id")) and bool(row.get("annotation")) and not row.get("skip")


def _candidate_match_key(text: str, annotator_idx: int) -> tuple[str, int]:
    return (text, annotator_idx)


def _sync_row_from_candidate(row: dict[str, Any], candidate: dict[str, Any]) -> None:
    row["example_id"] = candidate["example_id"]
    row["source_row_index"] = candidate["source_row_index"]
    row["generation_model_src"] = candidate["generation_model_src"]


def _dedupe_example_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the last successful row per example_id (drops retries and skips)."""
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        if _is_good_example_row(row=row):
            by_id[str(row["example_id"])] = row
    return list(by_id.values())


def _load_candidates(max_doc_tokens: int, output_split: str, dataset_id: str | None = None, source_split: str | None = None, origin_split_filter: str | None = None) -> list[dict[str, Any]]:
    tokenizer = load_tokenizer()
    ds_id = dataset_id or _DATASET_ID
    ds_split = source_split or _SOURCE_SPLIT
    ds = load_dataset(ds_id, split=ds_split)
    if origin_split_filter:
        ds = ds.filter(lambda r: r.get("origin_split") == origin_split_filter)
    candidates: list[dict[str, Any]] = []
    for row in ds:
        article = clean_document_text(str(row.get("article") or ""))
        if not article:
            continue
        article = truncate_document_text(tokenizer=tokenizer, text=article, max_doc_tokens=max_doc_tokens)
        label = _ground_truth_to_label(row.get("ground_truth", ""))
        target_label = "AI text" if label == 1 else "human text"
        source_id = row.get("id")
        source_row_index = row.get("source_row_index")
        source_sha256 = str(row.get("source_sha256") or "")
        generation_model_src = str(row.get("generation_model") or "")
        for idx in _ANNOTATOR_INDICES:
            annot = row.get(f"annotator_{idx}") or {}
            comment = str(annot.get("comment") or "").strip()
            if not comment or _comment_word_count(comment) < 50:
                continue
            comment = _to_curly_quotes(comment)
            example_id = f"{source_row_index}:annotator_{idx}"
            candidates.append({
                "example_id": example_id,
                "source_id": source_id,
                "source_row_index": source_row_index,
                "source_sha256": source_sha256,
                "generation_model_src": generation_model_src,
                "annotator_idx": idx,
                "source_dataset": _DATASET_ID,
                "source_split": _SOURCE_SPLIT,
                "split": output_split,
                "label": label,
                "ground_truth": row.get("ground_truth"),
                "expert_majority_vote": row.get("expert_majority_vote"),
                "text": article,
                "target_label": target_label,
                "annotator_comment": comment,
                "annotator_confidence": annot.get("confidence"),
                "annotator_guess": annot.get("guess"),
            })
    return candidates


async def generate(args: argparse.Namespace) -> None:
    load_dotenv()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    examples_path = output_dir / f"{args.split}.examples.jsonl"
    audit_path = output_dir / f"{args.split}.audit.jsonl"
    manifest_path = output_dir / f"{args.split}.manifest.json"

    candidates = _load_candidates(
        max_doc_tokens=args.max_doc_tokens,
        output_split=args.split,
        dataset_id=args.dataset_id or None,
        source_split=args.source_split or None,
        origin_split_filter=args.origin_split_filter or None,
    )
    logger.info(
        "Loaded %d candidates from %s split=%s (HF output split=%s)",
        len(candidates),
        _DATASET_ID,
        _SOURCE_SPLIT,
        args.split,
    )
    cand_by_key = {
        _candidate_match_key(text=c["text"], annotator_idx=int(c["annotator_idx"])): c
        for c in candidates
    }

    existing_examples = _read_jsonl(examples_path) if args.resume else []
    existing_audit = _read_jsonl(audit_path) if args.resume else []
    if args.resume and existing_examples:
        n_before = len(existing_examples)
        existing_examples = _dedupe_example_rows(rows=existing_examples)
        if len(existing_examples) < n_before:
            logger.info(
                "Deduped examples file: %d lines -> %d unique successful rows",
                n_before,
                len(existing_examples),
            )
        for row in existing_examples:
            key = _candidate_match_key(text=str(row["text"]), annotator_idx=int(row["annotator_idx"]))
            cand = cand_by_key.get(key)
            if cand is not None:
                _sync_row_from_candidate(row=row, candidate=cand)
        _write_jsonl(examples_path, existing_examples)
    completed_keys = {
        _candidate_match_key(text=str(r["text"]), annotator_idx=int(r["annotator_idx"]))
        for r in existing_examples
        if _is_good_example_row(row=r)
    }
    if completed_keys:
        candidates = [
            c
            for c in candidates
            if _candidate_match_key(text=c["text"], annotator_idx=int(c["annotator_idx"])) not in completed_keys
        ]
        logger.info(
            "Resuming: %d done (by text+annotator), %d left to generate",
            len(completed_keys),
            len(candidates),
        )

    rng = random.Random(_SEED)
    rng.shuffle(candidates)
    if args.max_candidates > 0:
        candidates = candidates[: args.max_candidates]
        logger.info("Smoke-test mode: capped at %d candidates", len(candidates))

    client = _build_client_for_model(args.generation_model)
    sem = asyncio.Semaphore(args.concurrency)

    example_rows: list[dict[str, Any]] = list(existing_examples)
    audit_rows: list[dict[str, Any]] = list(existing_audit)

    def _flush() -> None:
        _write_jsonl(examples_path, example_rows)
        _write_jsonl(audit_path, audit_rows)
        manifest = {
            "dataset_id": _DATASET_ID,
            "source_split": _SOURCE_SPLIT,
            "split": args.split,
            "seed": _SEED,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "generation_model": args.generation_model,
            "reasoning_effort": args.reasoning_effort,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "samples_per_doc": args.samples_per_doc,
            "max_doc_tokens": args.max_doc_tokens,
            "num_examples": len(example_rows),
            "num_audit_records": len(audit_rows),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    async def _run_candidate(c: dict[str, Any]) -> dict[str, Any]:
        best, audit_records = await _generate_annotation(
            client,
            sem=sem,
            model=args.generation_model,
            target_text=c["text"],
            target_label=c["target_label"],
            annotator_comment=c["annotator_comment"],
            annotator_confidence=float(c["annotator_confidence"] or 3.0),
            label=c["label"],
            source_id=c["source_id"],
            example_id=c["example_id"],
            annotator_idx=c["annotator_idx"],
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens if args.max_tokens > 0 else None,
            samples_per_doc=args.samples_per_doc,
            service_tier=args.service_tier if args.service_tier else None,
            reasoning_effort=args.reasoning_effort,
            verdict_agg_beta=args.verdict_agg_beta,
            verdict_agg_scale=args.verdict_agg_scale,
            verdict_agg_tau=args.verdict_agg_tau,
        )
        return {"candidate": c, "best": best, "audit_records": audit_records}

    pending: dict[asyncio.Task, dict[str, Any]] = {}
    pbar = tqdm(total=len(candidates) + len(existing_examples), desc="examples", unit="ex", initial=len(existing_examples))
    candidate_iter = iter(candidates)

    def _fill_pending() -> None:
        while len(pending) < args.concurrency:
            c = next(candidate_iter, None)
            if c is None:
                break
            task = asyncio.create_task(_run_candidate(c))
            pending[task] = c

    _fill_pending()

    while pending:
        done, _ = await asyncio.wait(set(pending.keys()), return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            c = pending.pop(task)
            try:
                result = task.result()
            except Exception:
                logger.exception("task failed example_id=%s", c["example_id"])
                for t in pending:
                    t.cancel()
                await asyncio.gather(*pending.keys(), return_exceptions=True)
                raise

            audit_rows.extend(result["audit_records"])

            best = result["best"]
            candidate = result["candidate"]
            if best is None:
                # Format validation failed — don't store; will be retried on next run.
                logger.warning("format failure example_id=%s — will retry on next run", candidate["example_id"])
            else:
                is_skip = bool(best.get("skip", False))
                annotation = None if is_skip else best.get("annotation")
                sft_text = None if is_skip else best.get("sft_text")

                example_rows.append({
                    "example_id": candidate["example_id"],
                    "source_id": candidate["source_id"],
                    "source_row_index": candidate["source_row_index"],
                    "source_sha256": candidate["source_sha256"],
                    "generation_model_src": candidate["generation_model_src"],
                    "annotator_idx": candidate["annotator_idx"],
                    "source_dataset": _DATASET_ID,
                    "split": candidate["split"],
                    "skip": is_skip,
                    "label": candidate["label"],
                    "ground_truth": candidate["ground_truth"],
                    "expert_majority_vote": candidate["expert_majority_vote"],
                    "text": candidate["text"],
                    "annotator_comment": candidate["annotator_comment"],
                    "annotator_confidence": candidate["annotator_confidence"],
                    "annotator_guess": candidate["annotator_guess"],
                    "annotation": annotation,
                    "verdict_type": best.get("verdict_type") if not is_skip else None,
                    "verdict_score": best.get("verdict_score") if not is_skip else None,
                    "verdict_why": best.get("verdict_why") if not is_skip else None,
                    "sft_text": sft_text,
                    "format_diag": None if is_skip else best["format_diag"],
                    "generation": {
                        "input_prompt": best["input_prompt"],
                        "model_output_text_exact": best["model_output_text_exact"],
                        **({"provider": best.get("provider"), "api_request": best.get("api_request"), "api_response": best.get("api_response"), "reasoning_trace_text": best.get("reasoning_trace_text")} if not is_skip else {}),
                    },
                    "generation_model": args.generation_model,
                    "seed": _SEED,
                })
                pbar.update(1)

            if args.checkpoint_every > 0 and len(example_rows) % args.checkpoint_every == 0:
                _flush()

            _fill_pending()

    pbar.close()
    _flush()
    logger.info("Done: %d examples, %d audit records", len(example_rows), len(audit_rows))
    logger.info("  examples → %s", examples_path)
    logger.info("  audit    → %s", audit_path)
    logger.info("  manifest → %s", manifest_path)


_LEGACY_GENERATION_MODEL = "gpt-5.5"


def _hf_row_key(row: dict[str, Any]) -> str:
    eid = str(row.get("example_id") or "")
    gm = str(row.get("generation_model") or _LEGACY_GENERATION_MODEL)
    return f"{eid}\0{gm}"


def _load_existing_hub_rows(hub_repo: str, token: str | None) -> list[dict[str, Any]]:
    ds = load_dataset(hub_repo, split="train", token=token)
    out: list[dict[str, Any]] = []
    for row in ds:
        r = dict(row)
        if not str(r.get("generation_model") or "").strip():
            r["generation_model"] = _LEGACY_GENERATION_MODEL
        out.append(r)
    return out


def push_to_hub(
    examples_paths: list[Path],
    hub_repo: str,
    train_only: bool,
    append_to_hub: bool,
) -> None:
    """Read generated JSONL files and push a HF dataset to hub_repo.

    When train_only=True, every row goes to the train split only.
    When append_to_hub=True, load existing train split and merge by (example_id, generation_model).
    """
    rows: list[dict[str, Any]] = []
    for p in examples_paths:
        rows.extend(_read_jsonl(p))

    if not rows:
        logger.warning("push_to_hub: no rows to push")
        return

    def _to_hf_row(r: dict[str, Any]) -> dict[str, Any]:
        return {
            "text": r["text"],
            "annotation": r["annotation"],
            "sft_text": str(r.get("sft_text") or ""),
            "label": int(r["label"]),
            "verdict_type": str(r.get("verdict_type") or ""),
            "verdict_score": float(r.get("verdict_score") or 0.0),
            "verdict_why": str(r.get("verdict_why") or ""),
            "example_id": str(r.get("example_id", "")),
            "source_id": str(r.get("source_id", "")),
            "source_row_index": r.get("source_row_index"),
            "source_sha256": str(r.get("source_sha256", "")),
            "generation_model_src": str(r.get("generation_model_src", "")),
            "annotator_idx": int(r.get("annotator_idx", 0)),
            "ground_truth": str(r.get("ground_truth", "")),
            "expert_majority_vote": str(r.get("expert_majority_vote", "")),
            "annotator_comment": str(r.get("annotator_comment", "")),
            "annotator_confidence": float(r.get("annotator_confidence") or 0.0),
            "annotator_guess": str(r.get("annotator_guess", "")),
            "generation_model": str(r.get("generation_model", "")),
        }

    hf_rows = [_to_hf_row(r) for r in rows if not r.get("skip") and r.get("annotation")]
    logger.info("Filtered to %d annotated (non-skip) rows for HF push", len(hf_rows))
    if not hf_rows:
        logger.warning("push_to_hub: no annotated rows after filtering skips")
        return

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

    if append_to_hub:
        existing = _load_existing_hub_rows(hub_repo=hub_repo, token=token)
        by_key: dict[str, dict[str, Any]] = {_hf_row_key(row=r): r for r in existing}
        n_before = len(by_key)
        for row in hf_rows:
            by_key[_hf_row_key(row=row)] = row
        hf_rows = list(by_key.values())
        logger.info("append_to_hub: %d existing on hub -> %d rows after merge", n_before, len(hf_rows))
    else:
        by_id: dict[str, dict[str, Any]] = {}
        for row in hf_rows:
            eid = str(row.get("example_id") or "")
            if eid:
                by_id[eid] = row
        if by_id:
            n_dupes = len(hf_rows) - len(by_id)
            hf_rows = list(by_id.values())
            if n_dupes:
                logger.info(
                    "Deduped by example_id: %d -> %d rows (dropped %d duplicates)",
                    len(hf_rows) + n_dupes,
                    len(hf_rows),
                    n_dupes,
                )

    rng = random.Random(_SEED)
    rng.shuffle(hf_rows)
    if train_only:
        from huggingface_hub import HfApi, CommitOperationDelete, list_repo_files

        api = HfApi(token=token)
        stale = [
            f for f in list_repo_files(hub_repo, repo_type="dataset", token=token)
            if "/validation" in f or f.startswith("validation/") or "validation-" in f.split("/")[-1]
        ]
        if stale:
            logger.info("Deleting %d stale validation shard(s) from %s", len(stale), hub_repo)
            api.create_commit(
                repo_id=hub_repo,
                repo_type="dataset",
                operations=[CommitOperationDelete(path_in_repo=p) for p in stale],
                commit_message="Remove stale validation split (train-only dataset)",
            )
        ds = DatasetDict({"train": Dataset.from_list(hf_rows)})
        logger.info("Pushing %d train-only rows to %s", len(hf_rows), hub_repo)
    else:
        split_idx = max(1, int(len(hf_rows) * 0.9))
        train_rows = hf_rows[:split_idx]
        val_rows = hf_rows[split_idx:]
        ds = DatasetDict({
            "train": Dataset.from_list(train_rows),
            "validation": Dataset.from_list(val_rows) if val_rows else Dataset.from_list(train_rows[:1]),
        })
        logger.info("Pushing %d train + %d val rows to %s", len(train_rows), len(val_rows), hub_repo)
    ds.push_to_hub(hub_repo, token=token)
    logger.info("Pushed to https://huggingface.co/datasets/%s", hub_repo)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate SFT annotations from tell-human-detectors dataset")
    p.add_argument(
        "--split",
        default="train",
        help="Output split name for JSONL filenames and HF push (use train for train-only hub)",
    )
    p.add_argument("--generation-model", default="gpt-5.5")
    p.add_argument("--output-dir", default="human_annot_sft_output")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--samples-per-doc", type=int, default=3)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=0, help="0 = no limit")
    p.add_argument("--max-doc-tokens", type=int, default=1200)
    p.add_argument("--service-tier", default="")
    p.add_argument("--reasoning-effort", default="medium", help="OpenAI reasoning.effort for gpt-5/o-series (low|medium|high)")
    p.add_argument("--verdict-agg-beta", type=float, default=3.0, help="softmax beta for verdict score from inner tells (0=plain mean)")
    p.add_argument("--verdict-agg-scale", type=float, default=0.45, help="tanh scale for directional verdict")
    p.add_argument("--verdict-agg-tau", type=float, default=1.25, help="tanh temperature on signed agg")
    p.add_argument("--checkpoint-every", type=int, default=25)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--push-to-hub", action="store_true", help="Push generated examples to HuggingFace Hub after generation")
    p.add_argument(
        "--append-to-hub",
        action="store_true",
        help="Merge with existing hub train split; dedupe by (example_id, generation_model); backfill missing generation_model as gpt-5.5",
    )
    p.add_argument("--hub-repo", default="acmc/expert-annotated-TELL", help="HF repo ID to push to")
    p.add_argument(
        "--train-only-hub",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Push all rows to HF train split only (no validation split)",
    )
    p.add_argument("--max-candidates", type=int, default=0, help="Cap number of candidates (0 = no limit); useful for smoke tests")
    p.add_argument("--dataset-id", default="", help="Override source HF dataset ID (default: acmc/expert-annotated-TELL)")
    p.add_argument("--source-split", default="", help="Override source HF split (default: validation)")
    p.add_argument("--origin-split-filter", default="", help="Filter rows by origin_split column value (e.g. 'test')")
    return p


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = _build_parser().parse_args()
    try:
        weave.init(project_name="rl-detector", settings={"print_call_link": False, "log_level": "WARNING"})
    except Exception:
        pass
    asyncio.run(generate(args))
    if args.push_to_hub:
        if not args.hub_repo:
            raise ValueError("--hub-repo must be set when using --push-to-hub")
        output_dir = Path(args.output_dir)
        examples_paths = list(output_dir.glob("*.examples.jsonl"))
        push_to_hub(
            examples_paths=examples_paths,
            hub_repo=args.hub_repo,
            train_only=bool(args.train_only_hub),
            append_to_hub=bool(args.append_to_hub),
        )


if __name__ == "__main__":
    main()
