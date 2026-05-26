"""SFT pre-training for TELL annotations on the acmc/TELL dataset.

Uses the exact same tokenizer prompt path as GRPO: ``format_prompt_for_model``
(logical ``row["text"]`` only; ``<<<>>>`` body from ``tell_xml.escape_document_piece``),

plus the analysis-channel stub so the supervised distribution matches GRPO sampling.

Usage:
    python -m rl_detector.sft.train_tinker_sft [hydra overrides]

Key overrides:
    sft.dataset_path="hf://acmc/TELL"
    sft.epochs=3
    wandb.name="sft_pretrain"
"""

from dotenv import load_dotenv
load_dotenv()

import asyncio
import datetime
import html
import json
import logging
import math
import pathlib
import random
import re

import hydra
import numpy as np
import tinker
import torch
import wandb
import weave
from omegaconf import DictConfig
from transformers import AutoTokenizer
from tqdm import tqdm

import rl_detector.config as config_module
import rl_detector.prompt_utils as prompt_utils_mod
import rl_detector.prompts as prompts_mod
import rl_detector.rewards as rewards_mod
from rl_detector.config import CFG
from rl_detector.annotation_utils import get_outer_bracket_metadata, strip_all_bracket_annotations
from rl_detector.prompts import label_think_continuation
from rl_detector.prompt_utils import (
    format_prompt_for_model,
    detect_assistant_generation_suffix,
    load_tokenizer,
)
from rl_detector.rewards import format_diagnostics, parse_indicators
from rl_detector.rollouts import (
    _MASK_ANN_CLOSE as _SFT_ANN_CLOSE,
    _MASK_ANN_OPEN as _SFT_ANN_OPEN,
    _MASK_ANN_WHY_Q as _SFT_WHY_Q,
    _MASK_TEXT_CLOSE as _SFT_TEXT_CLOSE,
    _MASK_VERDICT_OPEN as _SFT_VERDICT_OPEN,
)
from rl_detector.tell_xml import wrap_outer_logical_plain_mid
from rl_detector.sft.why_idf_dropout import (
    WhyCharNgramIdfScorer,
    apply_online_why_idf_nested_dropout,
    apply_paced_annotation_dropout,
)

LOGGER = logging.getLogger(__name__)
SEED = int(CFG.frozen.seed)
FINAL_EVAL_VALIDATION_EXAMPLES = 25

def _sft_trace_fields(ex: dict) -> dict:
    # strip huge token lists from weave payloads; text fields are what you inspect in the UI
    return {
        "text": ex["text"],
        "logical_doc": ex["logical_doc"],
        "formatted_prompt": ex["formatted_prompt"],
        "model_input_text": ex["model_input_text"],
        "target_text": ex["target_text"],
        "escaped_annotation": ex["escaped_annotation"],
        "annotation_before": ex.get("annotation_before", ""),
        "annotation_after": ex["annotation"],
        "nested_count_before": ex.get("nested_count_before", -1),
        "nested_count_after": ex.get("nested_count_after", -1),
        "coverage_fraction": ex.get("coverage_fraction", -1.0),
    }


@weave.op()
def trace_sft_fwdbwd_batch(step: int, epoch: int, batch_idx: int, examples: list[dict]) -> dict:
    return {
        "step": step,
        "epoch": epoch,
        "batch_idx": batch_idx,
        "examples": [_sft_trace_fields(e) for e in examples],
    }


@weave.op()
def trace_style_clm_batch(
    step: int, epoch: int, texts: list[str], model_inputs: list[str], loss: float
) -> dict:
    return {
        "step": step,
        "epoch": epoch,
        "loss": loss,
        "n_texts": len(texts),
        "texts": texts,
        "model_inputs": model_inputs,
    }


# -- Data loading --------------------------------------------------------------

def _normalize_comment_text(text: str) -> str:
    """XML-escape special chars, then convert straight quotes to curly typographic quotes."""
    # XML-escape & < > (leave quotes for curly-quote pass below)
    text = html.escape(text, quote=False)

    # Double quotes: opening after start/whitespace/open-bracket, closing elsewhere
    text = re.sub(r'(?:(?<=^)|(?<=\s)|(?<=[({\[]))"', "\u201c", text)
    text = text.replace('"', "\u201d")

    # Single quotes: contractions/possessives (letter'letter) -> curly apostrophe
    text = re.sub(r"(\w)'(\w)", r"\1" + "\u2019" + r"\2", text)
    # Opening single quote after start/whitespace/open-bracket
    text = re.sub(r"(?:(?<=^)|(?<=\s)|(?<=[({\[]))'", "\u2018", text)
    # Everything else (closing quote, stray apostrophe) -> right single quote
    text = text.replace("'", "\u2019")

    return text


def _load_style_clm_items(path: str) -> list[dict]:
    """Load structured style CLM items from a human_detectors JSONL file.

    DEPRECATED: superseded by _load_expert_annot_split / acmc/expert-annotated-TELL.
    Kept for backwards compatibility with existing config files; new runs should
    leave style_clm_path empty and set expert_annot_dataset_path instead.

    Each annotator entry becomes one item with: article, type_str ("AI"/"human"),
    score (confidence/5 + small noise, clamped 0..1), comment (normalized str).
    """
    _GUESS_MAP = {"Human-Generated": "human", "Machine-Generated": "AI"}
    p = pathlib.Path(path).expanduser()
    entries = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    items: list[dict] = []
    for entry in entries:
        article = (entry.get("article") or "").strip()
        if not article:
            continue
        for key in ("annotator_1", "annotator_2", "annotator_3", "annotator_4", "annotator_5"):
            ann = entry.get(key)
            if not isinstance(ann, dict):
                continue
            type_str = _GUESS_MAP.get(ann.get("guess", ""))
            if type_str is None:
                continue
            confidence = float(ann.get("confidence") or 3.0)
            score = max(0.0, min(1.0, confidence / 5.0 + random.uniform(-0.05, 0.05)))
            comment = _normalize_comment_text((ann.get("comment") or "").strip())
            items.append({"article": article, "type_str": type_str, "score": score, "comment": comment})
    LOGGER.info("loaded %d style CLM items from %s", len(items), path)
    return items



def _load_tell_split(dataset_path: str, split: str) -> list[dict]:
    """Load one split of the acmc/TELL dataset.

    Returns list of {"text", "annotation", "label"} dicts where label is 1=AI 0=human.
    """
    if not dataset_path.startswith("hf://"):
        raise ValueError(f"SFT dataset must be hf://..., got: {dataset_path!r}")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("datasets package required for hf:// dataset URLs") from exc

    spec = dataset_path.removeprefix("hf://")
    parts = spec.split("/")
    if len(parts) < 2:
        raise ValueError(f"Invalid HF dataset path: {dataset_path!r}")
    repo_id = f"{parts[0]}/{parts[1]}"

    LOGGER.info("loading dataset %s split=%s", repo_id, split)
    rows = list(load_dataset(repo_id, split=split))
    out: list[dict] = []
    for row in rows:
        row = dict(row)
        text = row.get("text") or ""
        raw_ann = row.get("annotation") or ""
        if not text or not raw_ann:
            continue
        annotation = raw_ann
        label = row.get("label")
        if label is None:
            label = row.get("is_ai")
        if label is None:
            outer = get_outer_bracket_metadata(annotation)
            label = 1 if (outer and outer.get("type") == "AI") else 0
        else:
            label = int(label)
        out.append({"text": text, "annotation": annotation, "label": label})
    LOGGER.info("loaded %d examples from %s/%s", len(out), repo_id, split)
    if not out:
        raise ValueError(f"No valid rows found in {dataset_path}/{split}")
    return out


def _load_expert_annot_split(dataset_path: str, split: str) -> list[dict]:
    """Load one split of acmc/expert-annotated-TELL.

    Each row uses sft_text (full <text>...<verdict.../></text>) as the annotation so
    the model learns the complete span + verdict completion, not just the verdict.
    Rows whose sft_text fails format_diagnostics are dropped before training.
    """
    if not dataset_path.startswith("hf://"):
        raise ValueError(f"expert_annot_dataset_path must be hf://..., got: {dataset_path!r}")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("datasets package required for hf:// dataset URLs") from exc

    spec = dataset_path.removeprefix("hf://")
    parts = spec.split("/")
    if len(parts) < 2:
        raise ValueError(f"Invalid HF dataset path: {dataset_path!r}")
    repo_id = f"{parts[0]}/{parts[1]}"

    import os

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    LOGGER.info("loading expert-annot dataset %s split=%s", repo_id, split)
    rows = list(load_dataset(repo_id, split=split, token=token))
    if rows and "generation_model" in rows[0]:
        models = {str(r.get("generation_model") or "") for r in rows[:50]}
        LOGGER.info("expert-annot generation_model sample (first 50 rows): %s", models)
    out: list[dict] = []
    n_skipped_format = 0
    for row in rows:
        row = dict(row)
        text = (row.get("text") or "").strip()
        sft_text = (row.get("sft_text") or "").strip()
        if not text or not sft_text:
            continue
        label = int(row.get("label") or 0)
        # Hard format gate: every example that enters training must be valid.
        diag = format_diagnostics(sft_text, text)
        if not diag["ok"]:
            n_skipped_format += 1
            continue
        out.append({"text": text, "annotation": sft_text, "label": label})
    if n_skipped_format:
        LOGGER.warning("expert-annot: dropped %d rows that failed format gate", n_skipped_format)
    LOGGER.info("loaded %d expert-annot examples from %s/%s", len(out), repo_id, split)
    if not out:
        raise ValueError(f"No valid rows found in {dataset_path}/{split}")
    return out


# -- Stub token detection -------------------------------------------------------

_stub_open_cache: list[int] | None = None
_stub_close_cache: list[int] | None = None
_think_already_open_cache: bool | None = None
_return_token_cache: list[int] | None = None


def _init_stub_tokens(tokenizer) -> tuple[list[int], list[int], bool]:
    global _stub_open_cache, _stub_close_cache, _think_already_open_cache
    if _stub_open_cache is None:
        suffix = detect_assistant_generation_suffix(tokenizer)
        think_already_open = "<|channel|>analysis<|message|>" in suffix
        _think_already_open_cache = think_already_open
        open_str = "" if think_already_open else "<|channel|>analysis<|message|>"
        close_str = "<|end|><|start|>assistant<|channel|>final<|message|>"
        _stub_open_cache = tokenizer.encode(open_str, add_special_tokens=False) if open_str else []
        _stub_close_cache = tokenizer.encode(close_str, add_special_tokens=False)
        LOGGER.info(
            "stub tokens: open=%d close=%d think_already_open=%s",
            len(_stub_open_cache), len(_stub_close_cache), think_already_open,
        )
    return _stub_open_cache, _stub_close_cache, _think_already_open_cache


def _init_return_tokens(tokenizer) -> list[int]:
    global _return_token_cache
    if _return_token_cache is None:
        _return_token_cache = tokenizer.encode("<|return|>", add_special_tokens=False)
        if not _return_token_cache:
            raise ValueError("could not tokenize <|return|> for SFT packing")
    return _return_token_cache


# -- Datum construction ---------------------------------------------------------

def _build_sft_datum(
    tokenizer,
    row: dict,
    inject_label_instruction: bool,
    annotation_xml: str,
) -> "tuple[tinker.Datum, list[tinker.Datum], dict] | None":
    """Build one SFT datum from a TELL row.

    Prompt tokens = tokenizer(``format_prompt_for_model(.., text=logical_doc)``) + stub_open + stub_close.
    Completion = annotation tokens (only these get loss weight 1.0).
    """
    stub_open, stub_close, _ = _init_stub_tokens(tokenizer)
    return_tokens = _init_return_tokens(tokenizer)

    # Full sequence: prompt | stub_open | optional_label_instruction | stub_close | annotation | return
    ann_xml = annotation_xml
    inner_plain = strip_all_bracket_annotations(ann_xml)
    logical_doc = row["text"]

    _, formatted_prompt = format_prompt_for_model(tokenizer=tokenizer, text=logical_doc)
    prompt_tokens = tokenizer.encode(formatted_prompt, add_special_tokens=False)

    if inner_plain != logical_doc:
        return None

    diag = format_diagnostics(ann_xml, logical_doc)
    if not diag["ok"]:
        return None
    # hard invariant: every example that enters model training must pass format gate
    assert diag["ok"], f"format gate failed for training example: reason={diag.get('reason')}"

    annotation_tokens = tokenizer.encode(ann_xml, add_special_tokens=False)
    packed_annotation_tokens = annotation_tokens + return_tokens

    analysis_content_tokens: list[int] = []
    if inject_label_instruction:
        # this mirrors sampling-time label stub style; no focus hint in SFT
        analysis_content_tokens = tokenizer.encode(
            label_think_continuation(int(row["label"])),
            add_special_tokens=False,
        )

    full_tokens = (
        prompt_tokens
        + stub_open
        + analysis_content_tokens
        + stub_close
        + packed_annotation_tokens
    )
    if len(full_tokens) < 2:
        return None

    n_prefix = len(prompt_tokens) + len(stub_open) + len(analysis_content_tokens) + len(stub_close)
    model_input = full_tokens[:-1]
    target_tokens = full_tokens[1:]
    # Only train on annotation tokens (skip the prefix and the first annotation token
    # that was shifted into position by the model_input/target pairing)
    n_prefix_in_target = max(0, n_prefix - 1)
    weights = [0.0] * n_prefix_in_target + [1.0] * (len(target_tokens) - n_prefix_in_target)

    datum = tinker.Datum(
        model_input=tinker.ModelInput.from_ints(model_input),
        loss_fn_inputs={
            "target_tokens": torch.tensor(target_tokens, dtype=torch.long),
            "weights": torch.tensor(weights, dtype=torch.float32),
        },
    )

    hint_ce_datums: list[tinker.Datum] = []
    if inject_label_instruction and bool(getattr(CFG.sft, "hint_outer_ce_enabled", False)):
        full_prefix = prompt_tokens + stub_open + analysis_content_tokens + stub_close
        true_label = int(row["label"])
        # Forward: original hint -> correct type
        fwd = _build_sft_outer_type_ce_datum(
            tokenizer=tokenizer,
            full_prefix_tokens=full_prefix,
            annotation_tokens=annotation_tokens,
            return_tokens=return_tokens,
            target_type=true_label,
        )
        if fwd is not None:
            hint_ce_datums.append(fwd)
        # Contrastive: flipped hint -> opposite type (same text, only type="" spliced)
        flipped_prefix = _flip_hint_in_prefix_tokens(full_prefix, true_label, tokenizer)
        if flipped_prefix is not None:
            contra = _build_sft_outer_type_ce_datum(
                tokenizer=tokenizer,
                full_prefix_tokens=flipped_prefix,
                annotation_tokens=annotation_tokens,
                return_tokens=return_tokens,
                target_type=1 - true_label,
            )
            if contra is not None:
                hint_ce_datums.append(contra)

    # keep raw packed text so we can inspect exacctly what fwd-bwd sees
    raw_full_text = tokenizer.decode(full_tokens, skip_special_tokens=False)
    raw_model_input_text = tokenizer.decode(model_input, skip_special_tokens=False)
    raw_target_text = tokenizer.decode(target_tokens, skip_special_tokens=False)
    audit_row = {
        "text": row["text"],
        "logical_doc": logical_doc,
        "annotation": annotation_xml,
        "escaped_annotation": ann_xml,
        "annotation_with_return": ann_xml + "<|return|>",
        "formatted_prompt": formatted_prompt,
        "full_text": raw_full_text,
        "model_input_text": raw_model_input_text,
        "target_text": raw_target_text,
        "model_input_token_ids": model_input,
        "target_token_ids": target_tokens,
        "weights": weights,
        "inject_label_instruction": inject_label_instruction,
        "hint_ce_n": len(hint_ce_datums),
    }
    return datum, hint_ce_datums, audit_row


def _build_datums(
    rows: list[dict], tokenizer
) -> tuple[list[tinker.Datum], list[list[tinker.Datum]], list[dict]]:
    datums: list[tinker.Datum] = []
    hint_ce_datums: list[list[tinker.Datum]] = []
    audit_rows: list[dict] = []
    skipped = 0
    mix_ratio = float(getattr(CFG.sft, "label_injection_mix_ratio", 0.0))
    mix_ratio = max(0.0, min(1.0, mix_ratio))
    inject_enabled = bool(getattr(CFG.sft, "label_injection_enabled", False))
    rng = random.Random(SEED)
    paced_enabled = bool(getattr(getattr(CFG.sft, "paced_annotation_dropout", object()), "enabled", True))
    paced_words_per_ann = float(getattr(getattr(CFG.sft, "paced_annotation_dropout", object()), "words_per_annotation", 20.0))
    paced_score_bonus = float(getattr(getattr(CFG.sft, "paced_annotation_dropout", object()), "high_score_keep_bonus", 3.0))

    for row_i, row in enumerate(rows):
        inject_label_instruction = inject_enabled and (rng.random() < mix_ratio)
        ann_xml = row["annotation"]
        if paced_enabled:
            ann_xml, _nb, _na = apply_paced_annotation_dropout(
                logical_document=row["text"],
                annotation_xml=ann_xml,
                rng=random.Random(SEED + row_i),
                words_per_annotation=paced_words_per_ann,
                high_score_keep_bonus=paced_score_bonus,
            )
        built = _build_sft_datum(
            tokenizer=tokenizer,
            row=row,
            inject_label_instruction=inject_label_instruction,
            annotation_xml=ann_xml,
        )
        if built is None:
            skipped += 1
            continue
        datum, row_hint_ce_datums, audit_row = built
        datums.append(datum)
        hint_ce_datums.append(row_hint_ce_datums)
        audit_rows.append(audit_row)
    if skipped:
        LOGGER.warning(
            "skipped %d rows (inner!=text / format_diag fail / empty tokens)",
            skipped,
        )
    if inject_enabled:
        n_injected = sum(1 for row in audit_rows if row.get("inject_label_instruction"))
        LOGGER.info(
            "SFT label injection mix: injected=%d/%d (ratio=%.3f target=%.3f)",
            n_injected,
            len(audit_rows),
            (n_injected / len(audit_rows)) if audit_rows else 0.0,
            mix_ratio,
        )
    # hard invariant: no non-gated rows are allowed past datum construction
    for row in audit_rows:
        check = format_diagnostics(row["annotation"], row["text"])
        assert check["ok"], f"format gate invariant broken in built datums: reason={check.get('reason')}"
    return datums, hint_ce_datums, audit_rows


def _flip_hint_in_prefix_tokens(
    prefix_tokens: list[int],
    hint: int,
    tokenizer,
) -> list[int] | None:
    """Return a copy of prefix_tokens with the hint text swapped to the opposite label.

    Mirrors _flip_hint_in_prompt_tokens in train.py.
    """
    current_toks = tokenizer.encode(label_think_continuation(hint), add_special_tokens=False)
    flipped_toks = tokenizer.encode(label_think_continuation(1 - hint), add_special_tokens=False)
    n = len(current_toks)
    toks = list(prefix_tokens)
    for i in range(len(toks) - n + 1):
        if toks[i : i + n] == current_toks:
            return toks[:i] + flipped_toks + toks[i + n :]
    return None


def _build_sft_outer_type_ce_datum(
    tokenizer,
    full_prefix_tokens: list[int],
    annotation_tokens: list[int],
    return_tokens: list[int],
    target_type: int,
) -> tinker.Datum | None:
    """CE datum with weight=1 ONLY on the outer annotation type value tokens.

    Targets the <verdict type="..."> block (new format) when present, falling back to the
    last <annotation type="..."> (old/legacy format).  Only the type token(s) are supervised;
    nothing after type="..." is ever trained on so the causal chain (explanation -> label)
    is preserved in the contrastive pair.

    full_prefix_tokens must include the hint text (prompt + stub_open + hint + stub_close).
    target_type: 0=human, 1=AI.
    """
    # Prefer the VERDICT block (new format): VERDICT_PREFIX ... TEXT_CLOSE
    last_open = max(
        (i for i, t in enumerate(annotation_tokens) if int(t) == _SFT_VERDICT_OPEN), default=-1
    )
    last_close = max(
        (i for i, t in enumerate(annotation_tokens) if int(t) == _SFT_TEXT_CLOSE), default=-1
    ) if last_open >= 0 else -1
    # Fall back to legacy outer span: last ANN_PREFIX ... ANN_CLOSE
    if last_open < 0 or last_close <= last_open:
        last_open = max(
            (i for i, t in enumerate(annotation_tokens) if int(t) == _SFT_ANN_OPEN), default=-1
        )
        last_close = max(
            (i for i, t in enumerate(annotation_tokens) if int(t) == _SFT_ANN_CLOSE), default=-1
        )
    if last_open < 0 or last_close <= last_open:
        return None

    attr_tokens = list(annotation_tokens[last_open + 1 : last_close])

    _ai_toks: list[int] = tokenizer.encode("AI", add_special_tokens=False)
    _human_toks: list[int] = tokenizer.encode("human", add_special_tokens=False)
    target_toks: list[int] = _ai_toks if target_type == 1 else _human_toks

    # Find the type-value span (either "AI" or "human") anchored by type=" pre-context.
    # Scan backward so the last match (outermost type=) wins if there are nested tells.
    orig_start = -1
    orig_end = -1
    for _cand in [_ai_toks, _human_toks]:
        _nc = len(_cand)
        for _s in range(len(attr_tokens) - _nc, -1, -1):
            if attr_tokens[_s : _s + _nc] == _cand:
                _pre = tokenizer.decode(
                    annotation_tokens[last_open : last_open + 1 + _s],
                    skip_special_tokens=False,
                )
                _post = tokenizer.decode(attr_tokens[_s + _nc : _s + _nc + 2], skip_special_tokens=False)
                if _pre.endswith('type="') and _post.startswith('"'):
                    orig_start = _s
                    orig_end = _s + _nc
                    break
        if orig_start >= 0:
            break

    if orig_start < 0:
        return None

    # Splice target type into attr_tokens (no-op when annotation already has target_type).
    spliced_attr = attr_tokens[:orig_start] + target_toks + attr_tokens[orig_end:]
    type_tok_start = orig_start
    type_tok_end = orig_start + len(target_toks)

    # Rebuild annotation_tokens with spliced attrs.
    spliced_ann = (
        list(annotation_tokens[: last_open + 1])
        + spliced_attr
        + list(annotation_tokens[last_close:])
    )

    full_tokens = list(full_prefix_tokens) + spliced_ann + list(return_tokens)
    if len(full_tokens) < 2:
        return None

    model_input = full_tokens[:-1]
    target_tokens_list = full_tokens[1:]

    # annotation_tokens[last_open] is at full_tokens[n_prefix + last_open].
    # spliced_attr[type_tok_start] is at full_tokens[n_prefix + last_open + 1 + type_tok_start].
    # In target_tokens (= full_tokens[1:]) subtract 1 from each index.
    n_prefix = len(full_prefix_tokens)
    val_start = n_prefix + last_open + type_tok_start      # = n_prefix + last_open + 1 + type_tok_start - 1
    val_end = n_prefix + last_open + type_tok_end

    weights = [0.0] * len(target_tokens_list)
    for idx in range(val_start, min(val_end, len(weights))):
        weights[idx] = 1.0

    if not any(w > 0.0 for w in weights):
        return None

    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(model_input),
        loss_fn_inputs={
            "target_tokens": torch.tensor(target_tokens_list, dtype=torch.long),
            "weights": torch.tensor(weights, dtype=torch.float32),
        },
    )


def _build_style_clm_datum(tokenizer, item: dict) -> tinker.Datum | None:
    """Build a style CLM datum supervised ONLY on the verdict why value + score.

    DEPRECATED: superseded by expert-annotated datums built via _build_sft_datum
    from acmc/expert-annotated-TELL, which supervise the full completion rather than
    just the verdict why/score.
    """
    stub_open, stub_close, _ = _init_stub_tokens(tokenizer)
    return_tokens = _init_return_tokens(tokenizer)

    article = item["article"]
    type_str = item["type_str"]
    score = item["score"]
    comment = item["comment"]

    _, formatted_prompt = format_prompt_for_model(tokenizer=tokenizer, text=article)
    prompt_tokens = tokenizer.encode(formatted_prompt, add_special_tokens=False)

    # wrap_outer_logical_plain_mid handles escape_document_piece + escape_attr_piece + tag format
    ann_xml = wrap_outer_logical_plain_mid(
        article, {"type": type_str, "why": comment, "score": f"{score:.2f}"}
    )
    ann_tokens = tokenizer.encode(ann_xml, add_special_tokens=False)

    # Supervise the VERDICT block (new format): VERDICT_PREFIX ... TEXT_CLOSE.
    # The verdict why/score are the signals that drive RL reward, so that's what the
    # style CLM should teach. Fall back to legacy outer span if verdict is absent.
    verdict_open = max(
        (i for i, t in enumerate(ann_tokens) if int(t) == _SFT_VERDICT_OPEN), default=-1
    )
    text_close = max(
        (i for i, t in enumerate(ann_tokens) if int(t) == _SFT_TEXT_CLOSE), default=-1
    ) if verdict_open >= 0 else -1

    if verdict_open >= 0 and text_close > verdict_open:
        # New format: find WHY_Q token inside verdict block, supervise from why value -> TEXT_CLOSE.
        why_q_pos = next(
            (i for i in range(verdict_open + 1, text_close) if int(ann_tokens[i]) == _SFT_WHY_Q),
            -1,
        )
        if why_q_pos < 0:
            return None
        last_open = verdict_open
        why_val_abs = why_q_pos + 1   # absolute index in ann_tokens of first why-value token
        last_close = text_close
    else:
        # Legacy format: scan for why=" in outer span annotation.
        last_open = max(
            (i for i, t in enumerate(ann_tokens) if int(t) == _SFT_ANN_OPEN), default=-1
        )
        last_close = max(
            (i for i, t in enumerate(ann_tokens) if int(t) == _SFT_ANN_CLOSE), default=-1
        )
        if last_open < 0 or last_close <= last_open:
            return None
        attr_tokens = ann_tokens[last_open + 1 : last_close]
        why_val_start_in_attr = -1
        for _end in range(1, len(attr_tokens) + 1):
            if tokenizer.decode(attr_tokens[:_end], skip_special_tokens=False).endswith('why="'):
                why_val_start_in_attr = _end
                break
        if why_val_start_in_attr < 0:
            return None
        why_val_abs = last_open + 1 + why_val_start_in_attr

    # No analysis content (no hint for style CLM); use empty stub channel for format consistency.
    full_tokens = prompt_tokens + stub_open + stub_close + ann_tokens + return_tokens
    if len(full_tokens) < 2:
        return None

    model_input = full_tokens[:-1]
    target_tokens_list = full_tokens[1:]

    # ann_tokens[why_val_abs] = full_tokens[n_prefix + why_val_abs]
    # In target_tokens_list (= full_tokens[1:]): subtract 1 -> n_prefix + why_val_abs - 1
    n_prefix = len(prompt_tokens) + len(stub_open) + len(stub_close)
    w_start = n_prefix + why_val_abs - 1
    w_end = n_prefix + last_close  # exclusive upper bound (includes TEXT_CLOSE / ANN_CLOSE in target)

    weights = [0.0] * len(target_tokens_list)
    for idx in range(w_start, min(w_end, len(weights))):
        weights[idx] = 1.0

    if not any(w > 0.0 for w in weights):
        return None

    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(model_input),
        loss_fn_inputs={
            "target_tokens": torch.tensor(target_tokens_list, dtype=torch.long),
            "weights": torch.tensor(weights, dtype=torch.float32),
        },
    )


def _build_style_clm_pool(items: list[dict], tokenizer) -> tuple[list[tinker.Datum], list[str]]:
    pairs = [(item, _build_style_clm_datum(tokenizer, item)) for item in items]
    pool_pairs = [(item["article"], d) for item, d in pairs if d is not None]
    texts = [t for t, _ in pool_pairs]
    datums = [d for _, d in pool_pairs]
    LOGGER.info("built %d style CLM datums from %d items", len(datums), len(items))
    return datums, texts


def _batch(items, batch_size: int) -> list:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def _extract_type_logprobs(datums: list[tinker.Datum], output) -> list[float]:
    """Extract mean logprob at weighted (type-value) positions for each CE datum."""
    loss_fn_outputs = getattr(output, "loss_fn_outputs", None) or []
    result: list[float] = []
    for datum, loss_out in zip(datums, loss_fn_outputs, strict=False):
        if not isinstance(loss_out, dict):
            continue
        logprobs_raw = loss_out.get("logprobs")
        if logprobs_raw is None:
            continue
        lp = np.asarray(getattr(logprobs_raw, "data", logprobs_raw), dtype=np.float32).reshape(-1)
        w = np.asarray(
            getattr(datum.loss_fn_inputs["weights"], "data", datum.loss_fn_inputs["weights"]),
            dtype=np.float32,
        ).reshape(-1)
        if lp.shape != w.shape:
            continue
        weighted_lp = lp[w > 0.0]
        if len(weighted_lp) == 0:
            continue
        result.append(float(weighted_lp.mean()))
    return result


def _hint_ce_proxy_metrics(datums: list[tinker.Datum], output) -> dict:
    """Cheap per-step proxy metrics from the CE forward_backward output.

    NOTE: follow_rate here uses P(correct_type) > 0.5, NOT P(correct) > P(wrong) for
    the same context.  It is used only to drive the adaptive EMA -- not the primary
    accuracy metric.  The proper measurement is eval_hint_follow_rate computed per epoch
    via paired forward passes (see _eval_hint_follow_rate).
    """
    lps = _extract_type_logprobs(datums, output)
    if not lps:
        return {"hint_ce_type_prob": float("nan"), "hint_ce_follow_rate_proxy": float("nan")}
    probs = [math.exp(lp) for lp in lps]
    return {
        "hint_ce_type_prob": sum(probs) / len(probs),
        "hint_ce_follow_rate_proxy": sum(1.0 for p in probs if p > 0.5) / len(probs),
    }


def _build_hint_follow_eval_pairs(
    rows: list[dict], tokenizer
) -> list[tuple[tinker.Datum, tinker.Datum]]:
    """Build (correct_datum, wrong_datum) pairs for proper hint-follow evaluation.

    Both datums in a pair share the same prefix (prompt + stub_open + hint + stub_close)
    but differ only in the spliced outer annotation type token.  Comparing their logprobs
    at the weighted position gives P(correct_type | hint_context) vs
    P(wrong_type | hint_context) for the SAME context -- the correct follow-rate signal.
    """
    stub_open, stub_close, _ = _init_stub_tokens(tokenizer)
    return_tokens = _init_return_tokens(tokenizer)
    pairs: list[tuple[tinker.Datum, tinker.Datum]] = []
    for row in rows:
        ann_xml = row["annotation"]
        if not format_diagnostics(ann_xml, row["text"])["ok"]:
            continue
        true_label = int(row["label"])
        _, formatted_prompt = format_prompt_for_model(tokenizer=tokenizer, text=row["text"])
        prompt_tokens = tokenizer.encode(formatted_prompt, add_special_tokens=False)
        hint_tokens = tokenizer.encode(
            label_think_continuation(true_label), add_special_tokens=False
        )
        full_prefix = prompt_tokens + stub_open + hint_tokens + stub_close
        annotation_tokens = tokenizer.encode(ann_xml, add_special_tokens=False)
        correct = _build_sft_outer_type_ce_datum(
            tokenizer=tokenizer,
            full_prefix_tokens=full_prefix,
            annotation_tokens=annotation_tokens,
            return_tokens=return_tokens,
            target_type=true_label,
        )
        wrong = _build_sft_outer_type_ce_datum(
            tokenizer=tokenizer,
            full_prefix_tokens=full_prefix,
            annotation_tokens=annotation_tokens,
            return_tokens=return_tokens,
            target_type=1 - true_label,
        )
        if correct is not None and wrong is not None:
            pairs.append((correct, wrong))
    return pairs


async def _eval_hint_follow_rate(
    training_client, pairs: list[tuple[tinker.Datum, tinker.Datum]], batch_size: int
) -> dict:
    """True hint-follow rate: fraction where P(correct_type|hint) > P(wrong_type|hint).

    Runs two forward passes (correct and wrong type) on the same hint-conditioned prefix
    and compares logprobs at the type-value positions.  Also reports mean log-odds
    (log P(correct) - log P(wrong)) as a calibration measure.
    """
    if not pairs:
        return {
            "eval_hint_follow_rate": float("nan"),
            "eval_hint_log_odds_mean": float("nan"),
        }
    correct_datums = [p[0] for p in pairs]
    wrong_datums = [p[1] for p in pairs]

    correct_lps: list[float] = []
    for batch in _batch(correct_datums, batch_size):
        fut = await training_client.forward_async(data=batch, loss_fn="cross_entropy")
        out = await fut.result_async()
        correct_lps.extend(_extract_type_logprobs(batch, out))

    wrong_lps: list[float] = []
    for batch in _batch(wrong_datums, batch_size):
        fut = await training_client.forward_async(data=batch, loss_fn="cross_entropy")
        out = await fut.result_async()
        wrong_lps.extend(_extract_type_logprobs(batch, out))

    n = min(len(correct_lps), len(wrong_lps))
    if n == 0:
        return {
            "eval_hint_follow_rate": float("nan"),
            "eval_hint_log_odds_mean": float("nan"),
        }
    follow = [correct_lps[i] > wrong_lps[i] for i in range(n)]
    log_odds = [correct_lps[i] - wrong_lps[i] for i in range(n)]
    return {
        "eval_hint_follow_rate": sum(follow) / n,
        "eval_hint_log_odds_mean": sum(log_odds) / n,
    }


def _scalar_loss(datums: list[tinker.Datum], output) -> float:
    """Compute the mean cross-entropy loss for a Tinker forward output."""
    loss = getattr(output, "loss", None)
    if loss is not None:
        return float(loss)

    metrics = getattr(output, "metrics", None) or {}
    for key in ("loss:mean", "loss", "cross_entropy:mean", "ce:mean"):
        if key in metrics:
            return float(metrics[key])

    loss_outputs = getattr(output, "loss_fn_outputs", None) or []
    values: list[float] = []
    for datum, loss_output in zip(datums, loss_outputs, strict=False):
        logprobs = loss_output.get("logprobs") if isinstance(loss_output, dict) else None
        if logprobs is None:
            continue
        logprob_values = np.asarray(getattr(logprobs, "data", logprobs), dtype=np.float32).reshape(-1)
        weights = np.asarray(
            getattr(datum.loss_fn_inputs["weights"], "data", datum.loss_fn_inputs["weights"]),
            dtype=np.float32,
        ).reshape(-1)
        if logprob_values.shape != weights.shape:
            continue
        weight_sum = float(weights.sum())
        if weight_sum <= 0.0:
            continue
        values.append(float(-(logprob_values * weights).sum() / weight_sum))

    if values:
        return sum(values) / len(values)

    raise AttributeError("Tinker forward output does not expose a scalar loss")


# -- Loss evaluation ------------------------------------------------------------

async def _mean_eval_loss(training_client, eval_batches: list) -> float:
    if not eval_batches:
        return float("nan")
    losses: list[float] = []
    for batch in tqdm(eval_batches, desc="eval loss", leave=False):
        future = await training_client.forward_async(data=batch, loss_fn="cross_entropy")
        out = await future.result_async()
        losses.append(_scalar_loss(batch, out))
    return sum(losses) / len(losses)


# -- Format pass-rate -----------------------------------------------------------

def _format_pass_rate(rows: list[dict]) -> float:
    """Fraction of rows whose annotation parses correctly as bracket format."""
    if not rows:
        return float("nan")
    n_ok = sum(
        1 for row in rows
        if format_diagnostics(row["annotation"], row["text"])["ok"]
    )
    return n_ok / len(rows)


# -- Config rebinding -----------------------------------------------------------

def _rebind_config(cfg: DictConfig) -> None:
    config_module.CFG = cfg
    prompt_utils_mod.CFG = cfg
    prompts_mod.CFG = cfg
    rewards_mod.CFG = cfg
    global CFG, SEED
    CFG = cfg
    SEED = int(CFG.frozen.seed)


# -- Main training loop ---------------------------------------------------------

async def _train() -> None:
    sft_cfg = CFG.sft
    output_dir = pathlib.Path(sft_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    _weave_enabled = bool(getattr(CFG.wandb, "weave_trace", True))
    if _weave_enabled:
        _ent = getattr(CFG.wandb, "entity", None) or ""
        _proj = f"{_ent}/{CFG.wandb.project}" if _ent else CFG.wandb.project
        weave.init(project_name=_proj, settings={"print_call_link": False, "log_level": "WARNING"})

    # WandB initialisation
    if getattr(CFG.wandb, "enabled", False):
        wandb.init(
            project=CFG.wandb.project,
            entity=CFG.wandb.entity or None,
            name=getattr(CFG.wandb, "name", "sft_tell"),
            config={
                "base_model": CFG.model.base_model,
                "lora_rank": CFG.model.lora_rank,
                "epochs": int(sft_cfg.epochs),
                "batch_size": int(sft_cfg.batch_size),
                "learning_rate": float(sft_cfg.learning_rate),
                "dataset_path": sft_cfg.dataset_path,
                "label_injection_enabled": bool(getattr(sft_cfg, "label_injection_enabled", False)),
                "label_injection_mix_ratio": float(getattr(sft_cfg, "label_injection_mix_ratio", 0.0)),
                "expert_annot_dataset_path": str(getattr(sft_cfg, "expert_annot_dataset_path", "") or ""),
                "expert_annot_oversample_factor": int(getattr(sft_cfg, "expert_annot_oversample_factor", 1)),
                "tell_max_rows": getattr(sft_cfg, "tell_max_rows", None),
                "style_clm_path": str(getattr(sft_cfg, "style_clm_path", "") or ""),
                "style_clm_every_n_steps": int(getattr(sft_cfg, "style_clm_every_n_steps", 5)),
                "style_clm_lr_scale": float(getattr(sft_cfg, "style_clm_lr_scale", 0.5)),
                "style_clm_max_tokens": int(getattr(sft_cfg, "style_clm_max_tokens", 512)),
                "why_idf_dropout_enabled": bool(sft_cfg.why_idf_dropout_enabled),
                "why_idf_genericity_mode": str(sft_cfg.why_idf_genericity_mode),
                "why_idf_why_bucket_max_chars": int(sft_cfg.why_idf_why_bucket_max_chars),
                "why_idf_why_bucket_key_chars": sft_cfg.why_idf_why_bucket_key_chars,
                "why_idf_span_why_jaccard_weight": float(sft_cfg.why_idf_span_why_jaccard_weight),
                "why_idf_span_why_min_bucket_n_for_log": int(sft_cfg.why_idf_span_why_min_bucket_n_for_log),
                "why_idf_span_why_overlap_p_mass": str(sft_cfg.why_idf_span_why_overlap_p_mass),
                "why_idf_span_why_excess_quantile": float(sft_cfg.why_idf_span_why_excess_quantile),
                "why_idf_span_lemmatizer": "en_blank_rule",
                "why_idf_spacy_pipe_batch_size": int(sft_cfg.why_idf_spacy_pipe_batch_size),
                "why_idf_span_spacy_exclude_stopwords": bool(sft_cfg.why_idf_span_spacy_exclude_stopwords),
                "why_idf_char_ngram_min": int(sft_cfg.why_idf_char_ngram_min),
                "why_idf_char_ngram_max": int(sft_cfg.why_idf_char_ngram_max),
                "why_idf_drop_strength": float(sft_cfg.why_idf_drop_strength),
                "why_idf_p_drop_cap": float(sft_cfg.why_idf_p_drop_cap),
                "why_idf_score_keep_weight": float(sft_cfg.why_idf_score_keep_weight),
                "why_idf_keep_min": int(getattr(sft_cfg, "why_idf_keep_min", 1)),
                "why_idf_keep_max": int(getattr(sft_cfg, "why_idf_keep_max", 5)),
            },
        )

    # Load dataset splits
    base_path = sft_cfg.dataset_path.rstrip("/")
    if base_path.endswith("/train"):
        base_path = base_path[: -len("/train")]
    if base_path:
        train_rows = _load_tell_split(base_path, "train")
        _tell_max = getattr(sft_cfg, "tell_max_rows", None)
        if _tell_max is not None and int(_tell_max) < len(train_rows):
            train_rows = random.Random(SEED).sample(train_rows, int(_tell_max))
            LOGGER.info("TELL undersampled to %d rows (from full %d)", len(train_rows), int(_tell_max))
        eval_rows = _load_tell_split(base_path, "validation")
        eval_rows = random.Random(SEED).sample(
            eval_rows,
            min(FINAL_EVAL_VALIDATION_EXAMPLES, len(eval_rows)),
        )
    else:
        LOGGER.info("dataset_path is empty -- skipping TELL dataset; using expert_annot_dataset_path only")
        train_rows = []
        eval_rows = []

    # Expert-annotated dataset (acmc/expert-annotated-TELL).
    # Teaches the model complete span+verdict completions grounded in human rationale.
    # Uses the same label-injection mix ratio as the TELL dataset.
    _expert_annot_path = str(getattr(sft_cfg, "expert_annot_dataset_path", "") or "")
    if _expert_annot_path:
        expert_train_rows = _load_expert_annot_split(_expert_annot_path, "train")
        _oversample = int(getattr(sft_cfg, "expert_annot_oversample_factor", 1))
        if _oversample > 1:
            expert_train_rows = expert_train_rows * _oversample
            LOGGER.info(
                "expert-annot oversample: %dx -> %d rows (from %d unique)",
                _oversample, len(expert_train_rows), len(expert_train_rows) // _oversample,
            )
        expert_eval_rows: list[dict] = []
        for split_name in ("validation", "test"):
            try:
                expert_eval_rows = _load_expert_annot_split(_expert_annot_path, split_name)
                LOGGER.info("expert-annot eval split=%s n=%d", split_name, len(expert_eval_rows))
                break
            except ValueError:
                continue
        if not expert_eval_rows:
            LOGGER.info(
                "expert-annot has no validation/test split; all %d rows go to train, SFT eval disabled",
                len(expert_train_rows),
            )
        train_rows = train_rows + expert_train_rows
        random.Random(SEED + 7).shuffle(train_rows)
        LOGGER.info(
            "merged expert-annot into training: +%d train rows",
            len(expert_train_rows),
        )
        if expert_eval_rows:
            eval_rows = eval_rows + random.Random(SEED + 7).sample(
                expert_eval_rows,
                min(FINAL_EVAL_VALIDATION_EXAMPLES, len(expert_eval_rows)),
            )

    tokenizer = load_tokenizer()

    # Log format pass-rate for ground-truth annotations (sanity check)
    train_fpr = _format_pass_rate(train_rows[:200])
    eval_fpr = _format_pass_rate(eval_rows)
    LOGGER.info("annotation format pass-rate: train=%.3f eval=%.3f", train_fpr, eval_fpr)

    LOGGER.info("building datums: train=%d eval=%d", len(train_rows), len(eval_rows))
    mix_ratio = float(getattr(sft_cfg, "label_injection_mix_ratio", 0.0))
    mix_ratio = max(0.0, min(1.0, mix_ratio))
    inject_enabled = bool(getattr(sft_cfg, "label_injection_enabled", False))
    why_idf_enabled = bool(sft_cfg.why_idf_dropout_enabled)
    _paced_cfg = getattr(sft_cfg, "paced_annotation_dropout", None)
    paced_enabled = bool(getattr(_paced_cfg, "enabled", True))
    paced_words_per_ann = float(getattr(_paced_cfg, "words_per_annotation", 20.0))
    paced_score_bonus = float(getattr(_paced_cfg, "high_score_keep_bonus", 3.0))
    if paced_enabled:
        LOGGER.info(
            "paced_annotation_dropout ENABLED: target 1 ann per %.0f words, high_score_keep_bonus=%.1f",
            paced_words_per_ann, paced_score_bonus,
        )
    why_idf_mode = str(sft_cfg.why_idf_genericity_mode)
    why_idf_bucket_chars = int(sft_cfg.why_idf_why_bucket_max_chars)
    _why_key_ov = sft_cfg.why_idf_why_bucket_key_chars
    why_idf_bucket_key_chars = (
        int(_why_key_ov) if _why_key_ov is not None else why_idf_bucket_chars
    )
    why_idf_jaccard_w = float(sft_cfg.why_idf_span_why_jaccard_weight)
    why_idf_min_bucket_n = int(sft_cfg.why_idf_span_why_min_bucket_n_for_log)
    why_idf_overlap_pm = str(sft_cfg.why_idf_span_why_overlap_p_mass)
    why_idf_excess_q = float(sft_cfg.why_idf_span_why_excess_quantile)
    why_idf_spacy_batch = int(sft_cfg.why_idf_spacy_pipe_batch_size)
    why_idf_spacy_no_stop = bool(sft_cfg.why_idf_span_spacy_exclude_stopwords)
    why_idf_n_min = int(sft_cfg.why_idf_char_ngram_min)
    why_idf_n_max = int(sft_cfg.why_idf_char_ngram_max)
    why_idf_strength = float(sft_cfg.why_idf_drop_strength)
    why_idf_p_cap = float(sft_cfg.why_idf_p_drop_cap)
    why_idf_score_keep_w = float(sft_cfg.why_idf_score_keep_weight)
    why_idf_keep_min = int(getattr(sft_cfg, "why_idf_keep_min", 1))
    why_idf_keep_max = int(getattr(sft_cfg, "why_idf_keep_max", 5))
    why_scorer: WhyCharNgramIdfScorer | None = None
    if why_idf_enabled:
        LOGGER.warning(
            "why_idf DROPOUT ENABLED: apply_online_why_idf_nested_dropout removes nested tells "
            "(keep_min=%d keep_max=%d per doc); set sft.why_idf_dropout_enabled=false to train on all annotations",
            why_idf_keep_min,
            why_idf_keep_max,
        )
        why_scorer = WhyCharNgramIdfScorer.from_train_rows(
            rows=train_rows,
            n_min=why_idf_n_min,
            n_max=why_idf_n_max,
            genericity_mode=why_idf_mode,
            why_key_max_chars=why_idf_bucket_chars,
            why_bucket_key_chars=why_idf_bucket_key_chars,
            span_why_jaccard_weight=why_idf_jaccard_w,
            span_why_min_bucket_n_for_log=why_idf_min_bucket_n,
            span_why_overlap_p_mass_style=why_idf_overlap_pm,
            span_why_excess_quantile=why_idf_excess_q,
            spacy_pipe_batch_size=why_idf_spacy_batch,
            spacy_exclude_stopwords=why_idf_spacy_no_stop,
        )
        assert why_scorer is not None
        LOGGER.info(
            "why_idf nested-why dropout (outer never touched): mode=%s overlap_p_mass=%s excess_q=%.3f "
            "span_lemma=en_blank_rule bucket_max=%d bucket_key=%d jaccard_w=%.3f min_bucket_n=%d ngram=%d-%d strength=%.3f p_cap=%.3f score_keep_w=%.3f",
            why_idf_mode,
            why_idf_overlap_pm,
            why_idf_excess_q,
            why_idf_bucket_chars,
            why_idf_bucket_key_chars,
            why_idf_jaccard_w,
            why_idf_min_bucket_n,
            why_idf_n_min,
            why_idf_n_max,
            why_idf_strength,
            why_idf_p_cap,
            why_idf_score_keep_w,
        )
        train_datums: list[tinker.Datum] = []
        train_hint_ce_datums: list[list[tinker.Datum]] = []
        train_audit_rows: list[dict] = []
    else:
        LOGGER.info(
            "why_idf dropout DISABLED: every training step uses the full annotation from the dataset "
            "(no nested tell removal)",
        )
        train_datums, train_hint_ce_datums, train_audit_rows = _build_datums(rows=train_rows, tokenizer=tokenizer)
        if train_rows:
            _nested_n = [
                len(parse_indicators(output=row["annotation"], document=row["text"]) or [])
                for row in train_rows
            ]
            LOGGER.info(
                "full-annotation nested tell counts: mean=%.2f min=%d max=%d n_rows=%d",
                sum(_nested_n) / len(_nested_n),
                min(_nested_n),
                max(_nested_n),
                len(_nested_n),
            )
    eval_datums, _, _ = _build_datums(rows=eval_rows, tokenizer=tokenizer)
    LOGGER.info(
        "datums: train=%d eval=%d",
        len(train_datums) if not why_idf_enabled else len(train_rows),
        len(eval_datums),
    )
    if not why_idf_enabled and not train_datums:
        raise ValueError("SFT produced 0 training datums after row filters")
    if why_idf_enabled and not train_rows:
        raise ValueError("SFT train_rows empty")

    batch_size = int(sft_cfg.batch_size)
    train_batches = [] if why_idf_enabled else _batch(train_datums, batch_size=batch_size)
    train_hint_ce_batches: list[list[list[tinker.Datum]]] = (
        [] if why_idf_enabled else _batch(train_hint_ce_datums, batch_size=batch_size)
    )
    eval_batches = _batch(eval_datums, batch_size=batch_size)

    # Hint outer-type CE config
    _hint_outer_ce_enabled = bool(getattr(sft_cfg, "hint_outer_ce_enabled", False))
    _hint_outer_ce_every = int(getattr(sft_cfg, "hint_outer_ce_every_n_steps", 1))
    _hint_outer_ce_cooldown_every = int(getattr(sft_cfg, "hint_outer_ce_cooldown_every_n", 8))
    _hint_outer_ce_target = float(getattr(sft_cfg, "hint_outer_ce_target_follow_rate", 0.92))
    _hint_outer_ce_ema_alpha = float(getattr(sft_cfg, "hint_outer_ce_ema_alpha", 0.1))
    _hint_outer_ce_lr_scale = float(getattr(sft_cfg, "hint_outer_ce_lr_scale", 0.3))
    # Start pessimistic so we fire aggressively from step 0.
    _hint_ce_follow_ema: float = 0.0
    if _hint_outer_ce_enabled:
        LOGGER.info(
            "hint_outer_ce enabled: every_n=%d cooldown_every_n=%d target=%.2f ema_alpha=%.3f lr_scale=%.3f",
            _hint_outer_ce_every, _hint_outer_ce_cooldown_every,
            _hint_outer_ce_target, _hint_outer_ce_ema_alpha, _hint_outer_ce_lr_scale,
        )

    # Style CLM setup - DEPRECATED.
    # Superseded by expert-annotated-TELL (expert_annot_dataset_path), which provides
    # full span+verdict completions grounded in human rationale rather than just
    # verdict why/score supervision. Leave style_clm_path empty for new runs.
    _style_clm_enabled = False
    _style_clm_pool: list[tinker.Datum] = []
    _style_clm_texts: list[str] = []
    _style_clm_ptr = 0
    _style_clm_every = 0
    _style_clm_lr_scale = 1.0
    _style_clm_batch_size = batch_size
    _style_clm_path = str(getattr(sft_cfg, "style_clm_path", "") or "")
    if _style_clm_path:
        _style_clm_every = int(getattr(sft_cfg, "style_clm_every_n_steps", 5))
        _style_clm_lr_scale = float(getattr(sft_cfg, "style_clm_lr_scale", 0.5))
        _style_clm_max_tokens = int(getattr(sft_cfg, "style_clm_max_tokens", 512))
        _style_clm_batch_size = int(getattr(sft_cfg, "style_clm_batch_size", batch_size))
        try:
            _style_clm_items = _load_style_clm_items(_style_clm_path)
            _style_clm_pool, _style_clm_texts = _build_style_clm_pool(_style_clm_items, tokenizer)
            combined = list(zip(_style_clm_pool, _style_clm_texts))
            random.Random(SEED + 99).shuffle(combined)
            _style_clm_pool, _style_clm_texts = [d for d, _ in combined], [t for _, t in combined]
            if _style_clm_pool:
                _style_clm_enabled = True
                LOGGER.info(
                    "style CLM enabled: %d datums (outer-annotation only), every_n=%d lr_scale=%.2f",
                    len(_style_clm_pool), _style_clm_every, _style_clm_lr_scale,
                )
        except Exception:
            LOGGER.exception("style CLM load failed -- skipping")

    hint_follow_eval_pairs = (
        _build_hint_follow_eval_pairs(eval_rows, tokenizer)
        if _hint_outer_ce_enabled else []
    )
    if _hint_outer_ce_enabled:
        LOGGER.info("hint-follow eval pairs built: %d", len(hint_follow_eval_pairs))

    service_client = tinker.ServiceClient()
    _sft_checkpoint = str(getattr(CFG.model, "checkpoint", None) or "").strip() or None
    if _sft_checkpoint:
        LOGGER.info("SFT: loading weights from checkpoint %s", _sft_checkpoint)
        training_client = await service_client.create_training_client_from_state_async(
            path=_sft_checkpoint,
        )
    else:
        training_client = await service_client.create_lora_training_client_async(
            base_model=CFG.model.base_model,
            rank=int(CFG.model.lora_rank),
            seed=SEED,
        )

    global_step = 0
    metrics_path = output_dir / "sft_metrics.jsonl"
    audit_path = output_dir / "sft_forward_backward_audit.jsonl"
    n_epochs = int(sft_cfg.epochs)
    lr = float(sft_cfg.learning_rate)
    log_every = int(sft_cfg.log_every_steps)
    ckpt_every = int(sft_cfg.checkpoint_every_steps)
    n_train = len(train_rows)
    for epoch_idx in range(n_epochs):
        LOGGER.info("epoch %d/%d start", epoch_idx + 1, n_epochs)
        if why_idf_enabled:
            row_order = list(range(n_train))
            random.Random(SEED + epoch_idx).shuffle(row_order)
            num_train_batches = (n_train + batch_size - 1) // batch_size
            epoch_batch_slots = list(range(num_train_batches))
            random.Random(SEED + epoch_idx + 41).shuffle(epoch_batch_slots)
            tqdm_iter = epoch_batch_slots
        else:
            indexed_epoch_batches = list(enumerate(train_batches))
            random.Random(SEED + epoch_idx).shuffle(indexed_epoch_batches)
            tqdm_iter = indexed_epoch_batches

        for batch_slot in tqdm(tqdm_iter, desc=f"epoch {epoch_idx + 1}/{n_epochs}", leave=False):
            if why_idf_enabled:
                batch_i = int(batch_slot)
                i0 = batch_i * batch_size
                idxs = row_order[i0 : i0 + batch_size]
                batch = []
                hint_ce_datums_this_batch: list[tinker.Datum] = []
                fwdbwd_audit_examples: list[dict] = []
                for j, row_i in enumerate(idxs):
                    row = train_rows[row_i]
                    inj = random.Random(SEED + epoch_idx * (n_train + 3) + row_i).random()
                    inject_label_instruction = inject_enabled and (inj < mix_ratio)
                    ann_before = row["annotation"]
                    ann_xml, n_before, n_after = apply_online_why_idf_nested_dropout(
                        logical_document=row["text"],
                        annotation_xml=ann_before,
                        rng=random.Random(SEED + epoch_idx * 500_000 + batch_i * batch_size + j),
                        scorer=why_scorer,
                        row_index=int(row_i),
                        drop_strength=why_idf_strength,
                        p_drop_cap=why_idf_p_cap,
                        score_keep_weight=why_idf_score_keep_w,
                        keep_min=why_idf_keep_min,
                        keep_max=why_idf_keep_max,
                    )
                    if paced_enabled:
                        ann_xml, n_before, n_after = apply_paced_annotation_dropout(
                            logical_document=row["text"],
                            annotation_xml=ann_xml,
                            rng=random.Random(SEED + epoch_idx * 500_000 + batch_i * batch_size + j + 1),
                            words_per_annotation=paced_words_per_ann,
                            high_score_keep_bonus=paced_score_bonus,
                        )
                    cov_frac = (n_before - n_after) / n_before if n_before > 0 else 0.0
                    built = _build_sft_datum(
                        tokenizer=tokenizer,
                        row=row,
                        inject_label_instruction=inject_label_instruction,
                        annotation_xml=ann_xml,
                    )
                    if built is None:
                        continue
                    main_datum, row_hint_ce, audit_row = built
                    batch.append(main_datum)
                    hint_ce_datums_this_batch.extend(row_hint_ce)
                    audit_row["annotation_before"] = ann_before
                    audit_row["nested_count_before"] = n_before
                    audit_row["nested_count_after"] = n_after
                    audit_row["coverage_fraction"] = cov_frac
                    fwdbwd_audit_examples.append(audit_row)
                batch_idx = batch_i
                if not batch:
                    continue
            else:
                batch_idx, batch = batch_slot
                fwdbwd_audit_examples = train_audit_rows[
                    batch_idx * batch_size : batch_idx * batch_size + len(batch)
                ]
                hint_ce_datums_this_batch = [
                    d for row_ces in train_hint_ce_batches[batch_idx] for d in row_ces
                ] if train_hint_ce_batches else []

            fwdbwd_future = await training_client.forward_backward_async(
                data=batch, loss_fn="cross_entropy"
            )
            fwdbwd_out = await fwdbwd_future.result_async()
            optim_future = await training_client.optim_step_async(
                adam_params=tinker.AdamParams(learning_rate=lr)
            )
            await optim_future.result_async()
            global_step += 1

            if _hint_outer_ce_enabled:
                _in_cooldown = _hint_ce_follow_ema >= _hint_outer_ce_target
                _effective_every = _hint_outer_ce_cooldown_every if _in_cooldown else _hint_outer_ce_every
                _should_fire_ce = _effective_every > 0 and global_step % _effective_every == 0
                if _should_fire_ce:
                    n_injected_this_batch = sum(
                        1 for r in fwdbwd_audit_examples if r.get("inject_label_instruction")
                    )
                    n_ce_expected = 2 * n_injected_this_batch
                    hint_ce_build_rate = (
                        len(hint_ce_datums_this_batch) / n_ce_expected
                        if n_ce_expected > 0 else float("nan")
                    )

                    if hint_ce_datums_this_batch:
                        hint_fwd_future = await training_client.forward_backward_async(
                            data=hint_ce_datums_this_batch, loss_fn="cross_entropy"
                        )
                        hint_fwd_out = await hint_fwd_future.result_async()
                        hint_opt_future = await training_client.optim_step_async(
                            adam_params=tinker.AdamParams(learning_rate=lr * _hint_outer_ce_lr_scale)
                        )
                        await hint_opt_future.result_async()
                        try:
                            hint_ce_loss = _scalar_loss(hint_ce_datums_this_batch, hint_fwd_out)
                        except AttributeError:
                            hint_ce_loss = float("nan")
                        follow_metrics = _hint_ce_proxy_metrics(hint_ce_datums_this_batch, hint_fwd_out)
                        # Update EMA using actual type_prob, not the binary proxy.
                        # type_prob is the mean P(correct token) -- this is what we actually care about.
                        _tp = follow_metrics["hint_ce_type_prob"]
                        if not math.isnan(_tp):
                            _hint_ce_follow_ema = (
                                (1 - _hint_outer_ce_ema_alpha) * _hint_ce_follow_ema
                                + _hint_outer_ce_ema_alpha * _tp
                            )
                    else:
                        hint_ce_loss = float("nan")
                        follow_metrics = {"hint_ce_type_prob": float("nan"), "hint_ce_follow_rate_proxy": float("nan")}

                    LOGGER.info(
                        "step=%d hint_outer_ce: loss=%.4f follow_proxy=%.3f ema=%.3f type_prob=%.3f "
                        "build_rate=%.2f n=%d cooldown=%s lr=%.2e",
                        global_step,
                        hint_ce_loss,
                        follow_metrics["hint_ce_follow_rate_proxy"],
                        _hint_ce_follow_ema,
                        follow_metrics["hint_ce_type_prob"],
                        hint_ce_build_rate,
                        len(hint_ce_datums_this_batch),
                        _in_cooldown,
                        lr * _hint_outer_ce_lr_scale,
                    )
                    if getattr(CFG.wandb, "enabled", False):
                        wandb.log(
                            {
                                "train/hint_outer_ce_loss": hint_ce_loss,
                                "train/hint_outer_ce_follow_rate_proxy": follow_metrics["hint_ce_follow_rate_proxy"],
                                "train/hint_outer_ce_follow_rate_ema": _hint_ce_follow_ema,
                                "train/hint_outer_ce_type_prob": follow_metrics["hint_ce_type_prob"],
                                "train/hint_outer_ce_build_rate": hint_ce_build_rate,
                                "train/hint_outer_ce_n": len(hint_ce_datums_this_batch),
                                "train/hint_outer_ce_in_cooldown": float(_in_cooldown),
                            },
                            step=global_step,
                        )

            if _style_clm_enabled and _style_clm_every > 0 and global_step % _style_clm_every == 0:
                n = len(_style_clm_pool)
                end = _style_clm_ptr + _style_clm_batch_size
                clm_batch = _style_clm_pool[_style_clm_ptr:end] + _style_clm_pool[:max(0, end - n)]
                clm_batch_texts = _style_clm_texts[_style_clm_ptr:end] + _style_clm_texts[:max(0, end - n)]
                _style_clm_ptr = end % n
                clm_fwd_future = await training_client.forward_backward_async(
                    data=clm_batch, loss_fn="cross_entropy"
                )
                clm_fwd_out = await clm_fwd_future.result_async()
                clm_opt_future = await training_client.optim_step_async(
                    adam_params=tinker.AdamParams(learning_rate=lr * _style_clm_lr_scale)
                )
                await clm_opt_future.result_async()
                try:
                    clm_loss = _scalar_loss(clm_batch, clm_fwd_out)
                except AttributeError:
                    clm_loss = float("nan")
                LOGGER.info("step=%d style_clm_loss=%.4f lr=%.2e", global_step, clm_loss, lr * _style_clm_lr_scale)
                if getattr(CFG.wandb, "enabled", False):
                    wandb.log({"train/style_clm_loss": clm_loss}, step=global_step)
                if _weave_enabled:
                    clm_decoded_inputs = [
                        tokenizer.decode(d.model_input.to_ints()) for d in clm_batch
                    ]
                    trace_style_clm_batch(
                        step=global_step,
                        epoch=epoch_idx,
                        texts=clm_batch_texts,
                        model_inputs=clm_decoded_inputs,
                        loss=clm_loss,
                    )

            if global_step % log_every == 0:
                train_loss = _scalar_loss(batch, fwdbwd_out)
                metric: dict = {
                    "step": global_step,
                    "epoch": epoch_idx,
                    "train_loss": float(train_loss),
                }
                with open(metrics_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(metric, ensure_ascii=False) + "\n")
                LOGGER.info(
                    "step=%d epoch=%d train_loss=%.4f",
                    global_step, epoch_idx, float(train_loss),
                )
                batch_start = batch_idx * batch_size
                batch_end = min(batch_start + len(batch), n_train if why_idf_enabled else len(train_audit_rows))
                fwdbwd_audit = {
                    "step": global_step,
                    "epoch": epoch_idx,
                    "batch_idx": batch_idx,
                    "batch_size": len(batch),
                    "batch_start": batch_start,
                    "batch_end": batch_end,
                    "examples": fwdbwd_audit_examples,
                }
                with open(audit_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(fwdbwd_audit, ensure_ascii=False) + "\n")
                if getattr(CFG.wandb, "enabled", False):
                    wandb.log({"train/loss": float(train_loss)}, step=global_step)

            # Weave trace every 5 steps -- independent of log_every so coverage/annotations are visible frequently
            if _weave_enabled and global_step % 5 == 0:
                trace_sft_fwdbwd_batch(
                    step=global_step,
                    epoch=epoch_idx,
                    batch_idx=batch_idx,
                    examples=fwdbwd_audit_examples,
                )

            if ckpt_every > 0 and global_step % ckpt_every == 0:
                ckpt_name = f"sft-step-{global_step}"
                save_future = await training_client.save_state_async(name=ckpt_name)
                save_out = await save_future.result_async()
                LOGGER.info("checkpoint step=%d path=%s", global_step, save_out.path)

        hint_follow_eval: dict = {}
        if eval_batches:
            epoch_eval_loss = await _mean_eval_loss(training_client, eval_batches)
        else:
            epoch_eval_loss = float("nan")
            LOGGER.info("epoch %d/%d: skipping eval (no eval set)", epoch_idx + 1, n_epochs)

        if _hint_outer_ce_enabled and hint_follow_eval_pairs:
            hint_follow_eval = await _eval_hint_follow_rate(
                training_client, hint_follow_eval_pairs, batch_size
            )
            LOGGER.info(
                "epoch %d/%d hint_follow: rate=%.3f log_odds=%.3f (n=%d pairs)",
                epoch_idx + 1, n_epochs,
                hint_follow_eval["eval_hint_follow_rate"],
                hint_follow_eval["eval_hint_log_odds_mean"],
                len(hint_follow_eval_pairs),
            )

        epoch_eval_metric: dict = {
            "step": global_step,
            "epoch": epoch_idx,
            "eval_loss_epoch": float(epoch_eval_loss),
            **hint_follow_eval,
        }
        with open(metrics_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(epoch_eval_metric, ensure_ascii=False) + "\n")
        LOGGER.info("epoch %d/%d eval_loss=%.4f", epoch_idx + 1, n_epochs, epoch_eval_loss)
        if getattr(CFG.wandb, "enabled", False):
            wandb.log(
                {"eval/loss": float(epoch_eval_loss), **{f"eval/{k}": v for k, v in hint_follow_eval.items()}},
                step=global_step,
            )

        # Save per-epoch checkpoint
        epoch_ckpt_name = f"sft-epoch-{epoch_idx + 1}"
        save_future = await training_client.save_state_async(name=epoch_ckpt_name)
        save_out = await save_future.result_async()
        LOGGER.info("epoch checkpoint path=%s", save_out.path)
        if getattr(CFG.wandb, "enabled", False):
            wandb.log({"checkpoint/path": save_out.path, "checkpoint/epoch": epoch_idx + 1}, step=global_step)

    # Final checkpoint
    final_future = await training_client.save_state_async(name="sft-final")
    final_out = await final_future.result_async()
    LOGGER.info("final checkpoint path=%s", final_out.path)

    if eval_batches:
        LOGGER.info(
            "running final loss-only eval on validation subset (n=%d)",
            len(eval_datums),
        )
        final_eval_loss = await _mean_eval_loss(training_client, eval_batches)
        final_eval_metric: dict = {
            "step": global_step,
            "eval_loss_final": float(final_eval_loss),
            "eval_n_validation_examples": len(eval_rows),
            "annotation_format_pass_rate": eval_fpr,
        }
        with open(metrics_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(final_eval_metric, ensure_ascii=False) + "\n")
        LOGGER.info(
            "final_eval_loss=%.4f annotation_fpr=%.3f n_val=%d",
            float(final_eval_loss),
            eval_fpr,
            len(eval_rows),
        )
        if getattr(CFG.wandb, "enabled", False):
            wandb.log(
                {
                    "eval/loss_final": float(final_eval_loss),
                    "eval/n_validation_examples": len(eval_rows),
                    "eval/annotation_format_pass_rate": eval_fpr,
                },
                step=global_step,
            )
    else:
        LOGGER.info("skipping final eval (no eval set)")

    manifest = {
        "finished_at": datetime.datetime.utcnow().isoformat(),
        "seed": SEED,
        "base_model": CFG.model.base_model,
        "dataset_path": sft_cfg.dataset_path,
        "n_train": len(train_rows),
        "n_eval": len(eval_rows),
        "epochs": n_epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "label_injection_enabled": bool(getattr(sft_cfg, "label_injection_enabled", False)),
        "label_injection_mix_ratio": float(getattr(sft_cfg, "label_injection_mix_ratio", 0.0)),
        "why_idf_dropout_enabled": bool(sft_cfg.why_idf_dropout_enabled),
        "why_idf_genericity_mode": str(sft_cfg.why_idf_genericity_mode),
        "why_idf_why_bucket_max_chars": int(sft_cfg.why_idf_why_bucket_max_chars),
        "why_idf_why_bucket_key_chars": sft_cfg.why_idf_why_bucket_key_chars,
        "why_idf_span_why_jaccard_weight": float(sft_cfg.why_idf_span_why_jaccard_weight),
        "why_idf_span_why_min_bucket_n_for_log": int(sft_cfg.why_idf_span_why_min_bucket_n_for_log),
        "why_idf_span_why_overlap_p_mass": str(sft_cfg.why_idf_span_why_overlap_p_mass),
        "why_idf_span_why_excess_quantile": float(sft_cfg.why_idf_span_why_excess_quantile),
        "why_idf_span_lemmatizer": "en_blank_rule",
        "why_idf_spacy_pipe_batch_size": int(sft_cfg.why_idf_spacy_pipe_batch_size),
        "why_idf_span_spacy_exclude_stopwords": bool(sft_cfg.why_idf_span_spacy_exclude_stopwords),
        "why_idf_char_ngram_min": int(sft_cfg.why_idf_char_ngram_min),
        "why_idf_char_ngram_max": int(sft_cfg.why_idf_char_ngram_max),
        "why_idf_drop_strength": float(sft_cfg.why_idf_drop_strength),
        "why_idf_p_drop_cap": float(sft_cfg.why_idf_p_drop_cap),
        "why_idf_score_keep_weight": float(sft_cfg.why_idf_score_keep_weight),
        "why_idf_keep_min": int(getattr(sft_cfg, "why_idf_keep_min", 1)),
        "why_idf_keep_max": int(getattr(sft_cfg, "why_idf_keep_max", 5)),
        "expert_annot_dataset_path": _expert_annot_path,
        "expert_annot_oversample_factor": int(getattr(sft_cfg, "expert_annot_oversample_factor", 1)),
        "tell_max_rows": getattr(sft_cfg, "tell_max_rows", None),
        "style_clm_enabled": _style_clm_enabled,
        "style_clm_path": _style_clm_path,
        "style_clm_every_n_steps": int(getattr(sft_cfg, "style_clm_every_n_steps", 5)),
        "style_clm_lr_scale": float(getattr(sft_cfg, "style_clm_lr_scale", 0.5)),
        "final_checkpoint_path": final_out.path,
    }
    manifest_path = output_dir / "sft_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    LOGGER.info("manifest written to %s", manifest_path)
    LOGGER.info("SFT DONE -- final checkpoint: %s", final_out.path)

    if getattr(CFG.wandb, "enabled", False):
        wandb.log({"sft/final_checkpoint": final_out.path})
        wandb.finish()
    if _weave_enabled:
        weave.finish()


@hydra.main(
    version_base=None,
    config_path="../../../conf",
    config_name="config",
)
def _hydra_run(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _rebind_config(cfg=cfg)
    asyncio.run(_train())


if __name__ == "__main__":
    _hydra_run()
