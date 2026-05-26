"""Listwise win-rate evaluation on human_detectors with blinded ranking.

Intermediate artifacts (resumable, committed under data/winrate_eval/):
  rollouts/{dataset}/{rollout_key}/{doc_id}.json
  style_rewrites/{dataset}/{style_key}/{doc_id}.json
  judge_rankings/{dataset}/{rollout_key}/{judge_id}/{doc_id}.json

Stages: --only-style -> tell_human_detectors_style_paraphrases_v3.json (frozen bank)
         rollouts + --skip-style + judge panel -> results/ and experiments/

Outputs:
  data/winrate_eval/experiments/{run_name}/results.json  <- share this (lean)
  data/winrate_eval/experiments/{run_name}/audit.jsonl   <- full per-doc replay (local audit)
  Per-stage caches under data/winrate_eval/{rollouts,style_rewrites,judge_rankings}/

Design note — K=1 model verdict per document
---------------------------------------------
Multiple rollouts from the same policy checkpoint on the same document are i.i.d.
samples from p(output | doc, θ). They vary only due to sampling stochasticity and are
NOT analogous to independent human annotators (who differ in background, priors, and
reading strategy). Including K>1 model outputs in the judge's ranking list risks leaking
source identity (5 stylistically similar items stand out) and inflates apparent duel count
without adding independent observations.

With N=200 fixed documents the bootstrap CI width is dominated by between-document
variance (∝ 1/√N_docs). Within-document variance reduction from K>1 is second-order and
does not meaningfully tighten the CI. K=1 also gives a cleaner estimand: "a single draw
from the model's output distribution vs a single human annotator on the same document."
"""

import argparse
import asyncio
import hashlib
import json
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WINRATE_DIR = Path(__file__).resolve().parent
for _path in (str(_REPO_ROOT / "src"), str(_WINRATE_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import tinker
from dotenv import load_dotenv
from tqdm import tqdm

from rl_detector.annotate import create_runtime
from rl_detector.config import CFG
from rl_detector.data import clean_document_text
from rl_detector.prompt_utils import (
    ANN_SPECIAL_ID_TEXT_OPEN,
    ANN_SPECIAL_ID_VERDICT_PREFIX,
    format_prompt_for_model,
)
from rl_detector.rollouts import _get_analysis_stub_tokens, extract_response_text
from rl_detector.tell_xml import escape_document_piece, get_outer_meta_dict, strip_return_token
from winrate_judges import (
    JUDGE_PANEL,
    JUDGE_PROMPT_VERSION,
    build_judge_openai_clients,
    judge_cache_key,
    judge_cache_path,
    judge_panel_by_id,
    load_judge_cache,
    run_listwise_judge_for_spec,
    write_judge_cache,
)
from scipy.stats import wilcoxon as scipy_wilcoxon


DEFAULT_EXPLANATION_DATASET = "hf://suraj-ranganath/tell-human-detectors/test"
DEFAULT_DATA_DIR = "data/winrate_eval"
DEFAULT_RESULTS_DIR = "results/winrate_eval"
DEFAULT_STYLE_PARAPHRASE_BANK = "data/winrate_eval/tell_human_detectors_style_paraphrases_v3.json"
STYLE_REWRITE_PROMPT_VERSION = "v3"


JUDGE_PROMPT_TEMPLATE = """You are ranking how convincing evidence explanations are.

Task:
Given a source document and {n_candidates} blinded evidence explanations from different annotators, rank all explanations from best to worst according to how convincing the cited evidence and explanation are.

Criteria:
1) Concrete local evidence: specific ideas, elements, or details that are specific to the document.
2) Falsifiability: the explanation should make a specific claim a skeptical reader can verify in the document. It's not enough to say what the evidence is, the explanation should explain why that evidence supports the claim.
3) Internal coherence: the evidence should support the explanation's own AI/human authorship claim without contradicting itself.
4) Human plausibility: a concise forensic observation can beat a polished checklist if it identifies a decisive cue.
5) Real world grounding: the evidence and explanation should fit the document's genre, situation, and general world knowledge.

Important:
- You do not know the true label. Do not use or assume any ground-truth AI/human answer.
- Only rate based on content, not writing style.
- Rank explanation convincingness only from the source document and the candidate explanation.
- You must rank all {n_candidates} items uniquely: rank 1 is most convincing, rank {n_candidates} is least convincing.
- Output JSON only. Do not write any rationale, summary, or commentary outside the ranking array.

Return exactly:
{{
  "ranking": [
    {{"item_id": "A1", "rank": 1, "quality_score": 0.93}},
    ...
  ]
}}

Constraints:
- Include exactly {n_candidates} entries in ranking.
- item_id must match one from the candidate list.
- rank must be integers 1..{n_candidates}, unique.
- quality_score should be float in [0,1], higher is better.

Document:
<document>
{document}
</document>

Candidates (blinded):
{candidates}
"""


STYLE_REWRITE_PROMPT = f"""Rewrite the comment below so it matches the writing style of the example comment.

Rules:
- Change wording, tone, and sentence flow only. Rewrite the style, but keep the same content.
- Do NOT add, remove, or alter factual claims, cited evidence, or the AI/human conclusion.
- Keep the same evidence points and the same guess direction.
- Do not mention rewriting, style matching, or the example.
- Put the rewritten comment only inside <<< and >>>. Nothing before <<< or after >>>.

Style example (match this voice and cadence, not the facts):
<<<
I think this is human, and I would even guess it comes from a real encyclopedia entry. The facts are very specific: a full birth date, a city, a job title, a war deployment, a time window, and a sports role. The sentence has a real compressed biographical rhythm, with several facts packed into one line. The bracketed citation marks are a strong clue that this was copied from a sourced page, not invented as a smooth paragraph.
>>>

Human comment to rewrite:
<<<
{{human_comment}}
>>>
"""


def build_run_config(args: argparse.Namespace) -> dict:
    return {
        "dataset_url": args.dataset_url,
        "checkpoint_path": args.checkpoint_path,
        "offset": args.offset,
        "sample_size": args.sample_size,
        "workers": args.workers,
        "model_rollouts_per_doc": args.model_rollouts_per_doc,
        "max_retries": args.max_retries,
        "sampling_temperature": args.sampling_temperature,
        "sampling_top_p": args.sampling_top_p,
        "verdict_max_tokens": args.verdict_max_tokens,
        "judge_max_tokens": args.judge_max_tokens,
        "judge_ids": [x.strip() for x in args.judge_ids.split(",") if x.strip()],
        "use_judge_cache": args.use_judge_cache,
        "invalidate_judges": args.invalidate_judges,
        "output_dir": args.output_dir,
        "run_name": args.run_name,
        "seed": args.seed,
        "style_normalize_human": args.style_normalize_human,
        "style_rewrite_max_tokens": args.style_rewrite_max_tokens,
        "style_rewrite_temperature": args.style_rewrite_temperature,
        "style_rewrite_top_p": args.style_rewrite_top_p,
        "style_rewrite_reasoning_effort": args.style_rewrite_reasoning_effort,
        "base_model": args.base_model,
        "data_dir": args.data_dir,
        "results_dir": args.results_dir,
        "use_rollout_cache": args.use_rollout_cache,
        "use_style_cache": args.use_style_cache,
        "invalidate_rollouts": args.invalidate_rollouts,
        "invalidate_style": args.invalidate_style,
        "skip_rollouts": args.skip_rollouts,
        "skip_style": args.skip_style,
        "only_judge": args.only_judge,
        "only_style": args.only_style,
        "style_paraphrase_bank_path": args.style_paraphrase_bank_path,
        "retry_row_indices": _parse_retry_row_indices(args=args),
        "merge_into_audit": args.merge_into_audit,
        "invalidate_judge_ids": sorted(
            _parse_judge_id_list(judge_ids_csv=args.invalidate_judge_ids)
        ),
    }


def _parse_judge_id_list(judge_ids_csv: str) -> set[str]:
    return {x.strip() for x in judge_ids_csv.split(",") if x.strip()}


def _parse_retry_row_indices(args: argparse.Namespace) -> list[int] | None:
    if args.retry_row_indices:
        return [int(x.strip()) for x in args.retry_row_indices.split(",") if x.strip()]
    if args.retry_failed_audit:
        return load_retry_row_indices_from_audit(audit_path=args.retry_failed_audit)
    return None


def load_retry_row_indices_from_audit(audit_path: str) -> list[int]:
    indices: list[int] = []
    for line in Path(audit_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("error"):
            indices.append(int(rec["index"]))
    return sorted(set(indices))


def load_records_from_audit(audit_path: str) -> list[dict]:
    records: list[dict] = []
    for line in Path(audit_path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return sorted(records, key=lambda r: r["index"])


def merge_retry_records(base_records: list[dict], retry_records: list[dict]) -> list[dict]:
    by_index = {int(rec["index"]): rec for rec in base_records}
    for rec in retry_records:
        by_index[int(rec["index"])] = rec
    return [by_index[i] for i in sorted(by_index)]


def slugify_path_piece(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._=-]+", "_", text.strip())
    return cleaned[:160] if cleaned else "unknown"


def dataset_cache_key(dataset_url: str) -> str:
    if dataset_url.startswith("hf://"):
        return slugify_path_piece(text=dataset_url.removeprefix("hf://"))
    return slugify_path_piece(text=Path(dataset_url).stem)


def checkpoint_cache_key(checkpoint_path: str) -> str:
    return slugify_path_piece(text=checkpoint_path.replace("tinker://", ""))


def style_rewrite_cache_key(config: dict) -> str:
    payload = {
        "prompt_version": STYLE_REWRITE_PROMPT_VERSION,
        "base_model": config["base_model"],
        "style_rewrite_max_tokens": config["style_rewrite_max_tokens"],
        "style_rewrite_temperature": config["style_rewrite_temperature"],
        "style_rewrite_top_p": config["style_rewrite_top_p"],
        "style_rewrite_reasoning_effort": config["style_rewrite_reasoning_effort"],
        "seed": config["seed"],
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return digest[:12]


def rollout_cache_key(config: dict) -> str:
    payload = {
        "checkpoint_path": config["checkpoint_path"],
        "model_rollouts_per_doc": config["model_rollouts_per_doc"],
        "sampling_temperature": config["sampling_temperature"],
        "sampling_top_p": config["sampling_top_p"],
        "verdict_max_tokens": config["verdict_max_tokens"],
        "seed": config["seed"],
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return digest[:12]


def doc_cache_id(row: dict, row_idx: int) -> str:
    source_id = row.get("id")
    if source_id is not None:
        return f"id{slugify_path_piece(text=str(source_id))}_row{row_idx:05d}"
    return f"row_{row_idx:05d}"


def build_cache_layout(config: dict) -> dict:
    dataset_key = dataset_cache_key(dataset_url=config["dataset_url"])
    rollout_key = rollout_cache_key(config=config)
    style_key = style_rewrite_cache_key(config=config)
    data_root = Path(config["data_dir"])
    judges_dir = data_root / "judge_rankings" / dataset_key / rollout_key
    return {
        "dataset_key": dataset_key,
        "rollout_key": rollout_key,
        "style_key": style_key,
        "rollout_dir": data_root / "rollouts" / dataset_key / rollout_key,
        "style_dir": data_root / "style_rewrites" / dataset_key / style_key,
        "judges_dir": judges_dir,
    }


def rollout_cache_path(cache_layout: dict, doc_id: str) -> Path:
    return cache_layout["rollout_dir"] / f"{doc_id}.json"


def style_cache_path(cache_layout: dict, doc_id: str) -> Path:
    return cache_layout["style_dir"] / f"{doc_id}.json"


def read_json_cache(path: Path) -> dict | None:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_json_cache(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_json_cache_if_allowed(path: Path, payload: dict, overwrite: bool) -> None:
    if path.is_file() and not overwrite:
        return
    write_json_cache(path=path, payload=payload)


def load_doc_cache(
    path: Path,
    doc_id: str,
    expected_cache_key: str,
    use_cache: bool,
    invalidate: bool,
    cache_label: str,
) -> dict | None:
    if not use_cache:
        return None
    if not path.is_file():
        return None
    cached = read_json_cache(path=path)
    if cached is None:
        return None
    if cached.get("doc_id") != doc_id:
        raise RuntimeError(
            f"doc_id mismatch in {cache_label} cache for doc_id={doc_id} at {path} "
            f"(found {cached.get('doc_id')!r})"
        )
    if cached.get("cache_key") != expected_cache_key:
        if invalidate:
            return None
        raise RuntimeError(
            f"stale {cache_label} cache for doc_id={doc_id} at {path} "
            f"(expected cache_key={expected_cache_key!r}, found {cached.get('cache_key')!r}); "
            f"pass --invalidate-{cache_label} to refresh"
        )
    return cached


def load_rollout_cache(path: Path, config: dict, doc_id: str) -> dict | None:
    return load_doc_cache(
        path=path,
        doc_id=doc_id,
        expected_cache_key=rollout_cache_key(config=config),
        use_cache=True,
        invalidate=False,
        cache_label="rollouts",
    )


def load_style_cache(path: Path, config: dict, doc_id: str) -> dict | None:
    return load_doc_cache(
        path=path,
        doc_id=doc_id,
        expected_cache_key=style_rewrite_cache_key(config=config),
        use_cache=True,
        invalidate=False,
        cache_label="style",
    )


def load_style_paraphrase_bank_index(bank_path: str, config: dict) -> dict[str, dict]:
    path = Path(bank_path)
    if not path.is_file():
        return {}
    bank = read_json_cache(path=path)
    if bank is None:
        return {}
    if bank.get("prompt_version") != STYLE_REWRITE_PROMPT_VERSION:
        return {}
    if bank.get("style_key") != style_rewrite_cache_key(config=config):
        return {}
    by_doc: dict[str, dict] = {}
    for doc in bank.get("docs", []):
        doc_id = doc.get("doc_id")
        if doc_id:
            by_doc[doc_id] = doc
    return by_doc


def write_style_paraphrase_bank(bank_path: str, config: dict, records: list[dict]) -> str:
    docs = []
    for rec in records:
        if rec.get("error") is not None:
            continue
        docs.append(
            {
                "doc_id": rec["doc_id"],
                "source_id": rec.get("source", {}).get("id"),
                "ground_truth": rec.get("original_label"),
                "human_annotations_raw": rec.get("human_annotations_raw", []),
                "human_annotations": rec.get("human_annotations", []),
                "style_rewrites": rec.get("style_rewrites", []),
            }
        )
    docs.sort(key=lambda row: row["doc_id"])
    payload = {
        "prompt_version": STYLE_REWRITE_PROMPT_VERSION,
        "style_key": style_rewrite_cache_key(config=config),
        "dataset_url": config["dataset_url"],
        "base_model": config["base_model"],
        "style_rewrite_example": STYLE_REWRITE_EXAMPLE,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_docs": len(docs),
        "docs": docs,
    }
    path = Path(bank_path)
    write_json_cache(path=path, payload=payload)
    return str(path.resolve())


def fetch_dataset(dataset_url: str) -> list[dict]:
    if dataset_url.startswith("hf://"):
        try:
            from datasets import load_dataset
        except Exception as exc:
            raise RuntimeError("datasets is required for hf:// dataset URLs") from exc
        spec = dataset_url.removeprefix("hf://")
        repo_id, split = spec.rsplit("/", 1)
        return [dict(row) for row in load_dataset(repo_id, split=split)]

    def rows_from_payload(payload) -> list[dict]:
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            return [payload[key] for key in sorted(payload.keys(), key=lambda x: int(x))]
        raise ValueError(f"Unsupported dataset payload type: {type(payload).__name__}")

    if "://" not in dataset_url and dataset_url.endswith(".jsonl"):
        with open(dataset_url, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    if "://" not in dataset_url:
        with open(dataset_url, encoding="utf-8") as f:
            payload = json.load(f)
        return rows_from_payload(payload)

    with urlopen(dataset_url) as response:
        if dataset_url.endswith(".jsonl"):
            return [json.loads(line) for line in response.read().decode("utf-8").splitlines() if line.strip()]
        payload = json.load(response)
    return rows_from_payload(payload)


def select_rows(rows: list[dict], offset: int, sample_size: int) -> list[dict]:
    start = max(0, offset)
    end = start + max(0, sample_size)
    return rows[start:end]


def select_rows_for_retry(
    rows: list[dict],
    offset: int,
    sample_size: int,
    retry_row_indices: list[int],
) -> tuple[list[dict], list[int]]:
    selected = select_rows(rows=rows, offset=offset, sample_size=sample_size)
    out_rows: list[dict] = []
    out_indices: list[int] = []
    for idx in retry_row_indices:
        if idx < 0 or idx >= len(selected):
            raise ValueError(
                f"retry row index {idx} out of range for slice "
                f"offset={offset} sample_size={sample_size} (len={len(selected)})"
            )
        out_rows.append(selected[idx])
        out_indices.append(idx)
    return out_rows, out_indices


def build_human_annotations(row: dict) -> list[dict]:
    items = []
    for idx in range(1, 6):
        annot = row[f"annotator_{idx}"]
        items.append(
            {
                "kind": "human",
                "source_id": f"annotator_{idx}",
                "text": annot.get("comment") or "",
                "guess": annot.get("guess"),
                "confidence": annot.get("confidence"),
            }
        )
    return items


async def sample_model_verdict(
    runtime: dict,
    document: str,
    seed: int,
    sampling_temperature: float,
    sampling_top_p: float,
    verdict_max_tokens: int,
) -> dict:
    """Sample only the verdict portion by prefilling the prompt through <verdict type=".

    The model only generates the verdict completion starting from type value onwards:
      AI|human" why="..." score="..." /></text>
    This skips full-rollout span generation and the separate synth step.
    """
    tokenizer = runtime["tokenizer"]
    sampling_client = runtime["sampling_client"]
    think_already_open = runtime.get("think_already_open", False)

    _, neutral_formatted = format_prompt_for_model(tokenizer=tokenizer, text=document)
    neutral_prompt_tokens = tokenizer.encode(neutral_formatted)
    stub_open, stub_close = _get_analysis_stub_tokens(tokenizer, think_already_open)

    escaped_doc_tokens = tokenizer.encode(escape_document_piece(document), add_special_tokens=False)
    prefill_tokens = (
        neutral_prompt_tokens
        + stub_open
        + stub_close
        + [ANN_SPECIAL_ID_TEXT_OPEN]
        + escaped_doc_tokens
        + [ANN_SPECIAL_ID_VERDICT_PREFIX]
    )

    sampled = await sampling_client.sample_async(
        prompt=tinker.ModelInput.from_ints(prefill_tokens),
        num_samples=1,
        sampling_params=tinker.SamplingParams(
            max_tokens=verdict_max_tokens,
            temperature=sampling_temperature,
            top_p=sampling_top_p,
            seed=seed,
            reasoning_effort=CFG.sampling.reasoning_effort,
        ),
    )
    completion_tokens = list(sampled.sequences[0].tokens)
    completion_text = tokenizer.decode(completion_tokens, skip_special_tokens=False).strip()
    completion_text = strip_return_token(completion_text)
    for _tail in (getattr(tokenizer, "eos_token", None), getattr(tokenizer, "pad_token", None)):
        if _tail and completion_text.endswith(_tail):
            completion_text = completion_text[: -len(_tail)].strip()

    # Reconstruct the full response text so we can reuse the existing parser.
    full_response = "<text>" + escape_document_piece(document) + '<verdict type="' + completion_text
    verdict_meta = get_outer_meta_dict(full_response)

    why_text = (verdict_meta or {}).get("explanation", "") or ""
    verdict_type = (verdict_meta or {}).get("type", "") or ""

    return {
        "seed": seed,
        "completion_text": completion_text,
        "full_response": full_response,
        "why_text": why_text,
        "verdict_type": verdict_type,
    }


def normalize_model_comment(text: str) -> str:
    return " ".join(text.split())


def build_style_rewrite_prompt(human_comment: str) -> str:
    return STYLE_REWRITE_PROMPT.format(human_comment=human_comment.strip())


def parse_rewritten_comment(text: str) -> str:
    body = extract_response_text(text)
    start = body.find("<<<")
    if start < 0:
        return ""
    start = start + 3
    end = body.find(">>>", start)
    if end < 0:
        return ""
    return body[start:end].strip()


def encode_style_rewrite_prompt_tokens(tokenizer, prompt_text: str, think_already_open: bool) -> list[int]:
    messages = [{"role": "user", "content": prompt_text}]
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    base_tokens = tokenizer.encode(formatted, add_special_tokens=False)
    stub_open, stub_close = _get_analysis_stub_tokens(tokenizer=tokenizer, think_already_open=think_already_open)
    return base_tokens + stub_open + stub_close


async def rewrite_human_comment_style(
    base_runtime: dict,
    human_comment: str,
    seed: int,
    rewrite_max_tokens: int,
    rewrite_temperature: float,
    rewrite_top_p: float,
    rewrite_reasoning_effort: str,
) -> dict:
    tokenizer = base_runtime["tokenizer"]
    sampling_client = base_runtime["sampling_client"]
    think_already_open = base_runtime["think_already_open"]
    original = human_comment.strip()
    if not original:
        return {
            "original_text": original,
            "rewritten_text": original,
            "parse_ok": True,
            "prompt": "",
            "completion_text": "",
        }
    prompt_text = build_style_rewrite_prompt(human_comment=original)
    prompt_tokens = encode_style_rewrite_prompt_tokens(
        tokenizer=tokenizer,
        prompt_text=prompt_text,
        think_already_open=think_already_open,
    )
    sampled = await sampling_client.sample_async(
        prompt=tinker.ModelInput.from_ints(prompt_tokens),
        num_samples=1,
        sampling_params=tinker.SamplingParams(
            max_tokens=rewrite_max_tokens,
            temperature=rewrite_temperature,
            top_p=rewrite_top_p,
            seed=seed,
            reasoning_effort=rewrite_reasoning_effort,
        ),
    )
    completion_tokens = list(sampled.sequences[0].tokens)
    completion_text = tokenizer.decode(completion_tokens, skip_special_tokens=False).strip()
    parsed = parse_rewritten_comment(text=completion_text)
    parsed = strip_return_token(parsed)
    rewritten = normalize_model_comment(text=parsed)
    parse_ok = bool(rewritten)
    if not rewritten:
        rewritten = original
    return {
        "original_text": original,
        "rewritten_text": rewritten,
        "parse_ok": parse_ok,
        "prompt": prompt_text,
        "completion_text": completion_text,
    }


async def apply_style_normalize_to_human_items(
    base_runtime: dict,
    human_items: list[dict],
    seed: int,
    rewrite_max_tokens: int,
    rewrite_temperature: float,
    rewrite_top_p: float,
    rewrite_reasoning_effort: str,
) -> tuple[list[dict], list[dict]]:
    rewrite_tasks = [
        rewrite_human_comment_style(
            base_runtime=base_runtime,
            human_comment=item["text"],
            seed=seed + idx,
            rewrite_max_tokens=rewrite_max_tokens,
            rewrite_temperature=rewrite_temperature,
            rewrite_top_p=rewrite_top_p,
            rewrite_reasoning_effort=rewrite_reasoning_effort,
        )
        for idx, item in enumerate(human_items)
    ]
    rewrite_results = await asyncio.gather(*rewrite_tasks)
    normalized_items = []
    rewrite_audit = []
    for item, rewrite in zip(human_items, rewrite_results):
        normalized_items.append(
            {
                **item,
                "text": rewrite["rewritten_text"],
            }
        )
        rewrite_audit.append(
            {
                "source_id": item["source_id"],
                "guess": item.get("guess"),
                "confidence": item.get("confidence"),
                **rewrite,
            }
        )
    return normalized_items, rewrite_audit


def prepare_blinded_candidates(
    human_items: list[dict],
    model_items: list[dict],
    seed: int,
) -> tuple[list[dict], list[dict]]:
    candidates = []
    for idx, item in enumerate(human_items, start=1):
        candidates.append(
            {
                "item_id": f"H{idx}",
                "true_kind": "human",
                "source_id": item["source_id"],
                "text": item["text"],
            }
        )
    for idx, item in enumerate(model_items, start=1):
        candidates.append(
            {
                "item_id": f"M{idx}",
                "true_kind": "model",
                "source_id": item["source_id"],
                "text": item["text"],
            }
        )
    rng = random.Random(seed)
    shuffled = list(candidates)
    rng.shuffle(shuffled)
    blinded = []
    mapping = []
    for idx, item in enumerate(shuffled, start=1):
        blind_id = f"A{idx}"
        blinded.append({"item_id": blind_id, "text": item["text"]})
        mapping.append(
            {
                "blind_item_id": blind_id,
                "original_item_id": item["item_id"],
                "true_kind": item["true_kind"],
                "source_id": item["source_id"],
            }
        )
    return blinded, mapping


def build_judge_prompt(document: str, candidates: list[dict]) -> str:
    candidates_blob = json.dumps(candidates, ensure_ascii=True)
    return JUDGE_PROMPT_TEMPLATE.format(
        document=document,
        candidates=candidates_blob,
        n_candidates=len(candidates),
    )


def finalize_judge_result(judge_result: dict, blinded_candidates: list[dict], mapping: list[dict]) -> dict:
    ranking = sanitize_ranking(
        parsed=judge_result["parsed"],
        valid_ids={item["item_id"] for item in blinded_candidates},
    )
    win_stats = compute_doc_win_stats(ranking=ranking, mapping=mapping)
    ranking_with_metadata = enrich_ranking_with_mapping(ranking=ranking, mapping=mapping)
    parsed_with_metadata = {**judge_result["parsed"], "ranking": ranking_with_metadata}
    return {
        **judge_result,
        "parsed": parsed_with_metadata,
        "ranking": ranking_with_metadata,
        "ranking_with_metadata": ranking_with_metadata,
        "head_to_heads": win_stats["head_to_heads"],
        "win_stats": win_stats,
    }


def sanitize_ranking(parsed: dict, valid_ids: set[str]) -> list[dict]:
    ranking = parsed.get("ranking") if isinstance(parsed, dict) else None
    rows = ranking if isinstance(ranking, list) else []
    clean = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        item_id = str(row.get("item_id", ""))
        rank = row.get("rank")
        score = row.get("quality_score")
        if item_id not in valid_ids:
            continue
        if rank is None:
            raise ValueError(f"judge ranking row missing rank for item_id={item_id!r}")
        if score is None:
            raise ValueError(f"judge ranking row missing quality_score for item_id={item_id!r}")
        rank_i = int(rank)
        score_f = float(score)
        clean.append({"item_id": item_id, "rank": rank_i, "quality_score": max(0.0, min(1.0, score_f))})

    ids_seen = {r["item_id"] for r in clean}
    missing = sorted(valid_ids - ids_seen)
    n = len(valid_ids)
    if missing:
        raise ValueError(f"judge ranking missing item_ids={missing} (got {len(clean)}/{n}, do not impute)")
    ranks = [r["rank"] for r in clean]
    if len(set(ranks)) != n or sorted(ranks) != list(range(1, n + 1)):
        raise ValueError(f"judge ranking has invalid ranks={ranks} for n={n}")
    clean = sorted(clean, key=lambda x: x["rank"])
    return clean


def enrich_ranking_with_mapping(ranking: list[dict], mapping: list[dict]) -> list[dict]:
    meta_by_blind_id = {row["blind_item_id"]: row for row in mapping}
    enriched = []
    for row in ranking:
        meta = meta_by_blind_id.get(row["item_id"], {})
        enriched.append(
            {
                **row,
                "original_item_id": meta.get("original_item_id"),
                "true_kind": meta.get("true_kind"),
                "source_id": meta.get("source_id"),
            }
        )
    return enriched


def compute_doc_win_stats(ranking: list[dict], mapping: list[dict]) -> dict:
    ranking_with_metadata = enrich_ranking_with_mapping(ranking=ranking, mapping=mapping)
    rank_by_blind_id = {row["item_id"]: row for row in ranking_with_metadata}
    model_ids = [row["item_id"] for row in ranking_with_metadata if row["true_kind"] == "model"]
    human_ids = [row["item_id"] for row in ranking_with_metadata if row["true_kind"] == "human"]
    wins = 0
    losses = 0
    ties = 0
    head_to_heads = []
    for mid in model_ids:
        for hid in human_ids:
            model_item = rank_by_blind_id[mid]
            human_item = rank_by_blind_id[hid]
            mr = model_item["rank"]
            hr = human_item["rank"]
            if mr < hr:
                wins += 1
                winner = model_item
            elif mr > hr:
                losses += 1
                winner = human_item
            else:
                ties += 1
                winner = None
            head_to_heads.append(
                {
                    "model_item": model_item,
                    "human_item": human_item,
                    "winner_item_id": winner["item_id"] if winner else None,
                    "winner_true_kind": winner["true_kind"] if winner else "tie",
                    "winner_source_id": winner["source_id"] if winner else None,
                    "outcome": "model" if winner is model_item else "human" if winner is human_item else "tie",
                }
            )
    total = wins + losses + ties
    win_rate = (wins + 0.5 * ties) / total if total > 0 else 0.0
    return {
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "total_duels": total,
        "win_rate": win_rate,
        "head_to_heads": head_to_heads,
    }


def permutation_test_win_rate(doc_win_rates: list[float], seed: int, n_perm: int = 10000) -> float:
    """One-sided bootstrap permutation p-value: P(permuted_mean >= observed) under H0=0.5.

    Under H0 each document's win-rate deviation from 0.5 is equally likely to be positive
    or negative, so we randomly flip signs of (rate - 0.5) and test whether the permuted
    mean meets or exceeds the observed mean.
    """
    if not doc_win_rates:
        return 1.0
    observed = sum(doc_win_rates) / len(doc_win_rates)
    centered = [v - 0.5 for v in doc_win_rates]
    n = len(centered)
    rng = random.Random(seed)
    count = 0
    for _ in range(n_perm):
        perm_mean = sum(rng.choice((-1, 1)) * c for c in centered) / n + 0.5
        if perm_mean >= observed:
            count += 1
    return count / n_perm


def wilcoxon_p(doc_win_rates: list[float]) -> float:
    """One-sided Wilcoxon signed-rank p-value: H1 = median win_rate > 0.5."""
    if not doc_win_rates:
        return 1.0
    diffs = [v - 0.5 for v in doc_win_rates]
    if all(d == 0.0 for d in diffs):
        return 1.0
    _, p = scipy_wilcoxon(diffs, alternative="greater")
    return float(p)


def win_rate_effect_sizes(doc_win_rates: list[float]) -> dict:
    """Effect sizes for doc-level win rates vs chance (0.5).

    mean_diff_from_chance is the primary effect in probability units.
    cohens_d / hedges_g are on per-document deviations (rate - 0.5).
    """
    n = len(doc_win_rates)
    if n == 0:
        return {
            "mean_diff_from_chance": 0.0,
            "median_diff_from_chance": 0.0,
            "cohens_d": 0.0,
            "hedges_g": 0.0,
            "std_diff_from_chance": 0.0,
            "fraction_docs_above_chance": 0.0,
            "fraction_docs_at_or_above_chance": 0.0,
        }
    diffs = [v - 0.5 for v in doc_win_rates]
    mean_diff = sum(diffs) / n
    sorted_rates = sorted(doc_win_rates)
    mid = n // 2
    if n % 2 == 1:
        median_rate = sorted_rates[mid]
    else:
        median_rate = 0.5 * (sorted_rates[mid - 1] + sorted_rates[mid])
    median_diff = median_rate - 0.5
    if n > 1:
        var = sum((d - mean_diff) ** 2 for d in diffs) / (n - 1)
        std_diff = var**0.5
    else:
        std_diff = 0.0
    if std_diff > 0:
        cohens_d = mean_diff / std_diff
        hedges_g = cohens_d * (1.0 - 3.0 / (4.0 * n - 1.0))
    elif mean_diff == 0.0:
        cohens_d = 0.0
        hedges_g = 0.0
    else:
        cohens_d = float("inf")
        hedges_g = float("inf")
    return {
        "mean_diff_from_chance": mean_diff,
        "median_diff_from_chance": median_diff,
        "cohens_d": cohens_d,
        "hedges_g": hedges_g,
        "std_diff_from_chance": std_diff,
        "fraction_docs_above_chance": sum(1 for v in doc_win_rates if v > 0.5) / n,
        "fraction_docs_at_or_above_chance": sum(1 for v in doc_win_rates if v >= 0.5) / n,
    }


def holm_adjust_pvalues(p_values: list[float]) -> list[float]:
    """Holm step-down adjustment for a family of one-sided tests."""
    n = len(p_values)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: p_values[i])
    adjusted = [1.0] * n
    prev = 0.0
    for rank, idx in enumerate(order):
        m = n - rank
        adj = min(1.0, max(prev, p_values[idx] * m))
        prev = adj
        adjusted[idx] = adj
    return adjusted


def panel_doc_win_rates(records: list[dict], judge_ids: list[str]) -> list[float]:
    """Per-document panel score: mean win rate across judges (unit of analysis = document)."""
    ok = [r for r in records if r.get("error") is None]
    rates = []
    for rec in ok:
        per_judge = [rec["judge_results"][jid]["win_stats"]["win_rate"] for jid in judge_ids]
        rates.append(sum(per_judge) / len(per_judge))
    return rates


def compute_panel_summary(records: list[dict], judge_ids: list[str], seed: int) -> dict:
    panel_rates = panel_doc_win_rates(records=records, judge_ids=judge_ids)
    n_ok = len(panel_rates)
    ok = [r for r in records if r.get("error") is None]
    agree = 0
    for rec in ok:
        per = [rec["judge_results"][jid]["win_stats"]["win_rate"] for jid in judge_ids]
        if all(v > 0.5 for v in per) or all(v < 0.5 for v in per):
            agree += 1
    if n_ok == 0:
        return {
            "estimand": "panel_mean_doc_win_rate",
            "n_documents": len(records),
            "n_success": 0,
            "n_failed": len(records),
            "panel_doc_win_rate_mean": 0.0,
            "panel_doc_win_rate_ci95": [0.0, 0.0],
            "judge_agreement_rate": 0.0,
            "permutation_p": 1.0,
            "wilcoxon_p": 1.0,
            "effect_size": win_rate_effect_sizes(doc_win_rates=[]),
        }
    lo, hi = bootstrap_ci(values=panel_rates, seed=seed)
    effect = win_rate_effect_sizes(doc_win_rates=panel_rates)
    return {
        "estimand": "panel_mean_doc_win_rate",
        "n_documents": len(records),
        "n_success": n_ok,
        "n_failed": len(records) - n_ok,
        "panel_doc_win_rate_mean": sum(panel_rates) / n_ok,
        "panel_doc_win_rate_ci95": [lo, hi],
        "judge_agreement_rate": agree / n_ok,
        "permutation_p": permutation_test_win_rate(doc_win_rates=panel_rates, seed=seed),
        "wilcoxon_p": wilcoxon_p(doc_win_rates=panel_rates),
        "null_hypothesis": "panel_mean_doc_win_rate <= 0.5",
        "inference_unit": "document",
        "effect_size": effect,
        "mean_diff_from_chance_ci95": [lo - 0.5, hi - 0.5],
    }


BOOTSTRAP_N_BOOT = 10000

JUDGE_DISPLAY_LATEX: dict[str, str] = {
    "openai_gpt54mini_flex": "GPT-5.4-mini",
    "ucsd_gemma4_26b": "Gemma 4 26B",
    "deepinfra_deepseek_v4_flash": "DeepSeek V4 Flash",
    "deepinfra_nemotron_super": "Nemotron Super",
    "tinker_gpt_oss_120b": "GPT-OSS 120B",
}


def format_tex_pvalue(p: float) -> str:
    if p < 0.0001:
        return r"$<10^{-4}$"
    if p < 0.001:
        return r"$<0.001$"
    return rf"\num{{{p:.3f}}}"


def format_tex_rate_ci(mean: float, ci: list[float]) -> str:
    pct = 100.0 * mean
    lo = 100.0 * ci[0]
    hi = 100.0 * ci[1]
    return rf"\num{{{pct:.1f}}} [\num{{{lo:.1f}}}, \num{{{hi:.1f}}}]"


def build_winrate_table_compact_tex(summary: dict, meta: dict | None = None) -> str:
    panel = summary["panel"]
    per_judge = summary["per_judge"]
    judge_ids = summary["judge_panel"]
    n_ok = panel["n_success"]
    n_failed = panel["n_failed"]

    lines = [
        r"% Requires: \usepackage{booktabs,siunitx}",
        r"% siunitx: \sisetup{mode=text,detect-weight}",
        r"\begin{table}[t]",
        r"\centering",
        (
            rf"\caption{{Listwise win rate vs.\ human annotators (TELL human-detectors test, "
            rf"$n=\num{{{n_ok}}}$ documents with complete 5-judge panel"
        )
        + (rf"; \num{{{n_failed}}} excluded" if n_failed else "")
        + rf"). 95\% CIs: document-level bootstrap ($B=\num{{{BOOTSTRAP_N_BOOT}}}$).}}"
        ,
        r"\label{tab:winrate_compact}",
        r"\begin{tabular}{l c}",
        r"\toprule",
        r"{Judge} & {Win rate (\%) [95\% CI]} \\",
        r"\midrule",
    ]
    rate_ci = format_tex_rate_ci(
        mean=panel["panel_doc_win_rate_mean"],
        ci=panel["panel_doc_win_rate_ci95"],
    )
    lines.append(rf"\bfseries Panel mean & \bfseries {rate_ci} \\")
    lines.append(r"\midrule")
    for judge_id in judge_ids:
        row = per_judge[judge_id]
        rate_ci = format_tex_rate_ci(
            mean=row["doc_win_rate_mean"],
            ci=row["doc_win_rate_ci95"],
        )
        name = JUDGE_DISPLAY_LATEX.get(judge_id, judge_id)
        lines.append(f"  {name} & {rate_ci} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def build_winrate_table_tex(summary: dict, meta: dict | None = None) -> str:
    panel = summary["panel"]
    per_judge = summary["per_judge"]
    judge_ids = summary["judge_panel"]
    n_ok = panel["n_success"]
    n_failed = panel["n_failed"]

    def row_cells(mean: float, ci: list[float], perm_p: float, wilc_p: float) -> tuple[str, str, str, str]:
        pct = 100.0 * mean
        lo = 100.0 * ci[0]
        hi = 100.0 * ci[1]
        return (
            rf"\num{{{pct:.1f}}}",
            rf"[\num{{{lo:.1f}}}, \num{{{hi:.1f}}}]",
            format_tex_pvalue(p=perm_p),
            format_tex_pvalue(p=wilc_p),
        )

    lines = [
        r"% Requires: \usepackage{booktabs,siunitx}",
        r"% siunitx: \sisetup{mode=text,detect-weight}",
        r"\begin{table}[t]",
        r"\centering",
        (
            rf"\caption{{Listwise win rate vs.\ human annotators (TELL human-detectors test, "
            rf"$n=\num{{{n_ok}}}$ documents with complete 5-judge panel"
        )
        + (rf"; \num{{{n_failed}}} excluded" if n_failed else "")
        + (
            rf"). 95\% CIs: document-level bootstrap ($B=\num{{{BOOTSTRAP_N_BOOT}}}$). "
            rf"$p_{{\mathrm{{perm}}}}$: one-sided sign-flip permutation vs.\ 50\%; "
            rf"$p_{{\mathrm{{Wilc}}}}$: one-sided Wilcoxon signed-rank vs.\ 50\%.}}"
        ),
        r"\label{tab:winrate}",
        r"\begin{tabular}{l S[table-format=2.1] c c c}",
        r"\toprule",
        r"{Judge} & {Win rate (\%)} & {95\% CI} & {$p_\mathrm{perm}$} & {$p_\mathrm{Wilc}$} \\",
        r"\midrule",
    ]
    rate, ci_cell, pp, wp = row_cells(
        mean=panel["panel_doc_win_rate_mean"],
        ci=panel["panel_doc_win_rate_ci95"],
        perm_p=panel["permutation_p"],
        wilc_p=panel["wilcoxon_p"],
    )
    lines.append(rf"\bfseries Panel mean & \bfseries {rate} & {ci_cell} & {pp} & {wp} \\")
    lines.append(r"\midrule")
    for judge_id in judge_ids:
        row = per_judge[judge_id]
        rate, ci_cell, pp, wp = row_cells(
            mean=row["doc_win_rate_mean"],
            ci=row["doc_win_rate_ci95"],
            perm_p=row["permutation_p"],
            wilc_p=row["wilcoxon_p"],
        )
        name = JUDGE_DISPLAY_LATEX.get(judge_id, judge_id)
        lines.append(f"  {name} & {rate} & {ci_cell} & {pp} & {wp} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def write_winrate_table_tex(summary: dict, out_path: Path, meta: dict | None = None) -> dict:
    out_path.write_text(build_winrate_table_tex(summary=summary, meta=meta), encoding="utf-8")
    compact_path = out_path.with_name("winrate_table_compact.tex")
    compact_path.write_text(build_winrate_table_compact_tex(summary=summary, meta=meta), encoding="utf-8")
    return {
        "winrate_table_tex": str(out_path.resolve()),
        "winrate_table_compact_tex": str(compact_path.resolve()),
    }


def bootstrap_ci(values: list[float], seed: int, n_boot: int = BOOTSTRAP_N_BOOT, alpha: float = 0.05) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    rng = random.Random(seed)
    means = []
    n = len(values)
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo_idx = int((alpha / 2.0) * len(means))
    hi_idx = int((1.0 - alpha / 2.0) * len(means)) - 1
    lo_idx = max(0, min(lo_idx, len(means) - 1))
    hi_idx = max(0, min(hi_idx, len(means) - 1))
    return means[lo_idx], means[hi_idx]


async def process_row(
    row_idx: int,
    row: dict,
    runtime: dict | None,
    base_runtime: dict | None,
    judge_specs: list[dict],
    openai_judge_clients: dict,
    tinker_judge_runtime: dict | None,
    config: dict,
    cache_layout: dict,
) -> dict:
    last_error = None
    article = clean_document_text(row.get("article", ""))
    doc_id = doc_cache_id(row=row, row_idx=row_idx)
    rollout_path = rollout_cache_path(cache_layout=cache_layout, doc_id=doc_id)
    style_path = style_cache_path(cache_layout=cache_layout, doc_id=doc_id)
    for attempt in range(1, config["max_retries"] + 1):
        try:
            doc_seed = config["seed"] + (1000 * row_idx)
            rollout_from_cache = False
            style_from_cache = False
            verdict_results = []
            model_annotations = []
            if not config["only_style"] and config["use_rollout_cache"] and not config["invalidate_rollouts"]:
                rollout_cached = load_doc_cache(
                    path=rollout_path,
                    doc_id=doc_id,
                    expected_cache_key=rollout_cache_key(config=config),
                    use_cache=True,
                    invalidate=config["invalidate_rollouts"],
                    cache_label="rollouts",
                )
                if rollout_cached is not None:
                    verdict_results = rollout_cached["model_verdicts"]
                    model_annotations = rollout_cached["model_annotations"]
                    rollout_from_cache = True
            if not config["only_style"] and config["skip_rollouts"] and not rollout_from_cache:
                raise RuntimeError(f"missing rollout cache for doc_id={doc_id} at {rollout_path}")
            if not config["only_style"] and not rollout_from_cache:
                if runtime is None:
                    raise RuntimeError("runtime is required to sample rollouts")
                verdict_tasks = [
                    sample_model_verdict(
                        runtime=runtime,
                        document=article,
                        seed=doc_seed + k,
                        sampling_temperature=config["sampling_temperature"],
                        sampling_top_p=config["sampling_top_p"],
                        verdict_max_tokens=config["verdict_max_tokens"],
                    )
                    for k in range(config["model_rollouts_per_doc"])
                ]
                verdict_results = await asyncio.gather(*verdict_tasks)
                model_annotations = [
                    {
                        "source_id": f"model_rollout_{k}",
                        "text": normalize_model_comment(text=vr["why_text"]),
                        "verdict_type": vr["verdict_type"],
                        "full_response": vr["full_response"],
                    }
                    for k, vr in enumerate(verdict_results, start=1)
                ]
                if config["use_rollout_cache"]:
                    write_json_cache_if_allowed(
                        path=rollout_path,
                        payload={
                            "doc_id": doc_id,
                            "cache_key": rollout_cache_key(config=config),
                            "checkpoint_path": config["checkpoint_path"],
                            "saved_at": datetime.now(timezone.utc).isoformat(),
                            "model_verdicts": verdict_results,
                            "model_annotations": model_annotations,
                        },
                        overwrite=config["invalidate_rollouts"],
                    )

            human_annotations_raw = build_human_annotations(row=row)
            style_rewrites = []
            human_annotations = human_annotations_raw
            style_from_bank = False
            if config["style_normalize_human"]:
                bank_entry = config.get("style_bank_by_doc", {}).get(doc_id)
                if bank_entry is not None and not config["invalidate_style"]:
                    style_rewrites = bank_entry["style_rewrites"]
                    human_annotations = bank_entry["human_annotations"]
                    style_from_cache = True
                    style_from_bank = True
                if config["use_style_cache"] and not config["invalidate_style"] and not style_from_bank:
                    style_cached = load_doc_cache(
                        path=style_path,
                        doc_id=doc_id,
                        expected_cache_key=style_rewrite_cache_key(config=config),
                        use_cache=True,
                        invalidate=config["invalidate_style"],
                        cache_label="style",
                    )
                    if style_cached is not None:
                        style_rewrites = style_cached["style_rewrites"]
                        human_annotations = style_cached["human_annotations"]
                        style_from_cache = True
                if config["skip_style"] and not style_from_cache:
                    bank_path = config.get("style_paraphrase_bank_path", "")
                    raise RuntimeError(
                        f"missing style paraphrase for doc_id={doc_id} "
                        f"(bank={bank_path!r}, cache={style_path})"
                    )
                if not style_from_cache:
                    if base_runtime is None:
                        raise RuntimeError("base_runtime is required to rewrite human comments")
                    human_annotations, style_rewrites = await apply_style_normalize_to_human_items(
                        base_runtime=base_runtime,
                        human_items=human_annotations_raw,
                        seed=doc_seed + 200,
                        rewrite_max_tokens=config["style_rewrite_max_tokens"],
                        rewrite_temperature=config["style_rewrite_temperature"],
                        rewrite_top_p=config["style_rewrite_top_p"],
                        rewrite_reasoning_effort=config["style_rewrite_reasoning_effort"],
                    )
                    if config["use_style_cache"]:
                        write_json_cache_if_allowed(
                            path=style_path,
                            payload={
                                "doc_id": doc_id,
                                "cache_key": style_rewrite_cache_key(config=config),
                                "prompt_version": STYLE_REWRITE_PROMPT_VERSION,
                                "saved_at": datetime.now(timezone.utc).isoformat(),
                                "human_annotations_raw": human_annotations_raw,
                                "style_rewrites": style_rewrites,
                                "human_annotations": human_annotations,
                            },
                            overwrite=config["invalidate_style"],
                        )
            if config["only_style"]:
                n_parse_ok = sum(1 for sr in style_rewrites if sr.get("parse_ok"))
                return {
                    "index": row_idx,
                    "attempt": attempt,
                    "error": None,
                    "only_style": True,
                    "doc_id": doc_id,
                    "style_from_cache": style_from_cache,
                    "style_from_bank": style_from_bank,
                    "style_cache_path": str(style_path),
                    "original_label": row.get("ground_truth"),
                    "source": {"id": row.get("id"), "ground_truth": row.get("ground_truth")},
                    "human_annotations_raw": human_annotations_raw,
                    "style_rewrites": style_rewrites,
                    "human_annotations": human_annotations,
                    "n_style_rewrites": len(style_rewrites),
                    "n_parse_ok": n_parse_ok,
                }
            blinded_candidates, mapping = prepare_blinded_candidates(
                human_items=human_annotations,
                model_items=model_annotations,
                seed=doc_seed + 500,
            )
            judge_prompt = build_judge_prompt(document=article, candidates=blinded_candidates)
            judge_results = {}
            for judge_spec in judge_specs:
                judge_id = judge_spec["judge_id"]
                jpath = judge_cache_path(cache_layout=cache_layout, judge_id=judge_id, doc_id=doc_id)
                judge_from_cache = False
                invalidate_this_judge = config["invalidate_judges"] or (
                    judge_id in config.get("invalidate_judge_ids", [])
                )
                if config["use_judge_cache"] and not invalidate_this_judge:
                    cached = load_judge_cache(
                        path=jpath,
                        config=config,
                        judge_id=judge_id,
                        doc_id=doc_id,
                        invalidate=invalidate_this_judge,
                    )
                    if cached is not None:
                        judge_results[judge_id] = cached["judge_result"]
                        judge_from_cache = True
                if not judge_from_cache:
                    raw_judge = await run_listwise_judge_for_spec(
                        judge_spec=judge_spec,
                        judge_max_tokens=config["judge_max_tokens"],
                        prompt=judge_prompt,
                        seed=doc_seed + 900 + hash(judge_id) % 1000,
                        openai_clients=openai_judge_clients,
                        tinker_runtime=tinker_judge_runtime,
                    )
                    finalized = finalize_judge_result(
                        judge_result=raw_judge,
                        blinded_candidates=blinded_candidates,
                        mapping=mapping,
                    )
                    if config["use_judge_cache"]:
                        write_judge_cache(
                            path=jpath,
                            payload={
                                "doc_id": doc_id,
                                "cache_key": judge_cache_key(config=config, judge_id=judge_id),
                                "judge_id": judge_id,
                                "judge_prompt_version": JUDGE_PROMPT_VERSION,
                                "saved_at": datetime.now(timezone.utc).isoformat(),
                                "blinded_candidates": blinded_candidates,
                                "blinded_mapping": mapping,
                                "judge_result": finalized,
                            },
                            overwrite=invalidate_this_judge,
                        )
                    judge_results[judge_id] = finalized
                judge_results[judge_id]["judge_from_cache"] = judge_from_cache
                judge_results[judge_id]["judge_cache_path"] = str(jpath)
            return {
                "index": row_idx,
                "attempt": attempt,
                "error": None,
                "doc_id": doc_id,
                "rollout_from_cache": rollout_from_cache,
                "style_from_cache": style_from_cache,
                "rollout_cache_path": str(rollout_path),
                "style_cache_path": str(style_path) if config["style_normalize_human"] else "",
                "original_label": row.get("ground_truth"),
                "majority_vote": row.get("majority_vote"),
                "expert_majority_vote": row.get("expert_majority_vote"),
                "generation_model": row.get("generation_model"),
                "source": {
                    "id": row.get("id"),
                    "ground_truth": row.get("ground_truth"),
                    "majority_vote": row.get("majority_vote"),
                    "expert_majority_vote": row.get("expert_majority_vote"),
                    "generation_model": row.get("generation_model"),
                    "article": article,
                    "annotator_1": row.get("annotator_1"),
                    "annotator_2": row.get("annotator_2"),
                    "annotator_3": row.get("annotator_3"),
                    "annotator_4": row.get("annotator_4"),
                    "annotator_5": row.get("annotator_5"),
                },
                "model_verdicts": verdict_results,
                "model_annotations": model_annotations,
                "human_annotations_raw": human_annotations_raw,
                "style_rewrites": style_rewrites,
                "human_annotations": human_annotations,
                "blinded_candidates": blinded_candidates,
                "blinded_mapping": mapping,
                "judge_results": judge_results,
            }
        except Exception as exc:
            last_error = str(exc)
            await asyncio.sleep(1.0 * attempt)
    return {
        "index": row_idx,
        "attempt": config["max_retries"],
        "error": last_error or "unknown_error",
        "doc_id": doc_id,
        "rollout_from_cache": False,
        "style_from_cache": False,
        "rollout_cache_path": str(rollout_path),
        "style_cache_path": str(style_path) if config["style_normalize_human"] else "",
        "original_label": row.get("ground_truth"),
        "majority_vote": row.get("majority_vote"),
        "expert_majority_vote": row.get("expert_majority_vote"),
        "generation_model": row.get("generation_model"),
        "source": {
            "id": row.get("id"),
            "ground_truth": row.get("ground_truth"),
            "majority_vote": row.get("majority_vote"),
            "expert_majority_vote": row.get("expert_majority_vote"),
            "generation_model": row.get("generation_model"),
            "article": article,
            "annotator_1": row.get("annotator_1"),
            "annotator_2": row.get("annotator_2"),
            "annotator_3": row.get("annotator_3"),
            "annotator_4": row.get("annotator_4"),
            "annotator_5": row.get("annotator_5"),
        },
        "model_verdicts": [],
        "model_annotations": [],
        "human_annotations_raw": [],
        "style_rewrites": [],
        "human_annotations": [],
        "blinded_candidates": [],
        "blinded_mapping": [],
        "judge_results": {},
    }


def compute_style_summary(records: list[dict]) -> dict:
    ok = [r for r in records if r.get("error") is None]
    rewrites = []
    for rec in ok:
        rewrites.extend(rec.get("style_rewrites", []))
    n_rewrites = len(rewrites)
    n_parse_ok = sum(1 for sr in rewrites if sr.get("parse_ok"))
    return {
        "n": len(records),
        "n_success": len(ok),
        "n_failed": len(records) - len(ok),
        "n_style_rewrites": n_rewrites,
        "n_parse_ok": n_parse_ok,
        "n_parse_fail": n_rewrites - n_parse_ok,
        "parse_ok_rate": n_parse_ok / n_rewrites if n_rewrites else 0.0,
    }


def compute_summary(records: list[dict], seed: int) -> dict:
    ok = [r for r in records if r.get("error") is None]
    if not ok:
        return {
            "n": len(records),
            "n_success": 0,
            "n_failed": len(records),
            "overall_win_rate": 0.0,
            "doc_win_rate_mean": 0.0,
            "doc_win_rate_ci95": [0.0, 0.0],
            "permutation_p": 1.0,
            "wilcoxon_p": 1.0,
            "effect_size": win_rate_effect_sizes(doc_win_rates=[]),
            "mean_diff_from_chance_ci95": [0.0, 0.0],
        }
    total_wins = sum(r["win_stats"]["wins"] for r in ok)
    total_ties = sum(r["win_stats"]["ties"] for r in ok)
    total_duels = sum(r["win_stats"]["total_duels"] for r in ok)
    overall = (total_wins + 0.5 * total_ties) / total_duels if total_duels > 0 else 0.0
    doc_rates = [r["win_stats"]["win_rate"] for r in ok]
    lo, hi = bootstrap_ci(values=doc_rates, seed=seed)
    effect = win_rate_effect_sizes(doc_win_rates=doc_rates)
    return {
        "n": len(records),
        "n_success": len(ok),
        "n_failed": len(records) - len(ok),
        "overall_win_rate": overall,
        "doc_win_rate_mean": sum(doc_rates) / len(doc_rates),
        "doc_win_rate_ci95": [lo, hi],
        "total_duels": total_duels,
        "permutation_p": permutation_test_win_rate(doc_rates, seed=seed),
        "wilcoxon_p": wilcoxon_p(doc_rates),
        "effect_size": effect,
        "mean_diff_from_chance_ci95": [lo - 0.5, hi - 0.5],
    }


def compute_summary_for_judge(records: list[dict], judge_id: str, seed: int) -> dict:
    proxy = []
    for rec in records:
        if rec.get("error") is not None:
            proxy.append(rec)
            continue
        jr = rec.get("judge_results", {}).get(judge_id)
        if jr is None:
            proxy.append({**rec, "error": f"missing judge {judge_id}"})
        else:
            proxy.append({**rec, "win_stats": jr["win_stats"]})
    out = compute_summary(records=proxy, seed=seed)
    out["judge_id"] = judge_id
    return out


def consensus_doc_win_rates(records: list[dict], judge_ids: list[str]) -> list[float]:
    ok = [r for r in records if r.get("error") is None]
    rates = []
    for rec in ok:
        per_judge = [rec["judge_results"][jid]["win_stats"]["win_rate"] for jid in judge_ids]
        if all(v > 0.5 for v in per_judge):
            rates.append(1.0)
        elif all(v < 0.5 for v in per_judge):
            rates.append(0.0)
        else:
            rates.append(0.5)
    return rates


def _attach_holm_per_judge(per_judge: dict, judge_ids: list[str]) -> None:
    perm_ps = [per_judge[jid].get("permutation_p", 1.0) for jid in judge_ids]
    wilc_ps = [per_judge[jid].get("wilcoxon_p", 1.0) for jid in judge_ids]
    perm_holm = holm_adjust_pvalues(p_values=perm_ps)
    wilc_holm = holm_adjust_pvalues(p_values=wilc_ps)
    for jid, p_adj, w_adj in zip(judge_ids, perm_holm, wilc_holm):
        per_judge[jid]["permutation_p_holm"] = p_adj
        per_judge[jid]["wilcoxon_p_holm"] = w_adj


def compute_multi_judge_summary(records: list[dict], judge_ids: list[str], seed: int) -> dict:
    per_judge = {
        jid: compute_summary_for_judge(records=records, judge_id=jid, seed=seed) for jid in judge_ids
    }
    _attach_holm_per_judge(per_judge=per_judge, judge_ids=judge_ids)
    panel = compute_panel_summary(records=records, judge_ids=judge_ids, seed=seed + 7)
    consensus_rates = consensus_doc_win_rates(records=records, judge_ids=judge_ids)
    ok = [r for r in records if r.get("error") is None]
    n_ok = len(ok)
    agree = 0
    for rec in ok:
        per = [rec["judge_results"][jid]["win_stats"]["win_rate"] for jid in judge_ids]
        if all(v > 0.5 for v in per) or all(v < 0.5 for v in per):
            agree += 1
    lo, hi = bootstrap_ci(values=consensus_rates, seed=seed + 17) if consensus_rates else (0.0, 0.0)
    consensus = {
        "judge_id": "consensus_all_win",
        "n": len(records),
        "n_success": n_ok,
        "doc_win_rate_mean": sum(consensus_rates) / len(consensus_rates) if consensus_rates else 0.0,
        "doc_win_rate_ci95": [lo, hi],
        "judge_agreement_rate": agree / n_ok if n_ok else 0.0,
        "permutation_p": permutation_test_win_rate(consensus_rates, seed=seed + 31),
        "wilcoxon_p": wilcoxon_p(consensus_rates),
    }
    return {
        "judge_panel": judge_ids,
        "judge_prompt_version": JUDGE_PROMPT_VERSION,
        "panel": panel,
        "per_judge": per_judge,
        "consensus": consensus,
    }


async def run_pipeline(config: dict) -> dict:
    load_dotenv()
    started_at = datetime.now(timezone.utc).isoformat()
    rows = fetch_dataset(dataset_url=config["dataset_url"])
    retry_row_indices = config.get("retry_row_indices")
    if retry_row_indices:
        selected, row_indices = select_rows_for_retry(
            rows=rows,
            offset=config["offset"],
            sample_size=config["sample_size"],
            retry_row_indices=retry_row_indices,
        )
    else:
        selected = select_rows(
            rows=rows,
            offset=config["offset"],
            sample_size=config["sample_size"],
        )
        row_indices = list(range(len(selected)))
    cache_layout = build_cache_layout(config=config)
    style_bank_by_doc = load_style_paraphrase_bank_index(
        bank_path=config["style_paraphrase_bank_path"],
        config=config,
    )
    config["style_bank_by_doc"] = style_bank_by_doc
    config["rollout_key"] = cache_layout["rollout_key"]

    judge_specs = judge_panel_by_id(judge_ids=config["judge_ids"])
    judge_ids = [spec["judge_id"] for spec in judge_specs]
    config["judge_panel"] = judge_ids

    need_rollouts = (not config["only_judge"]) and (not config["only_style"]) and (not config["skip_rollouts"])
    need_style = config["style_normalize_human"] and (not config["only_judge"]) and (not config["skip_style"])
    need_style_generation = need_style and (
        config["invalidate_style"] or config["only_style"] or not style_bank_by_doc
    )
    need_judge = not config["only_style"]
    need_tinker_judge = need_judge and any(spec["backend"] == "tinker" for spec in judge_specs)
    runtime = None
    if need_rollouts:
        runtime = await create_runtime(checkpoint_path=config["checkpoint_path"])
    base_runtime = None
    if need_style_generation:
        base_runtime = await create_runtime(checkpoint_path=None, base_model=config["base_model"])
    if need_tinker_judge and base_runtime is None:
        base_runtime = await create_runtime(checkpoint_path=None, base_model=config["base_model"])
    openai_judge_clients = build_judge_openai_clients(judge_specs=judge_specs) if need_judge else {}
    tinker_judge_runtime = base_runtime if need_tinker_judge else None

    semaphore = asyncio.Semaphore(config["workers"])

    async def worker(row_idx: int, row: dict) -> dict:
        async with semaphore:
            t0 = time.time()
            rec = await process_row(
                row_idx=row_idx,
                row=row,
                runtime=runtime,
                base_runtime=base_runtime,
                judge_specs=judge_specs,
                openai_judge_clients=openai_judge_clients,
                tinker_judge_runtime=tinker_judge_runtime,
                config=config,
                cache_layout=cache_layout,
            )
            rec["elapsed_s"] = time.time() - t0
            return rec

    tasks = [
        asyncio.create_task(worker(row_idx=row_indices[pos], row=row))
        for pos, row in enumerate(selected)
    ]
    records = []
    with tqdm(total=len(tasks), desc="winrate_eval", unit="doc") as pbar:
        for task in asyncio.as_completed(tasks):
            rec = await task
            records.append(rec)
            if rec["error"] is None:
                if config["only_style"]:
                    pbar.set_postfix_str(
                        f"id={rec['source']['id']} parse_ok={rec.get('n_parse_ok', 0)}/{rec.get('n_style_rewrites', 0)}"
                    )
                else:
                    wrs = [rec["judge_results"][jid]["win_stats"]["win_rate"] for jid in judge_ids if jid in rec.get("judge_results", {})]
                    panel_wr = sum(wrs) / len(wrs) if wrs else 0.0
                    pbar.set_postfix_str(f"id={rec['source']['id']} panel={panel_wr:.2f}")
            else:
                pbar.set_postfix_str(f"id={rec['source']['id']} failed")
            pbar.update(1)

    records.sort(key=lambda r: r["index"])
    if config.get("merge_into_audit"):
        base_records = load_records_from_audit(audit_path=config["merge_into_audit"])
        records = merge_retry_records(base_records=base_records, retry_records=records)
    if config["only_style"]:
        summary = compute_style_summary(records=records)
    else:
        summary = compute_multi_judge_summary(records=records, seed=config["seed"], judge_ids=judge_ids)
    finished_at = datetime.now(timezone.utc).isoformat()
    return {
        "meta": {
            "started_at": started_at,
            "finished_at": finished_at,
            "config": config,
            "cache_layout": {
                "dataset_key": cache_layout["dataset_key"],
                "rollout_key": cache_layout["rollout_key"],
                "style_key": cache_layout["style_key"],
                "rollout_dir": str(cache_layout["rollout_dir"]),
                "style_dir": str(cache_layout["style_dir"]),
                "judges_dir": str(cache_layout["judges_dir"]),
            },
            "judge_panel": JUDGE_PANEL,
        },
        "summary": summary,
        "records": records,
    }


def export_config(config: dict) -> dict:
    out = dict(config)
    out.pop("style_bank_by_doc", None)
    for key, value in list(out.items()):
        if isinstance(value, set):
            out[key] = sorted(value)
    return out


def export_meta(meta: dict) -> dict:
    meta_out = {
        "started_at": meta["started_at"],
        "finished_at": meta["finished_at"],
        "config": export_config(config=meta["config"]),
        "cache_layout": meta["cache_layout"],
        "judge_panel": meta.get("judge_panel", JUDGE_PANEL),
    }
    return meta_out


def compact_judge_ranking(judge_result: dict) -> list[dict]:
    rows = judge_result.get("ranking_with_metadata") or judge_result.get("ranking") or []
    compact = []
    for row in rows:
        compact.append(
            {
                "blind_id": row.get("item_id"),
                "rank": row.get("rank"),
                "true_kind": row.get("true_kind"),
                "source_id": row.get("source_id"),
                "quality_score": row.get("quality_score"),
            }
        )
    return compact


def compact_doc_record(rec: dict, judge_ids: list[str]) -> dict:
    row = {
        "doc_id": rec.get("doc_id"),
        "error": rec.get("error"),
        "ground_truth": rec.get("original_label"),
        "majority_vote": rec.get("majority_vote"),
        "expert_majority_vote": rec.get("expert_majority_vote"),
        "generation_model": rec.get("generation_model"),
        "rollout_cache_path": rec.get("rollout_cache_path"),
        "style_cache_path": rec.get("style_cache_path"),
    }
    if rec.get("error"):
        return row
    per_judge = {}
    for judge_id in judge_ids:
        jr = rec.get("judge_results", {}).get(judge_id)
        if jr is None:
            continue
        ranking = compact_judge_ranking(judge_result=jr)
        model_ranks = [r["rank"] for r in ranking if r.get("true_kind") == "model"]
        per_judge[judge_id] = {
            "doc_win_rate": jr["win_stats"]["win_rate"],
            "wins": jr["win_stats"]["wins"],
            "losses": jr["win_stats"]["losses"],
            "ties": jr["win_stats"]["ties"],
            "model_ranks": model_ranks,
            "ranking": ranking,
            "judge_cache_path": jr.get("judge_cache_path"),
            "from_cache": jr.get("judge_from_cache"),
        }
    doc_rates = [row["doc_win_rate"] for row in per_judge.values()]
    row["panel_doc_win_rate"] = sum(doc_rates) / len(doc_rates) if doc_rates else None
    row["per_judge"] = per_judge
    return row


def build_public_results(result: dict, run_name: str, audit_path: str) -> dict:
    meta = export_meta(meta=result["meta"])
    config = meta["config"]
    judge_ids = config.get("judge_panel") or [
        row["judge_id"] for row in meta.get("judge_panel", JUDGE_PANEL) if "judge_id" in row
    ]
    cache_layout = meta["cache_layout"]
    return {
        "run_name": run_name,
        "started_at": meta["started_at"],
        "finished_at": meta["finished_at"],
        "protocol": {
            "dataset_url": config["dataset_url"],
            "checkpoint_path": config["checkpoint_path"],
            "offset": config["offset"],
            "sample_size": config["sample_size"],
            "seed": config["seed"],
            "model_rollouts_per_doc": config["model_rollouts_per_doc"],
            "sampling_temperature": config["sampling_temperature"],
            "sampling_top_p": config["sampling_top_p"],
            "verdict_max_tokens": config["verdict_max_tokens"],
            "judge_max_tokens": config["judge_max_tokens"],
            "judge_ids": config.get("judge_ids") or judge_ids,
            "style_normalize_human": config["style_normalize_human"],
            "style_paraphrase_bank_path": config.get("style_paraphrase_bank_path"),
            "base_model": config["base_model"],
            "judge_prompt_version": result["summary"].get("judge_prompt_version", JUDGE_PROMPT_VERSION),
            "judge_panel": judge_ids,
            "primary_estimand": "panel_mean_doc_win_rate",
            "inference": (
                "Primary: one-sided sign-flip permutation and Wilcoxon on per-document "
                "panel averages (mean win rate across judges). Secondary: per-judge "
                "tests with Holm correction across the panel."
            ),
            "rollout_key": cache_layout["rollout_key"],
            "style_key": cache_layout["style_key"],
        },
        "summary": result["summary"],
        "documents": [
            compact_doc_record(rec=rec, judge_ids=judge_ids)
            for rec in sorted(result["records"], key=lambda r: r.get("index", 0))
        ],
        "artifacts": {
            "style_paraphrase_bank": config.get("style_paraphrase_bank_path"),
            "rollout_dir": cache_layout["rollout_dir"],
            "style_dir": cache_layout["style_dir"],
            "judges_dir": cache_layout["judges_dir"],
            "full_audit_jsonl": audit_path,
        },
    }


def write_outputs(result: dict, results_dir: str, run_name: str) -> dict:
    out_dir = Path(results_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    audit_path = out_dir / "audit.jsonl"
    manifest_path = out_dir / "manifest.json"
    meta_export = export_meta(meta=result["meta"])
    public = build_public_results(
        result=result,
        run_name=run_name,
        audit_path=str(audit_path),
    )
    results_path = out_dir / "results.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(public, f, ensure_ascii=False, indent=2)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "meta": meta_export,
                "summary": result["summary"],
                "results_path": str(results_path),
                "audit_path": str(audit_path),
                "manifest_path": str(manifest_path),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    with audit_path.open("w", encoding="utf-8") as f:
        for row in result["records"]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(meta_export, f, ensure_ascii=False, indent=2)
    tex_path = out_dir / "winrate_table.tex"
    tex_paths = write_winrate_table_tex(summary=result["summary"], out_path=tex_path, meta=meta_export)
    return {
        "results_dir": str(out_dir),
        "results_path": str(results_path),
        "summary_path": str(summary_path),
        "audit_path": str(audit_path),
        "manifest_path": str(manifest_path),
        **tex_paths,
    }


def write_experiment_mirror(result: dict, data_dir: str, run_name: str) -> dict:
    exp_dir = Path(data_dir) / "experiments" / run_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    results_path = exp_dir / "results.json"
    summary_path = exp_dir / "summary.json"
    audit_path = exp_dir / "audit.jsonl"
    manifest_path = exp_dir / "manifest.json"
    panel_path = exp_dir / "judge_panel.json"
    meta_export = export_meta(meta=result["meta"])
    public = build_public_results(
        result=result,
        run_name=run_name,
        audit_path=str(audit_path),
    )
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(public, f, ensure_ascii=False, indent=2)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "meta": meta_export,
                "summary": result["summary"],
                "results_path": str(results_path),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    with audit_path.open("w", encoding="utf-8") as f:
        for row in result["records"]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(meta_export, f, ensure_ascii=False, indent=2)
    with panel_path.open("w", encoding="utf-8") as f:
        json.dump(JUDGE_PANEL, f, ensure_ascii=False, indent=2)
    tex_path = exp_dir / "winrate_table.tex"
    tex_paths = write_winrate_table_tex(summary=result["summary"], out_path=tex_path, meta=meta_export)
    return {
        "experiment_dir": str(exp_dir),
        "experiment_results_path": str(results_path),
        "experiment_summary_path": str(summary_path),
        "experiment_audit_path": str(audit_path),
        "experiment_manifest_path": str(manifest_path),
        "experiment_panel_path": str(panel_path),
        "experiment_winrate_table_tex": tex_paths["winrate_table_tex"],
        "experiment_winrate_table_compact_tex": tex_paths["winrate_table_compact_tex"],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Listwise win-rate evaluation on human_detectors")
    parser.add_argument("--dataset-url", type=str, default=DEFAULT_EXPLANATION_DATASET)
    parser.add_argument("--checkpoint-path", type=str, default="")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--sample-size", type=int, default=25)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--model-rollouts-per-doc", type=int, default=1)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--sampling-temperature", type=float, default=0.8)
    parser.add_argument("--sampling-top-p", type=float, default=0.95)
    parser.add_argument("--verdict-max-tokens", type=int, default=300)
    parser.add_argument("--judge-max-tokens", type=int, default=512)
    parser.add_argument(
        "--judge-ids",
        type=str,
        default="",
        help="Comma-separated judge_id subset; default is full panel",
    )
    parser.add_argument("--use-judge-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--invalidate-judges",
        "--force-judges",
        action="store_true",
        dest="invalidate_judges",
        help="Ignore judge caches, recompute, and overwrite stale entries",
    )
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--results-dir", type=str, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--seed", type=int, default=2242)
    parser.add_argument(
        "--style-normalize-human",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Rewrite human comments with base model to match TELL annotation style (content preserved)",
    )
    parser.add_argument("--style-rewrite-max-tokens", type=int, default=512)
    parser.add_argument("--style-rewrite-temperature", type=float, default=0.0)
    parser.add_argument("--style-rewrite-top-p", type=float, default=1.0)
    parser.add_argument("--style-rewrite-reasoning-effort", type=str, default="low")
    parser.add_argument("--base-model", type=str, default=CFG.model.base_model)
    parser.add_argument("--use-rollout-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-style-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--invalidate-rollouts",
        "--force-rollouts",
        action="store_true",
        dest="invalidate_rollouts",
        help="Ignore rollout caches, recompute, and overwrite stale entries",
    )
    parser.add_argument(
        "--invalidate-style",
        "--force-style",
        action="store_true",
        dest="invalidate_style",
        help="Ignore style caches, recompute, and overwrite stale entries",
    )
    parser.add_argument(
        "--skip-rollouts",
        action="store_true",
        help="Do not sample rollouts; require cached rollouts for every document",
    )
    parser.add_argument(
        "--skip-style",
        action="store_true",
        help="Do not rewrite humans; require cached style rewrites when --style-normalize-human",
    )
    parser.add_argument(
        "--only-judge",
        action="store_true",
        help="Only run judge ranking using cached rollouts/style (if enabled)",
    )
    parser.add_argument(
        "--only-style",
        action="store_true",
        help="Only generate style-normalized human comments; write caches under data/winrate_eval/style_rewrites/",
    )
    parser.add_argument(
        "--style-paraphrase-bank-path",
        type=str,
        default=DEFAULT_STYLE_PARAPHRASE_BANK,
        help="Frozen JSON bank of style-normalized human comments (load for judge runs; written after --only-style)",
    )
    parser.add_argument(
        "--only-export-tex",
        type=str,
        default="",
        help="Path to summary.json; write winrate_table.tex and winrate_table_compact.tex next to it and exit",
    )
    parser.add_argument(
        "--retry-failed-audit",
        type=str,
        default="",
        help="audit.jsonl path; rerun only rows that have error",
    )
    parser.add_argument(
        "--retry-row-indices",
        type=str,
        default="",
        help="Comma-separated row indices within the offset/sample slice (overrides --retry-failed-audit)",
    )
    parser.add_argument(
        "--merge-into-audit",
        type=str,
        default="",
        help="audit.jsonl path; merge retry rows back into full run records before summary",
    )
    parser.add_argument(
        "--invalidate-judge-ids",
        type=str,
        default="",
        help="Comma-separated judge_ids to recompute even if cache exists (e.g. tinker_gpt_oss_120b)",
    )
    return parser


async def async_main(args: argparse.Namespace) -> None:
    if args.only_export_tex:
        summary_path = Path(args.only_export_tex)
        blob = json.loads(summary_path.read_text(encoding="utf-8"))
        tex_path = summary_path.parent / "winrate_table.tex"
        tex_paths = write_winrate_table_tex(
            summary=blob["summary"],
            out_path=tex_path,
            meta=blob.get("meta"),
        )
        print(json.dumps(tex_paths, indent=2))
        return

    config = build_run_config(args=args)
    if config["only_style"] and config["only_judge"]:
        raise ValueError("use only one of --only-style or --only-judge")
    if config["only_style"] and not config["style_normalize_human"]:
        raise ValueError("--only-style requires style normalization (drop --no-style-normalize-human)")
    if not config["only_style"] and not config["checkpoint_path"]:
        raise ValueError("--checkpoint-path is required unless --only-style")
    if config.get("retry_row_indices") and not config.get("merge_into_audit"):
        raise ValueError("--retry-failed-audit/--retry-row-indices requires --merge-into-audit")
    if config.get("merge_into_audit") and not config.get("retry_row_indices"):
        raise ValueError("--merge-into-audit requires --retry-failed-audit or --retry-row-indices")
    if not config["run_name"]:
        config["run_name"] = f"winrate_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    results_dir = config["results_dir"]
    if args.output_dir:
        results_dir = args.output_dir
    result = await run_pipeline(config=config)
    paths = write_outputs(result=result, results_dir=results_dir, run_name=config["run_name"])
    paths.update(
        write_experiment_mirror(
            result=result,
            data_dir=config["data_dir"],
            run_name=config["run_name"],
        )
    )
    if config["only_style"] and config["style_paraphrase_bank_path"]:
        bank_path = write_style_paraphrase_bank(
            bank_path=config["style_paraphrase_bank_path"],
            config=config,
            records=result["records"],
        )
        paths["style_paraphrase_bank_path"] = bank_path
    print(json.dumps({"summary": result["summary"], "paths": paths}, indent=2))


def main() -> None:
    args = build_arg_parser().parse_args()
    asyncio.run(async_main(args=args))


if __name__ == "__main__":
    main()
