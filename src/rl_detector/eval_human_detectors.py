"""Separate evaluation pipeline for human_detectors.json with LLM judging."""

import argparse
import asyncio
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

import tinker
from dotenv import load_dotenv
from openai import AsyncOpenAI
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from rl_detector.annotate import create_runtime
from rl_detector.config import CFG
from rl_detector.data import clean_document_text
from rl_detector.frozen import aggregate, self_score_from_output
from rl_detector.prompt_utils import format_prompt_for_model
from rl_detector.rewards import parse_indicators
from rl_detector.rollouts import extract_response_text


DEFAULT_EXPLANATION_DATASET = "hf://acmc/expert-annotated-TELL/validation"


JUDGE_PROMPT_TEMPLATE = """You are evaluating the quality of model explanations for AI-vs-human detection.

Goal:
Evaluate how well the model's explanations (bracket annotation spans + why text) capture the expert evidence present in the human annotations.
Treat all human annotations jointly as the expert reference signal, not as separate independent judges.

What to evaluate:
1) Evidence capture quality: does the model capture the important evidence discussed by experts.
2) Evidence precision: are model explanations grounded in expert-described evidence rather than unsupported claims.
3) Confidence alignment: does model confidence align with the joint expert confidence signal.

Rules:
- Be strict and conservative.
- Reward specific overlap in evidence, not vague thematic overlap.
- Do not reward style or fluency by itself.
- Use all annotator comments together as one expert evidence pool.
- Output JSON only.

Scoring formulas:
- evidence_capture_score = 0.65 * coverage_score + 0.35 * precision_score
- confidence_alignment_score = 1.0 - abs(model_confidence - expert_confidence)
- final_score = 0.75 * evidence_capture_score + 0.25 * confidence_alignment_score

All score fields must be floats in [0.0, 1.0].

Return exactly this schema:
{{
  "coverage_score": float,
  "precision_score": float,
  "evidence_capture_score": float,
  "confidence_alignment_score": float,
  "final_score": float,
  "short_rationale": string,
  "strengths": [string],
  "misses": [string]
}}

Data:
- document_id: {document_id}
- ground_truth: {ground_truth}
- expert_majority_vote: {expert_majority_vote}
- model_verdict: {model_verdict}
- model_confidence: {model_confidence}
- expert_confidence: {expert_confidence}

Model tagged output:
<model_output>
{model_output}
</model_output>

Model tells parsed:
{parsed_tells}

All human annotations:
{all_annotations}
"""


def build_run_config(args: argparse.Namespace) -> dict:
    checkpoint_path = args.checkpoint_path
    if checkpoint_path in ("", "null", "None", "none"):
        checkpoint_path = None
    return {
        "dataset_url": args.dataset_url,
        "checkpoint_path": checkpoint_path,
        "offset": args.offset,
        "sample_size": args.sample_size,
        "judge_model": args.judge_model,
        "judge_base_url": args.judge_base_url,
        "judge_max_tokens": args.judge_max_tokens,
        "output_dir": args.output_dir,
        "run_name": args.run_name,
        "workers": args.workers,
        "max_retries": args.max_retries,
    }


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
            return rows_from_payload(json.load(f))

    with urlopen(dataset_url) as response:
        if dataset_url.endswith(".jsonl"):
            return [json.loads(line) for line in response.read().decode("utf-8").splitlines() if line.strip()]
        payload = json.load(response)
    return rows_from_payload(payload)


def select_rows(rows: list[dict], offset: int, sample_size: int) -> list[dict]:
    start = max(0, offset)
    end = start + max(0, sample_size)
    return rows[start:end]


def compute_expert_confidence(row: dict) -> float:
    values = []
    for idx in range(1, 6):
        annot = row[f"annotator_{idx}"]
        conf = annot.get("confidence")
        if conf is None:
            continue
        values.append((float(conf) - 1.0) / 4.0)
    if not values:
        return 0.5
    return max(0.0, min(1.0, sum(values) / len(values)))


def _get_document_text(row: dict) -> str:
    article = row.get("article")
    if article is None or str(article).strip() == "":
        article = row.get("text", "")
    return clean_document_text(str(article))


def build_annotations_blob(row: dict) -> str:
    items = []
    for idx in range(1, 6):
        annot = row[f"annotator_{idx}"]
        items.append(
            {
                "annotator_id": f"annotator_{idx}",
                "guess": annot.get("guess"),
                "confidence": annot.get("confidence"),
                "comment": annot.get("comment"),
            }
        )
    return json.dumps(items, ensure_ascii=True)


async def run_model_prediction(runtime: dict, document: str) -> dict:
    tokenizer = runtime["tokenizer"]
    sampling_client = runtime["sampling_client"]
    _, prompt_text = format_prompt_for_model(tokenizer=tokenizer, text=document)
    prompt_tokens = tokenizer.encode(prompt_text)
    model_input = tinker.ModelInput.from_ints(prompt_tokens)
    sampled = await sampling_client.sample_async(
        prompt=model_input,
        num_samples=1,
        sampling_params=tinker.SamplingParams(
            max_tokens=CFG.sampling.max_tokens,
            temperature=0.0,
            top_p=1.0,
            seed=2242,
            reasoning_effort=CFG.sampling.reasoning_effort,
        ),
    )
    completion_text = tokenizer.decode(sampled.sequences[0].tokens)
    response_text = extract_response_text(completion_text)
    indicators = parse_indicators(output=response_text) or []
    tell_scored = self_score_from_output(response_text, indicators) if indicators else []
    aggregate_score = aggregate(scored=tell_scored)
    model_confidence = max(0.0, min(1.0, abs(float(aggregate_score))))
    model_verdict = "Machine-Generated" if aggregate_score > 0 else "Human-Generated"
    return {
        "prompt_text": prompt_text,
        "completion_text": completion_text,
        "response_text": response_text,
        "indicators": indicators,
        "tell_scored": tell_scored,
        "aggregate_score": aggregate_score,
        "model_confidence": model_confidence,
        "model_verdict": model_verdict,
    }


async def judge_prediction(
    judge_client: AsyncOpenAI,
    judge_model: str,
    judge_max_tokens: int,
    row: dict,
    model_result: dict,
) -> dict:
    expert_confidence = compute_expert_confidence(row=row)
    annotations_blob = build_annotations_blob(row=row)
    parsed_tells = json.dumps(model_result["indicators"], ensure_ascii=True)
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        document_id=row.get("id"),
        ground_truth=row.get("ground_truth"),
        expert_majority_vote=row.get("expert_majority_vote"),
        model_verdict=model_result["model_verdict"],
        model_confidence=f"{model_result['model_confidence']:.6f}",
        expert_confidence=f"{expert_confidence:.6f}",
        model_output=model_result["response_text"],
        parsed_tells=parsed_tells,
        all_annotations=annotations_blob,
    )
    response = await judge_client.chat.completions.create(
        model=judge_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=judge_max_tokens,
        response_format={"type": "json_object"},
        seed=2242,
    )
    raw_content = response.choices[0].message.content or "{}"
    reasoning_content = getattr(response.choices[0].message, "reasoning_content", "") or ""
    parsed = json.loads(raw_content)
    return {
        "judge_prompt": prompt,
        "judge_raw_content": raw_content,
        "judge_reasoning_content": reasoning_content,
        "judge_scores": parsed,
        "expert_confidence": expert_confidence,
    }


async def process_row(
    row_idx: int,
    row: dict,
    runtime: dict,
    judge_client: AsyncOpenAI,
    config: dict,
) -> dict:
    last_error = None
    article = _get_document_text(row=row)
    has_human_annotations = all(f"annotator_{idx}" in row for idx in range(1, 6))
    for attempt in range(1, config["max_retries"] + 1):
        try:
            t0 = time.time()
            model_result = await asyncio.wait_for(
                fut=run_model_prediction(runtime=runtime, document=article),
                timeout=360.0,
            )
            judge_result = None
            if has_human_annotations:
                judge_result = await asyncio.wait_for(
                    fut=judge_prediction(
                        judge_client=judge_client,
                        judge_model=config["judge_model"],
                        judge_max_tokens=config["judge_max_tokens"],
                        row=row,
                        model_result=model_result,
                    ),
                    timeout=240.0,
                )
            elapsed_s = time.time() - t0
            return {
                "index": row_idx,
                "elapsed_s": elapsed_s,
                "attempt": attempt,
                "source": {
                    "id": row.get("id"),
                    "ground_truth": row.get("ground_truth"),
                    "majority_vote": row.get("majority_vote"),
                    "expert_majority_vote": row.get("expert_majority_vote"),
                    "label": row.get("label"),
                    "generation_model": row.get("generation_model"),
                    "article": article,
                    "annotator_1": row.get("annotator_1"),
                    "annotator_2": row.get("annotator_2"),
                    "annotator_3": row.get("annotator_3"),
                    "annotator_4": row.get("annotator_4"),
                    "annotator_5": row.get("annotator_5"),
                },
                "model": model_result,
                "judge": judge_result,
                "error": None,
            }
        except Exception as exc:
            last_error = str(exc)
            await asyncio.sleep(delay=1.5 * attempt)
    return {
        "index": row_idx,
        "elapsed_s": 0.0,
        "attempt": config["max_retries"],
        "source": {
            "id": row.get("id"),
            "ground_truth": row.get("ground_truth"),
            "majority_vote": row.get("majority_vote"),
            "expert_majority_vote": row.get("expert_majority_vote"),
            "label": row.get("label"),
            "generation_model": row.get("generation_model"),
            "article": article,
            "annotator_1": row.get("annotator_1"),
            "annotator_2": row.get("annotator_2"),
            "annotator_3": row.get("annotator_3"),
            "annotator_4": row.get("annotator_4"),
            "annotator_5": row.get("annotator_5"),
        },
        "model": None,
        "judge": None,
        "error": last_error or "unknown_error",
    }


def compute_summary(records: list[dict]) -> dict:
    ok_records = [r for r in records if r.get("model") is not None]
    judged_records = [r for r in ok_records if r.get("judge") is not None]
    final_scores = [float(r["judge"]["judge_scores"].get("final_score", 0.0)) for r in judged_records]
    evidence_scores = [float(r["judge"]["judge_scores"].get("evidence_capture_score", 0.0)) for r in judged_records]
    conf_scores = [float(r["judge"]["judge_scores"].get("confidence_alignment_score", 0.0)) for r in judged_records]
    if not ok_records:
        return {
            "n": len(records),
            "n_success": 0,
            "n_failed": len(records),
            "final_score_mean": 0.0,
            "evidence_capture_mean": 0.0,
            "confidence_alignment_mean": 0.0,
            "eval_accuracy_ground_truth": 0.0,
            "eval_accuracy_expert_majority_vote": 0.0,
            "eval_auroc_ground_truth": 0.0,
            "eval_auroc_expert_majority_vote": 0.0,
        }
    def _to_binary_label(v) -> int | None:
        if v is None:
            return None
        s = str(v).strip().lower()
        if s in ("ai", "machine-generated", "machine_generated", "1", "true"):
            return 1
        if s in ("human", "human-generated", "human_generated", "0", "false"):
            return 0
        return None
    y_gt: list[int] = []
    y_expert: list[int] = []
    y_label: list[int] = []
    y_score_for_gt: list[float] = []
    y_score_for_expert: list[float] = []
    y_score_for_label: list[float] = []
    y_pred_gt: list[int] = []
    y_pred_expert: list[int] = []
    y_pred_label: list[int] = []
    for row in ok_records:
        score = float(row["model"]["aggregate_score"])
        pred = 1 if score > 0 else 0
        gt = _to_binary_label(row["source"].get("ground_truth"))
        expert = _to_binary_label(row["source"].get("expert_majority_vote"))
        label = row["source"].get("label")
        label_bin = int(label) if label in (0, 1) else None
        if gt is not None:
            y_gt.append(gt)
            y_score_for_gt.append(score)
            y_pred_gt.append(pred)
        if expert is not None:
            y_expert.append(expert)
            y_score_for_expert.append(score)
            y_pred_expert.append(pred)
        if label_bin is not None:
            y_label.append(label_bin)
            y_score_for_label.append(score)
            y_pred_label.append(pred)
    acc_gt = (sum(1 for a, b in zip(y_gt, y_pred_gt) if a == b) / len(y_gt)) if y_gt else 0.0
    acc_expert = (sum(1 for a, b in zip(y_expert, y_pred_expert) if a == b) / len(y_expert)) if y_expert else 0.0
    acc_label = (sum(1 for a, b in zip(y_label, y_pred_label) if a == b) / len(y_label)) if y_label else 0.0
    auroc_gt = roc_auc_score(y_gt, y_score_for_gt) if len(set(y_gt)) > 1 else 0.0
    auroc_expert = roc_auc_score(y_expert, y_score_for_expert) if len(set(y_expert)) > 1 else 0.0
    auroc_label = roc_auc_score(y_label, y_score_for_label) if len(set(y_label)) > 1 else 0.0
    return {
        "n": len(records),
        "n_success": len(ok_records),
        "n_failed": len(records) - len(ok_records),
        "final_score_mean": (sum(final_scores) / len(final_scores)) if final_scores else 0.0,
        "evidence_capture_mean": (sum(evidence_scores) / len(evidence_scores)) if evidence_scores else 0.0,
        "confidence_alignment_mean": (sum(conf_scores) / len(conf_scores)) if conf_scores else 0.0,
        "eval_accuracy_ground_truth": acc_gt,
        "eval_accuracy_expert_majority_vote": acc_expert,
        "eval_accuracy_label": acc_label,
        "eval_auroc_ground_truth": float(auroc_gt),
        "eval_auroc_expert_majority_vote": float(auroc_expert),
        "eval_auroc_label": float(auroc_label),
        "n_judged_records": len(judged_records),
    }


async def run_pipeline(config: dict) -> dict:
    load_dotenv()
    started_at = datetime.now(timezone.utc).isoformat()
    rows = fetch_dataset(dataset_url=config["dataset_url"])
    sampled_rows = select_rows(rows=rows, offset=config["offset"], sample_size=config["sample_size"])
    runtime = await create_runtime(checkpoint_path=config["checkpoint_path"])
    judge_client = AsyncOpenAI(
        api_key=os.environ["XAI_API_KEY"],
        base_url=config["judge_base_url"],
        timeout=3600.0,
    )
    semaphore = asyncio.Semaphore(config["workers"])

    async def worker(row_idx: int, row: dict) -> dict:
        async with semaphore:
            return await process_row(
                row_idx=row_idx,
                row=row,
                runtime=runtime,
                judge_client=judge_client,
                config=config,
            )

    tasks = [asyncio.create_task(worker(row_idx=idx, row=row)) for idx, row in enumerate(sampled_rows)]
    records = []
    with tqdm(total=len(tasks), desc="eval_human_detectors", unit="doc") as pbar:
        for coro in asyncio.as_completed(tasks):
            record = await coro
            records.append(record)
            if record["error"] is None:
                if record["judge"] is not None:
                    final_score = record["judge"]["judge_scores"].get("final_score")
                    pbar.set_postfix_str(f"id={record['source']['id']} final={final_score}")
                else:
                    pbar.set_postfix_str(f"id={record['source']['id']} score={record['model']['aggregate_score']:.3f}")
            else:
                pbar.set_postfix_str(f"id={record['source']['id']} failed")
            pbar.update(1)

    records.sort(key=lambda x: x["index"])
    finished_at = datetime.now(timezone.utc).isoformat()
    summary = compute_summary(records=records)
    return {
        "meta": {
            "started_at": started_at,
            "finished_at": finished_at,
            "config": config,
        },
        "summary": summary,
        "records": records,
    }


def write_outputs(result: dict, output_dir: str, run_name: str) -> dict:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"{run_name}.summary.json"
    audit_path = out_dir / f"{run_name}.audit.jsonl"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "meta": result["meta"],
                "summary": result["summary"],
                "audit_path": str(audit_path),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    with audit_path.open("w", encoding="utf-8") as f:
        for record in result["records"]:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"summary_path": str(summary_path), "audit_path": str(audit_path)}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate detector explanations on human_detectors.json")
    parser.add_argument("--dataset-url", type=str, required=False, default=DEFAULT_EXPLANATION_DATASET)
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=False,
        default="null",
        help="Checkpoint path for model eval. Use null for base model.",
    )
    parser.add_argument("--offset", type=int, required=False, default=0)
    parser.add_argument("--sample-size", type=int, required=False, default=25)
    parser.add_argument("--judge-model", type=str, required=False, default="grok-4-1-fast-reasoning")
    parser.add_argument("--judge-base-url", type=str, required=False, default="https://api.x.ai/v1")
    parser.add_argument("--judge-max-tokens", type=int, required=False, default=1400)
    parser.add_argument("--output-dir", type=str, required=False, default="human_detectors_eval")
    parser.add_argument("--run-name", type=str, required=False, default="")
    parser.add_argument("--workers", type=int, required=False, default=32)
    parser.add_argument("--max-retries", type=int, required=False, default=3)
    return parser


async def async_main(args: argparse.Namespace) -> None:
    config = build_run_config(args=args)
    if not config["run_name"]:
        config["run_name"] = f"run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    result = await run_pipeline(config=config)
    paths = write_outputs(result=result, output_dir=config["output_dir"], run_name=config["run_name"])
    print(json.dumps({"summary": result["summary"], "paths": paths}, indent=2))


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    asyncio.run(async_main(args=args))


if __name__ == "__main__":
    main()
