"""Run causal tell faithfulness metrics from an eval audit log."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from typing import Sequence

from openai import AsyncOpenAI

from rl_detector.causal_tell_metrics import (
    Example,
    classification_metrics,
    evaluate_causal_tells,
    locate_tells,
)
from rl_detector.rewards import strip_tags


VERDICT_PROMPT = """\
You are a strict evaluator for AI-vs-human authorship detection.

Return only JSON with this exact schema:
{{"p_ai": float}}

The value p_ai is your probability from 0.0 to 1.0 that the document was AI-written.
Be calibrated and conservative. Do not explain your reasoning.

Document:
<text>
{text}
</text>
"""

COUNTERFACTUAL_REWRITE_PROMPT = """\
Rewrite only the span below. Preserve its meaning and factual content as much as possible.
Remove the specific {cue_type} cue described in the explanation.
Do not add new claims. Do not rewrite surrounding context.
Return only JSON with this exact schema:
{{"rewritten_span": string}}

Span:
{span_text}

Explanation of cue:
{explanation}
"""

CAUSAL_RESULTS_HEADER = [
    "date_tag",
    "run_name",
    "source_path",
    "step",
    "n_examples",
    "n_tells",
    "verdict_provider",
    "verdict_model",
    "full_auroc",
    "full_tpr_at_fpr01",
    "tell_only_auroc",
    "tell_only_tpr_at_fpr01",
    "sufficiency_drop_mean",
    "comprehensiveness_drop_mean",
    "removed_auroc",
    "delta_auroc_removed",
    "signed_deletion_score",
    "ai_positive_deletion_diracc",
    "human_positive_deletion_diracc",
    "contradiction_rate_high_score",
    "weighted_contradiction_high_score",
    "genericity_rate",
    "weighted_genericity",
    "area_under_budget_curve_auroc",
    "cf_directional_accuracy",
    "cf_impact_mean",
    "cf_impact_weighted",
    "num_counterfactuals",
    "summary_path",
]


def _json_default(obj):
    if isinstance(obj, Path):
        return str(obj)
    try:
        import numpy as np

        if isinstance(obj, np.generic):
            return obj.item()
    except Exception:
        pass
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _json_clean(obj):
    if isinstance(obj, float) and (obj != obj):
        return None
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {key: _json_clean(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_json_clean(value) for value in obj]
    if isinstance(obj, tuple):
        return [_json_clean(value) for value in obj]
    return obj


def _sanitize_tsv(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if value != value:
            return ""
        return f"{value:.12g}"
    return str(value).replace("\t", " ").replace("\n", " ").replace("\r", " ")


def load_audit_step(path: Path, step: str) -> dict:
    entries = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    if not entries:
        raise ValueError(f"No audit entries found in {path}")
    if step == "latest":
        return entries[-1]
    for entry in reversed(entries):
        if str(entry.get("step")) == str(step):
            return entry
    raise ValueError(f"Step {step!r} not found in {path}")


def examples_from_audit(entry: dict, require_format_ok: bool = True) -> tuple[list[Example], list[dict]]:
    examples: list[Example] = []
    skipped: list[dict] = []
    for idx, doc in enumerate(entry.get("docs", [])):
        reason = str(doc.get("format_reason", ""))
        if require_format_ok and reason != "ok":
            skipped.append({"index": idx, "reason": reason or "not_ok"})
            continue
        response_text = str(doc.get("response_text") or "")
        text = strip_tags(response_text)
        if not text:
            skipped.append({"index": idx, "reason": "empty_text"})
            continue
        indicators = doc.get("indicators") or []
        tells = locate_tells(text, indicators)
        if not tells:
            skipped.append({"index": idx, "reason": "no_located_tells"})
            continue
        label = 1 if int(doc.get("label", 0)) == 1 else -1
        examples.append(
            Example(
                doc_id=str(doc.get("doc_id") if doc.get("doc_id") is not None else idx),
                text=text,
                y=label,
                tells=tells,
            )
        )
    return examples, skipped


def fake_score_texts(texts: Sequence[str]) -> list[float]:
    """Deterministic offline scorer for smoke tests."""
    probs = []
    for text in texts:
        lower = text.lower()
        ai_hits = sum(lower.count(term) for term in ["ai", "generated", "formal", "therefore", "overall"])
        human_hits = sum(lower.count(term) for term in ["typo", "lol", "i ", "cant", "human"])
        logit = 0.35 * ai_hits - 0.35 * human_hits
        probs.append(1.0 / (1.0 + pow(2.718281828, -logit)))
    return probs


def _parse_p_ai(raw: str) -> float:
    try:
        parsed = json.loads(raw)
        return max(0.0, min(1.0, float(parsed["p_ai"])))
    except Exception:
        match = re.search(r"0(?:\.\d+)?|1(?:\.0+)?", raw)
        if not match:
            raise
        return max(0.0, min(1.0, float(match.group(0))))


def _parse_rewritten_span(raw: str) -> str:
    try:
        parsed = json.loads(raw)
        value = parsed.get("rewritten_span", "")
        return str(value).strip()
    except Exception:
        return raw.strip()


async def _score_texts_live(
    texts: Sequence[str],
    provider: str,
    model: str,
    max_concurrent: int,
    max_tokens: int,
) -> list[float]:
    provider = provider.lower()
    if provider != "xai":
        raise ValueError("eval_causal_tells currently supports provider=xai for live verdict judging")
    client = AsyncOpenAI(api_key=os.environ["XAI_API_KEY"], base_url="https://api.x.ai/v1", timeout=3600.0)
    sem = asyncio.Semaphore(max(1, max_concurrent))

    async def score_one(text: str) -> float:
        async with sem:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": VERDICT_PROMPT.format(text=text)}],
                temperature=0.0,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
                seed=2242,
            )
            content = response.choices[0].message.content or "{}"
            return _parse_p_ai(content)

    return list(await asyncio.gather(*[score_one(text) for text in texts]))


class CachedVerdictScorer:
    def __init__(
        self,
        provider: str,
        model: str,
        offline_fake: bool,
        max_concurrent: int,
        max_tokens: int,
    ) -> None:
        self.provider = provider
        self.model = model
        self.offline_fake = offline_fake
        self.max_concurrent = max_concurrent
        self.max_tokens = max_tokens
        self.cache: dict[str, float] = {}

    def __call__(self, texts: list[str]) -> list[float]:
        missing = [text for text in texts if text not in self.cache]
        if missing:
            if self.offline_fake:
                scores = fake_score_texts(missing)
            else:
                scores = asyncio.run(
                    _score_texts_live(
                        missing,
                        provider=self.provider,
                        model=self.model,
                        max_concurrent=self.max_concurrent,
                        max_tokens=self.max_tokens,
                    )
                )
            for text, score in zip(missing, scores):
                self.cache[text] = float(score)
        return [self.cache[text] for text in texts]


def _replace_one_span(text: str, start: int, end: int, replacement: str) -> str:
    return (text[:start] + replacement + text[end:]).strip() or " "


async def _rewrite_tells_live(
    requests: Sequence[tuple[Example, int]],
    model: str,
    max_concurrent: int,
    max_tokens: int,
) -> list[str]:
    client = AsyncOpenAI(api_key=os.environ["XAI_API_KEY"], base_url="https://api.x.ai/v1", timeout=3600.0)
    sem = asyncio.Semaphore(max(1, max_concurrent))

    async def rewrite_one(ex: Example, tell_idx: int) -> str:
        tell = ex.tells[tell_idx]
        span_text = ex.text[tell.start : tell.end]
        cue_type = "AI-authorship" if tell.polarity == 1 else "human-authorship"
        prompt = COUNTERFACTUAL_REWRITE_PROMPT.format(
            cue_type=cue_type,
            span_text=span_text,
            explanation=tell.explanation,
        )
        async with sem:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
                seed=2242,
            )
        return _parse_rewritten_span(response.choices[0].message.content or "{}")

    return list(await asyncio.gather(*[rewrite_one(ex, tell_idx) for ex, tell_idx in requests]))


def run_counterfactual_rewrites(
    examples: Sequence[Example],
    scorer: CachedVerdictScorer,
    model: str,
    offline_fake: bool,
    max_concurrent: int,
    max_tokens: int,
    max_tells_per_doc: int,
    min_score: float,
) -> dict:
    requests: list[tuple[Example, int]] = []
    for ex in examples:
        candidates = [
            (idx, tell)
            for idx, tell in enumerate(ex.tells)
            if tell.score >= min_score and tell.polarity in {-1, 1}
        ]
        candidates.sort(key=lambda item: item[1].score, reverse=True)
        for idx, _tell in candidates[:max_tells_per_doc]:
            requests.append((ex, idx))
    if not requests:
        return {
            "counterfactual": {
                "cf_directional_accuracy": None,
                "cf_impact_mean": None,
                "cf_impact_weighted": None,
                "num_counterfactuals": 0,
                "records": [],
            }
        }

    if offline_fake:
        rewritten = []
        for ex, tell_idx in requests:
            tell = ex.tells[tell_idx]
            span = ex.text[tell.start : tell.end]
            rewritten.append(f"[rewritten without {'AI' if tell.polarity == 1 else 'human'} cue: {span}]")
    else:
        rewritten = asyncio.run(
            _rewrite_tells_live(
                requests,
                model=model,
                max_concurrent=max_concurrent,
                max_tokens=max_tokens,
            )
        )

    full_probs_by_doc = {ex.doc_id: prob for ex, prob in zip(examples, scorer([ex.text for ex in examples]))}
    cf_texts = []
    metadata = []
    records = []
    for (ex, tell_idx), new_span in zip(requests, rewritten):
        tell = ex.tells[tell_idx]
        old_span = ex.text[tell.start : tell.end]
        if not new_span or new_span.strip() == old_span.strip():
            records.append(
                {
                    "doc_id": ex.doc_id,
                    "tell_idx": tell_idx,
                    "skipped": True,
                    "reason": "empty_or_unchanged_rewrite",
                    "old_span": old_span,
                    "new_span": new_span,
                    "polarity": tell.polarity,
                    "score": tell.score,
                    "explanation": tell.explanation,
                }
            )
            continue
        cf_texts.append(_replace_one_span(ex.text, tell.start, tell.end, new_span))
        metadata.append((ex, tell_idx, old_span, new_span))

    if not cf_texts:
        return {
            "counterfactual": {
                "cf_directional_accuracy": None,
                "cf_impact_mean": None,
                "cf_impact_weighted": None,
                "num_counterfactuals": 0,
                "records": records,
            }
        }

    cf_probs = scorer(cf_texts)
    effects = []
    weights = []
    for cf_prob, (ex, tell_idx, old_span, new_span) in zip(cf_probs, metadata):
        tell = ex.tells[tell_idx]
        full_prob = full_probs_by_doc[ex.doc_id]
        effect = tell.polarity * (full_prob - cf_prob)
        effects.append(effect)
        weights.append(tell.score)
        records.append(
            {
                "doc_id": ex.doc_id,
                "tell_idx": tell_idx,
                "skipped": False,
                "old_span": old_span,
                "new_span": new_span,
                "polarity": tell.polarity,
                "score": tell.score,
                "explanation": tell.explanation,
                "score_before": full_prob,
                "score_after": cf_prob,
                "effect": effect,
                "direction_correct": effect > 1e-6,
            }
        )

    total_weight = sum(weights)
    weighted = sum(effect * weight for effect, weight in zip(effects, weights)) / max(total_weight, 1e-12)
    return {
        "counterfactual": {
            "model": model,
            "cf_directional_accuracy": sum(1 for effect in effects if effect > 1e-6) / len(effects),
            "cf_impact_mean": sum(effects) / len(effects),
            "cf_impact_weighted": weighted,
            "num_counterfactuals": len(effects),
            "records": records,
        }
    }


def write_outputs(
    output_dir: Path,
    run_name: str,
    summary: dict,
    records: Sequence[Example],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{run_name}.summary.json"
    records_path = output_dir / f"{run_name}.records.jsonl"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(_json_clean(summary), f, indent=2, ensure_ascii=False, default=_json_default)
    with records_path.open("w", encoding="utf-8") as f:
        for ex in records:
            f.write(
                json.dumps(
                    {
                        "doc_id": ex.doc_id,
                        "label": ex.y,
                        "text_len": len(ex.text),
                        "tells": [
                            {
                                "start": tell.start,
                                "end": tell.end,
                                "polarity": tell.polarity,
                                "score": tell.score,
                                "span_text": tell.span_text,
                                "explanation": tell.explanation,
                                "type": tell.tell_type,
                            }
                            for tell in ex.tells
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return summary_path, records_path


def append_causal_results(path: Path, row: dict) -> None:
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8") as f:
        if not exists:
            f.write("\t".join(CAUSAL_RESULTS_HEADER) + "\n")
        f.write("\t".join(_sanitize_tsv(row.get(col, "")) for col in CAUSAL_RESULTS_HEADER) + "\n")


def build_results_row(args: argparse.Namespace, step: str, metrics: dict, summary_path: Path) -> dict:
    return {
        "date_tag": datetime.now(timezone.utc).strftime("%Y%m%d"),
        "run_name": args.run_name,
        "source_path": args.eval_audit_path,
        "step": step,
        "n_examples": metrics.get("n_examples"),
        "n_tells": metrics.get("n_tells"),
        "verdict_provider": args.verdict_provider,
        "verdict_model": args.verdict_model,
        "full_auroc": metrics.get("full", {}).get("auroc"),
        "full_tpr_at_fpr01": metrics.get("full", {}).get("tpr_at_fpr_0.01"),
        "tell_only_auroc": metrics.get("tell_only", {}).get("auroc"),
        "tell_only_tpr_at_fpr01": metrics.get("tell_only", {}).get("tpr_at_fpr_0.01"),
        "sufficiency_drop_mean": metrics.get("sufficiency_drop_mean"),
        "comprehensiveness_drop_mean": metrics.get("comprehensiveness_drop_mean"),
        "removed_auroc": metrics.get("removed", {}).get("auroc"),
        "delta_auroc_removed": metrics.get("delta_auroc_removed"),
        "signed_deletion_score": metrics.get("signed_deletion_score"),
        "ai_positive_deletion_diracc": metrics.get("ai_positive_deletion_diracc"),
        "human_positive_deletion_diracc": metrics.get("human_positive_deletion_diracc"),
        "contradiction_rate_high_score": metrics.get("contradiction_rate_high_score"),
        "weighted_contradiction_high_score": metrics.get("weighted_contradiction_high_score"),
        "genericity_rate": metrics.get("genericity_rate"),
        "weighted_genericity": metrics.get("weighted_genericity"),
        "area_under_budget_curve_auroc": metrics.get("area_under_budget_curve_auroc"),
        "cf_directional_accuracy": metrics.get("counterfactual", {}).get("cf_directional_accuracy"),
        "cf_impact_mean": metrics.get("counterfactual", {}).get("cf_impact_mean"),
        "cf_impact_weighted": metrics.get("counterfactual", {}).get("cf_impact_weighted"),
        "num_counterfactuals": metrics.get("counterfactual", {}).get("num_counterfactuals"),
        "summary_path": summary_path,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate causal faithfulness of TELL spans from eval audit logs.")
    parser.add_argument("--eval-audit-path", required=True)
    parser.add_argument("--step", default="latest")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-dir", default="causal_eval")
    parser.add_argument("--causal-results-path", default="causal_results.tsv")
    parser.add_argument("--verdict-provider", default="xai")
    parser.add_argument("--verdict-model", default="grok-4-1-fast-reasoning")
    parser.add_argument("--enable-counterfactuals", action="store_true")
    parser.add_argument("--counterfactual-model", default="grok-4-1-fast-reasoning")
    parser.add_argument("--counterfactual-max-tells-per-doc", type=int, default=5)
    parser.add_argument("--counterfactual-min-score", type=float, default=0.5)
    parser.add_argument("--counterfactual-max-tokens", type=int, default=512)
    parser.add_argument("--max-concurrent", type=int, default=16)
    parser.add_argument("--verdict-max-tokens", type=int, default=64)
    parser.add_argument("--high-score-threshold", type=float, default=0.5)
    parser.add_argument("--mask-token", default=None)
    parser.add_argument("--offline-fake-scorer", action="store_true")
    parser.add_argument("--include-format-failures", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    audit_path = Path(args.eval_audit_path)
    entry = load_audit_step(audit_path, args.step)
    examples, skipped = examples_from_audit(entry, require_format_ok=not args.include_format_failures)
    if not examples:
        raise SystemExit("No valid examples found for causal tell evaluation.")

    scorer = CachedVerdictScorer(
        provider=args.verdict_provider,
        model=args.verdict_model,
        offline_fake=args.offline_fake_scorer,
        max_concurrent=args.max_concurrent,
        max_tokens=args.verdict_max_tokens,
    )
    metrics = evaluate_causal_tells(
        examples,
        score_fn=scorer,
        high_score_threshold=args.high_score_threshold,
        mask_token=args.mask_token,
    )

    if args.enable_counterfactuals:
        metrics.update(
            run_counterfactual_rewrites(
                examples,
                scorer=scorer,
                model=args.counterfactual_model,
                offline_fake=args.offline_fake_scorer,
                max_concurrent=args.max_concurrent,
                max_tokens=args.counterfactual_max_tokens,
                max_tells_per_doc=args.counterfactual_max_tells_per_doc,
                min_score=args.counterfactual_min_score,
            )
        )

    summary = {
        "meta": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "run_name": args.run_name,
            "eval_audit_path": str(audit_path),
            "step": entry.get("step"),
            "verdict_provider": args.verdict_provider,
            "verdict_model": args.verdict_model,
            "offline_fake_scorer": args.offline_fake_scorer,
            "skipped_docs": skipped,
            "scored_text_cache_size": len(scorer.cache),
            "human_agreement_status": "not_computed_requires_human_ratings",
        },
        "metrics": metrics,
        "references": {
            "rationales": "Lei et al. 2016, Rationalizing Neural Predictions",
            "sufficiency_comprehensiveness": "DeYoung et al. 2020, ERASER",
            "faithfulness": "Jacovi and Goldberg 2020; Jain and Wallace 2019",
            "counterfactuals_deferred": "Wu et al. 2021, Polyjuice; Ross et al. 2020, MiCE",
            "human_utility_deferred": "Hase and Bansal 2020",
        },
    }
    summary_path, records_path = write_outputs(Path(args.output_dir), args.run_name, summary, examples)
    row = build_results_row(args, str(entry.get("step")), metrics, summary_path)
    append_causal_results(Path(args.causal_results_path), row)
    print(json.dumps({"summary_path": str(summary_path), "records_path": str(records_path), "results_path": args.causal_results_path}, indent=2))
    print(json.dumps(_json_clean(row), indent=2, default=_json_default))


if __name__ == "__main__":
    main()
