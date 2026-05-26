"""Rewrite weak 'why' explanations in acmc/TELL using the XAI (Grok) API.

For each nested tell annotation, checks whether the 'why' field has a genuinely
mechanistic explanation — WHY does this feature indicate human vs AI authorship at
the model-behaviour / training level?  Weak whys are rewritten.

The default heuristic is intentionally strict: a why must reference BOTH the observed
pattern AND a concrete reason (training, human habit, model behaviour).  Anything
short, vague, or that ends with "is a sign of X" without explaining the mechanism gets
flagged.  Use --rewrite-all to rewrite every why regardless.

Usage (dry-run on 50 rows, print diffs only — no API calls):
    PYTHONPATH=src uv run python experiments/rewrite_why_explanations.py \
        --limit 50 --split train --dry-run

Smoke test (20 rows, actually calls API):
    PYTHONPATH=src uv run python experiments/rewrite_why_explanations.py \
        --limit 20 --split train --out-dir /tmp/tell_improved_test

Full run, both splits:
    PYTHONPATH=src uv run python experiments/rewrite_why_explanations.py \
        --out-dir data/tell_improved --concurrency 40

Push to HuggingFace when happy with the output:
    PYTHONPATH=src uv run python experiments/rewrite_why_explanations.py \
        --out-dir data/tell_improved --push-to-hub acmc/TELL-improved
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import pathlib
import re
import sys
import time
from dataclasses import dataclass, field

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from openai import AsyncOpenAI
from rl_detector.tell_xml import escape_attr_piece, root_splits

# ── Quality heuristic ─────────────────────────────────────────────────────────
#
# We want to flag any why that doesn't contain a concrete reason WHY the feature
# is more common in human vs AI text.  The heuristic is deliberately strict:
# having "AI" or "because" is NOT enough; we require at least one of:
#   (a) a model-behaviour phrase  ("AI tends to", "LLMs rarely", "trained to", etc.)
#   (b) a human-habit phrase      ("humans naturally", "human writers tend to", etc.)
#   (c) an explicit causal chain  ("because + mechanism", "unlike X which does Y")
#
# If NONE of those are present, we flag the why for rewriting.

_MODEL_BEHAVIOUR_RE = re.compile(
    r"(?:"
    r"AI\s+(tends?|rarely|never|often|usually|frequently|almost never|is trained|generates?|produce[s]?)"
    r"|LLM[s]?\s+(tend[s]?|rarely|never|often|generate[s]?|produce[s]?)"
    r"|language model[s]?\s+(tend[s]?|rarely|generate[s]?|produce[s]?|are trained)"
    r"|\btrained\s+(to|on\b|not\s+to)"
    r"|\bfine[- ]?tun(ed|ing)\b"
    r"|(?:GPT|Claude|Gemini|Grok|LLM[s]?|model[s]?)\s+(tend[s]?|often|rarely|generate[s]?|produce[s]?|are\s+trained|is\s+trained)"
    r"|models?\s+(tend[s]?|rarely|often|are trained|generate[s]?|produce[s]?)"
    r"|(?:generated|auto-?generated|AI-?generated)\s+text"
    r"|\bgenerat(es?|ed|ion)\b.{0,40}\b(AI|model|LLM|automat)"
    r"|\b(AI|LLM|model[s]?).{0,40}\b(generat|produc|avoid|omit|skip|rarely|seldom|never)"
    r")",
    re.IGNORECASE,
)

_HUMAN_HABIT_RE = re.compile(
    r"(?:"
    r"human[s]?\s+(naturally|tend[s]?|often|rarely|write[s]?|use[s]?|prefer[s]?|avoid[s]?)"
    r"|people\s+(tend[s]?|naturally|often|rarely)"
    r"|human\s+writer[s]?"
    r"|(?:humans|people)\s+(?:instinctively|spontaneously|unconsciously)"
    r")",
    re.IGNORECASE,
)

_CAUSAL_CHAIN_RE = re.compile(
    r"(?:"
    r"\bbecause\b.{5,}"          # "because" followed by actual content
    r"|\bunlike\b.{10,}"          # "unlike X which does Y"
    r"|\bwhereas\b.{10,}"
    r"|\bin contrast\b.{10,}"
    r"|\bunless prompted\b"
    r"|since\s+(?:AI|LLM|model|language)"
    r")",
    re.IGNORECASE,
)

# Tautological endings: the why just names the conclusion without explaining mechanism.
_TAUTOLOGY_RE = re.compile(
    r"(?:"
    r"is a\s+(?:small |subtle |clear |strong |notable |definitive |tell-?tale )?"
    r"(?:sign|tell\b|indicator|marker|clue|hallmark|hint|evidence)"
    r"|indicates?\s+(?:human|AI|machine|non-AI|that this)"
    r"|suggests?\s+(?:human|AI|machine|non-AI|that this)"
    r"|points?\s+to\s+(?:human|AI|machine)"
    r"|is\s+(?:consistent|inconsistent)\s+with\s+(?:human|AI|machine)"
    r")",
    re.IGNORECASE,
)


def needs_rewrite(why: str) -> bool:
    """True when the why lacks a concrete mechanistic explanation."""
    why = (why or "").strip()
    if not why or len(why) < 20:
        return True
    # Short whys almost never contain enough mechanism.
    if len(why) < 60:
        return True
    # Check for concrete mechanistic content.
    has_mechanism = (
        _MODEL_BEHAVIOUR_RE.search(why) is not None
        or _HUMAN_HABIT_RE.search(why) is not None
        or _CAUSAL_CHAIN_RE.search(why) is not None
    )
    if has_mechanism:
        # Even with mechanism words, ends-in-tautology is still weak.
        if _TAUTOLOGY_RE.search(why):
            return True
        return False
    # No mechanism at all → flag regardless of tautology.
    return True


# ── Claude / Grok rewriting ───────────────────────────────────────────────────

_SYSTEM = """\
You are improving annotation quality for a dataset that labels human-written vs AI-generated text.

Each annotation marks a text span and has a "why" field explaining why the span indicates
human authorship.  A GOOD "why" must explain the MECHANISM — the concrete reason the feature
appears in human text but not (or less often) in AI-generated text.

GOOD "why" examples (have mechanism):
  "AI language models are trained to produce clean, polished prose; a raw & in a headline slot
   almost never appears in LLM output unless the model is explicitly prompted to include markup."
  "Humans naturally use sentence fragments and informal register in personal reviews; models tend
   to generate grammatically complete sentences even in casual contexts."

BAD "why" examples (tautological, no mechanism):
  "The raw ampersand is a small sign of human-published copy."   ← just names the conclusion
  "This kind of phrasing indicates human writing."               ← says what, not why
  "Consistent with human editorial style."                       ← no mechanism at all

Rules for your rewrite:
• Keep it 1–3 sentences, concise and specific to this span.
• Explain WHY the feature is rare/absent in AI output OR natural in human writing.
• Reference AI training, model generation behaviour, or documented human writing habits.
• Do NOT say "this is a sign/indicator/tell" — explain the mechanism instead.
• Preserve the factual observation from the original; only add/replace the mechanism part.
• Output ONLY the improved why text — no labels, no quotes, no preamble, no explanation.\
"""


async def rewrite_one(
    client: AsyncOpenAI,
    span_text: str,
    current_why: str,
    doc_snippet: str,
    ann_type: str,
    score: float,
    model: str,
    semaphore: asyncio.Semaphore,
) -> str:
    user = (
        f"Document (excerpt):\n{doc_snippet[:600]}\n\n"
        f"Annotated span: {span_text[:300]!r}\n"
        f"Annotation type: {ann_type}  score: {score:.2f}\n"
        f"Current weak why:\n{current_why}\n\n"
        f"Rewrite with a mechanistic explanation:"
    )
    async with semaphore:
        resp = await client.chat.completions.create(
            model=model,
            max_tokens=250,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": user},
            ],
        )
    return (resp.choices[0].message.content or "").strip()


# ── XML patching ──────────────────────────────────────────────────────────────

def patch_why(annotation_xml: str, old_why: str, new_why: str) -> str:
    """Replace the first occurrence of why="<old>" with why="<new>" in the annotation XML."""
    old_attr = f'why="{escape_attr_piece(old_why)}"'
    new_attr = f'why="{escape_attr_piece(new_why)}"'
    if old_attr not in annotation_xml:
        return annotation_xml  # don't corrupt row if something unexpected happened
    return annotation_xml.replace(old_attr, new_attr, 1)


# ── Dataset loading ───────────────────────────────────────────────────────────

def load_split(split: str) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("acmc/TELL", split=split)
    return [dict(row) for row in ds]


# ── Per-row processing ────────────────────────────────────────────────────────

@dataclass
class RewriteJob:
    row_idx: int
    tell_order: int
    span_text: str
    old_why: str
    doc_snippet: str
    ann_type: str
    score: float


@dataclass
class RowResult:
    row_idx: int
    n_nested: int
    n_rewritten: int
    rewrites: list[tuple[int, str, str]] = field(default_factory=list)  # (order, old, new)


async def process_split(
    rows: list[dict],
    client: AsyncOpenAI,
    model: str,
    semaphore: asyncio.Semaphore,
    rewrite_all: bool,
    limit: int | None,
    dry_run: bool,
) -> tuple[list[dict], list[RowResult]]:
    if limit is not None:
        rows = rows[:limit]

    # Collect rewrite jobs.
    jobs: list[RewriteJob] = []
    for row_idx, row in enumerate(rows):
        ann = row.get("annotation", "")
        if not ann:
            continue
        _inn, desc, _meta, ok, _end = root_splits(tx=ann)
        if not ok or not desc:
            continue
        doc_snippet = (row.get("text") or "")[:600]
        for order, tell in enumerate(desc):
            why = tell.get("explanation", "")
            if rewrite_all or needs_rewrite(why):
                jobs.append(RewriteJob(
                    row_idx=row_idx,
                    tell_order=order,
                    span_text=tell.get("span_text", ""),
                    old_why=why,
                    doc_snippet=doc_snippet,
                    ann_type=tell.get("type", "human"),
                    score=float(tell.get("score", 0.0)),
                ))

    annotated_rows = sum(1 for r in rows if r.get("annotation"))
    print(f"  {len(jobs)} tells flagged for rewrite  (out of {annotated_rows} annotated rows, "
          f"{sum(len((root_splits(r['annotation'])[1] or [])) for r in rows if r.get('annotation'))} total nested tells)")

    if dry_run:
        print("\n  [dry-run] sample of flagged whys (no API calls made):")
        for job in jobs[:10]:
            print(f"    row={job.row_idx} order={job.tell_order}")
            print(f"      span:    {job.span_text[:80]!r}")
            print(f"      why:     {job.old_why[:120]}")
        return list(rows), [RowResult(row_idx=i, n_nested=0, n_rewritten=0) for i in range(len(rows))]

    # Fan-out all API calls.
    async def run_job(job: RewriteJob) -> tuple[RewriteJob, str]:
        new_why = await rewrite_one(
            client=client,
            span_text=job.span_text,
            current_why=job.old_why,
            doc_snippet=job.doc_snippet,
            ann_type=job.ann_type,
            score=job.score,
            model=model,
            semaphore=semaphore,
        )
        return job, new_why

    tasks = [asyncio.create_task(run_job(j)) for j in jobs]

    done = 0
    results: list[tuple[RewriteJob, str]] = []
    t0 = time.monotonic()
    for coro in asyncio.as_completed(tasks):
        job, new_why = await coro
        results.append((job, new_why))
        done += 1
        if done % 100 == 0 or done == len(tasks):
            elapsed = time.monotonic() - t0
            rate = done / elapsed if elapsed > 0 else 0
            print(f"  {done}/{len(tasks)} rewrites done  ({rate:.1f}/s)")

    # Group by row, apply patches.
    from collections import defaultdict
    row_patches: dict[int, list[tuple[int, str, str]]] = defaultdict(list)
    for job, new_why in results:
        row_patches[job.row_idx].append((job.tell_order, job.old_why, new_why))

    improved_rows = list(rows)
    row_results: list[RowResult] = []

    for row_idx, row in enumerate(rows):
        ann = row.get("annotation", "")
        if not ann:
            row_results.append(RowResult(row_idx=row_idx, n_nested=0, n_rewritten=0))
            continue
        _inn, desc, _meta, ok, _end = root_splits(tx=ann)
        n_nested = len(desc) if ok and desc else 0
        patches = row_patches.get(row_idx, [])
        new_ann = ann
        for _order, old_why, new_why in patches:
            new_ann = patch_why(new_ann, old_why, new_why)
        if patches:
            improved_rows[row_idx] = {**row, "annotation": new_ann}
        row_results.append(RowResult(
            row_idx=row_idx,
            n_nested=n_nested,
            n_rewritten=len(patches),
            rewrites=patches,
        ))

    return improved_rows, row_results


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="both", choices=["train", "validation", "both"])
    parser.add_argument("--limit", type=int, default=None, help="Process only this many rows per split (for dry-runs)")
    parser.add_argument("--model", default="grok-3-fast", help="Model to use (XAI API)")
    parser.add_argument("--concurrency", type=int, default=40, help="Max parallel API calls")
    parser.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("data/tell_improved"))
    parser.add_argument("--rewrite-all", action="store_true", help="Rewrite every why, not just weak ones")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be rewritten, skip API calls")
    parser.add_argument("--push-to-hub", default=None, metavar="REPO_ID", help="Push improved dataset to HF repo")
    args = parser.parse_args()

    if not args.dry_run:
        args.out_dir.mkdir(parents=True, exist_ok=True)

    client = AsyncOpenAI(
        api_key=os.environ["XAI_API_KEY"],
        base_url="https://api.x.ai/v1",
    )
    semaphore = asyncio.Semaphore(args.concurrency)

    splits = ["train", "validation"] if args.split == "both" else [args.split]
    all_stats: dict[str, dict] = {}

    for split in splits:
        print(f"\n=== {split} ===")
        print("  loading...")
        rows = load_split(split)
        print(f"  {len(rows)} rows loaded")

        improved, row_results = asyncio.run(process_split(
            rows=rows,
            client=client,
            model=args.model,
            semaphore=semaphore,
            rewrite_all=args.rewrite_all,
            limit=args.limit,
            dry_run=args.dry_run,
        ))

        if args.dry_run:
            continue

        # Save improved split.
        out_path = args.out_dir / f"{split}.jsonl"
        with out_path.open("w", encoding="utf-8") as fh:
            for row in improved:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  saved {len(improved)} rows → {out_path}")

        # Save diff log.
        diff_path = args.out_dir / f"{split}_diffs.jsonl"
        n_changed = 0
        with diff_path.open("w", encoding="utf-8") as fh:
            for rr in row_results:
                if rr.rewrites:
                    fh.write(json.dumps({
                        "row_idx": rr.row_idx,
                        "n_nested": rr.n_nested,
                        "n_rewritten": rr.n_rewritten,
                        "rewrites": [
                            {"order": o, "old": old, "new": new}
                            for o, old, new in rr.rewrites
                        ],
                    }, ensure_ascii=False) + "\n")
                    n_changed += 1

        total_nested = sum(rr.n_nested for rr in row_results)
        total_rewritten = sum(rr.n_rewritten for rr in row_results)
        all_stats[split] = {
            "rows": len(rows),
            "rows_changed": n_changed,
            "total_nested_tells": total_nested,
            "total_rewritten": total_rewritten,
            "rewrite_rate": total_rewritten / total_nested if total_nested else 0.0,
        }
        print(f"  rewritten {total_rewritten}/{total_nested} tells "
              f"({100 * total_rewritten / max(1, total_nested):.1f}%) across {n_changed} rows")
        print(f"  diff log → {diff_path}")

    if not args.dry_run:
        stats_path = args.out_dir / "rewrite_stats.json"
        with stats_path.open("w", encoding="utf-8") as fh:
            json.dump(all_stats, fh, indent=2, ensure_ascii=False)
        print(f"\nstats → {stats_path}")

        if args.push_to_hub:
            print(f"\npushing to {args.push_to_hub}...")
            from datasets import DatasetDict, Dataset
            dataset_dict = {}
            for split in splits:
                split_rows = [
                    json.loads(l)
                    for l in (args.out_dir / f"{split}.jsonl").read_text().splitlines()
                    if l.strip()
                ]
                dataset_dict[split] = Dataset.from_list(split_rows)
            DatasetDict(dataset_dict).push_to_hub(args.push_to_hub)
            print("done.")


if __name__ == "__main__":
    main()
