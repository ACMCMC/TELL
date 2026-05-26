# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "datasets",
# ]
# ///
"""Audit SFT nested-why dropout: genericity rankings, common whys, before/after cases.

``genericity_bernoulli_p`` for a **nested** tell is ``p_drop`` from ``why_idf_dropout`` (same as
training): either span–why overlap excess or char n-gram IDF excess (see ``--genericity-mode``),
each capped and multiplied by the score keep factor.  Each nested tell is an independent
Bernoulli trial at that probability.  The **outer** root tell is never removed and never
participates in that sampling.

With ``span_why_overlap`` (default), buckets use a **short normalized-why prefix** (see
``--why-bucket-key-chars``), not full-text equality.  ``span_why_g`` uses ``log(1+max(|bucket|,min_n))``
times ``(1 - λ·max_jaccard)`` (``--span-why-jaccard-weight`` λ; singleton buckets still get the log
term).  JSONL field ``p_mass`` is genericity mass **before** the score factor (historically misnamed
``p_idf`` in code paths): in overlap mode it comes from **G excess**, not from ``mean_idf``.
``mean_idf`` is always the char n-gram rarity of the **why** text (informative, separate signal).
``genericity_excess`` = ``max(0, span_why_g - span_why_g_median)``.  So high ``mean_idf`` with
``p_mass`` = 0 is normal: the why looks rare, but **G** is still at or below the corpus median **G**.

  uv run python experiments/audit_why_idf_dropout.py --diagnose-only

Dropout hyperparameters default from ``rl_detector.config.CFG`` (``conf/config.yaml`` ``sft`` block),
same as training.  CLI flags override.  Optional ``--config path.yaml`` uses that file’s ``sft``
instead.  Only audit-only limits (fit row cap, num jsonl examples, output paths) live in
``_AUDIT_ONLY`` inside this script.  ``fit_limit`` None = full HF split (same scope as SFT fit).
Pass ``--fit-limit 8000`` for a faster approximate audit.

Writes:
  - summary JSON (corpus stats + most/least generic nested fragments only),
  - JSONL lines with full logical document, inner wired text before/after nested dropout, full
    annotated XML before/after, ``nested_spans_before`` / ``nested_spans_after`` (span text,
    tell scores, why, removal flag), ``outer_tell``, and per-nested dropout diagnostics,
  - optional ``.txt`` per case under ``--cases-dir`` (document head + nested table + full XML).
"""

from __future__ import annotations

import argparse
import html
import json
import random
import re
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf

from rl_detector.annotation_utils import get_outer_bracket_metadata
from rl_detector.config import CFG
from rl_detector.rewards import format_diagnostics
from rl_detector.sft.why_idf_dropout import (
    WhyCharNgramIdfScorer,
    per_nested_tell_drop_probs,
    rebuild_annotation_nested_dropout,
    sample_drop_nested_orders,
)
from rl_detector.tell_xml import root_splits


_REPO_ROOT = Path(__file__).resolve().parents[1]
# Report layout and how much data to scan; not used by the trained model.
_AUDIT_ONLY = {
    "split": "train",
    "fit_limit": None,
    "num_examples": 25,
    "why_truncate_extremes": 900,
    "num_high_genericity": 40,
    "num_low_genericity": 40,
    "cases_dir": Path("experiments/why_idf_dropout_cases"),
    "doc_head_chars": 2500,
    "out_summary": Path("experiments/why_idf_dropout_audit_summary.json"),
    "out_examples": Path("experiments/why_idf_dropout_audit_examples.jsonl"),
}

SEED = int(CFG.frozen.seed)


def _hf_repo_from_dataset_path(path: str) -> str:
    p = str(path).strip()
    if p.startswith("hf://"):
        return p.removeprefix("hf://").lstrip("/")
    return p


def _sft_for_audit(args: argparse.Namespace) -> DictConfig:
    if args.config is not None:
        cfg_path = Path(args.config)
        if not cfg_path.is_absolute():
            cfg_path = (_REPO_ROOT / cfg_path).resolve()
        if not cfg_path.is_file():
            raise SystemExit(f"--config not found: {cfg_path}")
        full = OmegaConf.load(cfg_path)
        OmegaConf.resolve(full)
        return full.sft
    return CFG.sft


def _merge_audit_defaults(args: argparse.Namespace) -> None:
    sft = _sft_for_audit(args=args)
    if args.dataset is None:
        args.dataset = _hf_repo_from_dataset_path(path=str(sft.dataset_path))
    if args.n_min is None:
        args.n_min = int(sft.why_idf_char_ngram_min)
    if args.n_max is None:
        args.n_max = int(sft.why_idf_char_ngram_max)
    if args.drop_strength is None:
        args.drop_strength = float(sft.why_idf_drop_strength)
    if args.p_drop_cap is None:
        args.p_drop_cap = float(sft.why_idf_p_drop_cap)
    if args.score_keep_weight is None:
        args.score_keep_weight = float(sft.why_idf_score_keep_weight)
    if args.genericity_mode is None:
        args.genericity_mode = str(sft.why_idf_genericity_mode)
    if args.why_bucket_chars is None:
        args.why_bucket_chars = int(sft.why_idf_why_bucket_max_chars)
    if args.why_bucket_key_chars is None:
        kb = getattr(sft, "why_idf_why_bucket_key_chars", None)
        if kb is not None:
            args.why_bucket_key_chars = int(kb)
    if args.span_why_jaccard_weight is None:
        args.span_why_jaccard_weight = float(sft.why_idf_span_why_jaccard_weight)
    if args.span_why_min_bucket_n_for_log is None:
        args.span_why_min_bucket_n_for_log = int(sft.why_idf_span_why_min_bucket_n_for_log)
    if args.span_why_overlap_p_mass is None:
        args.span_why_overlap_p_mass = str(sft.why_idf_span_why_overlap_p_mass)
    if args.span_why_excess_quantile is None:
        args.span_why_excess_quantile = float(sft.why_idf_span_why_excess_quantile)
    if args.spacy_pipe_batch_size is None:
        args.spacy_pipe_batch_size = int(sft.why_idf_spacy_pipe_batch_size)
    if args.spacy_exclude_stopwords is None:
        args.spacy_exclude_stopwords = 1 if bool(sft.why_idf_span_spacy_exclude_stopwords) else 0

    ad = _AUDIT_ONLY
    if args.split is None:
        args.split = ad["split"]
    if args.fit_limit is None:
        args.fit_limit = ad["fit_limit"]
    if args.num_examples is None:
        args.num_examples = ad["num_examples"]
    if args.why_key_len is None:
        args.why_key_len = int(args.why_bucket_chars)
    if args.why_truncate_extremes is None:
        args.why_truncate_extremes = ad["why_truncate_extremes"]
    if args.num_high_genericity is None:
        args.num_high_genericity = ad["num_high_genericity"]
    if args.num_low_genericity is None:
        args.num_low_genericity = ad["num_low_genericity"]
    if args.cases_dir is None:
        args.cases_dir = ad["cases_dir"]
    if args.doc_head_chars is None:
        args.doc_head_chars = ad["doc_head_chars"]
    if args.out_summary is None:
        args.out_summary = ad["out_summary"]
    if args.out_examples is None:
        args.out_examples = ad["out_examples"]


def _norm_why_key(text: str, max_len: int) -> str:
    t = html.unescape(text).strip().lower()
    t = re.sub(r"\s+", " ", t)
    if len(t) > max_len:
        t = t[:max_len] + "…"
    return t


def _load_tell_rows(repo_id: str, split: str, limit: int | None) -> list[dict]:
    ds = load_dataset(path=repo_id, split=split)
    n = len(ds)
    if limit is not None:
        n = min(int(limit), n)
    out: list[dict] = []
    for i in range(n):
        row = dict(ds[i])
        text = row.get("text") or ""
        raw_ann = row.get("annotation") or ""
        if not text or not raw_ann:
            continue
        label = row.get("label")
        if label is None:
            label = row.get("is_ai")
        if label is None:
            outer = get_outer_bracket_metadata(raw_ann)
            label = 1 if (outer and outer.get("type") == "AI") else 0
        else:
            label = int(label)
        out.append({"text": text, "annotation": raw_ann, "label": label})
    return out


def _outer_why(annotation_xml: str) -> str:
    _inn, _desc, meta, ok, _end = root_splits(tx=annotation_xml)
    if not ok or meta is None:
        return ""
    return str(meta.get("why", ""))


def _nested_tell_public_dict(leaf: dict) -> dict:
    return {
        "type": leaf.get("type"),
        "tell_score_raw": leaf.get("score"),
        "tell_score": _parse_tell_score_value(leaf=leaf),
        "span_text": str(leaf.get("span_text", "")),
        "why": str(leaf.get("explanation", "")),
    }


def _parse_tell_score_value(leaf: dict) -> float:
    raw = str(leaf.get("score", "0.5")).strip()
    try:
        v = float(raw)
    except ValueError:
        return 0.5
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def _nested_spans_snapshot(desc: list[dict], drop_orders: set[int]) -> list[dict]:
    out: list[dict] = []
    for o, leaf in enumerate(desc):
        out.append(
            {
                "nested_order": int(o),
                "span_text": str(leaf.get("span_text", "")),
                "tell_score": float(_parse_tell_score_value(leaf=leaf)),
                "tell_score_raw": leaf.get("score"),
                "why": str(leaf.get("explanation", "")),
                "type": leaf.get("type"),
                "removed_in_bernoulli_sample": int(o) in drop_orders,
            }
        )
    return out


def _nested_spans_snapshot_survivors(desc: list[dict]) -> list[dict]:
    out: list[dict] = []
    for o, leaf in enumerate(desc):
        out.append(
            {
                "nested_order": int(o),
                "span_text": str(leaf.get("span_text", "")),
                "tell_score": float(_parse_tell_score_value(leaf=leaf)),
                "tell_score_raw": leaf.get("score"),
                "why": str(leaf.get("explanation", "")),
                "type": leaf.get("type"),
            }
        )
    return out


def _outer_tell_audit(meta: dict, scorer: WhyCharNgramIdfScorer) -> dict:
    why = str(meta.get("why", ""))
    sc = _parse_tell_score_value(leaf=meta)
    return {
        "role": "outer_root",
        "nested_dropout_never_removes_this": True,
        "type": meta.get("type"),
        "why": why,
        "tell_score": sc,
        "why_mean_idf_under_same_idf_table": float(scorer.mean_idf(why_text=why)),
    }


def _collect_corpus_fragments(
    rows: list[dict],
    scorer: WhyCharNgramIdfScorer,
    drop_strength: float,
    p_drop_cap: float,
    score_keep_weight: float,
    why_truncate: int,
) -> list[dict]:
    out: list[dict] = []
    for row_idx, row in enumerate(rows):
        inn, desc, meta, ok, _end = root_splits(tx=row["annotation"])
        if not ok or meta is None or not desc:
            continue
        probs = per_nested_tell_drop_probs(
            scorer=scorer,
            row_index=int(row_idx),
            nested_desc=desc,
            drop_strength=drop_strength,
            p_drop_cap=p_drop_cap,
            score_keep_weight=score_keep_weight,
        )
        for p in probs:
            g = float(p["p_drop"])
            why_full = str(p.get("why", ""))
            out.append(
                {
                    "row_index_in_fit_slice": int(row_idx),
                    "label": int(row["label"]),
                    "nested_order": int(p["order"]),
                    "genericity_bernoulli_p": g,
                    "mean_idf": float(p["mean_idf"]),
                    "span_why_g": float(p["span_why_g"]),
                    "genericity_excess": float(p["genericity_excess"]),
                    "p_mass": float(p["p_idf"]),
                    "score": float(p["score"]),
                    "score_factor": float(p["score_factor"]),
                    "type": p.get("type"),
                    "span_preview": str(p.get("span_preview", "")),
                    "why": why_full[: int(why_truncate)],
                    "why_len": len(why_full),
                }
            )
    return out


def _write_case_txt(
    path: Path,
    row_idx: int,
    label: int,
    doc: str,
    ann_before: str,
    ann_after: str,
    doc_head_chars: int,
    nested_table_lines: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    head = doc[: int(doc_head_chars)]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"row_index_in_fit_slice={row_idx} label={label}\n")
        fh.write(f"document_len={len(doc)}\n")
        fh.write("\nOUTER_ROOT_TELL: never a candidate for nested dropout (XML always kept).\n")
        fh.write(f"\n=== DOCUMENT (first {doc_head_chars} chars) ===\n")
        fh.write(head)
        fh.write("\n\n=== NESTED TELLS (genericity = Bernoulli p; sampled remove) ===\n")
        fh.write("\n".join(nested_table_lines))
        fh.write("\n\n=== ANNOTATION XML BEFORE (full) ===\n")
        fh.write(ann_before)
        fh.write("\n\n=== ANNOTATION XML AFTER (full) ===\n")
        fh.write(ann_after)
        fh.write("\n")


def _run_diagnose(
    rows: list[dict],
    scorer: WhyCharNgramIdfScorer,
    drop_strength: float,
    p_drop_cap: float,
    score_keep_weight: float,
    rng_seed: int,
    keep_min: int = 1,
    keep_max: int = 5,
) -> None:
    import random

    mode = str(scorer.genericity_mode)
    ref = float(scorer.median_mean_idf)
    g_med = float(scorer.span_why_g_median)
    nested_p: list[float] = []
    nested_m: list[float] = []
    nested_g: list[float] = []
    rows_with_nested = 0
    rows_any_positive_p = 0
    rows_any_drop_sample = 0
    rng = random.Random(rng_seed)
    for row_idx, row in enumerate(rows):
        inn, desc, meta, ok, _end = root_splits(tx=row["annotation"])
        if not ok or meta is None or not desc:
            continue
        rows_with_nested += 1
        probs = per_nested_tell_drop_probs(
            scorer=scorer,
            row_index=int(row_idx),
            nested_desc=desc,
            drop_strength=drop_strength,
            p_drop_cap=p_drop_cap,
            score_keep_weight=score_keep_weight,
        )
        pos_p = False
        any_drop = False
        for p in probs:
            pd = float(p["p_drop"])
            nested_p.append(pd)
            nested_m.append(float(p["mean_idf"]))
            nested_g.append(float(p["span_why_g"]))
            if pd > 1e-15:
                pos_p = True
            if rng.random() < pd:
                any_drop = True
        if pos_p:
            rows_any_positive_p += 1
        if any_drop:
            rows_any_drop_sample += 1
    n = len(nested_p)
    z = sum(1 for x in nested_p if x <= 1e-15)
    below_ref = sum(1 for m in nested_m if m < ref - 1e-12)
    below_gmed = sum(1 for g in nested_g if g < g_med - 1e-12)
    print("=== why_idf nested dropout diagnose ===")
    print("genericity_mode", mode)
    print("span_lemmatizer", "en_blank_rule")
    print("fit_rows", len(rows), "rows_with_nested_tells", rows_with_nested)
    print("median_mean_idf_ref (outer+nested fragments at fit time)", ref)
    print("span_why_g_median", g_med)
    print("nested_tells_total", n)
    print("fraction_nested_tells_with_mean_idf_strictly_below_ref", below_ref / max(1, n))
    if mode == "span_why_overlap":
        print("fraction_nested_tells_with_span_why_g_strictly_below_median_G", below_gmed / max(1, n))
    print("fraction_nested_tells_with_p_drop==0 (numerical)", z / max(1, n))
    print("rows_with_at_least_one_nested_tell_having_p_drop>0", rows_any_positive_p, "/", rows_with_nested)
    print(
        "rows_with_at_least_one_Bernoulli_drop_in_one_pass (seed=%d):" % rng_seed,
        rows_any_drop_sample,
        "/",
        rows_with_nested,
    )
    if n:
        s = sorted(nested_p)
        print("p_drop min / median / max", s[0], s[len(s) // 2], s[-1])
        if mode == "span_why_overlap":
            sg = sorted(nested_g)
            print("span_why_g min / median / max", sg[0], sg[len(sg) // 2], sg[-1])

    # Per-document coverage stats using the same enforced clamping as training.
    rng2 = random.Random(rng_seed)
    per_doc_coverage: list[float] = []
    for row_idx, row in enumerate(rows):
        inn, desc, meta, ok, _end = root_splits(tx=row["annotation"])
        if not ok or meta is None or not desc:
            continue
        n_doc = len(desc)
        drop_set = sample_drop_nested_orders(
            rng=rng2,
            scorer=scorer,
            row_index=int(row_idx),
            nested_desc=desc,
            drop_strength=drop_strength,
            p_drop_cap=p_drop_cap,
            score_keep_weight=score_keep_weight,
            keep_min=int(keep_min),
            keep_max=int(keep_max),
        )
        n_kept = n_doc - len(drop_set)
        per_doc_coverage.append(float(n_kept))
    if per_doc_coverage:
        cov_s = sorted(per_doc_coverage)
        print("per_doc_n_kept min / median / mean / max: {:.1f} / {:.1f} / {:.1f} / {:.1f}".format(
            cov_s[0], cov_s[len(cov_s) // 2],
            sum(cov_s) / len(cov_s), cov_s[-1]
        ))
        in_range = sum(1 for c in per_doc_coverage if int(keep_min) <= int(c) <= int(keep_max))
        print(f"docs with n_kept in [{keep_min}, {keep_max}]: {in_range}/{len(per_doc_coverage)} ({100*in_range/len(per_doc_coverage):.1f}%)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="optional yaml with top-level sft:; defaults to rl_detector.config.CFG (conf/config.yaml)",
    )
    p.add_argument(
        "--diagnose-only",
        action="store_true",
        help="load data, fit scorer, print dropout statistics, then exit (no JSON/cases)",
    )
    p.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="HF dataset id; default sft.dataset_path from config (strip hf://)",
    )
    p.add_argument("--split", type=str, default=None, help="default _AUDIT_ONLY['split']")
    p.add_argument(
        "--fit-limit",
        type=int,
        default=None,
        help="max rows for fit; default _AUDIT_ONLY['fit_limit']",
    )
    p.add_argument(
        "--num-examples",
        type=int,
        default=None,
        help="jsonl + case files; default _AUDIT_ONLY['num_examples']",
    )
    p.add_argument("--n-min", type=int, default=None, help="default sft.why_idf_char_ngram_min")
    p.add_argument("--n-max", type=int, default=None, help="default sft.why_idf_char_ngram_max")
    p.add_argument("--drop-strength", type=float, default=None, help="default sft.why_idf_drop_strength")
    p.add_argument("--p-drop-cap", type=float, default=None, help="default sft.why_idf_p_drop_cap")
    p.add_argument("--score-keep-weight", type=float, default=None, help="default sft.why_idf_score_keep_weight")
    p.add_argument(
        "--genericity-mode",
        type=str,
        default=None,
        choices=("span_why_overlap", "char_ngram_idf"),
        help="default sft.why_idf_genericity_mode",
    )
    p.add_argument(
        "--why-bucket-chars",
        type=int,
        default=None,
        help="default sft.why_idf_why_bucket_max_chars",
    )
    p.add_argument(
        "--why-bucket-key-chars",
        type=int,
        default=None,
        help="default sft.why_idf_why_bucket_key_chars if set, else same as why-bucket-chars",
    )
    p.add_argument(
        "--span-why-jaccard-weight",
        type=float,
        default=None,
        help="default sft.why_idf_span_why_jaccard_weight",
    )
    p.add_argument(
        "--span-why-min-bucket-n-for-log",
        type=int,
        default=None,
        help="default sft.why_idf_span_why_min_bucket_n_for_log",
    )
    p.add_argument(
        "--span-why-overlap-p-mass",
        type=str,
        default=None,
        choices=("rank", "scaled", "subtract_median", "subtract_quantile"),
        help="default sft.why_idf_span_why_overlap_p_mass",
    )
    p.add_argument(
        "--span-why-excess-quantile",
        type=float,
        default=None,
        help="default sft.why_idf_span_why_excess_quantile",
    )
    p.add_argument("--spacy-pipe-batch-size", type=int, default=None, help="default sft.why_idf_spacy_pipe_batch_size")
    p.add_argument(
        "--spacy-exclude-stopwords",
        type=int,
        choices=[0, 1],
        default=None,
        help="default 1 if sft.why_idf_span_spacy_exclude_stopwords else 0",
    )
    p.add_argument(
        "--why-key-len",
        type=int,
        default=None,
        help="summary key truncate; default same as resolved why-bucket-chars",
    )
    p.add_argument(
        "--why-truncate-extremes",
        type=int,
        default=None,
        help="default _AUDIT_ONLY['why_truncate_extremes']",
    )
    p.add_argument(
        "--num-high-genericity",
        type=int,
        default=None,
        help="default _AUDIT_ONLY['num_high_genericity']",
    )
    p.add_argument(
        "--num-low-genericity",
        type=int,
        default=None,
        help="default _AUDIT_ONLY['num_low_genericity']",
    )
    p.add_argument(
        "--cases-dir",
        type=Path,
        default=None,
        help="default _AUDIT_ONLY['cases_dir']",
    )
    p.add_argument(
        "--doc-head-chars",
        type=int,
        default=None,
        help="default _AUDIT_ONLY['doc_head_chars']",
    )
    p.add_argument("--out-summary", type=Path, default=None, help="default _AUDIT_ONLY['out_summary']")
    p.add_argument(
        "--out-examples",
        type=Path,
        default=None,
        help="default _AUDIT_ONLY['out_examples']",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _merge_audit_defaults(args=args)
    rows = _load_tell_rows(repo_id=args.dataset, split=args.split, limit=args.fit_limit)
    if not rows:
        raise SystemExit("no rows loaded")

    ds = float(args.drop_strength)
    pc = float(args.p_drop_cap)
    sw = float(args.score_keep_weight)
    _sft = _sft_for_audit(args=args)
    keep_min = int(getattr(_sft, "why_idf_keep_min", 1))
    keep_max = int(getattr(_sft, "why_idf_keep_max", 5))
    why_key_max = int(args.why_bucket_chars)
    why_bucket_key = int(args.why_bucket_key_chars) if args.why_bucket_key_chars is not None else why_key_max

    scorer = WhyCharNgramIdfScorer.from_train_rows(
        rows=rows,
        n_min=int(args.n_min),
        n_max=int(args.n_max),
        genericity_mode=str(args.genericity_mode),
        why_key_max_chars=why_key_max,
        why_bucket_key_chars=why_bucket_key,
        span_why_jaccard_weight=float(args.span_why_jaccard_weight),
        span_why_min_bucket_n_for_log=int(args.span_why_min_bucket_n_for_log),
        span_why_overlap_p_mass_style=str(args.span_why_overlap_p_mass),
        span_why_excess_quantile=float(args.span_why_excess_quantile),
        spacy_pipe_batch_size=int(args.spacy_pipe_batch_size),
        spacy_exclude_stopwords=int(args.spacy_exclude_stopwords) == 1,
    )

    if bool(args.diagnose_only):
        _run_diagnose(
            rows=rows,
            scorer=scorer,
            drop_strength=ds,
            p_drop_cap=pc,
            score_keep_weight=sw,
            rng_seed=SEED,
            keep_min=keep_min,
            keep_max=keep_max,
        )
        return

    nested_why_ctr: Counter[str] = Counter()
    outer_why_ctr: Counter[str] = Counter()
    key_len = int(args.why_key_len)
    for row in rows:
        ann = row["annotation"]
        ow = _outer_why(annotation_xml=ann)
        if ow.strip():
            outer_why_ctr[_norm_why_key(text=ow, max_len=key_len)] += 1
        inn, desc, meta, ok, _end = root_splits(tx=ann)
        if not ok or not desc:
            continue
        for leaf in desc:
            w = str(leaf.get("explanation", ""))
            if w.strip():
                nested_why_ctr[_norm_why_key(text=w, max_len=key_len)] += 1

    top_grams = [
        {"gram": g, "df_rows": int(d)}
        for g, d in scorer.top_ngrams_by_row_df(top_k=48)
    ]

    frags = _collect_corpus_fragments(
        rows=rows,
        scorer=scorer,
        drop_strength=ds,
        p_drop_cap=pc,
        score_keep_weight=sw,
        why_truncate=int(args.why_truncate_extremes),
    )
    fr_sorted_high = sorted(
        frags,
        key=lambda r: (
            -float(r["genericity_bernoulli_p"]),
            -float(r["span_why_g"]),
            -float(r["mean_idf"]),
            int(r["row_index_in_fit_slice"]),
            int(r["nested_order"]),
        ),
    )
    fr_sorted_low = sorted(
        frags,
        key=lambda r: (
            float(r["genericity_bernoulli_p"]),
            float(r["span_why_g"]),
            -float(r["mean_idf"]),
            int(r["row_index_in_fit_slice"]),
            int(r["nested_order"]),
        ),
    )
    n_high = int(args.num_high_genericity)
    n_low = int(args.num_low_genericity)
    most_generic = fr_sorted_high[:n_high]
    least_generic = fr_sorted_low[:n_low]

    mode = str(args.genericity_mode)
    if mode == "span_why_overlap":
        pm = str(scorer.span_why_overlap_p_mass_style)
        gen_def = (
            "Nested tells only: bucket by short normalized why prefix; "
            "G = log(1+max(|bucket|,min_n)) * (1 - λ * max Jaccard of spaCy rule-lemma span sets). "
            f"overlap_p_mass={pm}: rank → p_mass=min(cap, drop_strength*(1-rank(mean_idf))) — "
            "generic wording (low IDF) gets high dropout pressure, specific wording (high IDF) gets low; "
            "drop_strength≈0.45 targets ~22% median coverage; "
            "scaled → min(cap, strength * G/median(G)); "
            "subtract_* → excess above median or quantile ref. "
            "p_drop = p_mass * max(0,1-score_keep_weight*tell_score). Outer root never removed."
        )
    else:
        gen_def = (
            "Nested tells only: char n-gram mean IDF on why text; p_mass from excess below corpus median "
            "(capped), times score keep factor as above. Outer root never removed."
        )

    summary = {
        "dataset": args.dataset,
        "split": args.split,
        "fit_n_rows": len(rows),
        "genericity_mode": mode,
        "why_bucket_max_chars": why_key_max,
        "why_bucket_key_chars": why_bucket_key,
        "span_why_jaccard_weight": float(args.span_why_jaccard_weight),
        "span_why_min_bucket_n_for_log": int(args.span_why_min_bucket_n_for_log),
        "span_why_overlap_p_mass": str(args.span_why_overlap_p_mass),
        "span_why_excess_quantile": float(args.span_why_excess_quantile),
        "span_lemmatizer": "en_blank_rule",
        "spacy_pipe_batch_size": int(args.spacy_pipe_batch_size),
        "span_spacy_exclude_stopwords": int(args.spacy_exclude_stopwords) == 1,
        "median_mean_idf": scorer.median_mean_idf,
        "span_why_g_median": scorer.span_why_g_median,
        "ngram_range": [int(args.n_min), int(args.n_max)],
        "drop_strength": ds,
        "p_drop_cap": pc,
        "score_keep_weight": sw,
        "genericity_definition": gen_def,
        "nested_fragment_count_scored": len(frags),
        "most_generic_nested_fragments": most_generic,
        "least_generic_nested_fragments": least_generic,
        "top_char_ngrams_by_row_df": top_grams,
        "outer_why_most_common": outer_why_ctr.most_common(40),
        "nested_why_most_common": nested_why_ctr.most_common(60),
        "cases_dir": str(args.cases_dir),
    }

    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_examples.parent.mkdir(parents=True, exist_ok=True)

    with open(args.out_summary, "w", encoding="utf-8") as fh:
        json.dump(obj=summary, fp=fh, ensure_ascii=False, indent=2)

    order = list(range(len(rows)))
    random.Random(SEED).shuffle(order)
    want = int(args.num_examples)

    with open(args.out_examples, "w", encoding="utf-8") as exf:
        written = 0
        for row_idx in order:
            if written >= want:
                break
            row = rows[row_idx]
            doc = row["text"]
            ann = row["annotation"]
            inn, desc, meta, ok, _end = root_splits(tx=ann)
            if not ok or meta is None:
                continue
            rng = random.Random(SEED + 17_000 + written)
            probs = per_nested_tell_drop_probs(
                scorer=scorer,
                row_index=int(row_idx),
                nested_desc=desc,
                drop_strength=ds,
                p_drop_cap=pc,
                score_keep_weight=sw,
            )
            drop_orders = sample_drop_nested_orders(
                rng=rng,
                scorer=scorer,
                row_index=int(row_idx),
                nested_desc=desc,
                drop_strength=ds,
                p_drop_cap=pc,
                score_keep_weight=sw,
                keep_min=keep_min,
                keep_max=keep_max,
            )
            after_ann = ann
            if drop_orders:
                after_ann = rebuild_annotation_nested_dropout(
                    logical_document=doc,
                    annotation_xml=ann,
                    drop_nested_orders=drop_orders,
                )
            diag_after = format_diagnostics(output=after_ann, document=doc)
            if drop_orders:
                assert diag_after.get("ok"), diag_after

            inn_a, desc_a, meta_a, ok_a, _e2 = root_splits(tx=after_ann)
            nested_after_surviving = [_nested_tell_public_dict(leaf=x) for x in (desc_a or [])]

            assert len(probs) == len(desc), (len(probs), len(desc))

            nested_tells: list[dict] = []
            table_lines: list[str] = []
            for p in probs:
                o = int(p["order"])
                gen = float(p["p_drop"])
                rem = o in drop_orders
                leaf = desc[o]
                nt = {
                    "role": "nested",
                    "order": o,
                    "type": p.get("type"),
                    "tell_score": float(p["score"]),
                    "tell_score_raw": leaf.get("score"),
                    "span_text": str(leaf.get("span_text", "")),
                    "why": str(leaf.get("explanation", "")),
                    "mean_idf": float(p["mean_idf"]),
                    "span_why_g": float(p["span_why_g"]),
                    "genericity_excess": float(p["genericity_excess"]),
                    "p_mass": float(p["p_idf"]),
                    "score_factor_for_dropout": float(p["score_factor"]),
                    "genericity_bernoulli_p": gen,
                    "bernoulli_sampled_remove_nested_span": rem,
                }
                nested_tells.append(nt)
                sp_short = str(leaf.get("span_text", ""))[:120]
                why_short = str(leaf.get("explanation", ""))[: int(args.why_key_len)]
                table_lines.append(
                    f"order={o} bernoulli_remove={rem} genericity_p={gen:.4f} span_why_g={float(p['span_why_g']):.4f} "
                    f"mean_idf={float(p['mean_idf']):.4f} p_mass={float(p['p_idf']):.4f} tell_score={float(p['score']):.3f} "
                    f"score_factor={float(p['score_factor']):.4f} type={p.get('type')}\n  span: {sp_short}\n  why: {why_short}"
                )

            case_name = f"case_{written:02d}_row{row_idx}.txt"
            case_path = args.cases_dir / case_name
            _write_case_txt(
                path=case_path,
                row_idx=int(row_idx),
                label=int(row["label"]),
                doc=doc,
                ann_before=ann,
                ann_after=after_ann,
                doc_head_chars=int(args.doc_head_chars),
                nested_table_lines=table_lines,
            )

            rec = {
                "fit_genericity_calibration": {
                    "genericity_mode": str(scorer.genericity_mode),
                    "median_mean_idf": float(scorer.median_mean_idf),
                    "span_why_g_median": float(scorer.span_why_g_median),
                    "span_why_g_excess_ref": float(scorer.span_why_g_excess_ref),
                    "span_why_overlap_p_mass": str(scorer.span_why_overlap_p_mass_style),
                    "span_why_excess_quantile": float(scorer.span_why_excess_quantile),
                    "why_bucket_key_chars": int(scorer.why_bucket_key_chars),
                    "span_why_jaccard_weight": float(scorer.span_why_jaccard_weight),
                    "span_why_min_bucket_n_for_log": int(scorer.span_why_min_bucket_n_for_log),
                    "p_mass_meaning": (
                        "span_why_overlap rank: min(cap, drop_strength * bucket_size/total_nested_tells) — "
                        "fraction of corpus nested tells sharing this why prefix, scaled by drop_strength. "
                        "char_ngram_idf uses mean_idf excess directly."
                    ),
                },
                "sampling": {
                    "mechanism": "independent_Bernoulli_per_nested_tell_at_genericity_p",
                    "outer_root_never_removed": True,
                    "bernoulli_rng_seed": SEED + 17_000 + written,
                },
                "case_file": str(case_path),
                "row_index_in_fit_slice": int(row_idx),
                "label": row["label"],
                "logical_document": doc,
                "logical_document_full": doc,
                "inner_wired_xml_before_nested_dropout": str(inn),
                "inner_wired_xml_after_nested_dropout": str(inn_a) if ok_a else "",
                "nested_spans_before": _nested_spans_snapshot(desc=desc, drop_orders=drop_orders),
                "nested_spans_after": _nested_spans_snapshot_survivors(desc=desc_a or []),
                "annotated_document_before_full": ann,
                "annotated_document_after_independent_bernoulli_sample_full": after_ann,
                "outer_tell": _outer_tell_audit(meta=meta, scorer=scorer),
                "nested_tells": nested_tells,
                "nested_tells_surviving_after_sample": nested_after_surviving,
                "nested_count_before": len(desc),
                "nested_count_after": len(desc_a or []),
                "dropped_nested_orders": sorted(int(x) for x in drop_orders),
                "n_kept": len(desc) - len(drop_orders),
                "keep_min": int(keep_min),
                "keep_max": int(keep_max),
                "format_ok_after": bool(diag_after.get("ok")),
            }
            exf.write(json.dumps(obj=rec, ensure_ascii=False) + "\n")
            written += 1

    print("wrote", args.out_summary)
    print("wrote", args.out_examples)
    print("wrote", written, "case files under", args.cases_dir)
    if written < want:
        print(
            f"warning: only {written}/{want} examples written (exhausted {len(rows)} shuffled rows; "
            "increase --fit-limit or fix rows that fail root_splits)",
            flush=True,
        )


if __name__ == "__main__":
    main()
