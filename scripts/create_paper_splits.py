"""Create deterministic paper splits from the unified TELL dataset.

The split policy is intentionally explicit because these files are paper
artifacts. It produces exact requested sizes while prioritizing a balanced,
clean main-eval set and preserving metadata for subgroup reporting.

Split pool policy:
  For each source dataset, rows are routed to the eval or train pool:
  - If the source has native test/val splits: test/val → eval pool, train → train pool.
  - If the source has only a train split (RAID, OpenLLMText, ghostbuster_essay,
    DAIGTv2, acmc/cheat): 20% hash holdout → eval pool, 80% → train pool.
    This ensures all datasets are represented in the benchmark eval sets.
  See get_split_pool() for the full mapping.

Eval allocation: uniform (equal) weights across eligible (dataset_id, domain) strata.
  This is the Neyman-optimal allocation when within-stratum variance is equal, which
  holds here because every stratum is balanced (50% AI / 50% human) by construction.
  Equal allocation maximises coverage across all domain×dataset combinations.

Train allocation: √stratum_size weights (survey-statistics compromise between
  proportional and equal), which increases training diversity without fully
  over-weighting the largest strata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


POLICY_VERSION = "unified_v3_paper_splits_v5"
DEFAULT_SEED = "20260428"

QUOTAS: dict[str, dict[int, int]] = {
    "main_eval": {0: 2_500, 1: 2_500},    # 5,000 benchmark test (balanced)
    "validation": {0: 2_500, 1: 2_500},   # 5,000 validation (balanced)
    "train": {0: 100_000, 1: 100_000},    # 200,000 train (balanced)
}

SPLIT_ORDER = ("main_eval", "validation", "train")

DETECTOR_COLUMNS = (
    "unified_id",
    "text",
    "label",
    "dataset_id",
    "domain",
    "generator_model",
    "attack",
    "lang",
    "source_split",
    "source",
    "source_detail",
    "is_default_training_candidate",
    "is_adversarial",
)
METADATA_COLUMNS = (
    "unified_id",
    "label",
    "dataset_id",
    "domain",
    "source_split",
    "source",
    "source_detail",
    "generator_model",
    "attack",
    "lang",
    "is_default_training_candidate",
    "is_adversarial",
)

# Datasets excluded entirely from all splits (see docs/UNIFIED_DATASET_BUILD.md)
EXCLUDED_DATASET_IDS: frozenset[str] = frozenset({
    "yaful/DeepfakeTextDetect",           # MAGE: removed in v3 refresh
    "zenodo_14962653_pan_voight_kampff",  # removed in v3 refresh
})

# Per-split stratification constants (see docs/UNIFIED_DATASET_BUILD.md)
# All three splits use the same balanced √stratum_size allocation policy;
# only the scale parameters differ.
_EVAL_MIN_SIDE = 100      # min available per label to include a stratum in eval splits
_EVAL_MIN_PER_SIDE = 50   # guaranteed minimum per label per stratum (eval splits)
_EVAL_MAX_PER_SIDE = 250  # cap per label per stratum (eval splits)
_EVAL_TARGET = 5_000      # total per eval split (2500 AI + 2500 human)

_TRAIN_MIN_SIDE = 50      # min available per label to include a stratum in train
_TRAIN_MIN_PER_SIDE = 25  # guaranteed minimum per label per stratum (train)
_TRAIN_MAX_PER_SIDE = 20_000  # cap per label per stratum (train)
_TRAIN_TARGET = 200_000   # total train (100k AI + 100k human)


_HOLDOUT_FRAC_NUM = 1   # numerator: 1-in-5 rows go to eval holdout pool
_HOLDOUT_FRAC_DEN = 5   # denominator: 20% holdout for train-only datasets

def _is_holdout(unified_id: str) -> bool:
    """Deterministic 20% holdout: row goes to eval pool when hash % DEN < NUM."""
    h = int(stable_hash(unified_id)[:8], 16)
    return (h % _HOLDOUT_FRAC_DEN) < _HOLDOUT_FRAC_NUM


def get_split_pool(dataset_id: str, source_split: str, unified_id: str = "") -> str:
    """Map a source row to 'train', 'eval', or 'excluded'.

    Policy:
    - Datasets with native eval/test splits: respect the original designation
      (test/val → eval pool, train → train pool).
    - Datasets with no native eval split (train-only sources): use a
      deterministic 20% hash holdout so every dataset is represented in the
      benchmark evaluation sets. The held-out rows are not used for training.
    - RAID: its 'extra' split is entirely non-English/code and passes no
      is_default filter, so RAID is also treated as a train-only source with
      hash holdout (both 'extra' and 'train' source_splits).
    """
    if dataset_id in EXCLUDED_DATASET_IDS:
        return "excluded"

    # Datasets with native eval splits — respect the original designation
    if dataset_id == "Ateeqq/AI-and-Human-Generated-Text":
        return "eval" if source_split == "test" else "train"
    if dataset_id == "Jinyan1/COLING_2025_MGT_en":
        return "eval" if source_split == "dev" else "train"
    if dataset_id == "SJTU-CL/ArguGPT":
        # ArguGPT has no human examples in any split — contrib is zero regardless
        return "eval" if source_split in {"validation", "test"} else "train"
    if dataset_id == "pangram/editlens_iclr":
        return "eval" if source_split != "train" else "train"
    if dataset_id == "ryuryukke/OUTFOX":
        return "train" if source_split.startswith("train_") else "eval"
    if dataset_id == "symanto/autextification2023":
        return "eval" if source_split.startswith("test_") else "train"

    # Train-only sources (no usable native eval split): deterministic 20% holdout
    # Includes: liamdugan/raid (extra split is all non-English/code),
    #           TheItCrOw/OpenLLMText, acmc/ghostbuster_essay, acmc/cheat, DAIGTv2
    return "eval" if _is_holdout(unified_id) else "train"


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize_text_for_hash(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def row_sort_key(row: dict[str, Any], seed: str) -> str:
    key = "|".join(
        [
            seed,
            str(row["label"]),
            str(row.get("domain") or ""),
            str(row.get("dataset_id") or ""),
            str(row.get("source_split") or ""),
            str(row.get("attack") or ""),
            str(row["unified_id"]),
        ]
    )
    return stable_hash(key)


def strata_key(row: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(row.get("domain") or "unknown"),
        str(row.get("dataset_id") or "unknown"),
        str(row.get("source_split") or "unknown"),
        str(row.get("attack") or "none"),
        str(row.get("generator_model") or "unknown"),
    )


def _enrich_language(row: dict[str, Any]) -> str:
    """Infer normalized language when lang is absent from the parquet.

    The unified-v3 parquet predates proper lang backfill, so many rows have an
    empty lang column despite being English. Re-derive from known dataset defaults
    and RAID's domain-encoded language values.
    """
    lang = str(row.get("lang") or "").strip()
    if lang:
        return lang
    dataset_id = str(row.get("dataset_id") or "")
    if dataset_id == "liamdugan/raid":
        detail = str(row.get("source_detail") or "").strip().lower()
        if detail == "german":
            return "de"
        if detail == "czech":
            return "cs"
    # symanto/autextification2023 encodes language in source_split (e.g. train_en, test_es)
    if dataset_id == "symanto/autextification2023":
        source_split = str(row.get("source_split") or "")
        if source_split.endswith("_es"):
            return "es"
        return "en"
    # All other datasets in this corpus are English
    return "en"


def _enrich_generator_model(row: dict[str, Any]) -> str:
    """Fill generator_model from known dataset provenance when the source had no model field.

    Rule: human rows are always "human"; AI rows default to the documented generator
    for datasets that have no per-row model metadata.
    """
    val = str(row.get("generator_model") or "").strip()
    if val:
        return val
    label = int(row.get("label", 0))
    if label == 0:
        return "human"
    dataset_id = str(row.get("dataset_id") or "")
    # Ateeqq and DAIGTv2 are documented ChatGPT datasets with no per-row model field
    if dataset_id in ("Ateeqq/AI-and-Human-Generated-Text", "DAIGTv2"):
        return "chatgpt"
    return ""


def detector_row(row: dict[str, Any], split: str, policy_version: str) -> dict[str, Any]:
    return {
        "id": row["unified_id"],
        "text": row["text"],
        "label": int(row["label"]),
        "split": split,
        "dataset_id": row.get("dataset_id"),
        "domain": row.get("domain"),
        "generator_model": _enrich_generator_model(row),
        "attack": str(row.get("attack") or "none"),
        "language": _enrich_language(row),
        "source_split": row.get("source_split"),
        "source": row.get("source"),
        "source_detail": row.get("source_detail"),
        "is_default_training_candidate": bool(row.get("is_default_training_candidate")),
        "is_adversarial": bool(row.get("is_adversarial")),
        "split_policy": policy_version,
    }


def _eval_stratum_key(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row.get("dataset_id") or "unknown"), str(row.get("domain") or "unknown"))


def allocate_balanced(
    rows: list[dict[str, Any]],
    seed: str,
    target: int,
    min_side: int,
    min_per_side: int,
    max_per_side: int,
    weight_fn: str = "equal",
    split_name: str = "split",
) -> list[dict[str, Any]]:
    """Allocate rows with per-stratum balance (equal AI and human per stratum).

    Used for all three splits: main_eval, validation, and train.

    Steps:
      1. Exclude strata where either label has fewer than min_side examples available.
      2. Guarantee min_per_side examples per label per eligible stratum (floor).
      3. Distribute remaining budget according to weight_fn:
           "equal"  — uniform weights (Neyman-optimal when within-stratum variance is
                      equal across strata; correct for balanced binary classification).
           "sqrt"   — weights ∝ √available (survey-statistics compromise between
                      proportional and equal; preferred for training splits).
         Budget is capped at max_per_side per label per stratum.
      4. Collect exactly target//2 per label (total = target).
    """
    strata: dict[tuple[str, str], dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        strata[_eval_stratum_key(row)][int(row["label"])].append(row)

    for key in strata:
        for label in (0, 1):
            strata[key][label].sort(key=lambda r: row_sort_key(r, seed))

    # Step 1: exclude strata where either label side has fewer than min_side available
    eligible = {
        key: buckets
        for key, buckets in strata.items()
        if len(buckets[0]) >= min_side and len(buckets[1]) >= min_side
    }
    if not eligible:
        raise RuntimeError(
            f"No eligible strata for {split_name}: all strata have < {min_side} examples per label."
        )

    target_per_label = target // 2

    # Step 2: guarantee minimum per label per stratum
    alloc: dict[tuple[str, str], int] = {key: min_per_side for key in eligible}
    fill_budget = target_per_label - sum(alloc.values())

    if fill_budget < 0:
        # More eligible strata than per-label budget: skip minimum guarantee, √-allocate from scratch
        alloc = {key: 0 for key in eligible}
        fill_budget = target_per_label

    # Step 3: distribute remaining budget, capped at max_per_side per label.
    if fill_budget > 0:
        if weight_fn == "equal":
            weights = {key: 1.0 for key in eligible}
        elif weight_fn == "sqrt":
            weights = {
                key: math.sqrt(min(len(strata[key][0]), len(strata[key][1])))
                for key in eligible
            }
        else:
            raise ValueError(f"Unknown weight_fn: {weight_fn!r}. Use 'equal' or 'sqrt'.")
        total_w = sum(weights.values()) or 1.0
        raw_fill: dict[tuple[str, str], float] = {
            key: fill_budget * weights[key] / total_w for key in eligible
        }
        for key in eligible:
            avail = min(len(strata[key][0]), len(strata[key][1]))
            cap = min(max_per_side, avail) - alloc[key]
            alloc[key] += min(int(math.floor(raw_fill[key])), max(0, cap))

        # Redistribute remaining budget (fractional parts + excess lost to cap)
        # via multi-pass round-robin so capped strata don't starve uncapped ones.
        remaining = target_per_label - sum(alloc.values())
        frac_order = sorted(
            eligible.keys(),
            key=lambda k: (raw_fill[k] - math.floor(raw_fill[k]), stable_hash(seed + repr(k))),
            reverse=True,
        )
        while remaining > 0:
            progressed = False
            for key in frac_order:
                if remaining <= 0:
                    break
                avail = min(len(strata[key][0]), len(strata[key][1]))
                if alloc[key] < max_per_side and alloc[key] < avail:
                    alloc[key] += 1
                    remaining -= 1
                    progressed = True
            if not progressed:
                raise RuntimeError(
                    f"Could not allocate {target_per_label} per label for {split_name}; "
                    f"{remaining} unallocated (all eligible strata capped or exhausted). "
                    f"Reduce max_per_side or target."
                )

    # Step 4: collect equal AI + human per stratum
    selected: list[dict[str, Any]] = []
    for key, n in alloc.items():
        for label in (0, 1):
            selected.extend(strata[key][label][:n])
    selected.sort(key=lambda r: row_sort_key(r, seed + "|final"))
    return selected


def load_metadata(parquet_path: Path) -> list[dict[str, Any]]:
    table = pq.read_table(parquet_path, columns=list(METADATA_COLUMNS))
    return table.to_pylist()


def assign_splits(rows: list[dict[str, Any]], seed: str) -> dict[str, str]:
    assignments: dict[str, str] = {}

    # Partition all rows by split pool based on source dataset designation
    eval_rows: list[dict[str, Any]] = []
    train_rows: list[dict[str, Any]] = []
    for row in rows:
        pool = get_split_pool(
            str(row.get("dataset_id") or ""),
            str(row.get("source_split") or ""),
            str(row.get("unified_id") or ""),
        )
        if pool == "eval":
            eval_rows.append(row)
        elif pool == "train":
            train_rows.append(row)
        # pool == "excluded": skip

    # Apply is_default_training_candidate filter to each pool
    eval_clean = [row for row in eval_rows if bool(row.get("is_default_training_candidate"))]
    train_clean = [row for row in train_rows if bool(row.get("is_default_training_candidate"))]

    # main_eval: balanced √stratum_size allocation from eval pool
    main_eval_selected = allocate_balanced(
        eval_clean, f"{seed}|main_eval",
        target=_EVAL_TARGET,
        min_side=_EVAL_MIN_SIDE, min_per_side=_EVAL_MIN_PER_SIDE, max_per_side=_EVAL_MAX_PER_SIDE,
        weight_fn="equal",
        split_name="main_eval",
    )
    assignments.update({row["unified_id"]: "main_eval" for row in main_eval_selected})

    # validation: same balanced policy from remaining eval pool rows
    remaining_eval = [row for row in eval_clean if row["unified_id"] not in assignments]
    validation_selected = allocate_balanced(
        remaining_eval, f"{seed}|validation",
        target=_EVAL_TARGET,
        min_side=_EVAL_MIN_SIDE, min_per_side=_EVAL_MIN_PER_SIDE, max_per_side=_EVAL_MAX_PER_SIDE,
        weight_fn="equal",
        split_name="validation",
    )
    assignments.update({row["unified_id"]: "validation" for row in validation_selected})

    # train: balanced √stratum_size allocation from train pool
    remaining_train = [row for row in train_clean if row["unified_id"] not in assignments]
    train_selected = allocate_balanced(
        remaining_train, f"{seed}|train",
        target=_TRAIN_TARGET,
        min_side=_TRAIN_MIN_SIDE, min_per_side=_TRAIN_MIN_PER_SIDE, max_per_side=_TRAIN_MAX_PER_SIDE,
        weight_fn="sqrt",
        split_name="train",
    )
    assignments.update({row["unified_id"]: "train" for row in train_selected})

    return assignments


def summarize(rows: list[dict[str, Any]], assignments: dict[str, str]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "policy_version": POLICY_VERSION,
        "target_quotas": QUOTAS,
        "total_rows": len(rows),
        "assigned_rows": len(assignments),
        "splits": {},
        "text_hash_overlap_counts": {},
    }
    counts: dict[str, Counter] = defaultdict(Counter)
    domains: dict[str, Counter] = defaultdict(Counter)
    datasets: dict[str, Counter] = defaultdict(Counter)
    attacks: dict[str, Counter] = defaultdict(Counter)

    for row in rows:
        split = assignments.get(row["unified_id"], "remaining")
        label = int(row["label"])
        counts[split][f"label_{label}"] += 1
        counts[split]["total"] += 1
        domains[split][f"{row.get('domain')}::label_{label}"] += 1
        datasets[split][f"{row.get('dataset_id')}::label_{label}"] += 1
        attacks[split][f"{row.get('attack') or 'none'}::label_{label}"] += 1

    for split in (*SPLIT_ORDER, "remaining"):
        summary["splits"][split] = {
            "counts": dict(counts[split]),
            "domains": dict(domains[split]),
            "datasets": dict(datasets[split]),
            "attacks": dict(attacks[split]),
        }
    return summary


def write_splits(
    parquet_path: Path,
    output_dir: Path,
    assignments: dict[str, str],
    policy_version: str,
    main_eval_shards: int,
) -> None:
    split_dir = output_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    handles = {
        split: (split_dir / f"{split}.jsonl").open("w", encoding="utf-8")
        for split in (*SPLIT_ORDER, "remaining")
    }
    shard_dir = split_dir / "main_eval_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_handles = [
        (shard_dir / f"main_eval.{idx:03d}.jsonl").open("w", encoding="utf-8")
        for idx in range(main_eval_shards)
    ]
    main_eval_seen = 0
    writer: pq.ParquetWriter | None = None
    assignment_rows = []
    try:
        pf = pq.ParquetFile(parquet_path)
        for batch in pf.iter_batches(columns=list(DETECTOR_COLUMNS), batch_size=50_000):
            for row in batch.to_pylist():
                split = assignments.get(row["unified_id"], "remaining")
                out = detector_row(row, split, policy_version)
                line = json.dumps(out, ensure_ascii=False, sort_keys=True)
                handles[split].write(line + "\n")
                if split == "main_eval":
                    shard_idx = main_eval_seen % main_eval_shards
                    shard_handles[shard_idx].write(line + "\n")
                    main_eval_seen += 1
                assignment_rows.append(
                    {
                        "unified_id": row["unified_id"],
                        "paper_split": split,
                    }
                )
            if len(assignment_rows) >= 250_000:
                table = pa.Table.from_pylist(assignment_rows)
                if writer is None:
                    writer = pq.ParquetWriter(output_dir / "split_assignments.parquet", table.schema, compression="zstd")
                writer.write_table(table)
                assignment_rows = []
    finally:
        for handle in handles.values():
            handle.close()
        for handle in shard_handles:
            handle.close()
        if assignment_rows:
            table = pa.Table.from_pylist(assignment_rows)
            if writer is None:
                writer = pq.ParquetWriter(output_dir / "split_assignments.parquet", table.schema, compression="zstd")
            writer.write_table(table)
        if writer is not None:
            writer.close()


def validate_summary(summary: dict[str, Any]) -> None:
    targets = {
        "main_eval": _EVAL_TARGET,
        "validation": _EVAL_TARGET,
        "train": _TRAIN_TARGET,
    }
    for split, total_target in targets.items():
        counts = summary["splits"][split]["counts"]
        per_label = total_target // 2
        for label in (0, 1):
            actual = counts.get(f"label_{label}", 0)
            if actual != per_label:
                raise AssertionError(
                    f"{split} label {label}: expected {per_label}, found {actual}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Create deterministic paper splits from unified TELL parquet.")
    parser.add_argument("--parquet", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", default=DEFAULT_SEED)
    parser.add_argument("--main-eval-shards", type=int, default=16)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("Loading metadata ...", flush=True)
    rows = load_metadata(args.parquet)
    print(f"Loaded {len(rows):,} rows", flush=True)
    assignments = assign_splits(rows, args.seed)
    print(f"Assigned {len(assignments):,} rows to splits", flush=True)
    summary = summarize(rows, assignments)
    summary["seed"] = args.seed
    summary["parquet"] = str(args.parquet)
    summary["main_eval_shards"] = args.main_eval_shards
    validate_summary(summary)
    (args.output_dir / "split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("Writing split files ...", flush=True)
    write_splits(args.parquet, args.output_dir, assignments, POLICY_VERSION, args.main_eval_shards)
    print("Done.")
    for split in SPLIT_ORDER:
        c = summary["splits"][split]["counts"]
        print(f"  {split}: {c.get('label_0',0):,} human + {c.get('label_1',0):,} AI = {c.get('total',0):,} total")


if __name__ == "__main__":
    main()
