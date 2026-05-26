"""Sample 50/50-balanced train/val/test splits from the unified TELL parquet.

Reads unified_tell_dataset.parquet, filters to is_default_training_candidate=true
for the clean slice, and also samples an adversarial slice from RAID
(is_adversarial=True, domain != code/non_english).

Adversarial rows have domain="adversarial" and source_detail=attack_type so they
form their own sampler stratum. LODO experiments simply enumerate real domains
(which never includes "adversarial"), naturally excluding adversarial rows without
any conditional logic.

Writes:
  splits/{train,val,test}.jsonl          — clean rows (mage_score=null)
  adversarial_splits/{train,val,test}.jsonl — adversarial rows (mage_score=null)
  shards/{train,val,test}/               — clean harness shards for MAGE
  adversarial_shards/{train,val,test}/   — adversarial harness shards for MAGE
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

READ_COLUMNS = (
    "unified_id",
    "text",
    "label",
    "domain",
    "dataset_id",
    "source_split",
    "source",
    "source_detail",
    "generator_model",
    "attack",
    "language",
    "is_default_training_candidate",
    "is_adversarial",
)

SPLIT_POLICY = "balanced_splits_v1"
ADVERSARIAL_DATASET = "liamdugan/raid"
ADVERSARIAL_EXCLUDED_DOMAINS = {"code", "non_english"}
EXCLUDED_DATASETS = {"acmc/cheat"}


def _stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _strata_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("domain") or "unknown"),
        str(row.get("dataset_id") or "unknown"),
        str(row.get("source_detail") or "unknown"),
    )


def _strata_key_adversarial(row: dict[str, Any]) -> tuple[str, str]:
    return (
        str(row.get("domain") or "unknown"),
        str(row.get("attack") or "unknown"),
    )


def _allocate_stratified(
    rows: list[dict[str, Any]],
    quota: int,
    seed: str,
    key_fn=_strata_key,
) -> list[dict[str, Any]]:
    """Proportional stratified sample of quota rows, deterministic via SHA-256 ordering."""
    quota = min(quota, len(rows))
    strata: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        strata[key_fn(row)].append(row)

    for bucket in strata.values():
        bucket.sort(key=lambda r: _stable_hash(f"{seed}|{r['unified_id']}"))

    total = len(rows)
    raw = {key: quota * len(bucket) / total for key, bucket in strata.items()}
    alloc = {key: min(len(strata[key]), int(math.floor(v))) for key, v in raw.items()}
    remaining = quota - sum(alloc.values())

    order = sorted(
        strata,
        key=lambda k: (raw[k] - math.floor(raw[k]), len(strata[k]), _stable_hash(seed + repr(k))),
        reverse=True,
    )
    i = 0
    while remaining > 0:
        key = order[i % len(order)]
        if alloc[key] < len(strata[key]):
            alloc[key] += 1
            remaining -= 1
        i += 1

    selected: list[dict] = []
    for key, n in alloc.items():
        selected.extend(strata[key][:n])
    selected.sort(key=lambda r: _stable_hash(f"{seed}|final|{r['unified_id']}"))
    return selected


def _load_clean_rows(parquet_path: Path) -> tuple[list[dict], list[dict]]:
    """Return (human_rows, ai_rows) for is_default_training_candidate rows."""
    human: list[dict] = []
    ai: list[dict] = []
    pf = pq.ParquetFile(parquet_path)
    total = 0
    for batch in pf.iter_batches(columns=list(READ_COLUMNS), batch_size=50_000):
        for row in batch.to_pylist():
            total += 1
            if total % 500_000 == 0:
                print(f"  scanned {total:,} rows ...", flush=True)
            if not row.get("is_default_training_candidate"):
                continue
            if row.get("dataset_id") in EXCLUDED_DATASETS:
                continue
            text = (row.get("text") or "").strip()
            if not text:
                continue
            if int(row["label"]) == 0:
                human.append(row)
            else:
                ai.append(row)
    print(f"Clean rows: {len(human):,} human  {len(ai):,} AI  (from {total:,} total)", flush=True)
    return human, ai


def _load_adversarial_rows(parquet_path: Path) -> list[dict]:
    """Return adversarial AI rows from RAID, excluding code/non_english domains."""
    rows: list[dict] = []
    pf = pq.ParquetFile(parquet_path)
    for batch in pf.iter_batches(columns=list(READ_COLUMNS), batch_size=50_000):
        for row in batch.to_pylist():
            if not row.get("is_adversarial"):
                continue
            if row.get("dataset_id") != ADVERSARIAL_DATASET:
                continue
            if row.get("domain") in ADVERSARIAL_EXCLUDED_DOMAINS:
                continue
            if int(row.get("label", 0)) != 1:
                continue
            text = (row.get("text") or "").strip()
            if not text:
                continue
            rows.append(row)
    print(f"Adversarial rows: {len(rows):,} AI from RAID", flush=True)
    return rows


def _to_rl_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["unified_id"],
        "text": row["text"],
        "label": int(row["label"]),
        "domain": row.get("domain"),
        "dataset_id": row.get("dataset_id"),
        "source": row.get("source"),
        "source_detail": row.get("source_detail"),
        "generator_model": row.get("generator_model"),
        "attack": row.get("attack"),
        "language": row.get("language"),
        "mage_score": None,
    }


def _to_rl_row_adversarial(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["unified_id"],
        "text": row["text"],
        "label": int(row["label"]),
        "domain": "adversarial",
        "dataset_id": row.get("dataset_id"),
        "source": row.get("source"),
        "source_detail": row.get("attack") or "unknown",
        "generator_model": row.get("generator_model"),
        "attack": row.get("attack"),
        "language": row.get("language"),
        "mage_score": None,
    }


def _to_harness_row(row: dict[str, Any], split: str, is_adversarial: bool = False) -> dict[str, Any]:
    return {
        "id": row["unified_id"],
        "text": row["text"],
        "label": int(row["label"]),
        "split": split,
        "dataset": row.get("dataset_id"),
        "domain": "adversarial" if is_adversarial else row.get("domain"),
        "generator": row.get("generator_model"),
        "attack": row.get("attack"),
        "language": row.get("language"),
        "source_split": row.get("source_split"),
        "source": row.get("source"),
        "source_detail": row.get("attack") if is_adversarial else row.get("source_detail"),
        "is_default_training_candidate": not is_adversarial,
        "is_adversarial": is_adversarial,
        "split_policy": SPLIT_POLICY,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  {len(rows):,} rows → {path}", flush=True)


def _write_shards(
    shard_dir: Path,
    rows: list[dict[str, Any]],
    split: str,
    n_shards: int,
    is_adversarial: bool = False,
) -> None:
    shard_dir.mkdir(parents=True, exist_ok=True)
    handles = [
        (shard_dir / f"main_eval.{idx:03d}.jsonl").open("w", encoding="utf-8")
        for idx in range(n_shards)
    ]
    try:
        for i, row in enumerate(rows):
            harness_row = _to_harness_row(row, split, is_adversarial=is_adversarial)
            handles[i % n_shards].write(json.dumps(harness_row, ensure_ascii=False) + "\n")
    finally:
        for fh in handles:
            fh.close()
    print(f"  {len(rows):,} rows → {n_shards} shards in {shard_dir}", flush=True)


def _split_rows(rows: list[dict], train_n: int, val_n: int, test_n: int, seed: str):
    rows = sorted(rows, key=lambda r: _stable_hash(f"{seed}|split|{r['unified_id']}"))
    return rows[:test_n], rows[test_n:test_n + val_n], rows[test_n + val_n:test_n + val_n + train_n]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--train-per-class", type=int, default=100_000)
    parser.add_argument("--val-per-class", type=int, default=10_000)
    parser.add_argument("--test-per-class", type=int, default=10_000)
    parser.add_argument("--adv-train", type=int, default=50_000, help="Adversarial AI rows in train")
    parser.add_argument("--adv-val", type=int, default=5_000, help="Adversarial AI rows in val")
    parser.add_argument("--adv-test", type=int, default=5_000, help="Adversarial AI rows in test")
    parser.add_argument("--shards", type=int, default=16)
    parser.add_argument("--seed", default="20260428")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed = args.seed
    total_per_class = args.train_per_class + args.val_per_class + args.test_per_class
    total_adv = args.adv_train + args.adv_val + args.adv_test

    print("Loading clean rows from parquet ...", flush=True)
    human_rows, ai_rows = _load_clean_rows(args.parquet)

    print("Loading adversarial rows from parquet ...", flush=True)
    adv_rows = _load_adversarial_rows(args.parquet)

    print("Stratified sampling (clean) ...", flush=True)
    human_sample = _allocate_stratified(human_rows, min(total_per_class, len(human_rows)), f"{seed}|human")
    ai_sample = _allocate_stratified(ai_rows, min(total_per_class, len(ai_rows)), f"{seed}|ai")

    print("Stratified sampling (adversarial) ...", flush=True)
    adv_sample = _allocate_stratified(
        adv_rows, min(total_adv, len(adv_rows)), f"{seed}|adv", key_fn=_strata_key_adversarial
    )

    human_test, human_val, human_train = _split_rows(
        human_sample, args.train_per_class, args.val_per_class, args.test_per_class, seed)
    ai_test, ai_val, ai_train = _split_rows(
        ai_sample, args.train_per_class, args.val_per_class, args.test_per_class, seed)
    adv_test, adv_val, adv_train = _split_rows(
        adv_sample, args.adv_train, args.adv_val, args.adv_test, f"{seed}|adv")

    rng = random.Random(int(seed))
    clean_splits = {
        "train": human_train + ai_train,
        "val": human_val + ai_val,
        "test": human_test + ai_test,
    }
    adv_splits = {"train": adv_train, "val": adv_val, "test": adv_test}
    for rows in clean_splits.values():
        rng.shuffle(rows)
    for rows in adv_splits.values():
        rng.shuffle(rows)

    splits_dir = args.output_dir / "splits"
    adv_splits_dir = args.output_dir / "adversarial_splits"
    shards_dir = args.output_dir / "shards"
    adv_shards_dir = args.output_dir / "adversarial_shards"
    splits_dir.mkdir(parents=True, exist_ok=True)
    adv_splits_dir.mkdir(parents=True, exist_ok=True)

    print("Writing clean JSONL splits ...", flush=True)
    for split, rows in clean_splits.items():
        _write_jsonl(splits_dir / f"{split}.jsonl", [_to_rl_row(r) for r in rows])

    print("Writing adversarial JSONL splits ...", flush=True)
    for split, rows in adv_splits.items():
        _write_jsonl(adv_splits_dir / f"{split}.jsonl", [_to_rl_row_adversarial(r) for r in rows])

    print("Writing clean harness shards ...", flush=True)
    for split, rows in clean_splits.items():
        _write_shards(shards_dir / split, rows, split, args.shards, is_adversarial=False)

    print("Writing adversarial harness shards ...", flush=True)
    for split, rows in adv_splits.items():
        _write_shards(adv_shards_dir / split, rows, split, args.shards, is_adversarial=True)

    summary = {
        "seed": seed,
        "parquet": str(args.parquet),
        "split_policy": SPLIT_POLICY,
        "splits": {
            split: {
                "total": len(clean_splits[split]),
                "human": sum(1 for r in clean_splits[split] if int(r["label"]) == 0),
                "ai": sum(1 for r in clean_splits[split] if int(r["label"]) == 1),
                "adversarial_ai": len(adv_splits[split]),
            }
            for split in ("train", "val", "test")
        },
    }
    (args.output_dir / "split_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
