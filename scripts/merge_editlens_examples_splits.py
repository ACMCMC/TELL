"""Concatenate two EditLens SFT example JSONLs (keep duplicates), shuffle, split."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def load_jsonl_rows(*, path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def merge_concat_rows(*, path_old: Path, path_new: Path) -> list[dict]:
    return load_jsonl_rows(path=path_old) + load_jsonl_rows(path=path_new)


def shuffle_rows(*, rows: list[dict], seed: int) -> None:
    rng = random.Random(seed)
    rng.shuffle(rows)


def truncate_rows(*, rows: list[dict], max_len: int) -> list[dict]:
    return rows[:max_len]


def split_three(
    *,
    rows: list[dict],
    n_train: int,
    n_val: int,
    n_test: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    need = n_train + n_val + n_test
    if len(rows) != need:
        raise ValueError(f"rows length {len(rows)} != n_train+n_val+n_test ({need})")
    train_rows = rows[:n_train]
    val_rows = rows[n_train : n_train + n_val]
    test_rows = rows[n_train + n_val :]
    return train_rows, val_rows, test_rows


def write_jsonl(*, path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path-old", type=Path, required=True)
    parser.add_argument("--path-new", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n-train", type=int, required=True)
    parser.add_argument("--n-val", type=int, required=True)
    parser.add_argument("--n-test", type=int, required=True)
    parser.add_argument("--truncate-to", type=int, required=True)
    args = parser.parse_args()

    merged = merge_concat_rows(path_old=args.path_old, path_new=args.path_new)
    shuffle_rows(rows=merged, seed=args.seed)
    merged_t = truncate_rows(rows=merged, max_len=args.truncate_to)
    train_rows, val_rows, test_rows = split_three(
        rows=merged_t,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
    )

    out_dir = args.output_dir
    write_jsonl(path=out_dir / "merged_concat_shuffled_truncated.jsonl", rows=merged_t)
    write_jsonl(path=out_dir / "train.jsonl", rows=train_rows)
    write_jsonl(path=out_dir / "val.jsonl", rows=val_rows)
    write_jsonl(path=out_dir / "test.jsonl", rows=test_rows)

    manifest = {
        "path_old": str(args.path_old.resolve()),
        "path_new": str(args.path_new.resolve()),
        "seed": args.seed,
        "rows_concat": len(merged),
        "rows_after_truncate": len(merged_t),
        "n_train": args.n_train,
        "n_val": args.n_val,
        "n_test": args.n_test,
        "dedupe": False,
        "note": "duplicate example_id allowed when annotations differ across sources",
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
