"""Create deterministic train/validation/test JSONL splits for TELL and optionally upload."""

import argparse
import json
import random
from pathlib import Path


SEED = 2262


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            row = json.loads(raw)
            if "text" in row and "annotation" in row and "label" in row:
                rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _split_rows(rows: list[dict], val_frac: float, test_frac: float) -> tuple[list[dict], list[dict], list[dict]]:
    by_label: dict[int, list[dict]] = {0: [], 1: []}
    for row in rows:
        label = int(row["label"])
        by_label[label].append(row)

    rng = random.Random(SEED)
    train: list[dict] = []
    val: list[dict] = []
    test: list[dict] = []

    for label in (0, 1):
        group = by_label[label]
        rng.shuffle(group)
        n = len(group)
        n_test = int(round(n * test_frac))
        n_val = int(round(n * val_frac))
        n_test = min(n_test, n)
        n_val = min(n_val, max(0, n - n_test))
        test.extend(group[:n_test])
        val.extend(group[n_test : n_test + n_val])
        train.extend(group[n_test + n_val :])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def _upload_splits(repo_id: str, out_dir: Path) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    files = ["train.jsonl", "validation.jsonl", "test.jsonl"]
    for name in files:
        api.upload_file(
            path_or_fileobj=str(out_dir / name),
            path_in_repo=name,
            repo_id=repo_id,
            repo_type="dataset",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Split TELL JSONL into train/validation/test and upload to HF.")
    parser.add_argument("--input-jsonl", required=True, help="Source JSONL with at least text, annotation, label")
    parser.add_argument("--output-dir", required=True, help="Output directory for split JSONL files")
    parser.add_argument("--val-frac", type=float, required=True, help="Validation fraction, e.g. 0.1")
    parser.add_argument("--test-frac", type=float, required=True, help="Test fraction, e.g. 0.1")
    parser.add_argument("--upload-repo", default="", help="HF dataset repo id, e.g. acmc/TELL")
    args = parser.parse_args()

    src = Path(args.input_jsonl)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_jsonl(path=src)
    if not rows:
        raise SystemExit("No valid rows found in input JSONL.")
    train, val, test = _split_rows(rows=rows, val_frac=float(args.val_frac), test_frac=float(args.test_frac))

    _write_jsonl(path=out_dir / "train.jsonl", rows=train)
    _write_jsonl(path=out_dir / "validation.jsonl", rows=val)
    _write_jsonl(path=out_dir / "test.jsonl", rows=test)

    manifest = {
        "seed": SEED,
        "input_jsonl": str(src),
        "n_total": len(rows),
        "n_train": len(train),
        "n_validation": len(val),
        "n_test": len(test),
        "val_frac": float(args.val_frac),
        "test_frac": float(args.test_frac),
    }
    with (out_dir / "split_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    if args.upload_repo:
        _upload_splits(repo_id=args.upload_repo, out_dir=out_dir)
        print(f"Uploaded splits to hf://{args.upload_repo}")

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
