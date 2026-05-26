#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
from typing import Any

from datasets import load_dataset


def _row_to_detector_example(row: dict[str, Any], split: str) -> dict[str, Any]:
    text = row.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"missing non-empty text for id={row.get('id')!r}")
    label = int(row["label"])
    if label not in (0, 1):
        raise ValueError(f"label must be 0 or 1 for id={row.get('id')!r}: {label!r}")

    return {
        "id": str(row.get("id") or f"{split}:{hashlib.sha1(text.encode('utf-8')).hexdigest()}"),
        "text": text,
        "label": label,
        "split": split,
        "dataset": row.get("dataset_id"),
        "domain": row.get("domain"),
        "generator": row.get("generator_model"),
        "attack": row.get("attack"),
        "language": row.get("language"),
        "source_split": row.get("source_split"),
        "source": row.get("source"),
        "source_detail": row.get("source_detail"),
        "is_adversarial": row.get("is_adversarial"),
        "split_policy": row.get("split_policy"),
        "is_default_training_candidate": row.get("is_default_training_candidate"),
    }


def _inc(counter: collections.Counter, key: Any) -> None:
    counter[str(key) if key is not None else "null"] += 1


def export_split(repo: str, split: str, out_dir: Path, config: str | None = None, limit: int | None = None) -> dict[str, Any]:
    ds = load_dataset(repo, config, split=split) if config else load_dataset(repo, split=split)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    out_path = out_dir / f"{split}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256()
    labels: collections.Counter = collections.Counter()
    datasets: collections.Counter = collections.Counter()
    domains: collections.Counter = collections.Counter()
    attacks: collections.Counter = collections.Counter()
    generators: collections.Counter = collections.Counter()

    n = 0
    skipped_empty_text = 0
    skipped_bad_label = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for row in ds:
            try:
                ex = _row_to_detector_example(dict(row), split)
            except ValueError as exc:
                if "missing non-empty text" in str(exc):
                    skipped_empty_text += 1
                    continue
                if "label must be 0 or 1" in str(exc):
                    skipped_bad_label += 1
                    continue
                raise
            encoded = json.dumps(ex, ensure_ascii=False, sort_keys=True)
            fh.write(encoded + "\n")
            sha.update((encoded + "\n").encode("utf-8"))
            _inc(labels, ex["label"])
            _inc(datasets, ex.get("dataset"))
            _inc(domains, ex.get("domain"))
            _inc(attacks, ex.get("attack"))
            _inc(generators, ex.get("generator"))
            n += 1

    return {
        "split": split,
        "path": str(out_path),
        "rows": n,
        "source_rows": len(ds),
        "skipped_empty_text": skipped_empty_text,
        "skipped_bad_label": skipped_bad_label,
        "sha256": sha.hexdigest(),
        "labels": dict(sorted(labels.items())),
        "datasets": dict(sorted(datasets.items())),
        "domains": dict(sorted(domains.items())),
        "attacks": dict(sorted(attacks.items())),
        "generators_top20": dict(generators.most_common(20)),
        "columns": list(ds.column_names),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export HF benchmark splits to detector-harness JSONL.")
    parser.add_argument("--repo", default="acmc/multi_domain_ai_human_text")
    parser.add_argument("--config", default=None)
    parser.add_argument("--splits", nargs="+", default=["validation", "test", "train"])
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=None, help="Optional per-split cap for smoke tests.")
    args = parser.parse_args()

    manifest = {
        "repo": args.repo,
        "config": args.config or "default",
        "limit": args.limit,
        "splits": {},
    }
    for split in args.splits:
        manifest["splits"][split] = export_split(args.repo, split, args.output_dir, args.config, args.limit)

    out = args.output_dir / "manifest.json"
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
