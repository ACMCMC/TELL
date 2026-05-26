from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_DETECTORS = (
    "openai_roberta",
    "chatgpt_d",
    "argugpt",
    "radar",
    "mage_d",
    "detectllm_lrr",
    "mfd",
)


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def merge_detector(output_root: Path, detector: str, shards: int, expected_rows: int, bootstrap: int) -> dict:
    shard_root = output_root / "predictions_sharded" / detector
    merged_root = output_root / "merged_predictions"
    metrics_root = output_root / "metrics"
    manifest_root = output_root / "manifests"
    merged_root.mkdir(parents=True, exist_ok=True)
    metrics_root.mkdir(parents=True, exist_ok=True)
    manifest_root.mkdir(parents=True, exist_ok=True)

    merged_path = merged_root / f"{detector}.predictions.jsonl"
    seen_ids: set[str] = set()
    n_rows = 0
    n_errors = 0
    missing = []

    with merged_path.open("w", encoding="utf-8") as out:
        for shard_idx in range(shards):
            shard_path = shard_root / f"{detector}.s{shard_idx:03d}.predictions.jsonl"
            if not shard_path.exists():
                missing.append(str(shard_path))
                continue
            for row in read_jsonl(shard_path):
                row_id = row.get("id")
                if row_id in seen_ids:
                    raise ValueError(f"Duplicate id for {detector}: {row_id}")
                seen_ids.add(row_id)
                n_rows += 1
                if row.get("error") is not None:
                    n_errors += 1
                out.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

            manifest_path = shard_path.with_name(shard_path.name.replace(".predictions.jsonl", ".manifest.json"))
            if manifest_path.exists():
                target = manifest_root / manifest_path.name
                target.write_text(manifest_path.read_text(encoding="utf-8"), encoding="utf-8")

    summary = {
        "detector": detector,
        "merged_predictions": str(merged_path),
        "n_rows": n_rows,
        "n_errors": n_errors,
        "missing_shards": missing,
        "expected_rows": expected_rows,
        "complete": not missing and n_rows == expected_rows,
    }
    if summary["complete"]:
        metrics_path = metrics_root / f"{detector}.metrics.json"
        cmd = [
            sys.executable,
            "-m",
            "detectors_bench.run_benchmark",
            "--predictions",
            str(merged_path),
            "--output",
            str(metrics_path),
            "--bootstrap",
            str(bootstrap),
        ]
        subprocess.check_call(cmd)
        summary["metrics"] = str(metrics_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge and validate wave-1 fast detector shard predictions.")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--detectors", default=",".join(DEFAULT_DETECTORS))
    parser.add_argument("--shards", type=int, default=16)
    parser.add_argument("--expected-rows", type=int, default=200_000)
    parser.add_argument("--summary-name", default="wave1_merge_summary.json")
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=0,
        help="Bootstrap replicates for metric CIs. Use 0 for the main unattended merge; run CIs separately.",
    )
    args = parser.parse_args()

    summaries = []
    for detector in [x for x in args.detectors.split(",") if x]:
        summaries.append(merge_detector(args.output_root, detector, args.shards, args.expected_rows, args.bootstrap))

    out = args.output_root / args.summary_name
    out.write_text(json.dumps(summaries, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
