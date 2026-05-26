#!/usr/bin/env python3
"""Prepare and optionally upload the TELL split of human_detectors."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from urllib.request import urlopen


SOURCE_URL = "https://raw.githubusercontent.com/jenna-russell/human_detectors/refs/heads/main/human_detectors.json"
SOURCE_DATASET = "jenna-russell/human_detectors"
SOURCE_SHA256 = "1b8f9eeb6413f2541a320b09c7cc77b8d5c51a9e69a87ae0e91bd0e9ede5bd9a"
SPLIT_SEED = "tell-human-detectors-v1-2242"
VALIDATION_PROMPT_IDS = [4, 8, 13, 14, 15, 16, 20, 23, 25, 26]
TEST_PROMPT_IDS = [1, 2, 3, 5, 6, 7, 9, 10, 11, 12, 17, 18, 19, 21, 22, 24, 27, 28, 29, 30]
GENERATION_MODELS = ["claude", "gpt-4o", "humanized_o1-pro", "o1-pro", "paraphrased_gpt-4o"]
LABELS = ["AI-generated", "Human-written"]


def canonical_sha256(payload: object) -> str:
    data = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def load_source(source_url: str) -> tuple[dict, list[dict]]:
    with urlopen(source_url, timeout=60) as response:
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected source JSON object, got {type(payload).__name__}")
    rows = [payload[str(idx)] for idx in range(len(payload))]
    return payload, rows


def add_split_metadata(row: dict, source_row_index: int, split: str) -> dict:
    out = dict(row)
    out["issue_original_type"] = type(row.get("issue")).__name__
    if row.get("issue") is not None:
        out["issue"] = str(row["issue"])
    out.update(
        {
            "split": split,
            "source_row_index": source_row_index,
            "source_dataset": SOURCE_DATASET,
            "source_sha256": SOURCE_SHA256,
            "source_url": SOURCE_URL,
            "split_method": "prompt_id_grouped_sha256_seed",
            "split_seed": SPLIT_SEED,
        }
    )
    return out


def validate_source(payload: dict, rows: list[dict], allow_checksum_update: bool = False) -> str:
    actual_sha = canonical_sha256(payload)
    if actual_sha != SOURCE_SHA256 and not allow_checksum_update:
        raise ValueError(f"Source checksum mismatch: expected {SOURCE_SHA256}, got {actual_sha}")
    if len(rows) != 300:
        raise ValueError(f"Expected 300 rows, got {len(rows)}")

    required = {
        "generation_model",
        "prompt_id",
        "article",
        "id",
        "ground_truth",
        "annotator_1",
        "annotator_2",
        "annotator_3",
        "annotator_4",
        "annotator_5",
        "majority_vote",
        "expert_majority_vote",
    }
    for idx, row in enumerate(rows):
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"Row {idx} missing required fields: {missing}")
        if row["ground_truth"] not in LABELS:
            raise ValueError(f"Row {idx} has unexpected ground_truth={row['ground_truth']!r}")
        if row["generation_model"] not in GENERATION_MODELS:
            raise ValueError(f"Row {idx} has unexpected generation_model={row['generation_model']!r}")
        for annot_idx in range(1, 6):
            annot = row.get(f"annotator_{annot_idx}")
            if not isinstance(annot, dict):
                raise ValueError(f"Row {idx} annotator_{annot_idx} is missing or not an object")

    label_counts = Counter(row["ground_truth"] for row in rows)
    if label_counts != Counter({"AI-generated": 150, "Human-written": 150}):
        raise ValueError(f"Unexpected label counts: {label_counts}")

    model_counts = Counter(row["generation_model"] for row in rows)
    if model_counts != Counter({model: 60 for model in GENERATION_MODELS}):
        raise ValueError(f"Unexpected generation_model counts: {model_counts}")

    by_prompt: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_prompt[int(row["prompt_id"])].append(row)
    if set(by_prompt) != set(range(1, 31)):
        raise ValueError(f"Unexpected prompt_id set: {sorted(by_prompt)}")
    expected_pairs = {(label, model) for label in LABELS for model in GENERATION_MODELS}
    for prompt_id, group in by_prompt.items():
        pairs = Counter((row["ground_truth"], row["generation_model"]) for row in group)
        if len(group) != 10 or set(pairs) != expected_pairs or any(count != 1 for count in pairs.values()):
            raise ValueError(f"Prompt group {prompt_id} is not a complete label/model block: {pairs}")
    return actual_sha


def assign_splits(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    val_ids = set(VALIDATION_PROMPT_IDS)
    test_ids = set(TEST_PROMPT_IDS)
    if val_ids & test_ids:
        raise ValueError("Validation and test prompt IDs overlap")
    if val_ids | test_ids != set(range(1, 31)):
        raise ValueError("Validation and test prompt IDs do not cover all prompt IDs")

    validation = []
    test = []
    for idx, row in enumerate(rows):
        prompt_id = int(row["prompt_id"])
        if prompt_id in val_ids:
            validation.append(add_split_metadata(row, idx, "validation"))
        elif prompt_id in test_ids:
            test.append(add_split_metadata(row, idx, "test"))
        else:
            raise ValueError(f"Unassigned prompt_id={prompt_id}")
    return validation, test


def validate_split(validation: list[dict], test: list[dict]) -> None:
    if len(validation) != 100:
        raise ValueError(f"Expected 100 validation rows, got {len(validation)}")
    if len(test) != 200:
        raise ValueError(f"Expected 200 test rows, got {len(test)}")

    val_prompts = {int(row["prompt_id"]) for row in validation}
    test_prompts = {int(row["prompt_id"]) for row in test}
    if val_prompts != set(VALIDATION_PROMPT_IDS):
        raise ValueError(f"Unexpected validation prompt IDs: {sorted(val_prompts)}")
    if test_prompts != set(TEST_PROMPT_IDS):
        raise ValueError(f"Unexpected test prompt IDs: {sorted(test_prompts)}")
    if val_prompts & test_prompts:
        raise ValueError("Prompt leakage: prompt_id appears in both validation and test")

    expected_val_pairs = Counter({(label, model): 10 for label in LABELS for model in GENERATION_MODELS})
    expected_test_pairs = Counter({(label, model): 20 for label in LABELS for model in GENERATION_MODELS})
    val_pairs = Counter((row["ground_truth"], row["generation_model"]) for row in validation)
    test_pairs = Counter((row["ground_truth"], row["generation_model"]) for row in test)
    if val_pairs != expected_val_pairs:
        raise ValueError(f"Unexpected validation label/model balance: {val_pairs}")
    if test_pairs != expected_test_pairs:
        raise ValueError(f"Unexpected test label/model balance: {test_pairs}")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def build_manifest(actual_sha: str, validation: list[dict], test: list[dict]) -> dict:
    return {
        "dataset_name": "tell-human-detectors",
        "source_dataset": SOURCE_DATASET,
        "source_url": SOURCE_URL,
        "source_sha256": SOURCE_SHA256,
        "actual_source_sha256": actual_sha,
        "split_method": "prompt_id_grouped_sha256_seed",
        "split_seed": SPLIT_SEED,
        "validation_prompt_ids": VALIDATION_PROMPT_IDS,
        "test_prompt_ids": TEST_PROMPT_IDS,
        "num_rows": {"validation": len(validation), "test": len(test), "total": len(validation) + len(test)},
        "labels": dict(Counter(row["ground_truth"] for row in validation + test)),
        "generation_models": dict(Counter(row["generation_model"] for row in validation + test)),
        "license": "MIT",
        "citation": "Russell, Karpinska, and Iyyer. People who frequently use ChatGPT for writing tasks are accurate and robust detectors of AI-generated text. arXiv:2501.15654.",
        "artifact_credit": "TELL split artifact prepared by Suraj Ranganath and Aldan Creo.",
    }


def dataset_card() -> str:
    return f"""---
license: mit
language:
- en
task_categories:
- text-classification
pretty_name: TELL Human Detectors Split
size_categories:
- n<1K
configs:
- config_name: default
  data_files:
  - split: validation
    path: validation.jsonl
  - split: test
    path: test.jsonl
---

# TELL Human Detectors Split

This dataset is a prompt-disjoint validation/test split of `human_detectors.json` from
Jenna Russell, Marzena Karpinska, and Mohit Iyyer, "People who frequently use
ChatGPT for writing tasks are accurate and robust detectors of AI-generated text"
(arXiv:2501.15654).

The split is intended as a research artifact for evaluating AI-vs-human writing
detectors and explanation-quality methods. It preserves the upstream fields, with
one documented `issue` type normalization for Hugging Face compatibility, and adds
split provenance metadata.

This TELL split artifact was prepared by Suraj Ranganath and Aldan Creo. The
underlying Human Detectors dataset should be cited to Russell, Karpinska, and
Iyyer.

## Source

- Upstream repository: https://github.com/jenna-russell/human_detectors
- Source file: `{SOURCE_URL}`
- Source dataset license: MIT
- Canonical source SHA256: `{SOURCE_SHA256}`

## Split Protocol

The original dataset has 300 documents, 30 `prompt_id` groups, and 10 rows per
prompt group. Each prompt group contains exactly one row for each combination of
`ground_truth` in `AI-generated`, `Human-written` and `generation_model` in
`claude`, `gpt-4o`, `humanized_o1-pro`, `o1-pro`, and `paraphrased_gpt-4o`.

We split by `prompt_id`, not by individual row, to prevent the same underlying
prompt/topic from appearing in both validation and test.

- Split seed: `{SPLIT_SEED}`
- Validation prompt IDs: `{VALIDATION_PROMPT_IDS}`
- Test prompt IDs: `{TEST_PROMPT_IDS}`
- Validation rows: 100
- Test rows: 200

Both splits are balanced by label and generation model. Validation has 50
human-written and 50 AI-generated documents. Test has 100 human-written and 100
AI-generated documents.

## Fields

All upstream fields are preserved, including article metadata, `article`,
`ground_truth`, automatic detector outputs, five annotator objects, `majority_vote`,
and `expert_majority_vote`. The upstream `issue` field mixes strings and integers,
so this artifact stores `issue` as a string for Hugging Face/Arrow compatibility
and records the original Python type in `issue_original_type`.

Additional fields:

- `split`: `validation` or `test`
- `source_row_index`: integer row index in the sorted upstream JSON object
- `source_dataset`: upstream dataset identifier
- `source_url`: upstream source URL
- `source_sha256`: canonical upstream checksum used by this artifact
- `split_method`: `prompt_id_grouped_sha256_seed`
- `split_seed`: deterministic split seed
- `issue_original_type`: original upstream JSON type for `issue`

## Recommended Use

Use `validation` for prompt, metric, and model-selection decisions. Use `test` once
for final reporting. Do not tune on the test set, and do not merge validation and
test without reporting that choice.

## Citation

```bibtex
@misc{{russell2025humandetectors,
  title={{People who frequently use ChatGPT for writing tasks are accurate and robust detectors of AI-generated text}},
  author={{Russell, Jenna and Karpinska, Marzena and Iyyer, Mohit}},
  year={{2025}},
  eprint={{2501.15654}},
  archivePrefix={{arXiv}},
  primaryClass={{cs.CL}}
}}
```
"""


def write_dataset(output_dir: Path, actual_sha: str, validation: list[dict], test: list[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "validation.jsonl", validation)
    write_jsonl(output_dir / "test.jsonl", test)
    manifest = build_manifest(actual_sha=actual_sha, validation=validation, test=test)
    (output_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "README.md").write_text(dataset_card(), encoding="utf-8")


def validate_with_datasets(output_dir: Path) -> None:
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError("datasets is required for reload validation") from exc
    ds = load_dataset(
        "json",
        data_files={
            "validation": str(output_dir / "validation.jsonl"),
            "test": str(output_dir / "test.jsonl"),
        },
    )
    if set(ds.keys()) != {"validation", "test"}:
        raise ValueError(f"Unexpected DatasetDict splits: {list(ds.keys())}")
    if len(ds["validation"]) != 100 or len(ds["test"]) != 200:
        raise ValueError(f"Unexpected loaded sizes: validation={len(ds['validation'])}, test={len(ds['test'])}")
    val_prompts = set(ds["validation"]["prompt_id"])
    test_prompts = set(ds["test"]["prompt_id"])
    if val_prompts & test_prompts:
        raise ValueError("Reloaded dataset has prompt_id leakage")


def upload_to_hf(output_dir: Path, repo_id: str, private: bool) -> str:
    try:
        from huggingface_hub import HfApi
    except Exception as exc:
        raise RuntimeError("huggingface_hub is required for --upload") from exc

    api = HfApi()
    if "/" not in repo_id:
        whoami = api.whoami()
        namespace = whoami.get("name") or whoami.get("fullname")
        if not namespace:
            raise RuntimeError("Could not infer Hugging Face namespace; pass --repo-id namespace/tell-human-detectors")
        repo_id = f"{namespace}/{repo_id}"
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(output_dir),
        commit_message="Add prompt-disjoint validation/test split",
    )
    return repo_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the HF release artifact for the TELL human_detectors split.")
    parser.add_argument("--source-url", default=SOURCE_URL)
    parser.add_argument("--output-dir", default="data/human_detectors_hf")
    parser.add_argument("--upload", action="store_true", help="Upload the generated files to Hugging Face.")
    parser.add_argument("--repo-id", default="tell-human-detectors", help="HF dataset repo ID or repo name.")
    parser.add_argument("--private", action="store_true", help="Create/upload the HF dataset repo as private.")
    parser.add_argument("--allow-checksum-update", action="store_true", help="Only for inspection if upstream changed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    payload, rows = load_source(args.source_url)
    actual_sha = validate_source(payload=payload, rows=rows, allow_checksum_update=args.allow_checksum_update)
    validation, test = assign_splits(rows)
    validate_split(validation=validation, test=test)
    write_dataset(output_dir=output_dir, actual_sha=actual_sha, validation=validation, test=test)
    validate_with_datasets(output_dir)

    result = {
        "output_dir": str(output_dir),
        "source_sha256": actual_sha,
        "validation_rows": len(validation),
        "test_rows": len(test),
        "validation_prompt_ids": VALIDATION_PROMPT_IDS,
        "test_prompt_ids": TEST_PROMPT_IDS,
    }
    if args.upload:
        result["hf_repo_id"] = upload_to_hf(output_dir=output_dir, repo_id=args.repo_id, private=args.private)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
