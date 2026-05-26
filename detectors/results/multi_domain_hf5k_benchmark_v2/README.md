# ACMC HF Five-Detector Bootstrap Package

This package contains the exact test/validation examples and per-example detector scores for the five-detector ACMC HF addendum.

## Main files

- `input/test.jsonl`, `input/validation.jsonl`: exact examples that were scored, including `id`, `text`, `label`, dataset/domain/generator metadata, and split metadata.
- `predictions/test/*.predictions.jsonl`, `predictions/validation/*.predictions.jsonl`: raw merged per-detector prediction files.
- `joined/test_examples_with_five_detector_scores.jsonl`, `joined/validation_examples_with_five_detector_scores.jsonl`: one record per example, with the original example fields plus a nested `detectors` object containing `score_ai`, `raw_score`, `error`, `runtime_s`, and other raw detector fields.
- `joined/*.tsv`: flat TSV versions of the joined files. JSONL is the safer source of truth because it preserves full text robustly.
- `summaries/paper_five_requested_detectors_results.{json,tsv}`: detector-level aggregate metrics from the paper addendum.
- `package_manifest.json`: file hashes, row counts, and error/missing counts.

## Bootstrap note

Use `label` as the target label and each detector's `score_ai` as the continuous detector score. PHD has explicit error rows where no score is defined; those rows have a non-null PHD error and null PHD score.
