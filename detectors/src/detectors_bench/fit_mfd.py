from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from .io import batched
from .io import load_examples, write_json
from .registry import DEFAULT_REGISTRY, resolve_detector_config
from .wrappers.lm_features import LMFeatureExtractor


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fit the MFD logistic model on validation/train JSONL only.")
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--detector", default="mfd")
    args = parser.parse_args(argv)

    cfg = resolve_detector_config(args.detector, args.registry)
    feature_names = list(cfg.get("feature_names", ["log_likelihood", "log_rank", "entropy", "llm_deviation"]))
    extractor = LMFeatureExtractor(cfg.get("base_model_name", "gpt2"), cfg.get("device", "auto"))
    batch_size = int(cfg.get("fit_batch_size", cfg.get("batch_size", 16)))
    examples = [ex for ex in load_examples(args.train) if ex.label is not None]
    if len({ex.label for ex in examples}) < 2:
        raise ValueError("MFD fitting requires both human (0) and AI (1) labels.")
    rows = []
    total_batches = (len(examples) + batch_size - 1) // batch_size
    for batch in tqdm(batched(examples, batch_size), total=total_batches, desc="mfd_features"):
        rows.extend(extractor.features_batch([ex.text for ex in batch]))
    x = np.asarray([[row[name] for name in feature_names] for row in rows], dtype=float)
    y = np.asarray([ex.label for ex in examples], dtype=int)
    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=0)),
        ]
    )
    pipeline.fit(x, y)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, args.output)
    write_json(
        args.output.with_suffix(".metadata.json"),
        {
            "detector": args.detector,
            "train": str(args.train),
            "n_train": len(examples),
            "feature_names": feature_names,
            "base_model_name": cfg.get("base_model_name", "gpt2"),
            "feature_batch_size": batch_size,
            "note": "Fit on non-test data only. Do not refit on paper test split.",
        },
    )


if __name__ == "__main__":
    main()
