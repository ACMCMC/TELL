from __future__ import annotations

from pathlib import Path
import time

import joblib
import numpy as np

from .base import DetectorWrapper
from detectors_bench.io import batched
from .lm_features import LMFeatureExtractor
from detectors_bench.registry import DETECTORS_ROOT
from detectors_bench.schemas import Example, Prediction, attach_example_metadata


class MFDWrapper(DetectorWrapper):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.feature_names = list(cfg.get("feature_names", ["log_likelihood", "log_rank", "entropy", "llm_deviation"]))
        self.extractor = LMFeatureExtractor(cfg.get("base_model_name", "gpt2"), cfg.get("device", "auto"))
        model_path = Path(cfg.get("model_path", "results/models/mfd.joblib"))
        self.model_path = model_path if model_path.is_absolute() else DETECTORS_ROOT / model_path
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"MFD requires a validation-trained model at {self.model_path}. "
                "Run `python -m detectors_bench.fit_mfd --train <train.jsonl> --output <path>` first."
            )
        self.pipeline = joblib.load(self.model_path)
        self.batch_size = int(cfg.get("batch_size", 16))

    def feature_vector(self, text: str) -> tuple[np.ndarray, dict[str, float]]:
        feats = self.extractor.features(text)
        return np.asarray([[feats[name] for name in self.feature_names]], dtype=float), feats

    def predict_one(self, ex: Example) -> Prediction:
        return self.predict_batch([ex])[0]

    def predict_batch(self, examples: list[Example]) -> list[Prediction]:
        out: list[Prediction] = []
        for batch in batched(examples, self.batch_size):
            start = time.perf_counter()
            try:
                feats_batch = self.extractor.features_batch([ex.text for ex in batch])
                x = np.asarray(
                    [[feats[name] for name in self.feature_names] for feats in feats_batch],
                    dtype=float,
                )
                scores = self.pipeline.predict_proba(x)[:, 1]
                elapsed = (time.perf_counter() - start) / max(len(batch), 1)
                for ex, feats, score in zip(batch, feats_batch, scores):
                    pred = self._prediction_from_score(ex, float(score), feats)
                    pred.runtime_s = elapsed
                    out.append(attach_example_metadata(pred, ex))
            except Exception as exc:  # noqa: BLE001
                for ex in batch:
                    out.append(
                        attach_example_metadata(
                            Prediction(id=ex.id, detector=self.name, score_ai=None, error=repr(exc)),
                            ex,
                        )
                    )
        return out

    def _prediction_from_score(self, ex: Example, score: float, feats: dict[str, float]) -> Prediction:
        return Prediction(
            id=ex.id,
            detector=self.name,
            score_ai=score,
            raw_score=score,
            raw_label="machine-generated" if score >= 0.5 else "human-written",
            pred_builtin=int(score >= 0.5),
            features=feats,
        )
