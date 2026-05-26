from __future__ import annotations

import hashlib
import sys
import time

import numpy as np

from detectors_bench.io import batched
from detectors_bench.registry import vendor_path
from detectors_bench.schemas import Example, Prediction, attach_example_metadata

from .base import DetectorWrapper, sigmoid


class PHDWrapper(DetectorWrapper):
    """Persistent homology dimension detector from Tulchinskii et al."""

    def __init__(self, cfg):
        super().__init__(cfg)
        import torch
        from transformers import AutoModel, AutoTokenizer

        root = vendor_path(cfg)
        sys.path.insert(0, str(root))
        from IntrinsicDim import PHD  # type: ignore

        self.PHD = PHD
        self.torch = torch
        self.device = "cuda" if cfg.get("device", "auto") == "auto" and torch.cuda.is_available() else cfg.get("device", "cpu")
        if self.device == "auto":
            self.device = "cpu"
        self.model_id = cfg.get("model_id", "FacebookAI/roberta-base")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        dtype_kwargs = {}
        if str(self.device).startswith("cuda"):
            dtype_kwargs["torch_dtype"] = torch.float16
        self.model = AutoModel.from_pretrained(self.model_id, **dtype_kwargs).to(self.device)
        self.model.eval()
        self.batch_size = int(cfg.get("batch_size", 1))
        self.max_length = int(cfg.get("max_length", 512))
        self.min_subsample = int(cfg.get("min_subsample", 40))
        self.intermediate_points = int(cfg.get("intermediate_points", 7))
        self.alpha = float(cfg.get("alpha", 1.0))
        self.n_points = int(cfg.get("n_points", 9))
        self.metric = str(cfg.get("metric", "euclidean"))
        self.threshold = float(cfg.get("raw_threshold", 8.5))
        self.scale = float(cfg.get("score_scale", 1.0))
        self.seed = int(cfg.get("seed", 0))

    def predict_one(self, ex: Example) -> Prediction:
        return self.predict_batch([ex])[0]

    def predict_batch(self, examples: list[Example]) -> list[Prediction]:
        out: list[Prediction] = []
        for batch in batched(examples, self.batch_size):
            batch_start = time.perf_counter()
            try:
                preds = self._predict_batch_inner(batch)
            except Exception:
                preds = []
                for ex in batch:
                    start = time.perf_counter()
                    try:
                        pred = self._predict_one_inner(ex)
                    except Exception as exc:  # noqa: BLE001
                        pred = Prediction(id=ex.id, detector=self.name, score_ai=None, error=repr(exc))
                    pred.runtime_s = time.perf_counter() - start
                    preds.append(pred)
            batch_runtime = (time.perf_counter() - batch_start) / max(len(batch), 1)
            for ex, pred in zip(batch, preds):
                if pred.runtime_s is None:
                    pred.runtime_s = batch_runtime
                out.append(attach_example_metadata(pred, ex))
        return out

    def _predict_one_inner(self, ex: Example) -> Prediction:
        clean = ex.text.replace("\n", " ").replace("  ", " ")
        encoded = self.tokenizer(
            clean,
            truncation=True,
            max_length=min(self.max_length, getattr(self.model.config, "max_position_embeddings", self.max_length)),
            return_tensors="pt",
        ).to(self.device)
        with self.torch.inference_mode():
            hidden = self.model(**encoded).last_hidden_state[0]

        # Drop CLS/SEP-equivalent special tokens, matching the paper and reference implementation.
        points = hidden.detach().float().cpu().numpy()[1:-1]
        return self._prediction_from_points(ex, points)

    def _predict_batch_inner(self, examples: list[Example]) -> list[Prediction]:
        clean = [ex.text.replace("\n", " ").replace("  ", " ") for ex in examples]
        encoded = self.tokenizer(
            clean,
            padding=True,
            truncation=True,
            max_length=min(self.max_length, getattr(self.model.config, "max_position_embeddings", self.max_length)),
            return_tensors="pt",
        ).to(self.device)
        with self.torch.inference_mode():
            hidden = self.model(**encoded).last_hidden_state
        mask = encoded.attention_mask.detach().cpu().numpy().astype(bool)

        preds: list[Prediction] = []
        for row_idx, ex in enumerate(examples):
            valid = hidden[row_idx].detach().float().cpu().numpy()[mask[row_idx]]
            points = valid[1:-1]
            try:
                preds.append(self._prediction_from_points(ex, points))
            except Exception as exc:  # noqa: BLE001
                preds.append(Prediction(id=ex.id, detector=self.name, score_ai=None, error=repr(exc)))
        return preds

    def _prediction_from_points(self, ex: Example, points: np.ndarray) -> Prediction:
        n_points_available = int(points.shape[0])
        if n_points_available < self.min_subsample + 2:
            raise ValueError(
                f"PHD requires at least {self.min_subsample + 2} usable token embeddings; got {n_points_available}"
            )
        step = (n_points_available - self.min_subsample) // max(self.intermediate_points, 1)
        step = max(step, 1)
        test_sizes = list(range(self.min_subsample, n_points_available + 1, step))
        if len(test_sizes) < 2:
            raise ValueError(
                "PHD requires at least two subsample sizes for the log-log slope; "
                f"got min_subsample={self.min_subsample}, usable_token_embeddings={n_points_available}"
            )
        max_points_exclusive = test_sizes[-1] + step

        seed = self.seed + int(hashlib.sha256(ex.id.encode("utf-8")).hexdigest()[:8], 16)
        np.random.seed(seed % (2**32))
        solver = self.PHD(alpha=self.alpha, metric=self.metric, n_points=self.n_points)
        raw_dim = float(
            solver.fit_transform(
                points,
                min_points=self.min_subsample,
                max_points=max_points_exclusive,
                point_jump=step,
            )
        )
        score = sigmoid((self.threshold - raw_dim) / max(self.scale, 1e-12))
        return Prediction(
            id=ex.id,
            detector=self.name,
            score_ai=float(score),
            raw_score=raw_dim,
            raw_label="machine-generated" if score >= 0.5 else "human-written",
            pred_builtin=int(score >= 0.5),
            features={
                "phd_dimension": raw_dim,
                "model_id": self.model_id,
                "min_subsample": self.min_subsample,
                "intermediate_points": self.intermediate_points,
                "alpha": self.alpha,
                "n_points": self.n_points,
                "metric": self.metric,
                "raw_threshold": self.threshold,
                "score_direction": "lower_phd_dimension_is_more_ai",
                "usable_token_embeddings": n_points_available,
                "subsample_sizes": test_sizes,
            },
        )
