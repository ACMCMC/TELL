from __future__ import annotations

import sys
from argparse import Namespace

from .base import DetectorWrapper
from detectors_bench.registry import DETECTORS_ROOT, vendor_path
from detectors_bench.schemas import Example, Prediction


class FastDetectGPTWrapper(DetectorWrapper):
    def __init__(self, cfg):
        super().__init__(cfg)
        root = vendor_path(cfg)
        sys.path.insert(0, str(root / "scripts"))
        from local_infer import FastDetectGPT  # type: ignore

        args = Namespace(
            sampling_model_name=cfg.get("sampling_model_name", "gpt-neo-2.7B"),
            scoring_model_name=cfg.get("scoring_model_name", "gpt-neo-2.7B"),
            device=cfg.get("device", "cuda"),
            cache_dir=str(DETECTORS_ROOT / cfg.get("cache_dir", "cache/huggingface")),
        )
        self.detector = FastDetectGPT(args)

    def predict_one(self, ex: Example) -> Prediction:
        prob, crit, ntokens = self.detector.compute_prob(ex.text)
        score = float(prob)
        return Prediction(
            id=ex.id,
            detector=self.name,
            score_ai=score,
            raw_score=float(crit),
            raw_label="machine-generated" if score >= 0.5 else "human-written",
            pred_builtin=int(score >= 0.5),
            features={"criterion": float(crit), "ntokens": int(ntokens)},
        )
