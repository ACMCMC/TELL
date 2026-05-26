from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod

from detectors_bench.registry import DetectorConfig
from detectors_bench.schemas import Example, Prediction, attach_example_metadata


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


class DetectorWrapper(ABC):
    def __init__(self, cfg: DetectorConfig) -> None:
        self.cfg = cfg
        self.name = cfg.name

    @abstractmethod
    def predict_one(self, ex: Example) -> Prediction:
        raise NotImplementedError

    def predict_batch(self, examples: list[Example]) -> list[Prediction]:
        out = []
        for ex in examples:
            start = time.perf_counter()
            try:
                pred = self.predict_one(ex)
            except Exception as exc:  # noqa: BLE001 - per-example audit logging is intentional.
                pred = Prediction(id=ex.id, detector=self.name, score_ai=None, error=repr(exc))
            pred.runtime_s = time.perf_counter() - start
            out.append(attach_example_metadata(pred, ex))
        return out


def require_optional(package: str, install_hint: str) -> None:
    try:
        __import__(package)
    except ImportError as exc:
        raise RuntimeError(f"Missing optional dependency {package!r}. {install_hint}") from exc
