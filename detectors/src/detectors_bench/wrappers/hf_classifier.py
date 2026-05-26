from __future__ import annotations

import re
import time
from typing import Any

import numpy as np

from detectors_bench.io import batched
from detectors_bench.schemas import Example, Prediction, attach_example_metadata

from .base import DetectorWrapper, require_optional


class HFClassifierWrapper(DetectorWrapper):
    def __init__(self, cfg):
        super().__init__(cfg)
        require_optional("transformers", "Install with `pip install -e '.[hf]'`.")
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = cfg.require("model_id")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_id).to(self.device)
        self.model.eval()
        self.max_length = int(cfg.get("max_length", 512))
        self.ai_label_regex = cfg.get("ai_label_regex")
        self.ai_label_index = cfg.get("ai_label_index")
        self.threshold = float(cfg.get("builtin_threshold", 0.5))
        self.label_map = dict(getattr(self.model.config, "id2label", {}) or {})

    def _ai_index(self, probs: np.ndarray) -> int:
        if self.ai_label_index is not None:
            return int(self.ai_label_index)
        labels = [str(self.label_map.get(i, f"LABEL_{i}")) for i in range(probs.shape[-1])]
        pattern = re.compile(str(self.ai_label_regex or "(fake|generated|machine|gpt|label_1)"), re.I)
        matches = [i for i, label in enumerate(labels) if pattern.search(label)]
        if matches:
            return matches[0]
        if len(labels) == 2:
            return 1
        raise ValueError(f"Cannot infer AI label index for {self.name}: {labels}")

    def predict_one(self, ex: Example) -> Prediction:
        return self.predict_batch([ex])[0]

    def predict_batch(self, examples: list[Example]) -> list[Prediction]:
        out: list[Prediction] = []
        batch_size = int(self.cfg.get("batch_size", 4))
        for batch in batched(examples, batch_size):
            start = time.perf_counter()
            try:
                encoded = self.tokenizer(
                    [ex.text for ex in batch],
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                with self.torch.inference_mode():
                    logits = self.model(**encoded).logits
                    probs = self.torch.softmax(logits, dim=-1).detach().cpu().numpy()
                ai_idx = self._ai_index(probs)
                labels = [str(self.label_map.get(i, f"LABEL_{i}")) for i in range(probs.shape[-1])]
                for ex, row in zip(batch, probs):
                    score = float(row[ai_idx])
                    pred = Prediction(
                        id=ex.id,
                        detector=self.name,
                        score_ai=score,
                        raw_score=score,
                        raw_label=labels[int(np.argmax(row))],
                        pred_builtin=int(score >= self.threshold),
                        features={
                            "labels": labels,
                            "probabilities": [float(x) for x in row],
                            "ai_label_index": int(ai_idx),
                        },
                        runtime_s=(time.perf_counter() - start) / max(len(batch), 1),
                    )
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
