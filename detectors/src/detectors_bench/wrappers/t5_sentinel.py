from __future__ import annotations

from pathlib import Path
import time
import urllib.request

import numpy as np

from detectors_bench.io import batched
from detectors_bench.registry import DETECTORS_ROOT
from detectors_bench.schemas import Example, Prediction, attach_example_metadata

from .base import DetectorWrapper, require_optional


class T5SentinelWrapper(DetectorWrapper):
    """Official T5Sentinel next-token classification surface."""

    DEFAULT_LABEL_TOKENS = {
        "Human": "<extra_id_0>",
        "ChatGPT": "<extra_id_1>",
        "PaLM": "<extra_id_2>",
        "LLaMA": "<extra_id_3>",
        "GPT2": "<extra_id_4>",
    }

    def __init__(self, cfg):
        super().__init__(cfg)
        require_optional("torch", "Install with `pip install -e '.[hf]'`.")
        from transformers import T5ForConditionalGeneration, T5TokenizerFast
        import torch

        self.torch = torch
        self.device = "cuda" if cfg.get("device", "auto") == "auto" and torch.cuda.is_available() else cfg.get("device", "cpu")
        if self.device == "auto":
            self.device = "cpu"
        self.model_name = cfg.get("base_model_name", "t5-small")
        self.tokenizer = T5TokenizerFast.from_pretrained(self.model_name, model_max_length=512)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name, return_dict=True)
        self.checkpoint = self._resolve_checkpoint()
        self._load_checkpoint(self.checkpoint)
        self.model.eval()
        self.model.to(self.device)
        self.max_length = int(cfg.get("max_length", 512))
        self.batch_size = int(cfg.get("batch_size", 16))
        self.threshold = float(cfg.get("builtin_threshold", 0.5))
        self.label_tokens = dict(cfg.get("label_tokens", self.DEFAULT_LABEL_TOKENS))
        self.label_token_ids = {
            label: self.tokenizer.convert_tokens_to_ids(token) for label, token in self.label_tokens.items()
        }
        self.human_label = str(cfg.get("human_label", "Human"))

    def _resolve_checkpoint(self) -> Path:
        raw_path = Path(str(self.cfg.require("checkpoint")))
        path = raw_path if raw_path.is_absolute() else DETECTORS_ROOT / raw_path
        if path.exists():
            return path
        url = self.cfg.get("checkpoint_url")
        if not url:
            raise FileNotFoundError(f"T5Sentinel checkpoint not found: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(str(url), path)  # noqa: S310 - official checkpoint URL from registry.
        return path

    def _load_checkpoint(self, path: Path) -> None:
        obj = self.torch.load(path, map_location="cpu")
        state = obj.get("model", obj) if isinstance(obj, dict) else obj
        if not isinstance(state, dict):
            raise ValueError(f"Unexpected T5Sentinel checkpoint payload at {path}")
        direct = self.model.load_state_dict(state, strict=False)
        if not direct.missing_keys:
            return
        stripped = {k.removeprefix("backbone."): v for k, v in state.items()}
        loaded = self.model.load_state_dict(stripped, strict=False)
        if loaded.missing_keys:
            raise RuntimeError(f"T5Sentinel checkpoint did not load cleanly; missing keys: {loaded.missing_keys[:5]}")

    def predict_one(self, ex: Example) -> Prediction:
        return self.predict_batch([ex])[0]

    def predict_batch(self, examples: list[Example]) -> list[Prediction]:
        out: list[Prediction] = []
        for batch in batched(examples, self.batch_size):
            start = time.perf_counter()
            try:
                encoded = self.tokenizer(
                    [ex.text for ex in batch],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                ).to(self.device)
                with self.torch.inference_mode():
                    generated = self.model.generate(
                        **encoded,
                        max_length=2,
                        output_scores=True,
                        return_dict_in_generate=True,
                    )
                    label_ids = [self.label_token_ids[label] for label in self.label_tokens]
                    filtered_logits = generated.scores[0][:, label_ids]
                    probs = self.torch.softmax(filtered_logits, dim=-1).detach().cpu().numpy()
                labels = list(self.label_tokens)
                human_idx = labels.index(self.human_label)
                elapsed = (time.perf_counter() - start) / max(len(batch), 1)
                for ex, row in zip(batch, probs):
                    human_prob = float(row[human_idx])
                    score_ai = float(1.0 - human_prob)
                    pred = Prediction(
                        id=ex.id,
                        detector=self.name,
                        score_ai=score_ai,
                        raw_score=score_ai,
                        raw_label=labels[int(np.argmax(row))],
                        pred_builtin=int(score_ai >= self.threshold),
                        features={
                            "checkpoint": str(self.checkpoint),
                            "base_model_name": self.model_name,
                            "labels": labels,
                            "probabilities": [float(x) for x in row],
                            "human_label": self.human_label,
                            "score_definition": "1 - P(Human)",
                        },
                        runtime_s=elapsed,
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
