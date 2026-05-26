from __future__ import annotations

from functools import lru_cache
import time

import numpy as np

from detectors_bench.io import batched
from .base import DetectorWrapper, sigmoid
from detectors_bench.schemas import Example, Prediction, attach_example_metadata


MODEL_ALIASES = {
    "gpt2": "gpt2",
    "gpt2-medium": "gpt2-medium",
    "gpt2-xl": "gpt2-xl",
    "gpt-neo-2.7B": "EleutherAI/gpt-neo-2.7B",
    "gpt-j-6B": "EleutherAI/gpt-j-6B",
    "opt-2.7b": "facebook/opt-2.7b",
}


@lru_cache(maxsize=8)
def load_causal_lm(model_name: str, device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = MODEL_ALIASES.get(model_name, model_name)
    dtype_kwargs = {}
    if device.startswith("cuda") and any(x in model_name for x in ("2.7B", "6B", "xl")):
        dtype_kwargs["torch_dtype"] = torch.float16
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(model_id, **dtype_kwargs).to(device)
    model.eval()
    return model, tokenizer


class LMFeatureExtractor:
    def __init__(self, model_name: str, device: str = "auto", max_length: int = 512) -> None:
        import torch

        self.torch = torch
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
        self.model, self.tokenizer = load_causal_lm(model_name, self.device)
        self.max_length = max_length

    def features(self, text: str) -> dict[str, float]:
        return self.features_batch([text])[0]

    def features_batch(self, texts: list[str]) -> list[dict[str, float]]:
        torch = self.torch
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            outputs = self.model(**encoded)
            logits = outputs.logits[:, :-1]
            labels = encoded.input_ids[:, 1:]
            mask = encoded.attention_mask[:, 1:].bool()
            denom = mask.sum(dim=1).clamp_min(1)
            log_probs = torch.log_softmax(logits, dim=-1)
            token_ll = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
            label_logits = logits.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
            # Equivalent to argsort rank, but avoids sorting the full vocabulary.
            ranks = (logits > label_logits.unsqueeze(-1)).sum(dim=-1).float() + 1.0
            log_ranks = torch.log(ranks)
            entropy_by_token = -(torch.softmax(logits, dim=-1) * log_probs).sum(-1)

            ll = (token_ll * mask).sum(dim=1) / denom
            log_rank = (log_ranks * mask).sum(dim=1) / denom
            entropy = (entropy_by_token * mask).sum(dim=1) / denom
            llm_deviation = (torch.square(log_ranks) * mask).sum(dim=1) / denom

        return [
            {
                "log_likelihood": float(a),
                "log_rank": float(b),
                "entropy": float(c),
                "llm_deviation": float(d),
            }
            for a, b, c, d in zip(
                ll.detach().cpu().tolist(),
                log_rank.detach().cpu().tolist(),
                entropy.detach().cpu().tolist(),
                llm_deviation.detach().cpu().tolist(),
            )
        ]


class DetectLLMLRRWrapper(DetectorWrapper):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.extractor = LMFeatureExtractor(cfg.get("base_model_name", "gpt2-medium"), cfg.get("device", "auto"))
        self.scale = float(cfg.get("score_scale", 6.0))
        self.batch_size = int(cfg.get("batch_size", 8))

    def predict_one(self, ex: Example) -> Prediction:
        return self.predict_batch([ex])[0]

    def predict_batch(self, examples: list[Example]) -> list[Prediction]:
        out: list[Prediction] = []
        for batch in batched(examples, self.batch_size):
            start = time.perf_counter()
            try:
                feats_batch = self.extractor.features_batch([ex.text for ex in batch])
                elapsed = (time.perf_counter() - start) / max(len(batch), 1)
                for ex, feats in zip(batch, feats_batch):
                    pred = self._prediction_from_features(ex, feats)
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

    def _prediction_from_features(self, ex: Example, feats: dict[str, float]) -> Prediction:
        raw = -feats["log_likelihood"] / max(feats["log_rank"], 1e-12)
        score = sigmoid(raw / self.scale)
        return Prediction(
            id=ex.id,
            detector=self.name,
            score_ai=float(score),
            raw_score=float(raw),
            raw_label="machine-generated" if score >= 0.5 else "human-written",
            pred_builtin=int(score >= 0.5),
            features=feats,
        )


class LogRankWrapper(DetectorWrapper):
    """Ippolito et al. log-rank detector using the reference causal LM."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.extractor = LMFeatureExtractor(
            cfg.get("base_model_name", "gpt2-medium"),
            cfg.get("device", "auto"),
            max_length=int(cfg.get("max_length", 1024)),
        )
        # Lower mean log-rank is more model-like, hence more AI-like.
        self.threshold = float(cfg.get("raw_threshold", 2.75))
        self.scale = float(cfg.get("score_scale", 1.0))
        self.batch_size = int(cfg.get("batch_size", 8))

    def predict_one(self, ex: Example) -> Prediction:
        return self.predict_batch([ex])[0]

    def predict_batch(self, examples: list[Example]) -> list[Prediction]:
        out: list[Prediction] = []
        for batch in batched(examples, self.batch_size):
            start = time.perf_counter()
            try:
                feats_batch = self.extractor.features_batch([ex.text for ex in batch])
                elapsed = (time.perf_counter() - start) / max(len(batch), 1)
                for ex, feats in zip(batch, feats_batch):
                    raw = float(feats["log_rank"])
                    score = sigmoid((self.threshold - raw) / max(self.scale, 1e-12))
                    pred = Prediction(
                        id=ex.id,
                        detector=self.name,
                        score_ai=float(score),
                        raw_score=raw,
                        raw_label="machine-generated" if score >= 0.5 else "human-written",
                        pred_builtin=int(score >= 0.5),
                        features={
                            **feats,
                            "base_model_name": self.cfg.get("base_model_name", "gpt2-medium"),
                            "raw_threshold": self.threshold,
                            "score_direction": "lower_log_rank_is_more_ai",
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
