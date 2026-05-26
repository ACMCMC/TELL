from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

from detectors_bench.io import batched
from detectors_bench.schemas import Example, Prediction, attach_example_metadata

from .base import DetectorWrapper, sigmoid
from .lm_features import MODEL_ALIASES


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _ngram_counts(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[i : i + n]) for i in range(0, max(0, len(tokens) - n + 1)))


def _overlap_ratio(target: Counter, pred: Counter) -> float:
    denom = sum(target.values())
    if denom <= 0:
        return 0.0
    inter = sum(min(count, pred.get(ngram, 0)) for ngram, count in target.items())
    return inter / denom


class DNAGPTWrapper(DetectorWrapper):
    def __init__(self, cfg):
        super().__init__(cfg)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = cfg.get("generator_model_name", "gpt2-medium")
        model_id = MODEL_ALIASES.get(self.model_name, self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(model_id).to(self.device)
        self.model.eval()
        self.n_regenerations = int(cfg.get("n_regenerations", 5))
        self.max_new_tokens = int(cfg.get("max_new_tokens", 128))
        self.max_ngram = int(cfg.get("ngram_order", 4))
        self.threshold = float(cfg.get("threshold", 0.00025))
        self.threshold_scale = float(cfg.get("threshold_scale", 0.0001))
        self.batch_size = int(cfg.get("batch_size", 1))

    def _generate(self, prefix: str) -> str:
        encoded = self.tokenizer(prefix, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with self.torch.inference_mode():
            output = self.model.generate(
                **encoded,
                do_sample=True,
                temperature=0.7,
                top_p=0.96,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
            )[0]
        return self.tokenizer.decode(output[encoded.input_ids.shape[1] :], skip_special_tokens=True)

    def _generate_batch(self, prefixes: list[str]) -> list[str]:
        encoded = self.tokenizer(
            prefixes,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)
        with self.torch.inference_mode():
            outputs = self.model.generate(
                **encoded,
                do_sample=True,
                temperature=0.7,
                top_p=0.96,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        prompt_len = encoded.input_ids.shape[1]
        return self.tokenizer.batch_decode(outputs[:, prompt_len:], skip_special_tokens=True)

    def _score(self, original_suffix: str, generated_suffix: str) -> dict[str, float]:
        target = _tokens(original_suffix)
        pred = _tokens(generated_suffix)
        ratios = {}
        weighted = 0.0
        denom = 0.0
        for n in range(1, self.max_ngram + 1):
            ratio = _overlap_ratio(_ngram_counts(target, n), _ngram_counts(pred, n))
            ratios[f"ngram_{n}_ratio"] = ratio
            if n >= 3:
                w = n * math.log(n)
                weighted += w * ratio
                denom += w
        ratios["dna_score"] = weighted / max(denom, 1e-12)
        return ratios

    def predict_one(self, ex: Example) -> Prediction:
        midpoint = max(1, len(ex.text) // 2)
        prefix = ex.text[:midpoint]
        suffix = ex.text[midpoint:]
        all_scores = [self._score(suffix, self._generate(prefix)) for _ in range(self.n_regenerations)]
        raw = float(np.mean([row["dna_score"] for row in all_scores]))
        score = sigmoid((raw - self.threshold) / max(self.threshold_scale, 1e-12))
        mean_features = {
            key: float(np.mean([row[key] for row in all_scores]))
            for key in all_scores[0]
        }
        return Prediction(
            id=ex.id,
            detector=self.name,
            score_ai=float(score),
            raw_score=raw,
            raw_label="machine-generated" if score >= 0.5 else "human-written",
            pred_builtin=int(score >= 0.5),
            features={**mean_features, "n_regenerations": self.n_regenerations, "threshold": self.threshold},
        )

    def predict_batch(self, examples: list[Example]) -> list[Prediction]:
        out: list[Prediction] = []
        for batch in batched(examples, self.batch_size):
            prefixes: list[str] = []
            suffixes: list[str] = []
            for ex in batch:
                midpoint = max(1, len(ex.text) // 2)
                prefixes.append(ex.text[:midpoint])
                suffixes.append(ex.text[midpoint:])

            per_doc_scores: list[list[dict[str, float]]] = [[] for _ in batch]
            try:
                for _ in range(self.n_regenerations):
                    generated = self._generate_batch(prefixes)
                    for scores, suffix, generation in zip(per_doc_scores, suffixes, generated):
                        scores.append(self._score(suffix, generation))

                for ex, scores in zip(batch, per_doc_scores):
                    raw = float(np.mean([row["dna_score"] for row in scores]))
                    score = sigmoid((raw - self.threshold) / max(self.threshold_scale, 1e-12))
                    mean_features = {
                        key: float(np.mean([row[key] for row in scores]))
                        for key in scores[0]
                    }
                    out.append(
                        attach_example_metadata(
                            Prediction(
                                id=ex.id,
                                detector=self.name,
                                score_ai=float(score),
                                raw_score=raw,
                                raw_label="machine-generated" if score >= 0.5 else "human-written",
                                pred_builtin=int(score >= 0.5),
                                features={
                                    **mean_features,
                                    "n_regenerations": self.n_regenerations,
                                    "threshold": self.threshold,
                                    "batched_generation": True,
                                },
                            ),
                            ex,
                        )
                    )
            except Exception as exc:  # noqa: BLE001
                for ex in batch:
                    out.append(
                        attach_example_metadata(
                            Prediction(id=ex.id, detector=self.name, score_ai=None, error=repr(exc)),
                            ex,
                        )
                    )
        return out
