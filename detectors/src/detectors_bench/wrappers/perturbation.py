from __future__ import annotations

import random
import re
import sys

import numpy as np

from detectors_bench.io import batched
from detectors_bench.schemas import Example, Prediction, attach_example_metadata

from .base import DetectorWrapper, sigmoid
from .lm_features import LMFeatureExtractor


class T5Perturber:
    def __init__(self, model_name: str, device: str = "auto", seed: int = 0, max_length: int = 512) -> None:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.torch = torch
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.rng = random.Random(seed)
        self.max_length = max_length

    def _mask_words(self, text: str, pct_words_masked: float, span_length: int) -> str:
        words = text.split()
        if len(words) < max(4, span_length + 2):
            return text
        n_spans = max(1, int(round(len(words) * pct_words_masked / max(span_length, 1))))
        starts = sorted(self.rng.sample(range(0, max(1, len(words) - span_length)), k=min(n_spans, len(words))))
        masked = []
        i = 0
        span_id = 0
        start_set = set(starts)
        while i < len(words):
            if i in start_set:
                masked.append(f"<extra_id_{span_id}>")
                i += span_length
                span_id += 1
            else:
                masked.append(words[i])
                i += 1
        masked.append(f"<extra_id_{span_id}>")
        return " ".join(masked)

    @staticmethod
    def _clean_fill(fill: str) -> str:
        fill = re.sub(r"</?s>|<pad>", " ", fill)
        return re.sub(r"\s+", " ", fill).strip()

    def _extract_fills(self, generated: str) -> list[str]:
        parts = re.split(r"<extra_id_\d+>", generated)
        return [self._clean_fill(part) for part in parts[1:]]

    def _apply_fills(self, source: str, masked: str, generated: str) -> str:
        sentinels = re.findall(r"<extra_id_\d+>", masked)
        if not sentinels:
            return source
        fills = self._extract_fills(generated)
        if len(fills) < max(0, len(sentinels) - 1):
            return source

        pieces = re.split(r"<extra_id_\d+>", masked)
        rebuilt = [pieces[0]]
        for idx, _sentinel in enumerate(sentinels):
            if idx < len(sentinels) - 1:
                rebuilt.append(fills[idx])
            rebuilt.append(pieces[idx + 1])
        text = re.sub(r"\s+", " ", " ".join(part for part in rebuilt if part)).strip()
        return text or source

    def perturb(self, text: str, n: int, pct_words_masked: float = 0.3, span_length: int = 2) -> list[str]:
        return self.perturb_batch([text], n, pct_words_masked=pct_words_masked, span_length=span_length)[0]

    def perturb_batch(
        self,
        texts: list[str],
        n: int,
        pct_words_masked: float = 0.3,
        span_length: int = 2,
        generation_batch_size: int = 4,
    ) -> list[list[str]]:
        masked_records: list[tuple[int, str, str]] = []
        for doc_idx, text in enumerate(texts):
            for _ in range(n):
                masked_records.append((doc_idx, text, self._mask_words(text, pct_words_masked, span_length)))

        out: list[list[str]] = [[] for _ in texts]
        for chunk in batched(masked_records, generation_batch_size):
            encoded = self.tokenizer(
                [masked for _, _, masked in chunk],
                padding=True,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
            ).to(self.device)
            with self.torch.inference_mode():
                generated = self.model.generate(
                    **encoded,
                    do_sample=True,
                    top_p=0.96,
                    temperature=1.0,
                    max_length=self.max_length,
                )
            fills = self.tokenizer.batch_decode(generated, skip_special_tokens=False)
            for (doc_idx, source, masked), fill in zip(chunk, fills):
                out[doc_idx].append(self._apply_fills(source, masked, fill))
        return out


class PerturbationMetricWrapper(DetectorWrapper):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.method = cfg.require("method")
        self.extractor = LMFeatureExtractor(cfg.get("base_model_name", "gpt2-medium"), cfg.get("device", "auto"))
        self.perturber = T5Perturber(cfg.get("mask_filling_model_name", "t5-large"), cfg.get("device", "auto"))
        self.n_perturbations = int(cfg.get("n_perturbations", 10))
        self.pct_words_masked = float(cfg.get("pct_words_masked", 0.3))
        self.span_length = int(cfg.get("span_length", 2))
        self.batch_size = int(cfg.get("batch_size", 4))
        self.perturb_batch_size = int(cfg.get("perturb_batch_size", 4))

    def predict_one(self, ex: Example) -> Prediction:
        feats = self.extractor.features(ex.text)
        perturbed = self.perturber.perturb(
            ex.text,
            self.n_perturbations,
            pct_words_masked=self.pct_words_masked,
            span_length=self.span_length,
        )
        pert_feats = [self.extractor.features(text) for text in perturbed]
        if self.method == "detectgpt":
            vals = np.asarray([row["log_likelihood"] for row in pert_feats], dtype=float)
            std = float(np.std(vals)) if len(vals) > 1 else 1.0
            raw = (feats["log_likelihood"] - float(np.mean(vals))) / max(std, 1e-12)
        elif self.method == "npr":
            vals = np.asarray([row["log_rank"] for row in pert_feats], dtype=float)
            raw = float(np.mean(vals)) / max(feats["log_rank"], 1e-12)
        else:
            raise ValueError(f"Unknown perturbation method: {self.method}")
        score = sigmoid(float(raw))
        return Prediction(
            id=ex.id,
            detector=self.name,
            score_ai=float(score),
            raw_score=float(raw),
            raw_label="machine-generated" if score >= 0.5 else "human-written",
            pred_builtin=int(score >= 0.5),
            features={
                **feats,
                "method": self.method,
                "n_perturbations": self.n_perturbations,
                "perturbed_mean_log_likelihood": float(np.mean([r["log_likelihood"] for r in pert_feats])),
                "perturbed_mean_log_rank": float(np.mean([r["log_rank"] for r in pert_feats])),
            },
        )

    def predict_batch(self, examples: list[Example]) -> list[Prediction]:
        import time

        out: list[Prediction] = []
        for batch in batched(examples, self.batch_size):
            start = time.perf_counter()
            try:
                predictions = self._predict_batch_checked(batch)
                elapsed = (time.perf_counter() - start) / max(len(batch), 1)
                for pred, ex in zip(predictions, batch):
                    pred.runtime_s = elapsed
                    out.append(attach_example_metadata(pred, ex))
            except Exception as exc:  # noqa: BLE001
                # Fall back to per-row audit behavior so one pathological long text
                # does not drop an entire shard.
                print(f"[perturbation_metric] batch fallback for {self.name}: {exc!r}", file=sys.stderr)
                out.extend(super().predict_batch(batch))
        return out

    def _predict_batch_checked(self, examples: list[Example]) -> list[Prediction]:
        feats_batch = self.extractor.features_batch([ex.text for ex in examples])
        perturbed_by_doc = self.perturber.perturb_batch(
            [ex.text for ex in examples],
            self.n_perturbations,
            pct_words_masked=self.pct_words_masked,
            span_length=self.span_length,
            generation_batch_size=self.perturb_batch_size,
        )
        flat_perturbed = [text for rows in perturbed_by_doc for text in rows]
        flat_feats: list[dict[str, float]] = []
        for chunk in batched(flat_perturbed, self.batch_size * max(self.n_perturbations, 1)):
            flat_feats.extend(self.extractor.features_batch(chunk))

        out = []
        cursor = 0
        for ex, feats, perturbed in zip(examples, feats_batch, perturbed_by_doc):
            pert_feats = flat_feats[cursor : cursor + len(perturbed)]
            cursor += len(perturbed)
            if self.method == "detectgpt":
                vals = np.asarray([row["log_likelihood"] for row in pert_feats], dtype=float)
                std = float(np.std(vals)) if len(vals) > 1 else 1.0
                raw = (feats["log_likelihood"] - float(np.mean(vals))) / max(std, 1e-12)
            elif self.method == "npr":
                vals = np.asarray([row["log_rank"] for row in pert_feats], dtype=float)
                raw = float(np.mean(vals)) / max(feats["log_rank"], 1e-12)
            else:
                raise ValueError(f"Unknown perturbation method: {self.method}")
            score = sigmoid(float(raw))
            out.append(
                Prediction(
                    id=ex.id,
                    detector=self.name,
                    score_ai=float(score),
                    raw_score=float(raw),
                    raw_label="machine-generated" if score >= 0.5 else "human-written",
                    pred_builtin=int(score >= 0.5),
                    features={
                        **feats,
                        "method": self.method,
                        "n_perturbations": self.n_perturbations,
                        "perturbed_mean_log_likelihood": float(np.mean([r["log_likelihood"] for r in pert_feats])),
                        "perturbed_mean_log_rank": float(np.mean([r["log_rank"] for r in pert_feats])),
                    },
                )
            )
        return out
