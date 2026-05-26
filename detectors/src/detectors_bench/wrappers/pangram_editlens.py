from __future__ import annotations

import glob
import os
import re
import time

import numpy as np

from detectors_bench.io import batched
from detectors_bench.schemas import Example, Prediction, attach_example_metadata

from .base import DetectorWrapper, require_optional


BOILERPLATE_STARTS = [
    "Sure",
    "Here",
    "Abstract",
    "Title",
    "I'm happy to help",
    "Certainly",
]


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def count_words(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


class NormedLinear:
    """Official EditLens score head: LayerNorm followed by a bias-free linear layer."""

    def __new__(cls, hidden_size: int, num_labels: int, device=None, dtype=None):
        import torch

        class _NormedLinear(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.norm = torch.nn.LayerNorm(hidden_size, device=device, dtype=dtype)
                self.linear = torch.nn.Linear(hidden_size, num_labels, bias=False, device=device, dtype=dtype)

            def forward(self, x):
                return self.linear(self.norm(x))

        return _NormedLinear()


class PangramEditLensWrapper(DetectorWrapper):
    def __init__(self, cfg):
        super().__init__(cfg)
        for package in ["torch", "transformers", "peft", "safetensors"]:
            require_optional(package, "Install with `pip install -e '.[pangram]'`.")
        try:
            import emoji
        except ImportError as exc:
            raise RuntimeError("Missing optional dependency 'emoji'. Install with `pip install -e '.[pangram]'`.") from exc

        import torch
        from huggingface_hub import hf_hub_download
        from peft import PeftModel
        from safetensors import safe_open
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig

        self.emoji = emoji
        self.torch = torch
        self.hf_hub_download = hf_hub_download
        self.safe_open = safe_open
        self.PeftModel = PeftModel
        self.AutoModelForSequenceClassification = AutoModelForSequenceClassification
        self.AutoTokenizer = AutoTokenizer
        self.BitsAndBytesConfig = BitsAndBytesConfig

        self.checkpoint = cfg.require("checkpoint")
        self.base_model_name = cfg.require("base_model_name")
        self.hf_token_env = cfg.get("hf_token_env", "HF_TOKEN")
        self.max_length = int(cfg.get("max_length", 1024))
        self.batch_size = int(cfg.get("batch_size", 4))
        self.min_words = int(cfg.get("min_words", 0))
        self.threshold = float(cfg.get("builtin_threshold", 0.5))
        self.load_in_4bit = bool(cfg.get("load_in_4bit", True))
        self.bnb_4bit_quant_type = cfg.get("bnb_4bit_quant_type", "nf4")
        self.bnb_4bit_compute_dtype = cfg.get("bnb_4bit_compute_dtype", "bfloat16")
        self.device_map = cfg.get("device_map")
        self._loaded = False

    def _token(self) -> str | None:
        token = os.environ.get(str(self.hf_token_env))
        if token:
            return token
        try:
            from huggingface_hub import get_token

            return get_token()
        except Exception:
            return None

    def _download(self, filename: str) -> str:
        token = self._token()
        return self.hf_hub_download(self.checkpoint, filename, token=token)

    def _infer_n_buckets(self) -> int:
        if os.path.isdir(self.checkpoint):
            safetensor_files = glob.glob(os.path.join(self.checkpoint, "*.safetensors"))
            safetensor_path = safetensor_files[0] if safetensor_files else None
        else:
            safetensor_path = self._download("adapter_model.safetensors")

        if safetensor_path and os.path.exists(safetensor_path):
            with self.safe_open(safetensor_path, framework="pt") as f:
                for key in f.keys():
                    if "score" in key and "linear.weight" in key:
                        return int(f.get_tensor(key).shape[0])
        raise ValueError(f"Could not infer Pangram EditLens bucket count from {self.checkpoint}.")

    def _clean_text(self, text: str) -> str:
        text = self.emoji.demojize(text)
        if "</think>" in text:
            text = text.split("</think>", 1)[1].strip()
        paragraphs = [p for p in text.split("\n") if p.strip()]
        if paragraphs:
            first = re.sub(r"^[^a-zA-Z0-9]*", "", paragraphs[0])
            first = self.emoji.replace_emoji(first, "")
            if any(first.startswith(phrase) for phrase in BOILERPLATE_STARTS) and len(paragraphs) > 1:
                text = "\n".join(paragraphs[1:])
        text = text.lower()
        return normalize_whitespace(text)

    def _load(self) -> None:
        if self._loaded:
            return

        token = self._token()
        self.n_buckets = self._infer_n_buckets()
        self.tokenizer = self.AutoTokenizer.from_pretrained(self.base_model_name, token=token)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        quantization_config = None
        if self.load_in_4bit:
            compute_dtype = getattr(self.torch, str(self.bnb_4bit_compute_dtype))
            quantization_config = self.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=compute_dtype,
            )

        model_kwargs = {
            "num_labels": self.n_buckets,
            "token": token,
            "quantization_config": quantization_config,
        }
        if self.device_map is not None:
            model_kwargs["device_map"] = self.device_map
        base_model = self.AutoModelForSequenceClassification.from_pretrained(self.base_model_name, **model_kwargs)
        base_model.config.pad_token_id = self.tokenizer.pad_token_id
        if hasattr(base_model, "score") and isinstance(base_model.score, self.torch.nn.Linear):
            hidden_size = base_model.config.hidden_size
            param = next(base_model.parameters())
            base_model.score = NormedLinear(hidden_size, self.n_buckets, device=param.device, dtype=param.dtype)

        self.model = self.PeftModel.from_pretrained(base_model, self.checkpoint, token=token)
        self.model.eval()
        self.bucket_labels = np.arange(self.n_buckets, dtype=np.float32)
        self._loaded = True

    def predict_one(self, ex: Example) -> Prediction:
        return self.predict_batch([ex])[0]

    def predict_batch(self, examples: list[Example]) -> list[Prediction]:
        self._load()
        out: list[Prediction] = []
        for batch in batched(examples, self.batch_size):
            start = time.perf_counter()
            try:
                cleaned = [self._clean_text(ex.text) for ex in batch]
                if self.min_words > 0:
                    for ex, text in zip(batch, cleaned):
                        if count_words(text) < self.min_words:
                            raise ValueError(f"Example {ex.id!r} has fewer than min_words={self.min_words}.")
                encoded = self.tokenizer(
                    cleaned,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                model_device = next(self.model.parameters()).device
                encoded = {k: v.to(model_device) for k, v in encoded.items()}
                with self.torch.inference_mode():
                    logits = self.model(**encoded).logits.float()
                    probs = self.torch.softmax(logits, dim=-1).detach().cpu().numpy()

                for ex, text, row in zip(batch, cleaned, probs):
                    bucket = int(np.argmax(row))
                    score = float((row @ self.bucket_labels) / max(self.n_buckets - 1, 1))
                    pred = Prediction(
                        id=ex.id,
                        detector=self.name,
                        score_ai=score,
                        raw_score=score,
                        raw_label=f"bucket_{bucket}",
                        pred_builtin=int(score >= self.threshold),
                        features={
                            "checkpoint": self.checkpoint,
                            "base_model_name": self.base_model_name,
                            "n_buckets": self.n_buckets,
                            "bucket_pred": bucket,
                            "bucket_probabilities": [float(x) for x in row],
                            "cleaned_word_count": count_words(text),
                            "max_length": self.max_length,
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
