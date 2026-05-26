from __future__ import annotations

import math
import time

from .base import DetectorWrapper, sigmoid
from detectors_bench.registry import vendor_path
from detectors_bench.schemas import Example, Prediction, attach_example_metadata


class BinocularsWrapper(DetectorWrapper):
    def __init__(self, cfg):
        super().__init__(cfg)
        root = vendor_path(cfg)
        import sys

        self._patch_transformers_legacy_head_mask()

        sys.path.insert(0, str(root))
        from binoculars import Binoculars  # type: ignore
        import binoculars.detector as detector_module  # type: ignore

        self._patch_binoculars_metrics(detector_module)

        self.detector = Binoculars(
            observer_name_or_path=cfg.get("observer_name_or_path", "tiiuae/falcon-7b"),
            performer_name_or_path=cfg.get("performer_name_or_path", "tiiuae/falcon-7b-instruct"),
            max_token_observed=int(cfg.get("max_token_observed", 512)),
            mode=cfg.get("mode", "low-fpr"),
        )
        self._patch_falcon_negative_indexing()
        self.threshold = float(self.detector.threshold)
        self.scale = float(cfg.get("threshold_scale", 0.05))

    @staticmethod
    def _patch_transformers_legacy_head_mask() -> None:
        try:
            from transformers.modeling_utils import ModuleUtilsMixin, PreTrainedModel
        except Exception:
            return

        if hasattr(ModuleUtilsMixin, "get_head_mask"):
            return

        def get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
            if head_mask is None:
                return [None] * num_hidden_layers
            return head_mask

        ModuleUtilsMixin.get_head_mask = get_head_mask  # type: ignore[attr-defined]
        PreTrainedModel.get_head_mask = get_head_mask  # type: ignore[attr-defined]

    @staticmethod
    def _patch_binoculars_metrics(detector_module) -> None:
        detector_module.perplexity = BinocularsWrapper._stable_perplexity
        detector_module.entropy = BinocularsWrapper._stable_entropy

    def _patch_falcon_negative_indexing(self) -> None:
        """Avoid CUDA gather asserts from Falcon's legacy negative advanced indexing.

        Some PyTorch/CUDA builds used on the server reject `qkv[..., [-2], :]`
        inside Falcon attention even though the equivalent slice is valid. This
        keeps the official Falcon weights and Binoculars score unchanged while
        replacing only the indexing expression with `-2:-1` / `-1:`.
        """

        import types

        def patch_model(model) -> None:
            for module in model.modules():
                original = getattr(module, "_split_heads", None)
                if original is None or getattr(module, "_rl_detector_split_heads_patched", False):
                    continue

                def split_heads(attn, fused_qkv, _original=original):
                    if not getattr(attn, "new_decoder_architecture", False):
                        return _original(fused_qkv)

                    import torch

                    batch, seq_len, _ = fused_qkv.shape
                    qkv = fused_qkv.view(
                        batch,
                        seq_len,
                        -1,
                        attn.num_heads // attn.num_kv_heads + 2,
                        attn.head_dim,
                    )
                    query = qkv[:, :, :, :-2]
                    key = qkv[:, :, :, -2:-1]
                    value = qkv[:, :, :, -1:]
                    key = torch.broadcast_to(key, query.shape)
                    value = torch.broadcast_to(value, query.shape)
                    query, key, value = [x.flatten(2, 3) for x in (query, key, value)]
                    return query, key, value

                module._split_heads = types.MethodType(split_heads, module)
                module._rl_detector_split_heads_patched = True

        patch_model(self.detector.observer_model)
        patch_model(self.detector.performer_model)

    @staticmethod
    def _stable_perplexity(encoding, logits, median=False, temperature=1.0):
        import numpy as np
        import torch

        shifted_logits = logits[..., :-1, :].contiguous().float() / temperature
        shifted_labels = encoding.input_ids[..., 1:].contiguous()
        shifted_attention_mask = encoding.attention_mask[..., 1:].contiguous()
        if int(shifted_labels.max().item()) >= int(shifted_logits.shape[-1]) or int(shifted_labels.min().item()) < 0:
            raise ValueError(
                "Binoculars labels are outside the performer vocabulary "
                f"(min={int(shifted_labels.min().item())}, max={int(shifted_labels.max().item())}, "
                f"vocab={int(shifted_logits.shape[-1])})."
            )
        target_logits = torch.gather(shifted_logits, -1, shifted_labels.unsqueeze(-1)).squeeze(-1)
        ce = torch.logsumexp(shifted_logits, dim=-1) - target_logits

        if median:
            ce_nan = ce.masked_fill(~shifted_attention_mask.bool(), float("nan"))
            return np.nanmedian(ce_nan.cpu().float().numpy(), 1)

        token_count = shifted_attention_mask.sum(1).clamp_min(1)
        ppl = (ce * shifted_attention_mask).sum(1)
        return (ppl / token_count).to("cpu").float().numpy()

    @staticmethod
    def _stable_entropy(
        p_logits,
        q_logits,
        encoding,
        pad_token_id,
        median=False,
        sample_p=False,
        temperature=1.0,
    ):
        import numpy as np
        import torch

        vocab_size = p_logits.shape[-1]
        total_tokens_available = q_logits.shape[-2]
        p_scores = p_logits.float() / temperature
        q_scores = q_logits.float() / temperature

        p_proba = torch.softmax(p_scores, dim=-1)
        if sample_p:
            p_proba = torch.multinomial(p_proba.view(-1, vocab_size), replacement=True, num_samples=1).view(-1)

        ce = torch.logsumexp(q_scores, dim=-1) - (p_proba * q_scores).sum(dim=-1)
        padding_mask = (encoding.input_ids != pad_token_id).bool()

        if median:
            ce_nan = ce.masked_fill(~padding_mask, float("nan"))
            return np.nanmedian(ce_nan.cpu().float().numpy(), 1)

        ce = ce.masked_fill(~padding_mask, 0.0)
        ce = torch.nan_to_num(ce, nan=0.0, posinf=0.0, neginf=0.0)
        token_count = padding_mask.sum(1).clamp_min(1)
        return (ce.sum(1) / token_count).to("cpu").float().numpy()

    def predict_one(self, ex: Example) -> Prediction:
        raw = self._compute_raw_scores([ex.text])[0]
        return self._prediction_from_raw(ex, raw)

    def predict_batch(self, examples: list[Example]) -> list[Prediction]:
        out = []
        batch_size = max(1, int(self.cfg.get("batch_size", 4)))
        for start_idx in range(0, len(examples), batch_size):
            batch = examples[start_idx : start_idx + batch_size]
            start = time.perf_counter()
            try:
                raw_scores = self._compute_raw_scores([ex.text for ex in batch])
                predictions = [
                    self._prediction_from_raw(ex, float(raw))
                    for ex, raw in zip(batch, raw_scores, strict=True)
                ]
            except Exception as exc:  # noqa: BLE001 - keep per-example audit rows.
                if self._is_fatal_cuda_error(exc):
                    raise
                predictions = [
                    Prediction(id=ex.id, detector=self.name, score_ai=None, error=repr(exc))
                    for ex in batch
                ]
            elapsed = time.perf_counter() - start
            per_example = elapsed / max(len(batch), 1)
            for pred, ex in zip(predictions, batch, strict=True):
                pred.runtime_s = per_example
                out.append(attach_example_metadata(pred, ex))
        return out

    def _compute_raw_scores(self, texts: list[str]) -> list[float]:
        import numpy as np

        encodings = self._tokenize_on_cpu(texts)
        self._validate_token_ids(encodings)
        observer_logits, performer_logits = self._get_logits_without_mutating_encodings(encodings)
        ppl = self._stable_perplexity(
            self._clone_encodings_to(encodings, performer_logits.device),
            performer_logits,
        )
        observer_device = self.detector.observer_model.device
        x_ppl = self._stable_entropy(
            observer_logits.to(observer_device),
            performer_logits.to(observer_device),
            self._clone_encodings_to(encodings, observer_device),
            self.detector.tokenizer.pad_token_id,
        )
        fallback_denominator = float(np.log(performer_logits.shape[-1]))
        x_ppl = np.where(np.isfinite(x_ppl) & (x_ppl > 1e-12), x_ppl, fallback_denominator)
        return [float(x) for x in (ppl / x_ppl).tolist()]

    def _get_logits_without_mutating_encodings(self, encodings):
        import torch

        observer_device = self.detector.observer_model.device
        performer_device = self.detector.performer_model.device
        observer_inputs = self._clone_encodings_to(encodings, observer_device)
        performer_inputs = self._clone_encodings_to(encodings, performer_device)
        with torch.inference_mode():
            observer_logits = self.detector.observer_model(**observer_inputs).logits
            performer_logits = self.detector.performer_model(**performer_inputs).logits
        if observer_device.type == "cuda" or str(observer_device).startswith("cuda"):
            torch.cuda.synchronize(observer_device)
        if performer_device.type == "cuda" or str(performer_device).startswith("cuda"):
            torch.cuda.synchronize(performer_device)
        return observer_logits, performer_logits

    def _tokenize_on_cpu(self, texts: list[str]):
        batch_size = len(texts)
        return self.detector.tokenizer(
            texts,
            return_tensors="pt",
            padding="longest" if batch_size > 1 else False,
            truncation=True,
            max_length=self.detector.max_token_observed,
            return_token_type_ids=False,
        )

    @staticmethod
    def _clone_encodings_to(encodings, device):
        from transformers import BatchEncoding

        return BatchEncoding(
            {
                key: value.detach().clone().to(device)
                for key, value in encodings.items()
            }
        )

    def _validate_token_ids(self, encodings) -> None:
        max_token_id = int(encodings.input_ids.max().item())
        observer_vocab = int(self.detector.observer_model.get_input_embeddings().num_embeddings)
        performer_vocab = int(self.detector.performer_model.get_input_embeddings().num_embeddings)
        vocab_limit = min(observer_vocab, performer_vocab)
        if max_token_id >= vocab_limit:
            raise ValueError(
                "Binoculars tokenizer produced token ids outside the observer/performer vocabulary "
                f"(max_token_id={max_token_id}, observer_vocab={observer_vocab}, performer_vocab={performer_vocab}). "
                "This would otherwise trigger a CUDA device-side assert; check that the official Falcon tokenizer "
                "matches both pinned Falcon checkpoints."
            )

    @staticmethod
    def _is_fatal_cuda_error(exc: Exception) -> bool:
        msg = repr(exc).lower()
        return (
            "device-side assert" in msg
            or "cuda error" in msg
            or "acceleratorerror" in msg
            or "illegal memory access" in msg
        )

    def _prediction_from_raw(self, ex: Example, raw: float) -> Prediction:
        if not math.isfinite(raw):
            raise RuntimeError("Binoculars returned a non-finite score; input may be too short for stable perplexity.")
        score_ai = sigmoid((self.threshold - raw) / self.scale)
        return Prediction(
            id=ex.id,
            detector=self.name,
            score_ai=float(score_ai),
            raw_score=raw,
            raw_label="Most likely AI-generated" if raw < self.threshold else "Most likely human-generated",
            pred_builtin=int(raw < self.threshold),
            features={"threshold": self.threshold, "mode": self.cfg.get("mode", "low-fpr")},
        )
