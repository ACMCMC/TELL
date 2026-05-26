from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from detectors_bench.io import batched
from detectors_bench.schemas import Example, Prediction, attach_example_metadata

from .base import DetectorWrapper, require_optional


class MELDWrapper(DetectorWrapper):
    """MELD released-checkpoint inference using the main AI/Human head."""

    def __init__(self, cfg):
        super().__init__(cfg)
        require_optional("torch", "Install with `pip install -e '.[hf]'`.")
        require_optional("safetensors", "Install with `pip install -e '.[hf]'`.")
        import torch
        import torch.nn as nn
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.nn = nn
        self.device = "cuda" if cfg.get("device", "auto") == "auto" and torch.cuda.is_available() else cfg.get("device", "cpu")
        if self.device == "auto":
            self.device = "cpu"
        self.model_id = cfg.get("model_id", "anon-review-meld-2026/meld")
        self.revision = cfg.get("revision")
        cfg_path = hf_hub_download(self.model_id, "meld_config.json", revision=self.revision)
        self.meld_cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
        self.max_length = int(cfg.get("max_length", self.meld_cfg.get("max_length", 1024)))
        self.stride = int(cfg.get("stride", 512))
        self.batch_size = int(cfg.get("batch_size", 4))
        self.threshold = float(cfg.get("builtin_threshold", 0.5))
        self.backbone_id = str(self.meld_cfg["backbone"])
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, revision=self.revision)

        model = _MELDDetector(
            AutoModel,
            nn,
            backbone=self.backbone_id,
            n_generators=int(self.meld_cfg["n_generators"]),
            n_attacks=int(self.meld_cfg["n_attacks"]),
            n_domains=int(self.meld_cfg["n_domains"]),
            num_labels=int(self.meld_cfg.get("num_labels", 2)),
            dropout=float(self.meld_cfg.get("dropout", 0.1)),
        )
        weights_path = hf_hub_download(self.model_id, "model.safetensors", revision=self.revision)
        state = load_file(weights_path, device="cpu")
        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError:
            stripped = {k.removeprefix("module."): v for k, v in state.items()}
            model.load_state_dict(stripped, strict=True)
        self.model = model.to(self.device)
        self.model.eval()

    def predict_one(self, ex: Example) -> Prediction:
        return self.predict_batch([ex])[0]

    def predict_batch(self, examples: list[Example]) -> list[Prediction]:
        out: list[Prediction] = []
        for batch in batched(examples, self.batch_size):
            start = time.perf_counter()
            try:
                chunks, doc_map = self._prepare_chunks(batch)
                scores_by_doc: dict[int, list[float]] = defaultdict(list)
                global_chunk_idx = 0
                for chunk_batch in batched(chunks, self.batch_size):
                    encoded = self.tokenizer.pad(chunk_batch, padding=True, return_tensors="pt").to(self.device)
                    with self.torch.inference_mode():
                        logits = self.model(encoded["input_ids"], encoded["attention_mask"])
                        probs = self.torch.softmax(logits, dim=-1)[:, 1].detach().cpu().tolist()
                    for prob in probs:
                        doc_idx = doc_map[global_chunk_idx]
                        scores_by_doc[doc_idx].append(float(prob))
                        global_chunk_idx += 1
                elapsed = (time.perf_counter() - start) / max(len(batch), 1)
                for idx, ex in enumerate(batch):
                    doc_scores = scores_by_doc[idx]
                    score = float(np.mean(doc_scores))
                    pred = Prediction(
                        id=ex.id,
                        detector=self.name,
                        score_ai=score,
                        raw_score=score,
                        raw_label="AI" if score >= self.threshold else "human",
                        pred_builtin=int(score >= self.threshold),
                        features={
                            "model_id": self.model_id,
                            "revision": self.revision,
                            "backbone": self.backbone_id,
                            "max_length": self.max_length,
                            "stride": self.stride,
                            "num_chunks": len(doc_scores),
                            "chunk_scores_mean_aggregation": doc_scores,
                            "score_definition": "softmax(main_head)[AI] averaged over chunks",
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

    def _prepare_chunks(self, examples: list[Example]) -> tuple[list[dict], list[int]]:
        chunks: list[dict] = []
        doc_map: list[int] = []
        max_content = max(1, self.max_length - self.tokenizer.num_special_tokens_to_add(pair=False))
        step = max(1, min(self.stride, max_content))
        for doc_idx, ex in enumerate(examples):
            token_ids = self.tokenizer.encode(ex.text, add_special_tokens=False)
            if not token_ids:
                token_ids = [self.tokenizer.unk_token_id or self.tokenizer.pad_token_id or 0]
            starts = [0] if len(token_ids) <= max_content else list(range(0, len(token_ids), step))
            for start in starts:
                piece = token_ids[start : start + max_content]
                if not piece:
                    continue
                prepared = self._prepare_token_piece(piece)
                chunks.append(prepared)
                doc_map.append(doc_idx)
                if start + max_content >= len(token_ids):
                    break
        return chunks, doc_map

    def _prepare_token_piece(self, piece: list[int]) -> dict[str, list[int]]:
        if hasattr(self.tokenizer, "prepare_for_model"):
            return self.tokenizer.prepare_for_model(
                piece,
                add_special_tokens=True,
                max_length=self.max_length,
                truncation=True,
                return_attention_mask=True,
            )

        # transformers>=5 can return TokenizersBackend for tokenizer-only HF repos.
        # MELD's released tokenizer uses BERT-style [CLS] ... [SEP] framing.
        cls_id = getattr(self.tokenizer, "cls_token_id", None)
        sep_id = getattr(self.tokenizer, "sep_token_id", None)
        if cls_id is None or sep_id is None:
            encoded = self.tokenizer.decode(piece, skip_special_tokens=False)
            return self.tokenizer(
                encoded,
                add_special_tokens=True,
                max_length=self.max_length,
                truncation=True,
                return_attention_mask=True,
            )
        input_ids = [int(cls_id), *piece[: max(0, self.max_length - 2)], int(sep_id)]
        return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids)}


class _MELDDetector:
    def __new__(
        cls,
        AutoModel,
        nn,
        *,
        backbone: str,
        n_generators: int,
        n_attacks: int,
        n_domains: int,
        num_labels: int = 2,
        dropout: float = 0.1,
    ):
        import torch

        class Impl(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                try:
                    self.backbone = AutoModel.from_pretrained(backbone, attn_implementation="sdpa")
                except Exception:
                    self.backbone = AutoModel.from_pretrained(backbone)
                if hasattr(self.backbone.config, "reference_compile"):
                    self.backbone.config.reference_compile = False
                hidden = self.backbone.config.hidden_size
                self.dropout = nn.Dropout(dropout)
                self.head_main = nn.Sequential(
                    nn.Linear(hidden, hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, num_labels),
                )
                self.head_gen = nn.Linear(hidden, n_generators)
                self.head_att = nn.Linear(hidden, n_attacks)
                self.head_dom = nn.Linear(hidden, n_domains)
                self.log_var_main = nn.Parameter(torch.zeros(()))
                self.log_var_gen = nn.Parameter(torch.zeros(()))
                self.log_var_att = nn.Parameter(torch.zeros(()))
                self.log_var_dom = nn.Parameter(torch.zeros(()))

            def forward(self, input_ids, attention_mask):
                out = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
                mask = attention_mask.unsqueeze(-1).to(out.dtype)
                pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
                pooled = self.dropout(pooled)
                return self.head_main(pooled).float()

        return Impl()
