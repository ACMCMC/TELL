from __future__ import annotations

import json
import sys
import time

from detectors_bench.io import batched
from .base import DetectorWrapper, require_optional, sigmoid
from detectors_bench.registry import vendor_path
from detectors_bench.schemas import Example, Prediction, attach_example_metadata


class MageWrapper(DetectorWrapper):
    def __init__(self, cfg):
        super().__init__(cfg)
        require_optional("transformers", "Install the MAGE environment from detectors/vendor/mage/requirements.txt.")
        require_optional("torch", "Install PyTorch in the MAGE environment.")
        import torch
        from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, LongformerConfig

        root = vendor_path(cfg)
        deployment = root / "deployment"
        sys.path.insert(0, str(deployment))
        from utils import preprocess  # type: ignore

        self.preprocess = preprocess
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = cfg.require("model_id")
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.get("tokenizer_id", self.model_id))
        try:
            config = AutoConfig.from_pretrained(self.model_id)
        except Exception:
            from huggingface_hub import hf_hub_download

            config_path = hf_hub_download(self.model_id, "config.json")
            raw_config = json.loads(open(config_path, encoding="utf-8").read())
            raw_config["id2label"] = {"0": "machine-generated", "1": "human-written"}
            raw_config["label2id"] = {"machine-generated": 0, "human-written": 1}
            if raw_config.get("model_type") == "longformer":
                config = LongformerConfig.from_dict(raw_config)
            else:
                config = AutoConfig.for_model(raw_config.get("model_type", "longformer"), **raw_config)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_id, config=config).to(self.device)
        self.model.eval()
        self.threshold = float(cfg.get("threshold", -3.08583984375))
        self.scale = float(cfg.get("threshold_scale", 1.0))
        self.max_length = int(cfg.get("max_length", 4096))

    def predict_one(self, ex: Example) -> Prediction:
        return self.predict_batch([ex])[0]

    def predict_batch(self, examples: list[Example]) -> list[Prediction]:
        out: list[Prediction] = []
        batch_size = int(self.cfg.get("batch_size", 8))
        for batch in batched(examples, batch_size):
            start = time.perf_counter()
            try:
                texts = [self.preprocess(ex.text) for ex in batch]
                encoded = self.tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                with self.torch.inference_mode():
                    logits_batch = self.model(**encoded).logits.detach().cpu()
                elapsed = (time.perf_counter() - start) / max(len(batch), 1)
                for ex, logits in zip(batch, logits_batch):
                    pred = self._prediction_from_logits(ex, logits)
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

    def _prediction_from_logits(self, ex: Example, logits) -> Prediction:
        is_machine = -float(logits[0].item())
        score_ai = sigmoid((self.threshold - is_machine) / self.scale)
        return Prediction(
            id=ex.id,
            detector=self.name,
            score_ai=float(score_ai),
            raw_score=is_machine,
            raw_label="machine-generated" if is_machine < self.threshold else "human-written",
            pred_builtin=int(is_machine < self.threshold),
            features={"threshold": self.threshold, "logits": [float(x) for x in logits.tolist()]},
        )
