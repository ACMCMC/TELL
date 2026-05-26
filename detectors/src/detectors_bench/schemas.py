from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Example:
    id: str
    text: str
    label: int | None = None
    split: str | None = None
    dataset: str | None = None
    domain: str | None = None
    generator: str | None = None
    attack: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, row: dict[str, Any]) -> "Example":
        if "id" not in row:
            raise ValueError("Example row missing required field: id")
        if "text" not in row:
            raise ValueError(f"Example {row.get('id')} missing required field: text")
        known = {"id", "text", "label", "split", "dataset", "domain", "generator", "attack"}
        meta = {k: v for k, v in row.items() if k not in known}
        label = row.get("label")
        if label is not None:
            label = int(label)
            if label not in (0, 1):
                raise ValueError(f"Example {row['id']} label must be 0=human or 1=AI")
        return cls(
            id=str(row["id"]),
            text=str(row["text"]),
            label=label,
            split=row.get("split"),
            dataset=row.get("dataset"),
            domain=row.get("domain"),
            generator=row.get("generator"),
            attack=row.get("attack"),
            meta=meta,
        )


@dataclass
class Prediction:
    id: str
    detector: str
    score_ai: float | None
    raw_score: float | None = None
    raw_label: str | None = None
    pred_builtin: int | None = None
    features: dict[str, Any] = field(default_factory=dict)
    runtime_s: float | None = None
    error: str | None = None
    label: int | None = None
    split: str | None = None
    dataset: str | None = None
    domain: str | None = None
    generator: str | None = None
    attack: str | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def attach_example_metadata(pred: Prediction, ex: Example) -> Prediction:
    pred.label = ex.label
    pred.split = ex.split
    pred.dataset = ex.dataset
    pred.domain = ex.domain
    pred.generator = ex.generator
    pred.attack = ex.attack
    return pred
