from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator, TypeVar

from .schemas import Example, Prediction

T = TypeVar("T")


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc


def load_examples(path: str | Path) -> list[Example]:
    return [Example.from_json(row) for row in read_jsonl(path)]


def load_predictions(path: str | Path) -> list[Prediction]:
    return [Prediction(**row) for row in read_jsonl(path)]


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: str | Path, row: dict) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(row, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def batched(items: list[T], batch_size: int) -> Iterator[list[T]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]
