"""Build the unified TELL dataset with normalized domain metadata.

The output preserves every included source row while adding a common schema for
domain-transfer experiments. It writes:

- ``unified_tell_dataset.jsonl.gz``: JSON lines with an ``original`` object.
- ``unified_tell_dataset.parquet``: same normalized columns plus
  ``original_json`` for compact release/storage.
- ``dataset_summary.json``: counts, exclusions, and verification.
- ``README.md``: concise dataset notes.

Example:
    python scripts/build_unified_dataset.py --output-dir ~/data/tell_unified
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import pickle
import subprocess
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset


@dataclass(frozen=True)
class SourceSpec:
    dataset_id: str
    splits: tuple[str, ...]
    text_field: str
    label_field: str | None = "label"
    label_from_model: bool = False


@dataclass(frozen=True)
class BuildSource:
    dataset_id: str
    iter_records: Callable[["BuildSource", "BuildStats"], Iterator[dict[str, Any]]]


@dataclass
class BuildStats:
    source_rows_seen_by_dataset: Counter[str] = field(default_factory=Counter)
    source_rows_seen_by_dataset_split: Counter[str] = field(default_factory=Counter)
    excluded_rows_by_dataset: Counter[str] = field(default_factory=Counter)
    excluded_rows_by_dataset_reason: Counter[str] = field(default_factory=Counter)
    excluded_rows_by_dataset_split_reason: Counter[str] = field(default_factory=Counter)

    def mark_seen(self, dataset_id: str, split: str) -> None:
        self.source_rows_seen_by_dataset[dataset_id] += 1
        self.source_rows_seen_by_dataset_split[_summary_key(dataset_id, split)] += 1

    def mark_excluded(self, dataset_id: str, split: str, reason: str) -> None:
        self.excluded_rows_by_dataset[dataset_id] += 1
        self.excluded_rows_by_dataset_reason[_summary_key(dataset_id, reason)] += 1
        self.excluded_rows_by_dataset_split_reason[_summary_key(dataset_id, split, reason)] += 1


BASE_SOURCE_SPECS: tuple[SourceSpec, ...] = (
    SourceSpec(
        dataset_id="Ateeqq/AI-and-Human-Generated-Text",
        splits=("train", "test"),
        text_field="abstract",
    ),
    SourceSpec(
        dataset_id="Jinyan1/COLING_2025_MGT_en",
        splits=("train", "dev"),
        text_field="text",
    ),
    SourceSpec(
        dataset_id="liamdugan/raid",
        splits=("train", "extra"),
        text_field="generation",
        label_field=None,
        label_from_model=True,
    ),
)

ARGUGPT_SPEC = SourceSpec(
    dataset_id="SJTU-CL/ArguGPT",
    splits=("train", "validation", "test"),
    text_field="text",
    label_field=None,
)

OPENLLMTEXT_SPEC = SourceSpec(
    dataset_id="TheItCrOw/OpenLLMText",
    splits=("train",),
    text_field="text",
    label_field=None,
)

GHOSTBUSTER_ESSAY_SPEC = SourceSpec(
    dataset_id="acmc/ghostbuster_essay",
    splits=("train",),
    text_field="text",
    label_field=None,
)

PANGRAM_DATASET_ID = "pangram/editlens_iclr"
PANGRAM_SPLITS = ("train", "val", "test", "test_enron", "test_llama")
PANGRAM_INCLUDED_TEXT_TYPES = {"human_written", "ai_generated"}

DAIGTV2_DATASET_ID = "DAIGTv2"
DAIGTV2_SOURCE_URL = "https://media.githubusercontent.com/media/crusnix/ai_text_detector_final/main/data/merged_dataset(1).csv"
DAIGTV2_CACHE_FILENAME = "merged_dataset_1.csv"
DAIGTV2_INCLUDED_SOURCE_PREFIX = "DAIGT_v2_"

OUTFOX_DATASET_ID = "ryuryukke/OUTFOX"
OUTFOX_RAW_DATA_BASE_URL = "https://raw.githubusercontent.com/ryuryukke/OUTFOX/main/data"
OUTFOX_SPLITS = ("train", "valid", "test")
OUTFOX_MODELS = ("chatgpt", "text_davinci_003", "flan_t5_xxl")
OUTFOX_ATTACK_FILES = (
    ("outfox", "chatgpt", "chatgpt/test/test_outfox_attacks.pkl"),
    ("dipper", "chatgpt", "dipper/chatgpt/test_attacks.pkl"),
    ("dipper", "text_davinci_003", "dipper/text_davinci_003/test_attacks.pkl"),
    ("dipper", "flan_t5_xxl", "dipper/flan_t5_xxl/test_attacks.pkl"),
)

AUTEXTIFICATION_DATASET_ID = "symanto/autextification2023"
AUTEXTIFICATION_LANGUAGES = ("en", "es")
AUTEXTIFICATION_DETECTION_FILES = {
    ("train", "en"): "data/train/subtask_1/en/train.tsv",
    ("train", "es"): "data/train/subtask_1/es/train.tsv",
    ("test", "en"): "data/test/subtask_1/en/test.tsv",
    ("test", "es"): "data/test/subtask_1/es/test.tsv",
}
AUTEXTIFICATION_MODEL_BY_CODE = {
    "A": "bloom-1b7",
    "B": "bloom-3b",
    "C": "bloom-7b1",
    "D": "babbage",
    "E": "curie",
    "F": "text-davinci-003",
    "NO-MODEL": "human",
}

COLING_DOMAIN_BY_SUB_SOURCE = {
    "arxiv": "academic_abstract",
    "peerread": "academic_abstract",
    "pubmed": "academic_abstract",
    "sci_gen": "academic_abstract",
    "medicine": "academic_abstract",
    "wiki_csai": "academic_abstract",
    "wikipedia": "encyclopedic_reference",
    "wp": "encyclopedic_reference",
    "reddit": "forum_qa",
    "reddit_eli5": "forum_qa",
    "eli5": "forum_qa",
    "cmv": "forum_qa",
    "open_qa": "forum_qa",
    "squad": "forum_qa",
    "wikihow": "howto_instructional",
    "yelp": "review_opinion",
    "imdb": "review_opinion",
    "xsum": "news",
    "cnn": "news",
    "tldr": "news",
    "dialogsum": "news",
    "roct": "creative_writing",
    "outfox": "creative_writing",
    "hswag": "commonsense_completion",
    "finance": "finance",
}

RAID_DOMAIN_BY_SOURCE_DOMAIN = {
    "abstracts": "academic_abstract",
    "books": "creative_writing",
    "code": "code",
    "news": "news",
    "poetry": "creative_writing",
    "recipes": "howto_instructional",
    "reddit": "forum_qa",
    "reviews": "review_opinion",
    "wiki": "encyclopedic_reference",
    "german": "non_english",
    "czech": "non_english",
}

PANGRAM_DOMAIN_BY_SOURCE = {
    "reddit_writing_prompts": "creative_writing",
    "news": "news",
    "fineweb_edu": "educational_web",
    "amazon_reviews": "review_opinion",
    "google_reviews": "review_opinion",
    "enron_email": "email",
}

AUTEXTIFICATION_DOMAIN_BY_SOURCE_DOMAIN = {
    "tweets": "social_media",
    "reviews": "review_opinion",
    "wiki": "howto_instructional",
    "wikihow": "howto_instructional",
    "news": "news",
    "legal": "legal",
}

LANGUAGE_ALIASES = {
    "en": "en",
    "en-en": "en",
    "en-us": "en",
    "english": "en",
    "es": "es",
    "es-es": "es",
    "spanish": "es",
    "de": "de",
    "german": "de",
    "cs": "cs",
    "czech": "cs",
}
ENGLISH_LANGUAGE_VALUES = {"en"}
DEFAULT_CACHE_DIR = Path(os.environ.get("RL_DETECTOR_SOURCE_CACHE", "~/.cache/rl-detector-unified-sources"))

PARQUET_SCHEMA = pa.schema(
    [
        ("unified_id", pa.string()),
        ("dataset_id", pa.string()),
        ("source_split", pa.string()),
        ("row_index", pa.int64()),
        ("text", pa.string()),
        ("label", pa.int8()),
        ("label_text", pa.string()),
        ("domain", pa.string()),
        ("domain_detail", pa.string()),
        ("source", pa.string()),
        ("source_detail", pa.string()),
        ("generator_model", pa.string()),
        ("attack", pa.string()),
        ("lang", pa.string()),
        ("language", pa.string()),
        ("title", pa.string()),
        ("decoding", pa.string()),
        ("repetition_penalty", pa.string()),
        ("is_adversarial", pa.bool_()),
        ("is_default_training_candidate", pa.bool_()),
        ("original_json", pa.string()),
    ]
)

PARQUET_COLUMNS = [field.name for field in PARQUET_SCHEMA]


def _none_or_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _normalize_language(value: Any) -> str | None:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    key = raw.replace("_", "-").lower()
    if key in LANGUAGE_ALIASES:
        return LANGUAGE_ALIASES[key]
    prefix = key.split("-", 1)[0]
    if len(prefix) == 2 and prefix.isalpha():
        return prefix
    return key


def _infer_language(
    *,
    dataset_id: str,
    domain: str,
    domain_detail: Any,
    source_detail: Any,
    lang: Any,
) -> str:
    normalized = _normalize_language(lang)
    if normalized is not None:
        return normalized

    detail = (str(domain_detail or source_detail or "")).strip().lower()
    if dataset_id == "liamdugan/raid":
        if detail == "german":
            return "de"
        if detail == "czech":
            return "cs"

    if domain == "non_english":
        return "unknown"
    return "en"


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item())
        except Exception:
            pass
    return str(value)


def _json_dumps(value: Any, *, sort_keys: bool = False) -> str:
    return json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=sort_keys)


def _summary_key(*parts: Any) -> str:
    return "::".join("" if part is None else str(part) for part in parts)


def _label(row: dict[str, Any], spec: SourceSpec) -> int:
    if spec.label_from_model:
        return 0 if row.get("model") == "human" else 1
    if spec.label_field is None:
        raise ValueError(f"{spec.dataset_id} has no label field or label rule")
    return int(row[spec.label_field])


def _normalize_domain(dataset_id: str, row: dict[str, Any]) -> tuple[str, str | None]:
    if dataset_id == "Ateeqq/AI-and-Human-Generated-Text":
        return "academic_abstract", "abstract"
    if dataset_id == "Jinyan1/COLING_2025_MGT_en":
        detail = _none_or_str(row.get("sub_source"))
        domain = COLING_DOMAIN_BY_SUB_SOURCE.get((detail or "").lower(), "unknown")
        return domain, detail
    if dataset_id == "liamdugan/raid":
        detail = _none_or_str(row.get("domain"))
        domain = RAID_DOMAIN_BY_SOURCE_DOMAIN.get((detail or "").lower(), "unknown")
        return domain, detail
    return "unknown", None


def _source(dataset_id: str, row: dict[str, Any]) -> tuple[str | None, str | None]:
    if dataset_id == "Ateeqq/AI-and-Human-Generated-Text":
        return "ateeqq", "abstract"
    if dataset_id == "Jinyan1/COLING_2025_MGT_en":
        return _none_or_str(row.get("source")), _none_or_str(row.get("sub_source"))
    if dataset_id == "liamdugan/raid":
        return "raid", _none_or_str(row.get("domain"))
    return None, None


def make_record(
    *,
    dataset_id: str,
    split: str,
    row_index: int,
    row: dict[str, Any],
    text: Any,
    label: int,
    domain: str,
    domain_detail: Any = None,
    source: Any = None,
    source_detail: Any = None,
    generator_model: Any = None,
    attack: Any = None,
    lang: Any = None,
    title: Any = None,
    decoding: Any = None,
    repetition_penalty: Any = None,
) -> dict[str, Any]:
    normalized_label = int(label)
    normalized_attack = _none_or_str(attack)
    normalized_lang = _none_or_str(lang)
    normalized_language = _infer_language(
        dataset_id=dataset_id,
        domain=domain,
        domain_detail=domain_detail,
        source_detail=source_detail,
        lang=lang,
    )
    original = _jsonable(row)
    original_json = _json_dumps(original, sort_keys=True)
    unified_id = f"{dataset_id}::{split}::{row_index}"
    is_adversarial = normalized_attack not in (None, "none")
    is_default_training_candidate = (
        domain not in {"code", "non_english", "unknown"}
        and not is_adversarial
        and normalized_language in ENGLISH_LANGUAGE_VALUES
    )

    return {
        "unified_id": unified_id,
        "dataset_id": dataset_id,
        "source_split": split,
        "row_index": row_index,
        "text": _none_or_str(text) or "",
        "label": normalized_label,
        "label_text": "AI" if normalized_label == 1 else "human",
        "domain": domain,
        "domain_detail": _none_or_str(domain_detail),
        "source": _none_or_str(source),
        "source_detail": _none_or_str(source_detail),
        "generator_model": _none_or_str(generator_model),
        "attack": normalized_attack,
        "lang": normalized_lang or normalized_language,
        "language": normalized_language,
        "title": _none_or_str(title),
        "decoding": _none_or_str(decoding),
        "repetition_penalty": _none_or_str(repetition_penalty),
        "is_adversarial": is_adversarial,
        "is_default_training_candidate": is_default_training_candidate,
        "original": original,
        "original_json": original_json,
    }


def normalize_row(spec: SourceSpec, split: str, row_index: int, row: dict[str, Any]) -> dict[str, Any]:
    label = _label(row, spec)
    domain, domain_detail = _normalize_domain(spec.dataset_id, row)
    source, source_detail = _source(spec.dataset_id, row)
    generator_model = row.get("model")
    lang = row.get("lang")
    # Ateeqq has no per-row model field; it is a documented ChatGPT dataset.
    if spec.dataset_id == "Ateeqq/AI-and-Human-Generated-Text":
        if not generator_model:
            generator_model = "chatgpt" if label == 1 else "human"
        if not lang:
            lang = "en"
    return make_record(
        dataset_id=spec.dataset_id,
        split=split,
        row_index=row_index,
        row=row,
        text=row.get(spec.text_field),
        label=label,
        domain=domain,
        domain_detail=domain_detail,
        source=source,
        source_detail=source_detail,
        generator_model=generator_model,
        attack=row.get("attack"),
        lang=lang,
        title=row.get("title"),
        decoding=row.get("decoding"),
        repetition_penalty=row.get("repetition_penalty"),
    )


def normalize_argugpt_row(split: str, row_index: int, row: dict[str, Any]) -> dict[str, Any]:
    exam_type = _none_or_str(row.get("exam_type"))
    return make_record(
        dataset_id=ARGUGPT_SPEC.dataset_id,
        split=split,
        row_index=row_index,
        row=row,
        text=row.get("text"),
        label=1,
        domain="student_essay",
        domain_detail=exam_type,
        source="argugpt",
        source_detail=exam_type,
        generator_model=row.get("model"),
        lang="en",
    )


def normalize_openllmtext_row(split: str, row_index: int, row: dict[str, Any]) -> dict[str, Any]:
    raw_label = str(row.get("label", "")).lower()
    if raw_label not in {"human", "ai"}:
        raise ValueError(f"Unexpected OpenLLMText label: {row.get('label')!r}")
    agent = row.get("agent")
    return make_record(
        dataset_id=OPENLLMTEXT_SPEC.dataset_id,
        split=split,
        row_index=row_index,
        row=row,
        text=row.get("text"),
        label=1 if raw_label == "ai" else 0,
        domain="web_text",
        domain_detail=row.get("domain"),
        source=row.get("source"),
        source_detail=row.get("type"),
        generator_model=agent,
        lang=row.get("lang"),
    )


def normalize_pangram_row(split: str, row_index: int, row: dict[str, Any]) -> dict[str, Any]:
    text_type = str(row.get("text_type", ""))
    if text_type not in PANGRAM_INCLUDED_TEXT_TYPES:
        raise ValueError(f"Unsupported Pangram text_type for included row: {text_type!r}")
    source_detail = _none_or_str(row.get("source"))
    domain = PANGRAM_DOMAIN_BY_SOURCE.get((source_detail or "").lower(), "unknown")
    return make_record(
        dataset_id=PANGRAM_DATASET_ID,
        split=split,
        row_index=row_index,
        row=row,
        text=row.get("text"),
        label=1 if text_type == "ai_generated" else 0,
        domain=domain,
        domain_detail=source_detail,
        source="pangram",
        source_detail=source_detail,
        generator_model=row.get("model"),
        lang="en",
    )


def normalize_daigtv2_row(split: str, row_index: int, row: dict[str, Any]) -> dict[str, Any]:
    source_value = _none_or_str(row.get("source")) or ""
    domain_detail = source_value.removeprefix(DAIGTV2_INCLUDED_SOURCE_PREFIX)
    label = int(row["label"])
    return make_record(
        dataset_id=DAIGTV2_DATASET_ID,
        split=split,
        row_index=row_index,
        row=row,
        text=row.get("text"),
        label=label,
        domain="student_essay",
        domain_detail=domain_detail,
        source="daigt_v2",
        source_detail=domain_detail,
        # DAIGTv2 has no per-row model field; it is a documented ChatGPT dataset.
        generator_model="chatgpt" if label == 1 else "human",
        lang="en",
    )


def normalize_ghostbuster_essay_row(split: str, row_index: int, row: dict[str, Any]) -> dict[str, Any]:
    generated = bool(row.get("generated"))
    return make_record(
        dataset_id=GHOSTBUSTER_ESSAY_SPEC.dataset_id,
        split=split,
        row_index=row_index,
        row=row,
        text=row.get("text"),
        label=1 if generated else 0,
        domain="student_essay",
        domain_detail="essay",
        source="ghostbuster",
        source_detail="essay",
        generator_model=row.get("model") if generated else "human",
        lang="en",
    )


def normalize_outfox_row(
    split: str,
    row_index: int,
    row: dict[str, Any],
    *,
    label: int,
    generator_model: str,
    attack: str | None = None,
) -> dict[str, Any]:
    source_detail = generator_model if attack is None else f"{generator_model}:{attack}"
    return make_record(
        dataset_id=OUTFOX_DATASET_ID,
        split=split,
        row_index=row_index,
        row=row,
        text=row.get("text"),
        label=label,
        domain="student_essay",
        domain_detail="kaggle_feedback_prize",
        source="outfox",
        source_detail=source_detail,
        generator_model=generator_model,
        attack=attack,
        lang="en",
    )


def normalize_autextification_row(
    split: str,
    row_index: int,
    row: dict[str, Any],
    *,
    language: str,
) -> dict[str, Any]:
    raw_label = str(row.get("label", "")).lower()
    if raw_label not in {"human", "generated"}:
        raise ValueError(f"Unexpected AuTexTification label: {row.get('label')!r}")
    label = 1 if raw_label == "generated" else 0
    raw_domain = _none_or_str(row.get("domain"))
    domain = AUTEXTIFICATION_DOMAIN_BY_SOURCE_DOMAIN.get((raw_domain or "").lower(), "unknown")
    source_model = _none_or_str(row.get("model"))
    generator_model = AUTEXTIFICATION_MODEL_BY_CODE.get(source_model or "", source_model)
    if label == 0:
        generator_model = "human"
    return make_record(
        dataset_id=AUTEXTIFICATION_DATASET_ID,
        split=split,
        row_index=row_index,
        row=row,
        text=row.get("text"),
        label=label,
        domain=domain,
        domain_detail=raw_domain,
        source="autextification2023",
        source_detail=raw_domain,
        generator_model=generator_model,
        lang=language,
    )


def _iter_hf_source(spec: SourceSpec, stats: BuildStats) -> Iterator[dict[str, Any]]:
    for split in spec.splits:
        dataset = load_dataset(spec.dataset_id, split=split, streaming=True)
        for row_index, row in enumerate(dataset):
            row_dict = dict(row)
            stats.mark_seen(spec.dataset_id, split)
            yield normalize_row(spec=spec, split=split, row_index=row_index, row=row_dict)


def _iter_argugpt(source: BuildSource, stats: BuildStats) -> Iterator[dict[str, Any]]:
    for split in ARGUGPT_SPEC.splits:
        dataset = load_dataset(ARGUGPT_SPEC.dataset_id, split=split, streaming=True)
        for row_index, row in enumerate(dataset):
            row_dict = dict(row)
            stats.mark_seen(source.dataset_id, split)
            yield normalize_argugpt_row(split, row_index, row_dict)


def _iter_openllmtext(source: BuildSource, stats: BuildStats) -> Iterator[dict[str, Any]]:
    for split in OPENLLMTEXT_SPEC.splits:
        dataset = load_dataset(OPENLLMTEXT_SPEC.dataset_id, split=split, streaming=True)
        for row_index, row in enumerate(dataset):
            row_dict = dict(row)
            stats.mark_seen(source.dataset_id, split)
            yield normalize_openllmtext_row(split, row_index, row_dict)


def _iter_ghostbuster_essay(source: BuildSource, stats: BuildStats) -> Iterator[dict[str, Any]]:
    for split in GHOSTBUSTER_ESSAY_SPEC.splits:
        dataset = load_dataset(GHOSTBUSTER_ESSAY_SPEC.dataset_id, split=split, streaming=True)
        for row_index, row in enumerate(dataset):
            row_dict = dict(row)
            stats.mark_seen(source.dataset_id, split)
            yield normalize_ghostbuster_essay_row(split, row_index, row_dict)


def _hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def _iter_pangram(source: BuildSource, stats: BuildStats) -> Iterator[dict[str, Any]]:
    token = _hf_token()
    token_arg: str | bool = token if token else True
    for split in PANGRAM_SPLITS:
        dataset = load_dataset(PANGRAM_DATASET_ID, split=split, streaming=True, token=token_arg)
        for row_index, row in enumerate(dataset):
            row_dict = dict(row)
            stats.mark_seen(source.dataset_id, split)
            text_type = str(row_dict.get("text_type", ""))
            if text_type not in PANGRAM_INCLUDED_TEXT_TYPES:
                stats.mark_excluded(source.dataset_id, split, f"text_type={text_type}")
                continue
            yield normalize_pangram_row(split, row_index, row_dict)


def _download_url(url: str, destination: Path) -> Path:
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        subprocess.run(["wget", "-q", "-O", str(temporary), url], check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        urllib.request.urlretrieve(url, temporary)
    temporary.replace(destination)
    return destination


def _iter_csv_file(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_index, row in enumerate(reader):
            yield row_index, dict(row)


def _iter_tsv_file(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row_index, row in enumerate(reader):
            row_dict = dict(row)
            row_dict.pop("", None)
            yield row_index, row_dict


def _load_pickle_file(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _outfox_pickle_path(relative_path: str) -> Path:
    destination = source_cache_dir() / "outfox" / relative_path
    url = f"{OUTFOX_RAW_DATA_BASE_URL}/{relative_path}"
    return _download_url(url, destination)


def _outfox_common_rows(split: str) -> tuple[list[Any], list[Any], list[Any]]:
    base = f"common/{split}/{split}"
    humans = _load_pickle_file(_outfox_pickle_path(f"{base}_humans.pkl"))
    problem_statements = _load_pickle_file(_outfox_pickle_path(f"{base}_problem_statements.pkl"))
    contexts = _load_pickle_file(_outfox_pickle_path(f"{base}_contexts.pkl"))
    return humans, problem_statements, contexts


def _outfox_original_row(
    *,
    text: Any,
    split: str,
    model: str,
    row_index: int,
    problem_statements: list[Any],
    contexts: list[Any],
    attack: str | None = None,
) -> dict[str, Any]:
    return {
        "text": text,
        "split": split,
        "model": model,
        "attack": attack,
        "problem_statement": problem_statements[row_index] if row_index < len(problem_statements) else None,
        "context": contexts[row_index] if row_index < len(contexts) else None,
    }


def _iter_outfox(source: BuildSource, stats: BuildStats) -> Iterator[dict[str, Any]]:
    common_by_split: dict[str, tuple[list[Any], list[Any], list[Any]]] = {}
    for split in OUTFOX_SPLITS:
        humans, problem_statements, contexts = _outfox_common_rows(split)
        common_by_split[split] = (humans, problem_statements, contexts)
        source_split = f"{split}_human"
        for row_index, text in enumerate(humans):
            row = _outfox_original_row(
                text=text,
                split=split,
                model="human",
                row_index=row_index,
                problem_statements=problem_statements,
                contexts=contexts,
            )
            stats.mark_seen(source.dataset_id, source_split)
            yield normalize_outfox_row(source_split, row_index, row, label=0, generator_model="human")

        for model in OUTFOX_MODELS:
            model_rows = _load_pickle_file(_outfox_pickle_path(f"{model}/{split}/{split}_lms.pkl"))
            source_split = f"{split}_{model}"
            for row_index, text in enumerate(model_rows):
                row = _outfox_original_row(
                    text=text,
                    split=split,
                    model=model,
                    row_index=row_index,
                    problem_statements=problem_statements,
                    contexts=contexts,
                )
                stats.mark_seen(source.dataset_id, source_split)
                yield normalize_outfox_row(source_split, row_index, row, label=1, generator_model=model)

    _, test_problem_statements, test_contexts = common_by_split.get("test") or _outfox_common_rows("test")
    for attack, model, relative_path in OUTFOX_ATTACK_FILES:
        attack_rows = _load_pickle_file(_outfox_pickle_path(relative_path))
        source_split = f"test_{model}_{attack}_attack"
        for row_index, text in enumerate(attack_rows):
            row = _outfox_original_row(
                text=text,
                split="test",
                model=model,
                row_index=row_index,
                problem_statements=test_problem_statements,
                contexts=test_contexts,
                attack=attack,
            )
            stats.mark_seen(source.dataset_id, source_split)
            yield normalize_outfox_row(
                source_split,
                row_index,
                row,
                label=1,
                generator_model=model,
                attack=attack,
            )


def _autextification_tsv_path(filename: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=AUTEXTIFICATION_DATASET_ID,
            filename=filename,
            repo_type="dataset",
            cache_dir=str(source_cache_dir()),
            token=_hf_token(),
        )
    )


def _iter_autextification(source: BuildSource, stats: BuildStats) -> Iterator[dict[str, Any]]:
    for language in AUTEXTIFICATION_LANGUAGES:
        for split in ("train", "test"):
            source_split = f"{split}_{language}"
            path = _autextification_tsv_path(AUTEXTIFICATION_DETECTION_FILES[(split, language)])
            for row_index, row in _iter_tsv_file(path):
                stats.mark_seen(source.dataset_id, source_split)
                yield normalize_autextification_row(source_split, row_index, row, language=language)


def _iter_daigtv2(source: BuildSource, stats: BuildStats) -> Iterator[dict[str, Any]]:
    path = _download_url(DAIGTV2_SOURCE_URL, source_cache_dir() / DAIGTV2_CACHE_FILENAME)
    split = "merged_dataset(1).csv"
    for row_index, row in _iter_csv_file(path):
        stats.mark_seen(source.dataset_id, split)
        source_value = _none_or_str(row.get("source")) or ""
        if not source_value.startswith(DAIGTV2_INCLUDED_SOURCE_PREFIX):
            stats.mark_excluded(source.dataset_id, split, "source!=DAIGT_v2")
            continue
        yield normalize_daigtv2_row(split, row_index, row)


_SOURCE_CACHE_DIR = DEFAULT_CACHE_DIR.expanduser()


def source_cache_dir() -> Path:
    return _SOURCE_CACHE_DIR


def set_source_cache_dir(path: Path) -> None:
    global _SOURCE_CACHE_DIR
    _SOURCE_CACHE_DIR = path.expanduser()


SOURCE_ADAPTERS: tuple[BuildSource, ...] = (
    *(BuildSource(spec.dataset_id, lambda _source, stats, spec=spec: _iter_hf_source(spec, stats)) for spec in BASE_SOURCE_SPECS),
    BuildSource(ARGUGPT_SPEC.dataset_id, _iter_argugpt),
    BuildSource(OPENLLMTEXT_SPEC.dataset_id, _iter_openllmtext),
    BuildSource(GHOSTBUSTER_ESSAY_SPEC.dataset_id, _iter_ghostbuster_essay),
    BuildSource(PANGRAM_DATASET_ID, _iter_pangram),
    BuildSource(DAIGTV2_DATASET_ID, _iter_daigtv2),
    BuildSource(OUTFOX_DATASET_ID, _iter_outfox),
    BuildSource(AUTEXTIFICATION_DATASET_ID, _iter_autextification),
)


def iter_normalized_records(
    sources: Iterable[BuildSource] = SOURCE_ADAPTERS,
    stats: BuildStats | None = None,
) -> Iterable[dict[str, Any]]:
    build_stats = stats if stats is not None else BuildStats()
    for source in sources:
        yield from source.iter_records(source, build_stats)


def _parquet_record(record: dict[str, Any]) -> dict[str, Any]:
    return {column: record.get(column) for column in PARQUET_COLUMNS}


def build_dataset(
    output_dir: Path,
    batch_size: int = 10_000,
    max_rows: int | None = None,
    cache_dir: Path | None = None,
    sources: Iterable[BuildSource] = SOURCE_ADAPTERS,
) -> dict[str, Any]:
    if cache_dir is not None:
        set_source_cache_dir(cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "unified_tell_dataset.jsonl.gz"
    parquet_path = output_dir / "unified_tell_dataset.parquet"

    counts_by_dataset: Counter[str] = Counter()
    counts_by_dataset_split: Counter[str] = Counter()
    counts_by_domain: Counter[str] = Counter()
    counts_by_dataset_domain: Counter[str] = Counter()
    label_by_domain: dict[str, Counter[int]] = defaultdict(Counter)
    default_candidate_counts_by_domain: Counter[str] = Counter()
    stats = BuildStats()

    writer: pq.ParquetWriter | None = None
    parquet_batch: list[dict[str, Any]] = []
    total_rows = 0

    try:
        with gzip.open(jsonl_path, "wt", encoding="utf-8") as jsonl:
            for record in iter_normalized_records(sources, stats=stats):
                jsonl.write(_json_dumps({k: v for k, v in record.items() if k != "original_json"}) + "\n")
                parquet_batch.append(_parquet_record(record))

                dataset_id = record["dataset_id"]
                split = record["source_split"]
                domain = record["domain"]
                label = int(record["label"])
                counts_by_dataset[dataset_id] += 1
                counts_by_dataset_split[_summary_key(dataset_id, split)] += 1
                counts_by_domain[domain] += 1
                counts_by_dataset_domain[_summary_key(dataset_id, domain)] += 1
                label_by_domain[domain][label] += 1
                if record["is_default_training_candidate"]:
                    default_candidate_counts_by_domain[domain] += 1

                total_rows += 1
                if len(parquet_batch) >= batch_size:
                    table = pa.Table.from_pylist(parquet_batch, schema=PARQUET_SCHEMA)
                    if writer is None:
                        writer = pq.ParquetWriter(parquet_path, PARQUET_SCHEMA, compression="zstd")
                    writer.write_table(table)
                    parquet_batch = []

                if total_rows % 100_000 == 0:
                    print(f"wrote {total_rows:,} rows", flush=True)
                if max_rows is not None and total_rows >= max_rows:
                    break

        if parquet_batch:
            table = pa.Table.from_pylist(parquet_batch, schema=PARQUET_SCHEMA)
            if writer is None:
                writer = pq.ParquetWriter(parquet_path, PARQUET_SCHEMA, compression="zstd")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()

    sum_included_source_rows = sum(counts_by_dataset.values())
    sum_seen_source_rows = sum(stats.source_rows_seen_by_dataset.values())
    total_excluded_rows = sum(stats.excluded_rows_by_dataset.values())
    summary = {
        "total_unified_rows": total_rows,
        "sum_source_dataset_rows": sum_included_source_rows,
        "sum_included_source_rows": sum_included_source_rows,
        "sum_seen_source_rows": sum_seen_source_rows,
        "total_excluded_rows": total_excluded_rows,
        "row_count_matches_sum_source_datasets": total_rows == sum_included_source_rows,
        "seen_minus_excluded_matches_included": (sum_seen_source_rows - total_excluded_rows) == sum_included_source_rows,
        "counts_by_dataset": dict(counts_by_dataset),
        "counts_by_dataset_split": dict(counts_by_dataset_split),
        "source_rows_seen_by_dataset": dict(stats.source_rows_seen_by_dataset),
        "source_rows_seen_by_dataset_split": dict(stats.source_rows_seen_by_dataset_split),
        "excluded_rows_by_dataset": dict(stats.excluded_rows_by_dataset),
        "excluded_rows_by_dataset_reason": dict(stats.excluded_rows_by_dataset_reason),
        "excluded_rows_by_dataset_split_reason": dict(stats.excluded_rows_by_dataset_split_reason),
        "counts_by_domain": dict(counts_by_domain),
        "counts_by_dataset_domain": dict(counts_by_dataset_domain),
        "label_by_domain": {
            domain: {"human": counter[0], "ai": counter[1], "total": counter[0] + counter[1]}
            for domain, counter in sorted(label_by_domain.items())
        },
        "default_training_candidate_counts_by_domain": dict(default_candidate_counts_by_domain),
        "files": {
            "jsonl_gz": str(jsonl_path),
            "parquet": str(parquet_path),
            "source_cache_dir": str(source_cache_dir()),
        },
    }

    summary_path = output_dir / "dataset_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "README.md").write_text(_readme(summary), encoding="utf-8")
    return summary


def _readme(summary: dict[str, Any]) -> str:
    return f"""# Unified TELL Dataset

Rows: {summary["total_unified_rows"]}

Verification:

- total unified rows: {summary["total_unified_rows"]}
- sum included source rows: {summary["sum_included_source_rows"]}
- source rows seen: {summary["sum_seen_source_rows"]}
- excluded source rows: {summary["total_excluded_rows"]}
- total equals included source rows: {summary["row_count_matches_sum_source_datasets"]}
- seen minus excluded equals included: {summary["seen_minus_excluded_matches_included"]}

Files:

- `unified_tell_dataset.jsonl.gz`: release-friendly JSONL. Each row contains
  normalized metadata plus the complete original source row in `original`.
- `unified_tell_dataset.parquet`: compact Parquet. It contains the normalized
  metadata and `original_json`, a lossless JSON serialization of the source row.
- `dataset_summary.json`: count tables, exclusion tables, and verification.

Canonical experiment fields:

- `domain`: normalized leave-one-domain-out category.
- `domain_detail`: original fine-grained source category.
- `dataset_id`: source dataset.
- `source`: dataset-internal provenance.
- `source_detail`: original fine category used to derive `domain`.
- `lang`: raw source language value when available, otherwise the normalized
  language code.
- `language`: normalized language code for every row.
- `label`: `0 = human`, `1 = AI`.
- `is_default_training_candidate`: English/prose, non-adversarial row suitable
  for the primary leave-one-domain-out experiments.

Intentional exclusions:

- Pangram `text_type=ai_edited` rows are excluded because this release uses only
  AI-generated and human-written text.
- The DAIGTv2 GitHub CSV is a merged file. Only `source` values starting with
  `DAIGT_v2_` are included in the DAIGTv2 slice.
- AuTexTification model-attribution files are not loaded; this binary corpus
  uses its Subtask 1 human-vs-generated detection files for English and Spanish.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, help="Directory for dataset files.")
    parser.add_argument("--batch-size", type=int, default=10_000, help="Parquet write batch size.")
    parser.add_argument("--max-rows", type=int, default=None, help="Debug cap across all datasets.")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Directory for downloaded source files that are not loaded directly through datasets.",
    )
    args = parser.parse_args()

    summary = build_dataset(
        Path(args.output_dir).expanduser(),
        batch_size=args.batch_size,
        max_rows=args.max_rows,
        cache_dir=args.cache_dir.expanduser(),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
