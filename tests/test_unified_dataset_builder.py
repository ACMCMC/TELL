import gzip
import json

import pyarrow.parquet as pq

from scripts import build_unified_dataset as builder


def test_source_adapters_use_refreshed_dataset_set():
    dataset_ids = {source.dataset_id for source in builder.SOURCE_ADAPTERS}

    assert "yaful/DeepfakeTextDetect" not in dataset_ids
    assert "zenodo_14962653_pan_voight_kampff" not in dataset_ids
    assert builder.GHOSTBUSTER_ESSAY_SPEC.dataset_id in dataset_ids
    assert builder.OUTFOX_DATASET_ID in dataset_ids
    assert builder.AUTEXTIFICATION_DATASET_ID in dataset_ids


def test_argugpt_rows_are_ai_student_essays():
    record = builder.normalize_argugpt_row(
        "train",
        0,
        {"text": "Essay", "model": "gpt-3.5-turbo", "exam_type": "toefl"},
    )

    assert record["label"] == 1
    assert record["domain"] == "student_essay"
    assert record["domain_detail"] == "toefl"
    assert record["generator_model"] == "gpt-3.5-turbo"
    assert record["language"] == "en"


def test_raid_non_english_domain_infers_language():
    record = builder.normalize_row(
        builder.SourceSpec(
            dataset_id="liamdugan/raid",
            splits=("train",),
            text_field="generation",
            label_field=None,
            label_from_model=True,
        ),
        "train",
        0,
        {"generation": "Guten Tag", "model": "human", "domain": "german"},
    )

    assert record["language"] == "de"
    assert record["domain"] == "non_english"
    assert record["is_default_training_candidate"] is False


def test_openllmtext_label_mapping_and_language_candidate():
    record = builder.normalize_openllmtext_row(
        "train",
        1,
        {
            "text": "Web text",
            "label": "ai",
            "agent": "llama",
            "domain": "OpenLLMText",
            "source": "openweb",
            "type": "fulltext",
            "lang": "en-EN",
        },
    )

    assert record["label"] == 1
    assert record["domain"] == "web_text"
    assert record["source"] == "openweb"
    assert record["generator_model"] == "llama"
    assert record["lang"] == "en-EN"
    assert record["language"] == "en"
    assert record["is_default_training_candidate"] is True


def test_pangram_text_type_and_source_mapping():
    human = builder.normalize_pangram_row(
        "test_enron",
        3,
        {"text": "Email", "text_type": "human_written", "source": "enron_email", "model": "human"},
    )
    ai = builder.normalize_pangram_row(
        "test",
        4,
        {"text": "Review", "text_type": "ai_generated", "source": "google_reviews", "model": "gemini"},
    )

    assert human["label"] == 0
    assert human["domain"] == "email"
    assert ai["label"] == 1
    assert ai["domain"] == "review_opinion"


def test_daigtv2_iterator_filters_non_daigt_sources(tmp_path):
    builder.set_source_cache_dir(tmp_path)
    csv_path = tmp_path / builder.DAIGTV2_CACHE_FILENAME
    csv_path.write_text(
        "text,label,source\n"
        "keep,1,DAIGT_v2_Car-free cities\n"
        "skip,0,OtherSource\n",
        encoding="utf-8",
    )
    stats = builder.BuildStats()

    records = list(builder._iter_daigtv2(builder.BuildSource(builder.DAIGTV2_DATASET_ID, builder._iter_daigtv2), stats))

    assert len(records) == 1
    assert records[0]["text"] == "keep"
    assert records[0]["domain"] == "student_essay"
    assert records[0]["domain_detail"] == "Car-free cities"
    assert stats.source_rows_seen_by_dataset[builder.DAIGTV2_DATASET_ID] == 2
    assert stats.excluded_rows_by_dataset[builder.DAIGTV2_DATASET_ID] == 1


def test_ghostbuster_essay_mapping():
    record = builder.normalize_ghostbuster_essay_row(
        "train",
        0,
        {"text": "Essay", "model": "gpt_prompt1", "generated": True},
    )

    assert record["dataset_id"] == builder.GHOSTBUSTER_ESSAY_SPEC.dataset_id
    assert record["label"] == 1
    assert record["domain"] == "student_essay"
    assert record["source"] == "ghostbuster"
    assert record["generator_model"] == "gpt_prompt1"
    assert record["language"] == "en"


def test_outfox_mapping_marks_attacks():
    record = builder.normalize_outfox_row(
        "test_chatgpt_dipper_attack",
        2,
        {"text": "Attacked essay", "model": "chatgpt", "attack": "dipper"},
        label=1,
        generator_model="chatgpt",
        attack="dipper",
    )

    assert record["dataset_id"] == builder.OUTFOX_DATASET_ID
    assert record["label"] == 1
    assert record["domain"] == "student_essay"
    assert record["source_detail"] == "chatgpt:dipper"
    assert record["is_adversarial"] is True
    assert record["is_default_training_candidate"] is False
    assert record["language"] == "en"


def test_autextification_detection_mapping_and_language():
    record = builder.normalize_autextification_row(
        "train_es",
        4,
        {"text": "Reseña", "label": "generated", "model": "F", "domain": "reviews"},
        language="es",
    )

    assert record["dataset_id"] == builder.AUTEXTIFICATION_DATASET_ID
    assert record["label"] == 1
    assert record["domain"] == "review_opinion"
    assert record["generator_model"] == "text-davinci-003"
    assert record["language"] == "es"
    assert record["is_default_training_candidate"] is False


def test_build_dataset_tracks_included_seen_and_excluded_rows(tmp_path):
    def iter_synthetic(source, stats):
        stats.mark_seen(source.dataset_id, "train")
        yield builder.make_record(
            dataset_id=source.dataset_id,
            split="train",
            row_index=0,
            row={"text": "keep", "label": 0},
            text="keep",
            label=0,
            domain="forum_qa",
            domain_detail="unit",
            source="unit",
            source_detail="unit",
        )
        stats.mark_seen(source.dataset_id, "train")
        stats.mark_excluded(source.dataset_id, "train", "filtered")

    source = builder.BuildSource("synthetic", iter_synthetic)

    summary = builder.build_dataset(tmp_path, batch_size=1, sources=(source,))

    assert summary["total_unified_rows"] == 1
    assert summary["sum_included_source_rows"] == 1
    assert summary["sum_seen_source_rows"] == 2
    assert summary["total_excluded_rows"] == 1
    assert summary["row_count_matches_sum_source_datasets"] is True
    assert summary["seen_minus_excluded_matches_included"] is True

    with gzip.open(tmp_path / "unified_tell_dataset.jsonl.gz", "rt", encoding="utf-8") as handle:
        lines = [json.loads(line) for line in handle]
    assert len(lines) == 1
    assert lines[0]["original"] == {"text": "keep", "label": 0}
    assert lines[0]["language"] == "en"

    parquet_file = pq.ParquetFile(tmp_path / "unified_tell_dataset.parquet")
    assert parquet_file.metadata.num_rows == 1
    assert "language" in parquet_file.schema_arrow.names
