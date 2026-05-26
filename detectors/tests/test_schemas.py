import pytest

from detectors_bench.schemas import Example


def test_example_schema_preserves_metadata():
    ex = Example.from_json({"id": "x", "text": "hello", "label": 1, "source_id": "abc"})
    assert ex.label == 1
    assert ex.meta == {"source_id": "abc"}


def test_example_schema_rejects_invalid_label():
    with pytest.raises(ValueError):
        Example.from_json({"id": "x", "text": "hello", "label": 2})
