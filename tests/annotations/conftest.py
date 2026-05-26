"""Shared fixtures for annotation / special-token tests."""

import pytest

from rl_detector.prompt_utils import (
    ANNOTATION_TOKEN_REMAP,
    _apply_token_remap,
    assert_annotation_remap_ids_are_reserved_placeholders,
)

BASE_MODEL = "openai/gpt-oss-120b"


@pytest.fixture(scope="module")
def remapped_tok():
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    except Exception:
        pytest.skip("base tokenizer not available")
    assert_annotation_remap_ids_are_reserved_placeholders(tok=tok, remap=ANNOTATION_TOKEN_REMAP)
    _apply_token_remap(tok=tok, remap=ANNOTATION_TOKEN_REMAP)
    return tok
