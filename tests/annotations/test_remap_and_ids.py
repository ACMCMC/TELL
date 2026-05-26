"""Remap application, added-token tables, and id alignment with training code."""

import pytest

from rl_detector.prompt_utils import (
    ANNOTATION_TOKEN_REMAP,
    ANN_SPECIAL_ID_ANN_PREFIX,
    ANN_SPECIAL_ID_CLOSE,
    ANN_SPECIAL_ID_SCORE_Q,
    ANN_SPECIAL_ID_SPAN_OPEN,
    ANN_SPECIAL_ID_WHY_Q,
    assert_annotation_remap_ids_are_reserved_placeholders,
)


class TestRemapUsesReservedPlaceholdersOnly:
    def test_remap_ids_are_tokenizer_reserved_slots(self):
        from transformers import AutoTokenizer

        try:
            tok = AutoTokenizer.from_pretrained("openai/gpt-oss-120b")
        except Exception:
            pytest.skip("base tokenizer not available")
        assert_annotation_remap_ids_are_reserved_placeholders(tok=tok, remap=ANNOTATION_TOKEN_REMAP)


class TestRemapSingleTokenEncode:
    def test_each_structural_piece_is_one_id(self, remapped_tok):
        for tok_id, token_str in ANNOTATION_TOKEN_REMAP.items():
            ids = remapped_tok.encode(text=token_str, add_special_tokens=False)
            assert ids == [tok_id], f"{token_str!r} → {ids}, want [{tok_id}]"

    def test_span_open_does_not_merge_with_following_text(self, remapped_tok):
        ids = remapped_tok.encode(text="<span>Hello", add_special_tokens=False)
        assert ids[0] == ANN_SPECIAL_ID_SPAN_OPEN

    def test_close_chunk_is_atomic_after_score_tail(self, remapped_tok):
        ids = remapped_tok.encode(text='score="0.5" /></span>', add_special_tokens=False)
        assert ids[-1] == ANN_SPECIAL_ID_CLOSE

    def test_annotation_prefix_is_first_id_of_partial_tag(self, remapped_tok):
        ids = remapped_tok.encode(text='<annotation type="AI"', add_special_tokens=False)
        assert ids[0] == ANN_SPECIAL_ID_ANN_PREFIX


class TestAddedTokenTables:
    def test_encoder_decoder_dicts_match_remap(self, remapped_tok):
        for tok_id, token_str in ANNOTATION_TOKEN_REMAP.items():
            assert remapped_tok.added_tokens_encoder.get(token_str) == tok_id
            assert remapped_tok.added_tokens_decoder[tok_id].content == token_str


class TestRolloutsMaskConstants:
    def test_mask_ids_equal_remap_keys(self, remapped_tok):
        from rl_detector.rollouts import (
            _MASK_ANN_CLOSE,
            _MASK_ANN_OPEN,
            _MASK_ANN_SCORE_Q,
            _MASK_ANN_WHY_Q,
            _MASK_SPAN_OPEN,
            _MASK_TEXT_CLOSE,
            _MASK_TEXT_OPEN,
            _MASK_VERDICT_OPEN,
        )

        mask_ids = (
            _MASK_TEXT_OPEN,
            _MASK_VERDICT_OPEN,
            _MASK_TEXT_CLOSE,
            _MASK_SPAN_OPEN,
            _MASK_ANN_OPEN,
            _MASK_ANN_WHY_Q,
            _MASK_ANN_SCORE_Q,
            _MASK_ANN_CLOSE,
        )
        assert frozenset(mask_ids) == frozenset(ANNOTATION_TOKEN_REMAP.keys())
        for tid in mask_ids:
            want = ANNOTATION_TOKEN_REMAP[tid]
            got = remapped_tok.decode(
                token_ids=[tid],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            assert got == want


class TestSftImportsSameIdsAsRollouts:
    def test_sft_aliases_are_rollouts_objects(self):
        pytest.importorskip("spacy")
        from rl_detector.rollouts import _MASK_ANN_CLOSE, _MASK_ANN_OPEN
        from rl_detector.sft import train_tinker_sft as sft

        assert sft._SFT_ANN_OPEN is _MASK_ANN_OPEN
        assert sft._SFT_ANN_CLOSE is _MASK_ANN_CLOSE
