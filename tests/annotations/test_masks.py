"""``compute_annotation_token_mask`` and structural mask tests."""

import pytest

from rl_detector.prompt_utils import ANNOTATION_TOKEN_REMAP
from rl_detector.rollouts import (
    _MASK_ANN_CLOSE,
    _MASK_ANN_OPEN,
    _MASK_ANN_SCORE_Q,
    _MASK_ANN_WHY_Q,
    _MASK_SPAN_OPEN,
    _MASK_TEXT_CLOSE,
    _MASK_TEXT_OPEN,
    _MASK_VERDICT_OPEN,
    compute_annotation_token_mask,
    compute_structural_token_mask,
)

from . import wire_samples as ws

S = _MASK_SPAN_OPEN
A = _MASK_ANN_OPEN
W = _MASK_ANN_WHY_Q
Q = _MASK_ANN_SCORE_Q
C = _MASK_ANN_CLOSE
T0 = _MASK_TEXT_OPEN
TV = _MASK_VERDICT_OPEN
TX = _MASK_TEXT_CLOSE
D = 42
_STRUCT = frozenset(ANNOTATION_TOKEN_REMAP.keys())


def _mask(tokens, R=0):
    return compute_annotation_token_mask(tokenizer=None, completion_tokens=tokens, n_reasoning_tokens=R)


class TestAnnotationTokenMaskSynthetic:
    def test_empty_response(self):
        assert _mask(tokens=[], R=0) == []

    def test_all_reasoning_skipped(self):
        assert _mask(tokens=[D, D, D], R=3) == []

    def test_reasoning_prefix_skipped(self):
        assert _mask(tokens=[D, D, S, D, A, D, C], R=2) == [1.0, 0.0, 1.0, 1.0, 1.0]

    def test_doc_tokens_zero(self):
        assert _mask(tokens=[D, D, D]) == [0.0, 0.0, 0.0]

    def test_span_ann_close_singletons(self):
        assert _mask(tokens=[S]) == [1.0]
        assert _mask(tokens=[A]) == [1.0]
        assert _mask(tokens=[C]) == [1.0]

    def test_why_and_score_delimiter_tokens_inside_attrs(self):
        tokens = [S, D, A, 99, W, 100, Q, 101, C]
        assert _mask(tokens=tokens) == [1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    def test_doc_between_span_and_annotation(self):
        tokens = [S, D, D, A, D, D, C]
        assert _mask(tokens=tokens) == [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]

    def test_attrs_between_open_and_close(self):
        tokens = [S, A, 10, 11, 12, C]
        assert _mask(tokens=tokens) == [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    def test_doc_after_close(self):
        tokens = [S, D, A, D, C, D, D]
        assert _mask(tokens=tokens) == [1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0]

    def test_two_sequential_spans(self):
        tokens = [S, D, A, D, C, S, D, A, D, C]
        assert _mask(tokens=tokens) == [1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0]

    def test_nested_spans(self):
        tokens = [S, D, S, D, A, D, C, A, D, C]
        assert _mask(tokens=tokens) == [1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    def test_reasoning_larger_than_tokens(self):
        assert _mask(tokens=[S, D, A], R=10) == []

    def test_r_equals_token_count(self):
        tokens = [S, D, A, C]
        assert _mask(tokens=tokens, R=4) == []

    def test_mask_length_matches_response_for_all_R(self):
        tokens = [S, D, A, D, C, D]
        for R in range(len(tokens) + 1):
            m = _mask(tokens=tokens, R=R)
            assert len(m) == max(0, len(tokens) - R)

    def test_unclosed_annotation_stays_attrs_until_end(self):
        tokens = [S, D, D, A, D, D, D]
        assert _mask(tokens=tokens) == [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]

    def test_doc_only_no_structure(self):
        assert _mask(tokens=[D, D, D, D, D]) == [0.0] * 5

    def test_many_attr_tokens(self):
        attr_tokens = list(range(1000, 1050))
        tokens = [S, A] + attr_tokens + [C]
        m = _mask(tokens=tokens)
        assert m[0] == 1.0
        assert m[1] == 1.0
        assert all(v == 1.0 for v in m[2:-1])
        assert m[-1] == 1.0

    def test_close_then_open_outer_span(self):
        tokens = [S, A, C, S, D, A, C]
        assert _mask(tokens=tokens) == [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0]

    def test_text_verdict_wrapper_mask(self):
        tokens = [T0, S, D, A, D, C, TV, D, D, TX, D]
        assert _mask(tokens=tokens) == [1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]


class TestMaskOnRealEncodedWire:
    @pytest.mark.parametrize("wire", ws.ALL_ROUNDTRIP_WIRES)
    def test_structural_ids_always_weighted(self, remapped_tok, wire):
        ids = remapped_tok.encode(text=wire, add_special_tokens=False)
        m = compute_structural_token_mask(completion_tokens=ids, n_reasoning_tokens=0)
        for i, tid in enumerate(ids):
            assert m[i] == (tid in _STRUCT)

    @pytest.mark.parametrize("wire", ws.ALL_ROUNDTRIP_WIRES)
    def test_mask_has_both_doc_zero_and_attr_one_regions(self, remapped_tok, wire):
        ids = remapped_tok.encode(text=wire, add_special_tokens=False)
        am = compute_annotation_token_mask(tokenizer=None, completion_tokens=ids, n_reasoning_tokens=0)
        assert 0.0 in am and 1.0 in am

    @pytest.mark.parametrize("wire", ws.ALL_ROUNDTRIP_WIRES)
    def test_structural_tokens_one_in_annotation_mask(self, remapped_tok, wire):
        ids = remapped_tok.encode(text=wire, add_special_tokens=False)
        am = compute_annotation_token_mask(tokenizer=None, completion_tokens=ids, n_reasoning_tokens=0)
        for i, tid in enumerate(ids):
            if tid in _STRUCT:
                assert am[i] == 1.0


