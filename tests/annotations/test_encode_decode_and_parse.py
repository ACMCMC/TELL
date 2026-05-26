"""Encode/decode roundtrip, ``tell_xml`` parse, and checkpoint-specific golden id lists."""

import pytest

from rl_detector.prompt_utils import (
    ANNOTATION_TOKEN_REMAP,
    ANN_SPECIAL_ID_ANN_PREFIX,
    ANN_SPECIAL_ID_CLOSE,
    ANN_SPECIAL_ID_SCORE_Q,
    ANN_SPECIAL_ID_SPAN_OPEN,
    ANN_SPECIAL_ID_TEXT_CLOSE,
    ANN_SPECIAL_ID_TEXT_OPEN,
    ANN_SPECIAL_ID_VERDICT_PREFIX,
    ANN_SPECIAL_ID_WHY_Q,
)
from rl_detector.rollouts import (
    _MASK_ANN_CLOSE,
    _MASK_ANN_OPEN,
    _MASK_SPAN_OPEN,
)
from rl_detector.tell_xml import root_splits, wrap_outer_logical_plain_mid

from . import wire_samples as ws

# ---------------------------------------------------------------------------
# Checkpoint goldens (``openai/gpt-oss-120b`` + remap). Refresh when vocab changes.
# ---------------------------------------------------------------------------

_GOLDEN_MINIMAL_IDS = [
    ANN_SPECIAL_ID_SPAN_OPEN,
    87,
    ANN_SPECIAL_ID_ANN_PREFIX,
    17527,
    ANN_SPECIAL_ID_WHY_Q,
    525,
    ANN_SPECIAL_ID_SCORE_Q,
    15,
    13,
    1434,
    ANN_SPECIAL_ID_CLOSE,
]

_GOLDEN_NESTED_IDS = [
    ANN_SPECIAL_ID_SPAN_OPEN,
    64,
    ANN_SPECIAL_ID_SPAN_OPEN,
    65,
    ANN_SPECIAL_ID_ANN_PREFIX,
    17527,
    ANN_SPECIAL_ID_WHY_Q,
    77,
    ANN_SPECIAL_ID_SCORE_Q,
    15,
    13,
    19,
    ANN_SPECIAL_ID_CLOSE,
    ANN_SPECIAL_ID_ANN_PREFIX,
    51527,
    ANN_SPECIAL_ID_WHY_Q,
    78,
    ANN_SPECIAL_ID_SCORE_Q,
    15,
    13,
    20,
    ANN_SPECIAL_ID_CLOSE,
]


_GOLDEN_MINIMAL_OUTER_IDS = [
    ANN_SPECIAL_ID_TEXT_OPEN,
    ANN_SPECIAL_ID_SPAN_OPEN,
    87,
    ANN_SPECIAL_ID_ANN_PREFIX,
    17527,
    ANN_SPECIAL_ID_WHY_Q,
    525,
    ANN_SPECIAL_ID_SCORE_Q,
    15,
    13,
    1434,
    ANN_SPECIAL_ID_CLOSE,
    ANN_SPECIAL_ID_VERDICT_PREFIX,
    17527,
    ANN_SPECIAL_ID_WHY_Q,
    525,
    ANN_SPECIAL_ID_SCORE_Q,
    15,
    13,
    1434,
    ANN_SPECIAL_ID_TEXT_CLOSE,
]


class TestRemapKeySet:
    def test_eight_distinct_ids(self):
        assert set(ANNOTATION_TOKEN_REMAP.keys()) == {
            ANN_SPECIAL_ID_TEXT_OPEN,
            ANN_SPECIAL_ID_VERDICT_PREFIX,
            ANN_SPECIAL_ID_TEXT_CLOSE,
            ANN_SPECIAL_ID_SPAN_OPEN,
            ANN_SPECIAL_ID_ANN_PREFIX,
            ANN_SPECIAL_ID_WHY_Q,
            ANN_SPECIAL_ID_SCORE_Q,
            ANN_SPECIAL_ID_CLOSE,
        }


class TestCheckpointGoldens:
    def test_minimal_wire_encode_matches_golden(self, remapped_tok):
        ids = remapped_tok.encode(text=ws.WIRE_MINIMAL, add_special_tokens=False)
        assert ids == _GOLDEN_MINIMAL_IDS

    def test_nested_two_tells_encode_matches_golden(self, remapped_tok):
        ids = remapped_tok.encode(text=ws.WIRE_NESTED_TWO_TELLS, add_special_tokens=False)
        assert ids == _GOLDEN_NESTED_IDS

    def test_minimal_outer_wire_encode_matches_golden(self, remapped_tok):
        ids = remapped_tok.encode(text=ws.WIRE_MINIMAL_OUTER, add_special_tokens=False)
        assert ids == _GOLDEN_MINIMAL_OUTER_IDS


class TestRoundtripAllSamples:
    @pytest.mark.parametrize("wire", ws.ALL_ROUNDTRIP_WIRES)
    def test_encode_decode_identity(self, remapped_tok, wire):
        ids = remapped_tok.encode(text=wire, add_special_tokens=False)
        out = remapped_tok.decode(
            token_ids=ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        assert out == wire


class TestStructuralIdCounts:
    _INNER = frozenset(
        {
            ANN_SPECIAL_ID_SPAN_OPEN,
            ANN_SPECIAL_ID_ANN_PREFIX,
            ANN_SPECIAL_ID_WHY_Q,
            ANN_SPECIAL_ID_SCORE_Q,
            ANN_SPECIAL_ID_CLOSE,
        }
    )
    _OUTER = frozenset(
        {
            ANN_SPECIAL_ID_TEXT_OPEN,
            ANN_SPECIAL_ID_VERDICT_PREFIX,
            ANN_SPECIAL_ID_TEXT_CLOSE,
        }
    )

    def test_minimal_inner_once_outer_absent(self, remapped_tok):
        ids = remapped_tok.encode(text=ws.WIRE_MINIMAL, add_special_tokens=False)
        for tid in self._INNER:
            assert ids.count(tid) == 1
        for tid in self._OUTER:
            assert ids.count(tid) == 0

    def test_nested_two_tells_inner_twice_outer_absent(self, remapped_tok):
        ids = remapped_tok.encode(text=ws.WIRE_NESTED_TWO_TELLS, add_special_tokens=False)
        for tid in self._INNER:
            assert ids.count(tid) == 2
        for tid in self._OUTER:
            assert ids.count(tid) == 0

    def test_minimal_outer_wire_each_outer_once(self, remapped_tok):
        ids = remapped_tok.encode(text=ws.WIRE_MINIMAL_OUTER, add_special_tokens=False)
        for tid in self._OUTER:
            assert ids.count(tid) == 1
        for tid in self._INNER:
            if tid in (ANN_SPECIAL_ID_WHY_Q, ANN_SPECIAL_ID_SCORE_Q):
                assert ids.count(tid) == 2
            else:
                assert ids.count(tid) == 1

    def test_review_nested_has_expected_special_multiplicity(self, remapped_tok):
        ids = remapped_tok.encode(text=ws.WIRE_REVIEW_NESTED, add_special_tokens=False)
        assert ids.count(ANN_SPECIAL_ID_SPAN_OPEN) == 2
        assert ids.count(ANN_SPECIAL_ID_ANN_PREFIX) == 2
        assert ids.count(ANN_SPECIAL_ID_CLOSE) == 2
        for tid in self._OUTER:
            assert ids.count(tid) == 0


class TestBoundaryAtomicity:
    def test_plain_text_before_span_does_not_absorb_span_token(self, remapped_tok):
        wire = "pre<span>z"
        ids = remapped_tok.encode(text=wire, add_special_tokens=False)
        assert remapped_tok.decode(token_ids=ids, skip_special_tokens=False, clean_up_tokenization_spaces=False) == wire
        assert ids.count(_MASK_SPAN_OPEN) == 1
        assert ids[ids.index(_MASK_SPAN_OPEN) - 1] != _MASK_SPAN_OPEN

    def test_plain_text_after_close_suffix_chunk(self, remapped_tok):
        wire = '0.1" /></span>post'
        ids = remapped_tok.encode(text=wire, add_special_tokens=False)
        assert remapped_tok.decode(token_ids=ids, skip_special_tokens=False, clean_up_tokenization_spaces=False) == wire
        assert ids.count(_MASK_ANN_CLOSE) == 1
        assert ids.index(_MASK_ANN_CLOSE) == len(ids) - 2


class TestWrapOuterLogicalPlainMid:
    def test_roundtrip_and_root_splits(self, remapped_tok):
        meta = {"type": "AI", "why": "short", "score": "0.33"}
        wire = wrap_outer_logical_plain_mid(mid_logical_plaintext="Hello world", meta=meta)
        ids = remapped_tok.encode(text=wire, add_special_tokens=False)
        back = remapped_tok.decode(
            token_ids=ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        assert back == wire
        inn, desc, outer, ok, _aft = root_splits(tx=back)
        assert ok and outer is not None
        assert inn == "Hello world"
        assert outer["type"] == "AI"
        assert outer["why"] == "short"
        assert outer["score"] == "0.33"
        assert isinstance(desc, list)


class TestRootSplitsRealisticWires:
    @pytest.mark.parametrize(
        "wire",
        [ws.WIRE_REVIEW_NESTED, ws.WIRE_TRIPLE_SPAN, ws.WIRE_NESTED_TWO_TELLS],
    )
    def test_parse_ok_after_token_roundtrip(self, remapped_tok, wire):
        ids = remapped_tok.encode(text=wire, add_special_tokens=False)
        back = remapped_tok.decode(
            token_ids=ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        _inn, _desc, outer, ok, _aft = root_splits(tx=back)
        assert ok and outer is not None
