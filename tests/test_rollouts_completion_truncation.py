"""Max-token budget truncation on normal rollouts yields invalid markup after extraction."""

from rl_detector.rewards import format_diagnostics
from rl_detector.rollouts import extract_response_text
from rl_detector.tell_xml import escape_document_piece, wrap_outer_logical_plain_mid

_COMPLETION_STUB = (
    "<|end|><|start|>assistant<|channel|>analysis<|message|>"
    "Text origin is human."
    "<|end|><|start|>assistant<|channel|>final<|message|>"
)


def test_truncated_budget_completion_fails_format_after_extract_response_text():
    document = "Hello world."
    outer_meta = {"type": "human", "why": "blah blah", "score": "0.82"}
    wired = wrap_outer_logical_plain_mid(mid_logical_plaintext=document, meta=outer_meta)
    completion_complete = _COMPLETION_STUB + wired
    assert extract_response_text(text=completion_complete) == wired
    assert format_diagnostics(output=wired, document=document)["ok"] is True

    # truncate: remove the closing " /></text>" so it's malformed
    from rl_detector.tell_xml import _TEXT_CLOSE_CHUNK
    truncated_wired = wired[: -len(_TEXT_CLOSE_CHUNK)]
    assert not truncated_wired.endswith(_TEXT_CLOSE_CHUNK)
    completion_truncated = _COMPLETION_STUB + truncated_wired
    response_truncated = extract_response_text(text=completion_truncated)
    diag = format_diagnostics(output=response_truncated, document=document)
    assert diag["ok"] is False
    assert diag["reason"] in ("missing_outer_annotation", "annotation_parse_failed")
