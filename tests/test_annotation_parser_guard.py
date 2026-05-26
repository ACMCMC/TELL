from rl_detector.tell_xml import wrap_span_piece
from rl_detector.rewards import format_diagnostics


def test_deep_span_nesting_hits_limit_gracefully():
    md = {"type": "AI", "why": "x", "score": "0.1"}
    out = "X"
    for _ in range(350):
        out = wrap_span_piece(out, md)
    diag = format_diagnostics(out, "X")
    assert diag["ok"] is False
