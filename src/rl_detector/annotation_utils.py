"""Public XML surface for tell parsing — thin re-export of ``tell_xml`` symbols.

Older call sites use the ``bracket_*`` / ``*_raw`` names; this module aliases them so that
new code can import either spelling.
"""

from __future__ import annotations

from rl_detector.tell_xml import (
    SP_CL,
    SP_OP,
    collect_bracket_tells_raw as collect_bracket_tells,
    collect_inner_rollout_fragment as _collect_inner,
    get_outer_meta_dict as get_outer_bracket_metadata,
    strip_all_marks_raw as strip_all_bracket_annotations,
    strip_score_attrs,
    wrap_outer_logical_plain_mid,
    wrap_span_piece,
)

# Legacy alias (older callers used the *_raw spelling).
strip_all_bracket_annotations_raw = strip_all_bracket_annotations


__all__ = [
    "SP_CL",
    "SP_OP",
    "_collect_inner",
    "collect_bracket_tells",
    "get_outer_bracket_metadata",
    "strip_all_bracket_annotations",
    "strip_all_bracket_annotations_raw",
    "strip_score_attrs",
    "wrap_outer_logical_plain_mid",
    "wrap_span_piece",
]

