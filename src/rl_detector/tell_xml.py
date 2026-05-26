"""Nested ``<span>`` + void ``<annotation type= AI|human why=… score=… />``.

Exactly one escaping rule turns LOGICAL dataset UTF‑8 into the XML the model reads and emits
inside fences and span bodies:

- ``escape_document_piece(logical_plaintext)`` is that single source (``<<< >>>`` payloads AND every
  plaintext fragment inside ``<span>…`` before markup).
- Prefer ``wrap_logical_leaf_span`` (nested leaves) or ``wrap_outer_logical_plain_mid`` (whole logical
  body) when emitting XML from dataset text so callers never duplicate ``escape_document_piece`` +
  ``wrap_span_piece``.
- When ``mid`` is already concatenated wired fragments (escaped runs + nested ``<span>`` XML),
  keep using ``wrap_span_piece`` alone.
"""

from __future__ import annotations

import re
from xml.sax.saxutils import escape as xml_esc

_O = "<span>"
_C = "</span>"
_TEXT_O = "<text>"
_VERDICT_PREF = '<verdict type="'
_TEXT_CLOSE_CHUNK = '" /></text>'
_RETURN_TOKEN = "<|return|>"
_PREF = "<annotation "
_DMAX = 256
_SCNUM = re.compile(r"^[0-9]*\.?[0-9]+$")
_ANN_RE = re.compile(
    r'^<annotation type="(AI|human)" why="((?:[^"\\]|\\.)*)"(?: score="([0-9]*\.?[0-9]+)")?\s*/>'
)
_VERDICT_RE = re.compile(
    r'^<verdict type="(AI|human)" why="((?:[^"\\]|\\.)*)" score="([0-9]*\.?[0-9]+)'
    + re.escape(_TEXT_CLOSE_CHUNK)
    + r"$",
)

SP_OP = _O
SP_CL = _C
AT_OP = _PREF


_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _strip_ctrl(tx: str) -> str:
    """Remove XML 1.0 forbidden control characters (all except \t \n \r)."""
    return _CTRL_RE.sub("", tx)


def _escape_xml_text_and_attr_piece(tx: str) -> str:
    """Single source of truth for XML 5-entity escaping after XML control stripping."""
    return xml_esc(_strip_ctrl(tx), {'"': "&quot;", "'": "&apos;"})


def canonical_logical_document(tx: str) -> str:
    """Canonical logical form expected from the model: what ``escape_document_piece`` encodes, decoded back.

    This is the comparison target for format checks — it equals the model's inner span text when the
    model faithfully echoes what it saw in the ``<<<>>>`` fence.  Differs from raw ``tx`` only when
    ``tx`` contains XML-forbidden control characters (stripped by ``escape_document_piece``).
    """
    return _strip_ctrl(tx)


def escape_document_piece(tx: str) -> str:
    """LOGICAL plaintext -> XML wire inside ``<<<>>>`` fences and ``<span>`` text nodes (attrs: ``escape_attr_piece``)."""
    return _escape_xml_text_and_attr_piece(tx)


def escape_attr_piece(tx: str) -> str:
    """Escape for XML attribute values — delegates to the same canonical escaper."""
    return _escape_xml_text_and_attr_piece(tx)


def _dent(tx: str) -> str:
    import html as _html

    return _html.unescape(tx)


def format_annotation_tag(meta: dict[str, str]) -> str:
    t = escape_attr_piece(meta["type"])
    w = escape_attr_piece((meta.get("why") or "").strip())
    sc = escape_attr_piece(str(meta["score"]))
    return f'<annotation type="{t}" why="{w}" score="{sc}" />'


def wrap_span_piece(mid: str, meta: dict[str, str]) -> str:
    return f"{_O}{mid}{format_annotation_tag(meta)}{_C}"


def wrap_logical_leaf_span(span_logical_plaintext: str, meta: dict[str, str]) -> str:
    """One nested leaf span from LOGICAL document substring."""
    return wrap_span_piece(escape_document_piece(span_logical_plaintext), meta)


def wrap_outer_logical_plain_mid(mid_logical_plaintext: str, meta: dict[str, str]) -> str:
    """Outer ``<text>`` wrapper + escaped doc + ``<verdict/>`` tail (new format, no outer span).

    ``mid`` is LOGICAL document text only (HF row), never pre-escaped.
    Produces: ``<text>escaped_doc<verdict type=… /></text>``.
    """
    t = escape_attr_piece(meta["type"])
    w = escape_attr_piece((meta.get("why") or "").strip())
    sc = escape_attr_piece(str(meta["score"]))
    return _TEXT_O + escape_document_piece(mid_logical_plaintext) + f'{_VERDICT_PREF}{t}" why="{w}" score="{sc}' + _TEXT_CLOSE_CHUNK


def strip_return_token(tx: str) -> str:
    """Strip trailing ``<|return|>`` (model EOS marker) if present."""
    return tx[: -len(_RETURN_TOKEN)] if tx.endswith(_RETURN_TOKEN) else tx


def _canon_tell_type_attr(raw: str) -> str:
    u = raw.strip().upper()
    return "AI" if u == "AI" else "human"


def normalize_loose_annotation_verdict_attr_spacing(tx: str) -> str:
    """Strip stray spaces inside ``type="…"`` and ``score="…"`` on ``<annotation/>`` and ``<verdict/>`` wire."""
    out = re.sub(
        r'type="\s*((?:AI|human))\s*"',
        lambda m: 'type="' + _canon_tell_type_attr(m.group(1)) + '"',
        tx,
        flags=re.IGNORECASE,
    )
    out = re.sub(r'score="\s*([0-9]*\.?[0-9]+)\s*"', r'score="\1"', out)
    return out


def strip_text_wrapper(tx: str) -> str | None:
    """If ``tx`` is ``<text>`` … ``\" /></text>`` (with optional trailing ``<|return|>``), return
    the inner slice before the ``<verdict`` tail; else None."""
    tx = strip_return_token(tx)
    if not tx.startswith(_TEXT_O) or not tx.endswith(_TEXT_CLOSE_CHUNK):
        return None
    inner = tx[len(_TEXT_O) : -len(_TEXT_CLOSE_CHUNK)]
    vp = inner.rfind(_VERDICT_PREF)
    if vp < 0:
        return None
    return inner[:vp]


def _rd_verdict_wrapped(suf: str) -> tuple[dict[str, str], bool] | None:
    scan = suf if suf.endswith(_TEXT_CLOSE_CHUNK) else suf + _TEXT_CLOSE_CHUNK
    mx = _VERDICT_RE.match(scan)
    if not mx:
        return None
    md, ok = _meta_from_re(mx)
    return md, ok


def _twy(typ: str, wy: str) -> str:
    r0 = wy.strip()
    r0 = re.sub(r"^id=\d+\s+", "", r0)
    return r0


def _meta_from_re(m: re.Match[str]) -> tuple[dict[str, str], bool]:
    wy_raw = ""
    bs = False
    for ch in m.group(2):
        if bs:
            wy_raw += ch
            bs = False
        elif ch == "\\":
            bs = True
        else:
            wy_raw += ch
    typ = m.group(1)
    wy = _twy(typ, _dent(wy_raw))
    sc_raw = m.group(3)
    sc = (sc_raw or "0.5").strip()
    md = {"type": typ, "why": wy, "score": sc}
    ok = bool(wy.strip()) and _SCNUM.match(sc) is not None
    fv = float(sc)
    ok = ok and 0.0 <= fv <= 1.0
    return md, ok


def _rd_ann(tx: str, i: int) -> tuple[dict[str, str], bool, int] | None:
    suf = tx[i:]
    mx = _ANN_RE.match(suf)
    if not mx:
        return None
    md, ok = _meta_from_re(mx)
    return md, ok, i + mx.end()


def _walk_until_ann(tx: str, i: int, dep: int) -> tuple[str, list[dict], int, bool]:
    """Accum decoded text before OWN <annotation>; list is preorder subtree tells for nested <span>s (each incl its node)."""
    if dep > _DMAX:
        return "", [], i, False
    pcs: list[str] = []
    pcs_len = 0
    desc_acc: list[dict] = []
    while i < len(tx):
        if tx.startswith(_PREF, i):
            return "".join(pcs), desc_acc, i, True
        if tx.startswith(_C, i):
            return "", [], i, False
        if tx.startswith(_O, i):
            inn_p, preorder_here, aft, ok_sub = _parse_span_bundle(tx, i, dep + 1)
            if not ok_sub:
                return "", [], i, False
            if preorder_here:
                # Root tell of this subtree starts at pcs_len in our accumulated plain text.
                preorder_here[0]["_inner_pos"] = pcs_len
                # Nested tells have _inner_pos relative to child's inner_plain; shift to our level.
                for t in preorder_here[1:]:
                    if "_inner_pos" in t:
                        t["_inner_pos"] += pcs_len
            pcs.append(inn_p)
            pcs_len += len(inn_p)
            desc_acc.extend(preorder_here)
            i = aft
            continue
        nx = tx.find("<", i)
        if nx < 0:
            return "", [], i, False
        if nx == i:
            return "", [], i, False
        chunk = _dent(tx[i:nx])
        pcs.append(chunk)
        pcs_len += len(chunk)
        i = nx
    return "", [], i, False


def _parse_span_bundle(tx: str, idx: int, dep: int) -> tuple[str, list[dict], int, bool]:
    """One <span…</span>; preorder list starts with THIS span meta row then nested preorder segments."""
    if not tx.startswith(_O, idx):
        return "", [], idx, False
    inn_plain, nested_pre, ai, ook = _walk_until_ann(tx, idx + len(_O), dep)
    if not ook:
        return "", [], idx, False
    apr = _rd_ann(tx, ai)
    if apr is None:
        return "", [], idx, False
    md_m, mk_ok, ap_end = apr
    if not tx.startswith(_C, ap_end):
        return "", [], idx, False
    aft_all = ap_end + len(_C)
    leaf = {"span_text": inn_plain, "explanation": md_m.get("why", ""), "type": md_m.get("type"), "score": md_m.get("score", "0.0")}
    if dep > 0:
        # Nested span: mark so SPAN_OPEN gets reward=0 in per-span-open advantage computation.
        # Content tokens (ann_why, ann_type, ann_score) are still scored normally by the rubric.
        leaf["_nested"] = True
    if mk_ok:
        preorder = [leaf] + nested_pre
    else:
        preorder = nested_pre
    return inn_plain, preorder, aft_all, True


def _walk_inner_xml_flat(tx: str) -> tuple[str, list[dict], bool]:
    """Parse new-format inner doc content: XML-escaped plain text + inline ``<span>`` tells.

    Returns (plain_doc_text, tells_preorder, ok).
    ok=False on any unexpected ``<`` tag or malformed tell.
    """
    pcs: list[str] = []
    pcs_len = 0
    tells: list[dict] = []
    i = 0
    n = len(tx)
    while i < n:
        if tx.startswith(_O, i):
            inn_p, preorder_here, aft, ok_sub = _parse_span_bundle(tx, i, 0)
            if not ok_sub:
                return "", [], False
            if preorder_here:
                preorder_here[0]["_inner_pos"] = pcs_len
                for t in preorder_here[1:]:
                    if "_inner_pos" in t:
                        t["_inner_pos"] += pcs_len
            pcs.append(inn_p)
            pcs_len += len(inn_p)
            tells.extend(preorder_here)
            i = aft
        elif tx.startswith("<", i):
            return "", [], False
        else:
            nx = tx.find("<", i)
            if nx < 0:
                pcs.append(_dent(tx[i:]))
                break
            chunk = _dent(tx[i:nx])
            pcs.append(chunk)
            pcs_len += len(chunk)
            i = nx
    return "".join(pcs), tells, True


def _esc_ok_flat_inner(tx: str) -> bool:
    """True iff all plaintext runs in flat inner doc content have canonical XML escaping."""
    i = 0
    n = len(tx)
    while i < n:
        if tx.startswith(_O, i):
            ok_sub, aft = _esc_ok_nested_span_bundle(tx, i, 0)
            if not ok_sub:
                return False
            i = aft
        elif tx.startswith("<", i):
            return False
        else:
            nx = tx.find("<", i)
            if nx < 0:
                raw = tx[i:]
                return raw == escape_document_piece(_dent(raw))
            raw = tx[i:nx]
            if raw != escape_document_piece(_dent(raw)):
                return False
            i = nx
    return True


def _ann_attrs_ok_flat_inner(tx: str) -> bool:
    """True iff all annotation attr escaping in flat inner doc content is canonical."""
    i = 0
    n = len(tx)
    while i < n:
        if tx.startswith(_O, i):
            ok_sub, aft = _ann_attr_nested_span_bundle(tx, i, 0)
            if not ok_sub:
                return False
            i = aft
        elif tx.startswith("<", i):
            return False
        else:
            nx = tx.find("<", i)
            if nx < 0:
                break
            i = nx
    return True



def root_splits(tx: str) -> tuple[str, list[dict], dict[str, str] | None, bool, int]:
    """root inner doc str, nested-only tells preorder, outer meta or None."""
    tx = strip_return_token(tx)
    legacy = strip_text_wrapper(tx)
    if legacy is not None:
        inner_full = tx[len(_TEXT_O) : -len(_TEXT_CLOSE_CHUNK)]
        vp = inner_full.rfind(_VERDICT_PREF)
        if vp < 0:
            return "", [], None, False, 0
        vr = _rd_verdict_wrapped(inner_full[vp:])
        if vr is None:
            return "", [], None, False, 0
        meta_v, ok_v = vr
        inn, desc, ok_flat = _walk_inner_xml_flat(legacy)
        if not ok_flat:
            return "", [], None, False, 0
        if not ok_v:
            return inn, desc, None, False, 0
        return inn, desc, meta_v, True, len(tx)
    if not tx.startswith(_O, 0):
        return "", [], None, False, 0
    inn_plain, desc_only_pre, ai, ook = _walk_until_ann(tx, len(_O), 0)
    if not ook:
        return "", [], None, False, 0
    apr = _rd_ann(tx, ai)
    if apr is None:
        return "", [], None, False, 0
    outer_m, outer_ok, ap_end = apr
    if not tx.startswith(_C, ap_end):
        return "", [], None, False, 0
    aft = ap_end + len(_C)
    if aft != len(tx):
        return "", [], None, False, aft
    # drop root tell from preorder: desc_only_pre lists [nested nodes in doc order preorder]
    return inn_plain, desc_only_pre, outer_m if outer_ok else None, bool(outer_ok), aft


def strip_all_marks_raw(tx: str) -> str:
    inn, _, meta, ok, _ = root_splits(tx)
    if not ok or meta is None:
        return tx
    return inn


def collect_bracket_tells_raw(tx: str) -> list[dict] | None:
    _inn, desc, meta, ok, _ = root_splits(tx)
    if not ok or meta is None:
        return None
    return desc


def get_outer_meta_dict(tx: str) -> dict | None:
    _inn, _desc, meta, ok, _ = root_splits(tx)
    if not ok or meta is None:
        return None
    return {"type": meta.get("type"), "explanation": meta.get("why", ""), "score_magnitude": float(meta["score"])}


def collect_inner_rollout_fragment(tx: str, tells_acc: list) -> tuple[str, int, bool]:
    """Like old _collect_inner: inner doc BEFORE root annot, nested tells appended, idx at annot start."""
    leg = strip_text_wrapper(tx=tx)
    if leg is not None:
        inn, desc, ok_flat = _walk_inner_xml_flat(leg)
        if not ok_flat:
            return "", 0, False
        tells_acc.extend(desc)
        return inn, len(tx), True
    if not tx.startswith(_O, 0):
        return "", 0, False
    inn_plain, desc, ai, ok_int = _walk_until_ann(tx, len(_O), 0)
    if not ok_int:
        return "", 0, False
    tells_acc.extend(desc)
    return inn_plain, ai, True


def strip_score_attrs(tx: str) -> str:
    out: list[str] = []
    i = 0
    ln = len(tx)
    while i < ln:
        j = tx.find(_PREF, i)
        if j < 0:
            out.append(tx[i:])
            break
        out.append(tx[i:j])
        ap = _rd_ann(tx, j)
        if ap is None:
            out.append(tx[j])
            i = j + 1
            continue
        md_m, _okx, endm = ap
        w2 = escape_attr_piece(md_m["why"])
        t2 = md_m["type"]
        out.append(f'<annotation type="{t2}" why="{w2}" />')
        i = endm
    return "".join(out)




def _annotation_tag_attrs_match_canonical_escaping(tx: str, ai: int) -> bool:
    suf = tx[ai:]
    mx = _ANN_RE.match(suf)
    if not mx:
        return False
    md, _mk = _meta_from_re(mx)
    g1, g2, g3 = mx.group(1), mx.group(2), mx.group(3)
    if escape_attr_piece(md["type"]) != g1:
        return False
    if escape_attr_piece((md.get("why") or "").strip()) != g2:
        return False
    esc_sc = escape_attr_piece(str(md["score"]))
    if g3 is not None and g3 != esc_sc:
        return False
    return True


def _ann_attr_walk_plain_runs(tx: str, i: int, dep: int) -> tuple[bool, int | None]:
    """Find OWN ``<annotation`` for this span; plaintext runs not inspected."""
    if dep > _DMAX:
        return False, None
    while i < len(tx):
        if tx.startswith(_PREF, i):
            return True, i
        if tx.startswith(_C, i):
            return False, None
        if tx.startswith(_O, i):
            ok_sub, aft = _ann_attr_nested_span_bundle(tx, i, dep + 1)
            if not ok_sub:
                return False, None
            i = aft
            continue
        nx = tx.find("<", i)
        if nx < 0:
            return False, None
        if nx == i:
            return False, None
        i = nx
    return False, None


def _ann_attr_nested_span_bundle(tx: str, idx: int, dep: int) -> tuple[bool, int]:
    if not tx.startswith(_O, idx):
        return False, idx
    hs, ai = _ann_attr_walk_plain_runs(tx, idx + len(_O), dep)
    if not hs or ai is None:
        return False, idx
    if not _annotation_tag_attrs_match_canonical_escaping(tx, ai):
        return False, idx
    apr = _rd_ann(tx, ai)
    if apr is None:
        return False, idx
    _md, _mk, ap_end = apr
    if not tx.startswith(_C, ap_end):
        return False, idx
    return True, ap_end + len(_C)



def _esc_ok_walk_plain_runs(tx: str, i: int, dep: int) -> tuple[bool, int | None]:
    """Return (hits_own_annotation_here, annot_start_abs_idx). Mirrors _walk_until_ann shape."""
    if dep > _DMAX:
        return False, None
    while i < len(tx):
        if tx.startswith(_PREF, i):
            return True, i
        if tx.startswith(_C, i):
            return False, None
        if tx.startswith(_O, i):
            ok_sub, aft = _esc_ok_nested_span_bundle(tx, i, dep + 1)
            if not ok_sub:
                return False, None
            i = aft
            continue
        nx = tx.find("<", i)
        if nx < 0:
            return False, None
        if nx == i:
            return False, None
        raw = tx[i:nx]
        if raw != escape_document_piece(_dent(raw)):
            return False, None
        i = nx
    return False, None


def _esc_ok_nested_span_bundle(tx: str, idx: int, dep: int) -> tuple[bool, int]:
    """True when nested span subtree has canonical escaped plaintext runs."""
    if not tx.startswith(_O, idx):
        return False, idx
    hs, ai = _esc_ok_walk_plain_runs(tx, idx + len(_O), dep)
    if not hs or ai is None:
        return False, idx
    apr = _rd_ann(tx, ai)
    if apr is None:
        return False, idx
    _md, _mk, ap_end = apr
    if not tx.startswith(_C, ap_end):
        return False, idx
    return True, ap_end + len(_C)



def _verdict_wire_attrs_match_canonical_escaping(suf: str) -> bool:
    scan = suf if suf.endswith(_TEXT_CLOSE_CHUNK) else suf + _TEXT_CLOSE_CHUNK
    mx = _VERDICT_RE.match(scan)
    if mx is None:
        return False
    md, ok = _meta_from_re(mx)
    if not ok:
        return False
    g1, g2, g3 = mx.group(1), mx.group(2), mx.group(3)
    if escape_attr_piece(md["type"]) != g1:
        return False
    if escape_attr_piece((md.get("why") or "").strip()) != g2:
        return False
    esc_sc = escape_attr_piece(str(md["score"]))
    if g3 != esc_sc:
        return False
    return True


def full_markup_wire_escaping_ok(full: str) -> bool:
    leg = strip_text_wrapper(tx=full)
    if leg is None:
        return False
    if not full.startswith(_TEXT_O) or not full.endswith(_TEXT_CLOSE_CHUNK):
        return False
    inner_full = full[len(_TEXT_O) : -len(_TEXT_CLOSE_CHUNK)]
    vp = inner_full.rfind(_VERDICT_PREF)
    if vp < 0:
        return False
    if not _verdict_wire_attrs_match_canonical_escaping(suf=inner_full[vp:]):
        return False
    return _esc_ok_flat_inner(leg) and _ann_attrs_ok_flat_inner(leg)
