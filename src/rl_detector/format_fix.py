"""Repair malformed ``<text>…<verdict/></text>`` completions using the dataset document as ground truth."""

import logging
import re

_OOB_SCORE_RE = re.compile(r'score="(-?[0-9]*\.?[0-9]+)"')

from rl_detector.annotation_utils import SP_CL, SP_OP
from rl_detector.config import CFG
from rl_detector.rewards import format_diagnostics, stripped_char_diff_count
from rl_detector.tell_xml import (
    _TEXT_CLOSE_CHUNK,
    _TEXT_O,
    _VERDICT_PREF,
    _parse_span_bundle,
    _rd_verdict_wrapped,
    _walk_inner_xml_flat,
    canonical_logical_document,
    escape_attr_piece,
    escape_document_piece,
    normalize_loose_annotation_verdict_attr_spacing,
    strip_return_token,
    strip_text_wrapper,
    wrap_outer_logical_plain_mid,
    wrap_span_piece,
)

logger = logging.getLogger(__name__)


def _collapse_double_entity_attrs(tx: str) -> str:
    """Collapse one layer of double-escaped entities (``&amp;apos;`` → ``&apos;``)."""
    out = tx
    for _ in range(6):
        prev = out
        out = out.replace("&amp;apos;", "&apos;").replace("&amp;quot;", "&quot;")
        if out == prev:
            break
    return out


def _fuzzy_anchor_span(
    document: str,
    span_raw: str,
    min_logical_start: int,
) -> tuple[int, str, int] | None:
    """Pick a substring of ``document`` near ``span_raw`` by edit distance."""
    n = len(document)
    if not span_raw:
        return None
    L0 = len(span_raw)
    best_i: int | None = None
    best_L: int | None = None
    best_d = 10**9
    for L in range(max(1, L0 - 2), L0 + 4):
        if L > n:
            continue
        max_dist = max(5, (L0 + L) // 6)
        for i in range(min_logical_start, n - L + 1):
            piece = document[i : i + L]
            dist = stripped_char_diff_count(text_a=piece, text_b=span_raw)
            if dist > max_dist:
                continue
            if dist < best_d or (dist == best_d and best_i is not None and i < best_i):
                best_d = dist
                best_i, best_L = i, L
    if best_i is None or best_L is None:
        return None
    piece = document[best_i : best_i + best_L]
    return best_i, piece, len(piece)


def _clamp_verdict_score(suf: str) -> str:
    """Replace an out-of-range score attribute with its [0.0, 1.0]-clamped value."""
    def _clamp(m: re.Match) -> str:
        try:
            v = float(m.group(1))
        except ValueError:
            return m.group(0)
        return f'score="{str(max(0.0, min(1.0, v)))}"'
    return _OOB_SCORE_RE.sub(_clamp, suf, count=1)


_VERDICT_DANGLING_RE = re.compile(
    r'^(<verdict\s+type="(?:AI|human)"\s+why="(?:[^"\\]|\\.)*"\s+score="[0-9]*\.?[0-9]+)'
    r'"\s*/>'
    r'.+</text>$',
    re.DOTALL,
)


def _strip_verdict_dangling_tail(suf: str) -> str | None:
    """Remove garbage tokens between a verdict's self-close /> and the outer </text>.

    Handles the case where the model emits extra tokens (e.g. a stray ``</span>`` and
    residual attribute fragments) after the verdict tag but before ``</text>``.
    Returns the cleaned suffix, or ``None`` if nothing was stripped or the pattern
    doesn't match.
    """
    m = _VERDICT_DANGLING_RE.match(suf)
    if not m:
        return None
    clean = m.group(1) + _TEXT_CLOSE_CHUNK
    if clean == suf:
        return None
    return clean


def _verdict_wire_suffix(meta: dict[str, str]) -> str:
    t = escape_attr_piece(tx=meta["type"])
    w = escape_attr_piece(tx=(meta.get("why") or "").strip())
    sc = escape_attr_piece(tx=str(meta["score"]))
    return _VERDICT_PREF + f'{t}" why="{w}" score="{sc}' + _TEXT_CLOSE_CHUNK


def _tell_intervals_non_overlapping(ordered: list[dict]) -> bool:
    iv = sorted(
        (
            int(t.get("_inner_pos", 0)),
            int(t.get("_inner_pos", 0)) + len(t.get("span_text") or ""),
        )
        for t in ordered
    )
    for k in range(1, len(iv)):
        if iv[k][0] < iv[k - 1][1]:
            return False
    return True


def _anchor_span_in_doc(
    *,
    doc_c: str,
    st: str,
    cursor_doc: int,
    hint_pos: int,
) -> int | None:
    """First ``>= cursor_doc`` occurrence of ``st`` whose index is closest to ``hint_pos``."""
    if not st:
        return None
    cands: list[int] = []
    j = doc_c.find(st, cursor_doc)
    while j >= 0:
        cands.append(j)
        j = doc_c.find(st, j + 1)
    if not cands:
        return None
    return min(cands, key=lambda x: abs(x - hint_pos))


def _drop_orphaned_annotation_closers(text: str) -> str:
    """Remove depth-1 ``<annotation .../></span>`` that orphan before more content (legacy inner inside ``<text>``)."""
    if not text.startswith(SP_OP):
        return text
    sp_op_len = len(SP_OP)
    sp_cl_len = len(SP_CL)
    ann_pref = "<annotation "
    ann_pref_len = len(ann_pref)
    out: list[str] = []
    i = 0
    depth = 0
    n = len(text)
    while i < n:
        if text.startswith(SP_OP, i):
            depth += 1
            out.append(SP_OP)
            i += sp_op_len
        elif text.startswith(SP_CL, i):
            out.append(SP_CL)
            if depth > 0:
                depth -= 1
            i += sp_cl_len
        elif depth == 1 and text.startswith(ann_pref, i):
            gt = text.find(">", i + ann_pref_len)
            if gt >= 0 and text[gt - 1] == "/":
                ann_tag_end = gt + 1
                j = ann_tag_end
                while j < n and text[j] in " \t\n\r":
                    j += 1
                if text.startswith(SP_CL, j) and j + sp_cl_len < n:
                    i = j + sp_cl_len
                    continue
                out.append(text[i:ann_tag_end])
                i = ann_tag_end
            else:
                out.append(text[i])
                i += 1
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def _drop_flat_orphaned_annotation_closers(text: str) -> str:
    """Remove depth-0 ``<annotation .../></span>`` orphans in new-format flat inner content.

    In new-format inner, orphaned annotations appear directly at depth 0 (not inside a ``<span>``
    opener), typically when the model forgot a ``<span>`` opener before some annotated text.  This
    drops both the orphaned ``<annotation/>`` and the unpaired ``</span>`` that follows.
    """
    sp_op_len = len(SP_OP)
    sp_cl_len = len(SP_CL)
    ann_pref = "<annotation "
    ann_pref_len = len(ann_pref)
    out: list[str] = []
    i = 0
    depth = 0
    n = len(text)
    while i < n:
        if text.startswith(SP_OP, i):
            depth += 1
            out.append(SP_OP)
            i += sp_op_len
        elif text.startswith(SP_CL, i):
            if depth > 0:
                depth -= 1
                out.append(SP_CL)
            # else: orphaned </span> at depth=0 — skip silently
            i += sp_cl_len
        elif text.startswith(ann_pref, i):
            gt = text.find(">", i + ann_pref_len)
            if gt >= 0:
                ann_end = gt + 1
                if depth == 0 and text[gt - 1] == "/":
                    # Orphaned annotation at top level: drop annotation + following </span>
                    j = ann_end
                    while j < n and text[j] in " \t\n\r":
                        j += 1
                    if text.startswith(SP_CL, j):
                        i = j + sp_cl_len
                        continue
                out.append(text[i:ann_end])
                i = ann_end
            else:
                out.append(text[i])
                i += 1
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def _surgical_text_repair(shell: str, inn: str, doc_c: str, max_fix_ratio: float) -> str | None:
    """Repair plain text in shell by applying diffs between inn and doc_c, preserving all XML tags.

    Builds a map from plain-text character positions in inn to byte ranges in shell, then
    uses SequenceMatcher opcodes to insert/delete/replace plain text while keeping every
    <span>, <annotation/>, </span> byte in place.  Returns None when the edit budget is
    exceeded or the mapping is inconsistent.
    """
    from difflib import SequenceMatcher

    diff = stripped_char_diff_count(inn, doc_c)
    fix_denom = max(1, len(inn) + len(doc_c))
    if diff / fix_denom > float(max_fix_ratio) and diff > 8:
        return None
    if diff == 0:
        return shell

    # Map each decoded plain-text character in inn to its byte span in shell.
    plain_spans: list[tuple[int, int]] = []
    i = 0
    n = len(shell)
    while i < n:
        if shell[i] == "<":
            j = shell.find(">", i)
            if j < 0:
                return None
            i = j + 1
        elif shell[i] == "&":
            j = shell.find(";", i)
            if j < 0:
                return None
            plain_spans.append((i, j + 1))
            i = j + 1
        else:
            plain_spans.append((i, i + 1))
            i += 1

    if len(plain_spans) != len(inn):
        return None

    def _xml_end(plain_idx: int) -> int:
        """Byte position just after the plain_idx-th decoded character in shell."""
        if plain_idx < len(plain_spans):
            return plain_spans[plain_idx][0]
        return len(shell)

    def _xml_span_end(plain_idx: int) -> int:
        """Byte position just after plain_spans[plain_idx]."""
        if plain_idx < len(plain_spans):
            return plain_spans[plain_idx][1]
        return len(shell)

    matcher = SequenceMatcher(None, inn, doc_c, autojunk=False)
    opcodes = matcher.get_opcodes()
    out: list[str] = []
    shell_cursor = 0

    for op_idx, (tag, a1, a2, b1, b2) in enumerate(opcodes):
        if tag == "equal":
            # Emit shell bytes up through the end of the last equal char (tags included).
            end = _xml_span_end(a2 - 1) if a2 > a1 else _xml_end(a1)
            out.append(shell[shell_cursor:end])
            shell_cursor = end
        elif tag == "insert":
            # Right-shift the insert point past XML tag clusters (e.g. nested </span>
            # closers) by finding how many insert-text chars also appear as the next chars
            # of inn at the same doc positions.  This moves the insertion past closing tags
            # that belong to already-copied span content.
            insert_len = b2 - b1
            next_a2 = opcodes[op_idx + 1][2] if op_idx + 1 < len(opcodes) and opcodes[op_idx + 1][0] == "equal" else a1
            k = 0
            while (
                k < insert_len
                and a1 + k < next_a2
                and b2 + k < len(doc_c)
                and inn[a1 + k] == doc_c[b1 + k]
            ):
                k += 1
            if next_a2 > a1:
                # Non-terminal: copy any pre-tags for the shifted position (correct for
                # nested span structure where opening tags precede the shifted equal chars).
                xml_pos = _xml_end(a1 + k) if a1 + k < len(plain_spans) else len(shell)
                out.append(shell[shell_cursor:xml_pos])
                shell_cursor = xml_pos
            # else: terminal insert — don't advance past trailing tags; insert at shell_cursor.
            out.append(escape_document_piece(doc_c[b1 + k : b2 + k]))
        elif tag == "delete":
            # Emit tags between shell_cursor and the first deleted char, skip deleted chars.
            xml_start = _xml_end(a1)
            xml_end = _xml_span_end(a2 - 1) if a2 > a1 else xml_start
            out.append(shell[shell_cursor:xml_start])
            shell_cursor = xml_end
        elif tag == "replace":
            xml_start = _xml_end(a1)
            xml_end = _xml_span_end(a2 - 1) if a2 > a1 else xml_start
            out.append(shell[shell_cursor:xml_start])
            shell_cursor = xml_end
            out.append(escape_document_piece(doc_c[b1:b2]))

    out.append(shell[shell_cursor:])
    return "".join(out)


def _rebuild_inner_preserving_spans(
    *,
    shell: str,
    logical_document: str,
    max_fix_ratio: float,
) -> str | None:
    """Rebuild ``<text>`` inner from ``logical_document`` while keeping flat ``<span>`` tells.

    Span payloads are re-anchored on the document (exact match, else fuzzy) so minor wire typos
    (e.g. double spaces) snap to dataset text.  Tells that cannot be anchored in forward order
    are skipped; an overrun guard prevents a tell from consuming document positions that belong
    to later tells when the model emitted redundant or hallucinated content.

    Returns None when inner XML does not parse, the edit budget is exceeded (no-tells case), or
    the rebuilt plain text does not equal the canonical document.
    """
    doc_c = canonical_logical_document(tx=logical_document)
    inn, tells, ok = _walk_inner_xml_flat(tx=shell)
    if not ok:
        return None
    if not tells:
        # Budget check: refuse to silently snap completely different plain text to the doc.
        diff = stripped_char_diff_count(inn, doc_c)
        fix_denom = max(1, len(inn) + len(doc_c))
        if (diff / fix_denom) > float(max_fix_ratio) and diff > 8:
            return None
        return escape_document_piece(tx=doc_c)
    ordered = sorted(tells, key=lambda t: t.get("_inner_pos", 0))
    # Overrun guard: when inner_plain is longer than the doc (model repeated/hallucinated content),
    # a tell whose anchor lands more than slack chars past its proportional doc position is skipped.
    ip_len = max(1, len(inn))
    doc_len = max(1, len(doc_c))
    ip_ratio = ip_len / doc_len
    if ip_ratio > 1.0:
        overrun_slack = max(15.0, 0.1 * doc_len / ip_ratio)
    else:
        overrun_slack = max(30.0, 0.3 * doc_len)

    def _is_overrun(idx: int, inner_pos: int | None) -> bool:
        if inner_pos is None:
            return False
        return idx > (inner_pos * doc_len / ip_len) + overrun_slack

    pos_inn = 0
    cursor_doc = 0
    out: list[str] = []
    last_i = len(ordered) - 1
    for ti, t in enumerate(ordered):
        pos = int(t.get("_inner_pos", 0))
        st = t.get("span_text") or ""
        if not st:
            if ti != last_i:
                return None
            out.append(escape_document_piece(tx=doc_c[cursor_doc:len(doc_c)]))
            out.append(
                wrap_span_piece(
                    mid=escape_document_piece(tx=""),
                    meta={
                        "type": t["type"],
                        "why": t["explanation"],
                        "score": str(t["score"]),
                    },
                )
            )
            cursor_doc = len(doc_c)
            pos_inn = pos
            continue
        if st not in inn or pos < pos_inn or pos + len(st) > len(inn):
            continue
        hint_doc = int(pos * doc_len / ip_len)
        idx = _anchor_span_in_doc(doc_c=doc_c, st=st, cursor_doc=cursor_doc, hint_pos=hint_doc)
        doc_piece = st
        span_len = len(st)
        if idx is None or idx < cursor_doc or _is_overrun(idx, pos):
            fz = _fuzzy_anchor_span(
                document=doc_c,
                span_raw=st,
                min_logical_start=cursor_doc,
            )
            if fz is None or fz[0] < cursor_doc or _is_overrun(fz[0], pos):
                continue
            idx, doc_piece, span_len = fz[0], fz[1], fz[2]
        out.append(escape_document_piece(tx=doc_c[cursor_doc:idx]))
        cursor_doc = idx + span_len
        out.append(
            wrap_span_piece(
                mid=escape_document_piece(tx=doc_piece),
                meta={
                    "type": t["type"],
                    "why": t["explanation"],
                    "score": str(t["score"]),
                },
            )
        )
        pos_inn = pos + len(st)
    out.append(escape_document_piece(tx=doc_c[cursor_doc:]))
    wired = "".join(out)
    inn2, _, ok2 = _walk_inner_xml_flat(tx=wired)
    if not ok2 or inn2 != doc_c:
        return None
    return wired


def _truncate_at_first_bad_span(text: str) -> str | None:
    """Keep the valid prefix of inner content up to (but not including) the first unparseable span.

    When a span fails to parse (e.g. thousands of nested unclosed ``<span>`` tags that exceed the
    depth limit), the existing ``_strip_unclosed_span_openers`` only drops the opener and keeps the
    malformed content, which still fails.  This function stops at the first bad span entirely,
    preserving all valid tells that appeared before it.

    Returns the truncated string if at least one bad span was dropped and the prefix is non-empty,
    otherwise ``None``.
    """
    i = 0
    n = len(text)
    pieces: list[str] = []
    dropped = False
    while i < n:
        if text.startswith(SP_OP, i):
            _, _, aft, ok = _parse_span_bundle(text, i, 0)
            if ok:
                pieces.append(text[i:aft])
                i = aft
            else:
                dropped = True
                break
        elif text.startswith("<", i):
            break
        else:
            nx = text.find("<", i)
            if nx < 0:
                pieces.append(text[i:])
                break
            pieces.append(text[i:nx])
            i = nx
    if not dropped:
        return None
    result = "".join(pieces)
    return result if result.strip() else None


def _strip_unclosed_span_openers(text: str) -> str | None:
    """Drop bare ``<span>`` openers whose span never closes with an annotation.

    Handles the budget-truncation case: model opened an outer ``<span>``, wrote inner
    annotated spans, but ran out of tokens before annotating the outer one.  The opener
    is removed and the inner content is promoted to the enclosing level.

    Returns the flattened string if at least one opener was stripped and the result
    parses cleanly, otherwise ``None``.
    """
    pieces: list[str] = []
    i = 0
    n = len(text)
    changed = False
    while i < n:
        if text.startswith(SP_OP, i):
            _, _, aft, ok = _parse_span_bundle(text, i, 0)
            if ok:
                pieces.append(text[i:aft])
                i = aft
            else:
                # Unclosed/unannotated outer span — drop its opener, keep its content.
                changed = True
                i += len(SP_OP)
        elif text.startswith("<", i):
            next_lt = text.find("<", i + 1)
            end = next_lt if next_lt >= 0 else n
            pieces.append(text[i:end])
            i = end
        else:
            next_lt = text.find("<", i)
            end = next_lt if next_lt >= 0 else n
            pieces.append(text[i:end])
            i = end
    if not changed:
        return None
    result = "".join(pieces)
    _, _, ok2 = _walk_inner_xml_flat(result)
    return result if ok2 else None


def try_fix_response(response_text: str, document: str, max_fix_ratio: float) -> str | None:
    """Try to repair a malformed new-format ``<text>…<verdict/></text>`` completion.

    Reads verdict metadata from the tail, rebuilds inner XML from the logical document while
    re-anchoring flat ``<span>`` tells (including fuzzy match to the document), or falls back to
    escaped document only when inner plaintext is within ``max_fix_ratio`` of the document.

    Returns ``None`` when inner XML does not parse (e.g. unclosed tags), when plaintext has no tells
    and is too far from the document (no silent ``<text>``-only snap), or when metadata cannot be read.
    """
    logical_document = str(document)
    doc_c = canonical_logical_document(tx=logical_document)

    work0 = strip_return_token(tx=response_text)
    diag = format_diagnostics(output=work0, document=logical_document)
    if diag["ok"]:
        return None

    reason = diag["reason"]
    if reason == "empty_final":
        return None
    if reason == "missing_outer_annotation" and not work0.startswith(_TEXT_O):
        return None

    work = normalize_loose_annotation_verdict_attr_spacing(tx=work0)
    work = _collapse_double_entity_attrs(tx=work)
    if work != work0:
        nd = format_diagnostics(output=work, document=logical_document)
        if nd["ok"]:
            return work

    shell = strip_text_wrapper(tx=work)
    if shell is None and not work.startswith(_TEXT_O):
        return None

    inner_full = work[len(_TEXT_O) :]
    vp = inner_full.rfind(_VERDICT_PREF)
    if vp < 0:
        return None
    verdict_suf = inner_full[vp:]
    first_close = verdict_suf.find(_TEXT_CLOSE_CHUNK)
    if first_close >= 0:
        verdict_suf = verdict_suf[: first_close + len(_TEXT_CLOSE_CHUNK)]
    vr = _rd_verdict_wrapped(suf=verdict_suf)
    if vr is None or not vr[1]:
        clamped = _clamp_verdict_score(verdict_suf)
        if clamped != verdict_suf:
            vr2 = _rd_verdict_wrapped(suf=clamped)
            if vr2 is not None and vr2[1]:
                vr = vr2
    if vr is None or not vr[1]:
        stripped = _strip_verdict_dangling_tail(verdict_suf)
        if stripped is not None:
            vr2 = _rd_verdict_wrapped(suf=stripped)
            if vr2 is not None and vr2[1]:
                vr = vr2
    if vr is None:
        return None
    meta_v, ok_v = vr
    if not ok_v:
        return None

    if shell is not None:
        shell_work = shell
        # Parse inner XML as-is first; only apply orphan-dropper passes when parse fails.
        inn_gate, tells_gate, ok_gate = _walk_inner_xml_flat(tx=shell_work)
        if not ok_gate:
            # New-format inner with depth-0 orphaned <annotation .../></span> pairs.
            # Try this first so flat tells (e.g. <span>A<ann/></span> text <span>B<ann/></span>)
            # are not incorrectly collapsed by the legacy depth-1 dropper below.
            cand = _drop_flat_orphaned_annotation_closers(text=shell_work)
            ig, tg, og = _walk_inner_xml_flat(tx=cand)
            if og:
                shell_work, inn_gate, tells_gate, ok_gate = cand, ig, tg, True
        if not ok_gate and shell_work.startswith(SP_OP):
            # Legacy-shaped inner (outer <span> wrapper) or depth-1 orphaned annotations.
            cand = _drop_orphaned_annotation_closers(text=shell_work)
            ig, tg, og = _walk_inner_xml_flat(tx=cand)
            if og:
                shell_work, inn_gate, tells_gate, ok_gate = cand, ig, tg, True
        if not ok_gate:
            # Unclosed outer <span> openers (budget truncation before the outer annotation).
            cand = _strip_unclosed_span_openers(text=shell_work)
            if cand is not None:
                ig, tg, og = _walk_inner_xml_flat(tx=cand)
                if og:
                    shell_work, inn_gate, tells_gate, ok_gate = cand, ig, tg, True
        if not ok_gate:
            # Malformed span content (e.g. thousands of nested unclosed <span> tags that exceed
            # the depth limit).  Drop everything from the first unparseable span onwards; the
            # rebuild step below will fill in the remaining document text without those tells.
            cand = _truncate_at_first_bad_span(text=shell_work)
            if cand is not None:
                ig, tg, og = _walk_inner_xml_flat(tx=cand)
                if og:
                    shell_work, inn_gate, tells_gate, ok_gate = cand, ig, tg, True
        if not ok_gate and _VERDICT_PREF in shell_work:
            # Embedded verdict in the shell (double-verdict case): the model emitted an extra
            # <verdict> mid-stream before the real one. Truncate at the first embedded verdict.
            ev_pos = shell_work.find(_VERDICT_PREF)
            cand = shell_work[:ev_pos]
            ig, tg, og = _walk_inner_xml_flat(tx=cand)
            if og:
                shell_work, inn_gate, tells_gate, ok_gate = cand, ig, tg, True
            elif not cand.strip():
                # Empty inner after stripping embedded verdict — treat as no-tells case.
                inn_gate, tells_gate, ok_gate = cand, [], True
        if not ok_gate:
            # Inner XML is unparseable even after cleanup; try a plain-text budget check.
            from rl_detector.rewards import strip_tags as _strip_tags_fb
            inner_plain_fb = _strip_tags_fb(shell_work)
            lim_fb = float(max_fix_ratio) * float(max(len(inner_plain_fb), len(doc_c), 1))
            if stripped_char_diff_count(text_a=inner_plain_fb, text_b=doc_c) > lim_fb:
                return None
            fixed = wrap_outer_logical_plain_mid(mid_logical_plaintext=logical_document, meta=meta_v)
            fixed_diag = format_diagnostics(output=fixed, document=logical_document)
            return fixed if fixed_diag["ok"] else None
        surgical = _surgical_text_repair(
            shell=shell_work,
            inn=inn_gate,
            doc_c=doc_c,
            max_fix_ratio=max_fix_ratio,
        )
        if surgical is not None:
            fixed_cand = _TEXT_O + surgical + _verdict_wire_suffix(meta=meta_v)
            if format_diagnostics(output=fixed_cand, document=logical_document)["ok"]:
                return fixed_cand
        rebuilt = _rebuild_inner_preserving_spans(
            shell=shell_work,
            logical_document=logical_document,
            max_fix_ratio=max_fix_ratio,
        )
        if rebuilt is not None:
            fixed = _TEXT_O + rebuilt + _verdict_wire_suffix(meta=meta_v)
        elif tells_gate:
            # Anchoring failed with tells present (e.g. hallucinated content); fall back to bare doc.
            fixed = wrap_outer_logical_plain_mid(
                mid_logical_plaintext=logical_document,
                meta=meta_v,
            )
        else:
            # No tells and budget check already rejected the rebuild — propagate the None.
            return None
    else:
        fixed = wrap_outer_logical_plain_mid(
            mid_logical_plaintext=logical_document,
            meta=meta_v,
        )
    fixed_diag = format_diagnostics(output=fixed, document=logical_document)
    return fixed if fixed_diag["ok"] else None


def _apply_format_fix_to_text_fields(
    response_text: str,
    completion_text: str,
    completion_tokens: list[int],
    completion_logprobs: list[float],
    document: str,
    tokenizer,
) -> tuple[str, str, list[int], list[float], bool, str | None]:
    """Try bracket-format fix; return repaired tokens with placeholder logprobs (rescored in batch after).

    The repaired token sequence replaces the broken one.  Logprobs are set to 0.0 as placeholders,
    the batch rescore pass in ``generate_rollouts`` calls ``compute_logprobs_async`` to fill in real values
    before any datum is built, so IS-ratios are correct and the gradient flows through the repaired tokens.

    Never synthesizes annotations or explanations that are not in the model output.
    This wrapper is allowed to remove mismatches and re-align copied document text only.
    """
    try:
        max_fix_ratio = float(getattr(CFG.training, "format_fix_max_ratio", 0.50))
        fixed = try_fix_response(
            response_text=response_text,
            document=document,
            max_fix_ratio=max_fix_ratio,
        )
        if fixed is None:
            return response_text, completion_text, completion_tokens, completion_logprobs, False, None
        wrong = response_text
        _pre = format_diagnostics(output=wrong, document=document)
        _post = format_diagnostics(output=fixed, document=document)
        assert not bool(_pre["ok"]), (_pre, wrong[:200])
        assert bool(_post["ok"]), (_post, fixed[:200])
        new_completion = completion_text.replace(response_text, fixed, 1) if response_text in completion_text else fixed
        new_tokens = tokenizer.encode(new_completion, add_special_tokens=False)
        new_logprobs = [0.0] * len(new_tokens)
        return fixed, new_completion, new_tokens, new_logprobs, True, wrong
    except Exception:
        logger.exception("format fix wrapper crashed; leaving response as formatting error")
        return response_text, completion_text, completion_tokens, completion_logprobs, False, None
