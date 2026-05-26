"""Online dropout of generic nested ``why`` explanations over train rows.

Default ``genericity_mode=span_why_overlap``: nested tells are grouped by a **short normalized
why prefix** (``why_bucket_key_chars``; a coarse first approximation, not exact-why identity).
Genericity mass ``G`` is ``log(1+max(|bucket|, min_bucket_n)) * (1 - λ * max_jaccard(span, other spans in bucket))``
with **λ** = ``span_why_jaccard_weight`` (0 = ignore span overlap; 1 = full penalty).  Singleton
buckets still get ``log(1+min_bucket_n)`` instead of a hard zero.  Batched ``nlp.pipe`` over
deduplicated span strings (``why_span_tokenize``).  Overlap ``p_mass`` default ``rank`` uses a single signal: ``1 - rank(mean_idf)`` over all nested
tells in the fit corpus.  Generic wording (low IDF = common words) gets high dropout pressure;
specific wording (high IDF = rare words) gets low pressure.  ``p_mass = min(cap, drop_strength *
(1 - rank(mean_idf)))``.  With ``drop_strength≈0.45`` the median nested tell gets ``p_mass≈0.22``;
target 15–30% overall coverage by tuning ``why_idf_drop_strength`` (0.3→~15%, 0.45→~22%,
0.6→~30%).  ``scaled`` uses ``G/median(G)``.  Then multiply by the usual score factor.

Legacy ``char_ngram_idf``: row-union char n-gram IDF on ``why`` text with median-mean_idf excess.

Only nested tells are candidates for removal; the outer root annotation is never removed or edited.
"""

from __future__ import annotations

import bisect
import html
import logging
import math
import random
import re
from collections import Counter, defaultdict

from rl_detector.annotation_utils import _collect_inner
from rl_detector.rewards import format_diagnostics
from rl_detector.format_fix import _fuzzy_anchor_span
from rl_detector.sft.why_span_tokenize import build_span_str_to_lemma_frozenset, normalize_fragment
from rl_detector.tell_xml import (
    escape_document_piece,
    escape_attr_piece,
    root_splits,
    strip_text_wrapper,
    wrap_span_piece,
    _TEXT_O,
    _VERDICT_PREF,
    _TEXT_CLOSE_CHUNK,
)

LOGGER = logging.getLogger(__name__)

# ── Content-type bias for dropout sort order ──────────────────────────────────
# Substance tells (hallucinations, factual errors, inconsistencies) get a
# negative sort-key offset so they are always picked before style-only markers.
# Style markers ("this feels", "sounds human") get a positive offset so they are
# dropped first when k < n.  The offsets only affect the keep-sort key, not the
# p_drop value itself.

_SUBSTANCE_WHY_RE = re.compile(
    r"(?:"
    r"\bhallucin"                           # hallucination / hallucinates
    r"|\binconsisten"                        # inconsistency / inconsistent
    r"|\bcontradict"                         # contradiction / contradicts
    r"|\bfactual(ly)?\b.{0,30}\b(error|wrong|incorrect|inaccurate|mistake)"
    r"|\b(wrong|incorrect|inaccurate)\b.{0,30}\b(fact|date|year|name|number|claim)"
    r"|\b(fact|date|year|name|number|claim)\b.{0,20}\b(wrong|incorrect|inaccurate)\b"
    r"|\banachronism"
    r"|\bimplausible\b|\bimpossible\b"
    r"|\bfabricat|\bmade.?up\b"
    r"|\bdoes not exist\b|\bnever existed\b|\bnever happened\b"
    r"|\bmisattribut"
    r"|\bhistorically\b.{0,20}\b(wrong|incorrect|inaccurate)"
    r"|\bfactual error\b|\bfact.?check"
    r")",
    re.IGNORECASE,
)

_STYLE_MARKER_WHY_RE = re.compile(
    r"(?:"
    r"\bthis feels\b"
    r"|\bfeels (human|natural|real|genuine|authentic|personal|organic|casual)\b"
    r"|\bsounds (human|natural|real|genuine|authentic|organic)\b"
    r"|\bsounds like (a |an )?(human|genuine|real|authentic|natural|organic)\b"
    r"|\bgives (the impression|a (natural|human|casual|organic|personal|authentic))\b"
    r"|\bhas a (natural|human|casual|organic|personal|authentic) (feel|flow|tone|vibe|touch)\b"
    r"|\bhas that (human|natural|real|genuine|casual|authentic)\b"
    r"|\b(vibe|tone) of (human|natural|real|genuine)\b"
    r"|\bstylistic(ally)?\b"
    r"|\btypical of human writing\b"
    r"|\bcharacteristic of human\b"
    r")",
    re.IGNORECASE,
)

# How far substance / style offsets shift the effective p_drop sort key.
# Substance tells are pulled toward 0 (always kept first);
# style-only markers are pushed toward 1 (dropped first).
_SUBSTANCE_SORT_OFFSET = -0.6
_STYLE_SORT_OFFSET = +0.4


def _content_sort_key(p_drop: float, why: str) -> float:
    """Effective p_drop for keep-sort only; does not change logged p_drop."""
    if _SUBSTANCE_WHY_RE.search(why):
        return max(0.0, p_drop + _SUBSTANCE_SORT_OFFSET)
    if _STYLE_MARKER_WHY_RE.search(why):
        return min(1.0, p_drop + _STYLE_SORT_OFFSET)
    return p_drop


def _normalize_why_fragment(text: str) -> str:
    return normalize_fragment(text=str(text))


def _char_ngrams_for_text(text: str, n_min: int, n_max: int) -> set[str]:
    t = _normalize_why_fragment(text=text)
    if len(t) < n_min:
        return set()
    out: set[str] = set()
    for n in range(n_min, n_max + 1):
        if len(t) < n:
            break
        for i in range(0, len(t) - n + 1):
            out.add(t[i : i + n])
    return out


def _row_union_grams(annotation_xml: str, n_min: int, n_max: int) -> set[str]:
    inn, desc, meta, ok, _end = root_splits(tx=annotation_xml)
    if not ok or meta is None:
        return set()
    acc: set[str] = set()
    acc |= _char_ngrams_for_text(text=meta.get("why", ""), n_min=n_min, n_max=n_max)
    for leaf in desc:
        acc |= _char_ngrams_for_text(text=str(leaf.get("explanation", "")), n_min=n_min, n_max=n_max)
    return acc


def _mean_idf_for_text(text: str, n_min: int, n_max: int, gram_df: dict[str, int], n_rows: int) -> float:
    grams = list(_char_ngrams_for_text(text=text, n_min=n_min, n_max=n_max))
    if not grams:
        return 0.0
    s = 0.0
    for g in grams:
        df = int(gram_df.get(g, 0))
        s += math.log((1.0 + float(n_rows)) / (1.0 + float(df)))
    return s / float(len(grams))


def _why_bucket_key(explanation: str, max_chars: int) -> str:
    t = _normalize_why_fragment(text=str(explanation))
    return t[:max_chars] if len(t) > max_chars else t


def _jaccard_words(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    u = len(a | b)
    if u == 0:
        return 0.0
    return len(a & b) / float(u)


def _span_why_genericity_table(
    rows: list[dict],
    why_bucket_key_chars: int,
    spacy_pipe_batch_size: int,
    spacy_exclude_stopwords: bool,
    span_why_jaccard_weight: float,
    span_why_min_bucket_n_for_log: int,
) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int], int]]:
    buckets: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    wkc = int(why_bucket_key_chars)
    lam = float(span_why_jaccard_weight)
    min_n = max(1, int(span_why_min_bucket_n_for_log))
    for row_idx, row in enumerate(rows):
        ann = row.get("annotation") or ""
        inn, desc, meta, ok, _end = root_splits(tx=ann)
        if not ok or meta is None or not desc:
            continue
        for order, leaf in enumerate(desc):
            wk = _why_bucket_key(explanation=str(leaf.get("explanation", "")), max_chars=wkc)
            buckets[wk].append((int(row_idx), int(order), str(leaf.get("span_text", ""))))
    all_spans: list[str] = []
    for _wk, members in buckets.items():
        for _ri, _oi, s in members:
            all_spans.append(str(s))
    uniq = sorted(set(all_spans))
    LOGGER.info(
        "span_why_overlap spaCy English+rule_lemma unique_spans=%d pipe_batch=%d exclude_stopwords=%s "
        "why_bucket_key_chars=%d jaccard_weight=%.4f min_bucket_n_for_log=%d",
        len(uniq),
        int(spacy_pipe_batch_size),
        bool(spacy_exclude_stopwords),
        wkc,
        lam,
        min_n,
    )
    if not uniq:
        span_to_set: dict[str, frozenset[str]] = {}
    else:
        span_to_set = build_span_str_to_lemma_frozenset(
            unique_spans=uniq,
            pipe_batch_size=int(spacy_pipe_batch_size),
            exclude_stopwords=bool(spacy_exclude_stopwords),
        )

    def log_for_bucket_size(nn: int) -> float:
        return math.log(1.0 + float(max(int(nn), min_n)))

    out: dict[tuple[int, int], float] = {}
    bucket_size_table: dict[tuple[int, int], int] = {}
    for _wk, members in buckets.items():
        n = len(members)
        sets = [span_to_set[str(s)] for _, _, s in members]
        log_part = log_for_bucket_size(nn=n)
        for i, (ri, oi, _) in enumerate(members):
            bucket_size_table[(ri, oi)] = n
            if n < 2:
                out[(ri, oi)] = log_part
                continue
            best = 0.0
            for j in range(n):
                if j == i:
                    continue
                best = max(best, _jaccard_words(a=sets[i], b=sets[j]))
            div = 1.0 - lam * best
            if div < 0.0:
                div = 0.0
            if div > 1.0:
                div = 1.0
            out[(ri, oi)] = log_part * div
    return out, bucket_size_table


def _median_float(vals: list[float]) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    mid = len(s) // 2
    if len(s) % 2 == 1:
        return float(s[mid])
    return 0.5 * (float(s[mid - 1]) + float(s[mid]))


def _tie_mid_rank(sorted_vals: list[float], x: float) -> float:
    """Fraction in ``[0, 1]`` from tie-aware mid-ranks on ascending ``sorted_vals``."""
    if not sorted_vals:
        return 0.0
    lo = bisect.bisect_left(sorted_vals, x)
    hi = bisect.bisect_right(sorted_vals, x)
    return (float(lo) + float(hi)) / (2.0 * float(len(sorted_vals)))


def _quantile_linear(vals: list[float], q: float) -> float:
    """q in [0, 1]; linear interpolation between sorted values."""
    if not vals:
        return 0.0
    qq = float(q)
    if qq <= 0.0:
        return float(min(vals))
    if qq >= 1.0:
        return float(max(vals))
    s = sorted(vals)
    n = len(s)
    pos = qq * (n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(s[lo])
    w = pos - lo
    return float(s[lo]) * (1.0 - w) + float(s[hi]) * w


class WhyCharNgramIdfScorer:
    """Char n-gram IDF table plus span–why overlap genericity (see module docstring)."""

    def __init__(
        self,
        gram_df: dict[str, int],
        n_rows: int,
        n_min: int,
        n_max: int,
        median_mean_idf: float,
        genericity_mode: str,
        why_key_max_chars: int,
        why_bucket_key_chars: int,
        span_why_jaccard_weight: float,
        span_why_min_bucket_n_for_log: int,
        span_why_overlap_p_mass_style: str,
        span_why_excess_quantile: float,
        span_why_g_by_row_order: dict[tuple[int, int], float],
        span_why_g_sorted: list[float],
        nested_mean_idf_sorted: list[float],
        span_why_g_median: float,
        span_why_g_excess_ref: float,
        bucket_size_by_row_order: dict[tuple[int, int], int],
        total_nested_tells: int,
    ) -> None:
        self._gram_df = gram_df
        self._n_rows = n_rows
        self._n_min = n_min
        self._n_max = n_max
        self.median_mean_idf = median_mean_idf
        self.genericity_mode = str(genericity_mode)
        self.why_key_max_chars = int(why_key_max_chars)
        self.why_bucket_key_chars = int(why_bucket_key_chars)
        self.span_why_jaccard_weight = float(span_why_jaccard_weight)
        self.span_why_min_bucket_n_for_log = int(span_why_min_bucket_n_for_log)
        self.span_why_overlap_p_mass_style = str(span_why_overlap_p_mass_style)
        self.span_why_excess_quantile = float(span_why_excess_quantile)
        self._span_g = dict(span_why_g_by_row_order)
        self._span_g_sorted = list(span_why_g_sorted)
        self._nested_mean_idf_sorted = list(nested_mean_idf_sorted)
        self.span_why_g_median = float(span_why_g_median)
        self.span_why_g_excess_ref = float(span_why_g_excess_ref)
        self._bucket_size = dict(bucket_size_by_row_order)
        self.total_nested_tells = int(total_nested_tells)
        self._bucket_size_sorted = sorted(bucket_size_by_row_order.values())

    def span_why_g(self, key: tuple[int, int]) -> float:
        return float(self._span_g.get(key, 0.0))

    def bucket_fraction(self, key: tuple[int, int]) -> float:
        """Fraction of all nested tells in the fit corpus that share this why-prefix bucket."""
        n = self._bucket_size.get(key, 1)
        return float(n) / float(max(1, self.total_nested_tells))

    @classmethod
    def from_train_rows(
        cls,
        rows: list[dict],
        n_min: int,
        n_max: int,
        genericity_mode: str,
        why_key_max_chars: int,
        why_bucket_key_chars: int,
        span_why_jaccard_weight: float,
        span_why_min_bucket_n_for_log: int,
        span_why_overlap_p_mass_style: str,
        span_why_excess_quantile: float,
        spacy_pipe_batch_size: int,
        spacy_exclude_stopwords: bool,
    ) -> WhyCharNgramIdfScorer:
        row_sets: list[set[str]] = []
        for row in rows:
            ann = row.get("annotation") or ""
            row_sets.append(_row_union_grams(annotation_xml=ann, n_min=n_min, n_max=n_max))
        n_rows = len(row_sets)
        gram_df: Counter[str] = Counter()
        for rs in row_sets:
            for g in rs:
                gram_df[g] += 1
        gdict = dict(gram_df)
        frag_scores: list[float] = []
        for row in rows:
            ann = row.get("annotation") or ""
            inn, desc, meta, ok, _end = root_splits(tx=ann)
            if not ok or meta is None:
                continue
            frag_scores.append(
                _mean_idf_for_text(
                    text=meta.get("why", ""),
                    n_min=n_min,
                    n_max=n_max,
                    gram_df=gdict,
                    n_rows=n_rows,
                )
            )
            for leaf in desc:
                frag_scores.append(
                    _mean_idf_for_text(
                        text=str(leaf.get("explanation", "")),
                        n_min=n_min,
                        n_max=n_max,
                        gram_df=gdict,
                        n_rows=n_rows,
                    )
                )
        frag_sorted = sorted(frag_scores)
        mid = len(frag_sorted) // 2
        if not frag_sorted:
            med = 0.0
        elif len(frag_sorted) % 2 == 1:
            med = float(frag_sorted[mid])
        else:
            med = 0.5 * (float(frag_sorted[mid - 1]) + float(frag_sorted[mid]))
        mode = str(genericity_mode)
        if mode not in ("char_ngram_idf", "span_why_overlap"):
            raise ValueError("genericity_mode must be char_ngram_idf or span_why_overlap")
        wk = int(why_key_max_chars)
        wbk = int(why_bucket_key_chars)
        sjw = float(span_why_jaccard_weight)
        smn = int(span_why_min_bucket_n_for_log)
        g_table, bucket_size_table = _span_why_genericity_table(
            rows=rows,
            why_bucket_key_chars=wbk,
            spacy_pipe_batch_size=int(spacy_pipe_batch_size),
            spacy_exclude_stopwords=bool(spacy_exclude_stopwords),
            span_why_jaccard_weight=sjw,
            span_why_min_bucket_n_for_log=smn,
        )
        total_nested_tells = len(g_table)
        nested_midfs: list[float] = []
        for row in rows:
            ann = row.get("annotation") or ""
            inn, desc, meta, ok, _end = root_splits(tx=ann)
            if not ok or meta is None or not desc:
                continue
            for leaf in desc:
                wtex = str(leaf.get("explanation", ""))
                nested_midfs.append(
                    _mean_idf_for_text(
                        text=wtex,
                        n_min=n_min,
                        n_max=n_max,
                        gram_df=gdict,
                        n_rows=n_rows,
                    )
                )
        g_vals = list(g_table.values())
        g_med = _median_float(vals=g_vals)
        pm_style = str(span_why_overlap_p_mass_style)
        q_ex = float(span_why_excess_quantile)
        if pm_style not in ("subtract_median", "subtract_quantile", "scaled", "rank"):
            raise ValueError(
                "span_why_overlap_p_mass_style must be subtract_median, subtract_quantile, scaled, or rank"
            )
        if not (0.0 <= q_ex <= 1.0):
            raise ValueError("span_why_excess_quantile must be in [0, 1]")
        g_ref = _quantile_linear(vals=g_vals, q=q_ex)
        g_sorted = sorted(g_vals)
        mf_sorted = sorted(nested_midfs)
        if len(mf_sorted) != len(g_vals):
            raise ValueError("nested mean_idf count must match span_why_g table length")
        LOGGER.info(
            "why_idf: mode=%s span_lemma=en_blank_rule n_rows=%d ngrams=%d median_mean_idf=%.4f span_why_g_median=%.5f "
            "overlap_p_mass=%s excess_q=%.3f span_why_g_excess_ref=%.5f total_nested_tells=%d "
            "ngram_range=%d-%d why_key_max=%d why_bucket_key=%d jaccard_w=%.4f min_bucket_n=%d",
            mode,
            n_rows,
            len(gdict),
            med,
            g_med,
            pm_style,
            q_ex,
            g_ref,
            total_nested_tells,
            n_min,
            n_max,
            wk,
            wbk,
            sjw,
            smn,
        )
        return cls(
            gram_df=gdict,
            n_rows=n_rows,
            n_min=n_min,
            n_max=n_max,
            median_mean_idf=med,
            genericity_mode=mode,
            why_key_max_chars=wk,
            why_bucket_key_chars=wbk,
            span_why_jaccard_weight=sjw,
            span_why_min_bucket_n_for_log=smn,
            span_why_overlap_p_mass_style=pm_style,
            span_why_excess_quantile=q_ex,
            span_why_g_by_row_order=g_table,
            span_why_g_sorted=g_sorted,
            nested_mean_idf_sorted=mf_sorted,
            span_why_g_median=g_med,
            span_why_g_excess_ref=g_ref,
            bucket_size_by_row_order=bucket_size_table,
            total_nested_tells=total_nested_tells,
        )

    def mean_idf(self, why_text: str) -> float:
        return _mean_idf_for_text(
            text=why_text,
            n_min=self._n_min,
            n_max=self._n_max,
            gram_df=self._gram_df,
            n_rows=self._n_rows,
        )

    def top_ngrams_by_row_df(self, top_k: int) -> list[tuple[str, int]]:
        """Char n-grams with highest row-level DF (show up in many train examples)."""
        ranked = sorted(self._gram_df.items(), key=lambda kv: (-kv[1], kv[0]))
        return ranked[: int(top_k)]


def _placement_nodes(
    annotation_xml: str,
    logical_document: str,
) -> tuple[list[dict], bool]:
    tells: list[dict] = []
    inner_plain, _ai, inner_ok = _collect_inner(annotation_xml, tells)
    if not inner_ok:
        return [], False
    _ip_len = max(1, len(inner_plain))
    _doc_len = max(1, len(logical_document))
    _ip_ratio = _ip_len / _doc_len
    if _ip_ratio > 1.0:
        _overrun_slack = max(15.0, 0.1 * _doc_len / _ip_ratio)
    else:
        _overrun_slack = max(30.0, 0.3 * _doc_len)

    def _is_overrun(idx: int, inner_pos: int | None) -> bool:
        if inner_pos is None:
            return False
        return idx > (inner_pos * _doc_len / _ip_len) + _overrun_slack

    placements: list[tuple[int, int, int, dict]] = []
    search_start = 0
    for order, t in enumerate(tells):
        span_raw = t.get("span_text") or ""
        if not span_raw:
            inner_pos = t.get("_inner_pos")
            if inner_pos is not None:
                if inner_pos >= len(inner_plain):
                    idx = len(logical_document)
                else:
                    idx = min(max(search_start, inner_pos), len(logical_document))
            else:
                idx = min(search_start, len(logical_document))
            t["span_text"] = ""
            placements.append((idx, idx, order, t))
            search_start = idx
            continue
        inner_pos = t.get("_inner_pos")
        anchor = max(search_start, inner_pos) if inner_pos is not None else search_start
        idx = logical_document.find(span_raw, anchor)
        if idx < 0 and inner_pos is not None and anchor > search_start:
            idx = logical_document.rfind(span_raw, search_start, anchor + len(span_raw))
        if idx < 0:
            idx = logical_document.find(span_raw, search_start)
        if _is_overrun(idx=idx, inner_pos=inner_pos):
            idx = -1
        if idx >= 0:
            logical_span = logical_document[idx : idx + len(span_raw)]
            wire_len = len(logical_span)
        else:
            fuzzy = _fuzzy_anchor_span(
                document=logical_document,
                span_raw=span_raw,
                min_logical_start=search_start,
            )
            if fuzzy is None:
                continue
            idx, logical_span, wire_len = fuzzy
            if _is_overrun(idx=idx, inner_pos=inner_pos):
                continue
        t["span_text"] = logical_span
        placements.append((idx, idx + wire_len, order, t))
        search_start = idx + wire_len

    if len(placements) != len(tells):
        return [], False

    nodes = [
        {"start": start, "end": end, "order": order, "tell": t, "children": []}
        for start, end, order, t in placements
    ]
    nodes.sort(key=lambda x: (int(x["start"]), -int(x["end"]), int(x["order"])))
    roots: list[dict] = []
    stack: list[dict] = []
    for node in nodes:
        start = int(node["start"])
        end = int(node["end"])
        while stack and not (
            int(stack[-1]["start"]) <= start < int(stack[-1]["end"]) and end <= int(stack[-1]["end"])
        ):
            stack.pop()
        if stack:
            stack[-1]["children"].append(node)
        else:
            roots.append(node)
        stack.append(node)
    return roots, True


def _render_region(
    logical_document: str,
    start: int,
    end: int,
    children: list[dict],
    drop_orders: set[int],
) -> str:
    out: list[str] = []
    pos = start
    for child in sorted(children, key=lambda x: (int(x["start"]), -int(x["end"]), int(x["order"]))):
        cstart = int(child["start"])
        cend = int(child["end"])
        if cstart < pos or cend > end:
            continue
        out.append(escape_document_piece(logical_document[pos:cstart]))
        inner = _render_region(
            logical_document=logical_document,
            start=cstart,
            end=cend,
            children=child["children"],
            drop_orders=drop_orders,
        )
        t = child["tell"]
        if int(child["order"]) in drop_orders:
            out.append(inner)
        else:
            typ = t.get("type") or "AI"
            score = str(t.get("score", "0.0"))
            out.append(
                wrap_span_piece(
                    inner,
                    {
                        "type": typ,
                        "why": html.unescape(str((t.get("explanation") or "")).strip()),
                        "score": score,
                    },
                )
            )
        pos = cend
    out.append(escape_document_piece(logical_document[pos:end]))
    return "".join(out)


def rebuild_annotation_nested_dropout(
    logical_document: str,
    annotation_xml: str,
    drop_nested_orders: set[int],
) -> str:
    inn, desc, meta, ok, _end = root_splits(tx=annotation_xml)
    if not ok or meta is None:
        return annotation_xml
    if not desc:
        return annotation_xml
    roots, ok_nodes = _placement_nodes(
        annotation_xml=annotation_xml,
        logical_document=logical_document,
    )
    if not ok_nodes:
        return annotation_xml
    inner_with_tells = _render_region(
        logical_document=logical_document,
        start=0,
        end=len(logical_document),
        children=roots,
        drop_orders=drop_nested_orders,
    )
    if strip_text_wrapper(annotation_xml) is not None:
        t = escape_attr_piece(meta["type"])
        w = escape_attr_piece((meta.get("why") or "").strip())
        sc = escape_attr_piece(str(meta["score"]))
        return _TEXT_O + inner_with_tells + f'{_VERDICT_PREF}{t}" why="{w}" score="{sc}' + _TEXT_CLOSE_CHUNK
    return wrap_span_piece(inner_with_tells, meta)


def _nested_tell_score(leaf: dict) -> float:
    raw = str(leaf.get("score", "0.5")).strip()
    try:
        v = float(raw)
    except ValueError:
        v = 0.5
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def per_nested_tell_drop_probs(
    scorer: WhyCharNgramIdfScorer,
    row_index: int,
    nested_desc: list[dict],
    drop_strength: float,
    p_drop_cap: float,
    score_keep_weight: float,
) -> list[dict]:
    """Per nested tell: ``genericity_excess``, genericity mass (``p_idf``), score factor, ``p_drop``."""
    mode = scorer.genericity_mode
    ref_idf = float(scorer.median_mean_idf)
    denom_idf = ref_idf + 1e-6
    g_med = float(scorer.span_why_g_median)
    denom_g = max(1e-6, g_med)
    g_ref = float(scorer.span_why_g_excess_ref)
    pm_style = str(scorer.span_why_overlap_p_mass_style)
    w = float(score_keep_weight)
    out: list[dict] = []
    for order, leaf in enumerate(nested_desc):
        why = str(leaf.get("explanation", ""))
        m = scorer.mean_idf(why_text=why)
        g_span = scorer.span_why_g(key=(int(row_index), int(order)))
        if mode == "char_ngram_idf":
            excess = max(0.0, ref_idf - m)
            p_mass = min(float(p_drop_cap), float(drop_strength) * excess / denom_idf)
        elif mode == "span_why_overlap":
            if pm_style == "rank":
                # Generic wording (low IDF rank) → high p_mass; specific wording → low p_mass.
                # drop_strength scales the IDF rank; drop_strength≈0.45 → ~22% median coverage.
                mf_s = scorer._nested_mean_idf_sorted
                if not mf_s:
                    excess = 0.0
                    p_mass = 0.0
                else:
                    rm = _tie_mid_rank(sorted_vals=mf_s, x=m)
                    excess = max(0.0, 1.0 - rm)
                    p_mass = min(float(p_drop_cap), float(drop_strength) * excess)
            elif pm_style == "scaled":
                excess = g_span / denom_g
                p_mass = min(float(p_drop_cap), float(drop_strength) * excess)
            elif pm_style == "subtract_quantile":
                excess = max(0.0, g_span - g_ref)
                p_mass = min(float(p_drop_cap), float(drop_strength) * excess / denom_g)
            else:
                excess = max(0.0, g_span - g_med)
                p_mass = min(float(p_drop_cap), float(drop_strength) * excess / denom_g)
        else:
            raise ValueError("unknown genericity_mode")
        sc = _nested_tell_score(leaf=leaf)
        score_factor = max(0.0, 1.0 - w * sc)
        p_final = p_mass * score_factor
        span = str(leaf.get("span_text", ""))
        out.append(
            {
                "order": int(order),
                "type": leaf.get("type"),
                "score": sc,
                "span_preview": span[:160],
                "why": why,
                "mean_idf": m,
                "span_why_g": g_span,
                "genericity_excess": excess,
                "p_idf": p_mass,
                "score_factor": score_factor,
                "p_drop": p_final,
            }
        )
    return out


def sample_drop_nested_orders(
    rng: random.Random,
    scorer: WhyCharNgramIdfScorer,
    row_index: int,
    nested_desc: list[dict],
    drop_strength: float,
    p_drop_cap: float,
    score_keep_weight: float,
    keep_min: int = 1,
    keep_max: int = 5,
) -> set[int]:
    """Drop nested tells so that exactly k survive, where k ∈ [keep_min, keep_max].

    k is biased by document genericity: generic docs (high mean p_drop) → k near keep_min;
    specific docs (low mean p_drop) → k near keep_max.  k is chosen by stochastic rounding
    of the continuous bias target, then the k least-generic tells (lowest p_drop) are kept.
    At least 1 tell always survives.
    """
    if not nested_desc:
        return set()
    n = len(nested_desc)

    probs_list = per_nested_tell_drop_probs(
        scorer=scorer,
        row_index=int(row_index),
        nested_desc=nested_desc,
        drop_strength=drop_strength,
        p_drop_cap=p_drop_cap,
        score_keep_weight=score_keep_weight,
    )

    # Clamp keep bounds to what's feasible for this doc (always keep at least 1).
    eff_keep_min = max(1, min(int(keep_min), n))
    eff_keep_max = max(eff_keep_min, min(int(keep_max), n))

    # Continuous target k: generic doc (high mean p_drop) → keep_min; specific → keep_max.
    mean_p_drop = sum(float(pr["p_drop"]) for pr in probs_list) / len(probs_list)
    # Normalize mean_p_drop by cap so the scale is always [0, 1].
    normalized_genericity = min(1.0, max(0.0, mean_p_drop / max(float(p_drop_cap), 1e-9)))
    target_k = eff_keep_min + (eff_keep_max - eff_keep_min) * (1.0 - normalized_genericity)

    # Stochastic rounding so k is unbiased in expectation.
    k_floor = int(math.floor(target_k))
    k = k_floor + (1 if rng.random() < (target_k - k_floor) else 0)
    k = max(eff_keep_min, min(eff_keep_max, k))

    # Keep the k most-valuable tells.
    # Sort key: effective p_drop after content-type bias — substance tells (hallucinations,
    # factual errors) are pulled to the front; style-only markers are pushed to the back.
    sorted_by_priority = sorted(
        probs_list,
        key=lambda p: _content_sort_key(
            float(p["p_drop"]),
            str(nested_desc[int(p["order"])].get("explanation", "")),
        ),
    )
    keep_orders = {int(pr["order"]) for pr in sorted_by_priority[:k]}
    drop = {int(pr["order"]) for pr in probs_list if int(pr["order"]) not in keep_orders}

    n_kept = n - len(drop)
    assert n_kept >= 1, f"all {n} nested tells dropped — keep_min={keep_min} keep_max={keep_max}"
    assert eff_keep_min <= n_kept <= eff_keep_max, (
        f"kept {n_kept} tells but eff bounds are [{eff_keep_min}, {eff_keep_max}] "
        f"(keep_min={keep_min}, keep_max={keep_max}, n={n})"
    )

    return drop


def apply_online_why_idf_nested_dropout(
    logical_document: str,
    annotation_xml: str,
    rng: random.Random,
    scorer: WhyCharNgramIdfScorer,
    row_index: int,
    drop_strength: float,
    p_drop_cap: float,
    score_keep_weight: float,
    keep_min: int = 1,
    keep_max: int = 5,
) -> tuple[str, int, int]:
    """Return (annotation_xml_after_dropout, n_nested_before, n_nested_after).

    Keeps between keep_min and keep_max nested tells per document; docs with
    more generic tells lean toward keep_min, specific docs toward keep_max.
    """
    inn, desc, meta, ok, _end = root_splits(tx=annotation_xml)
    if not ok or meta is None or not desc:
        return annotation_xml, 0, 0
    n_before = len(desc)
    drop_orders = sample_drop_nested_orders(
        rng=rng,
        scorer=scorer,
        row_index=int(row_index),
        nested_desc=desc,
        drop_strength=drop_strength,
        p_drop_cap=p_drop_cap,
        score_keep_weight=score_keep_weight,
        keep_min=int(keep_min),
        keep_max=int(keep_max),
    )
    n_after = n_before - len(drop_orders)
    assert n_after >= 1, f"n_after={n_after} for n_before={n_before}"
    assert int(keep_min) <= n_after <= max(int(keep_min), min(int(keep_max), n_before)), (
        f"n_after={n_after} outside [{keep_min}, min({keep_max}, {n_before})]"
    )
    if not drop_orders:
        return annotation_xml, n_before, n_after
    rebuilt = rebuild_annotation_nested_dropout(
        logical_document=logical_document,
        annotation_xml=annotation_xml,
        drop_nested_orders=drop_orders,
    )
    diag = format_diagnostics(output=rebuilt, document=logical_document)
    assert diag["ok"], diag
    return rebuilt, n_before, n_after


def apply_paced_annotation_dropout(
    logical_document: str,
    annotation_xml: str,
    rng: random.Random,
    words_per_annotation: float = 20.0,
    high_score_keep_bonus: float = 3.0,
) -> tuple[str, int, int]:
    """Drop spans to target density of ~1 per ``words_per_annotation`` words.

    If the doc already has <= target annotations, returns unchanged.
    When dropping, high-score spans are ``high_score_keep_bonus`` times more
    likely to survive than score-0 spans (linear interpolation).

    Returns (xml_after, n_before, n_after).
    """
    inn, desc, meta, ok, _end = root_splits(tx=annotation_xml)
    if not ok or meta is None or not desc:
        return annotation_xml, 0, 0

    n_before = len(desc)
    n_words = len(logical_document.split()) if logical_document else 0
    target_n = max(1, int(n_words / max(1.0, words_per_annotation)))

    if n_before <= target_n:
        return annotation_xml, n_before, n_before

    # Weighted sample: keep `target_n` spans, biased toward high-score ones.
    # keep_weight = 1 + (bonus - 1) * score  → ranges from 1.0 (score=0) to bonus (score=1)
    keep_weights = []
    for leaf in desc:
        try:
            sc = max(0.0, min(1.0, float(str(leaf.get("score", "0.5")).strip())))
        except ValueError:
            sc = 0.5
        keep_weights.append(1.0 + (float(high_score_keep_bonus) - 1.0) * sc)

    # Sample without replacement using weighted reservoir
    indices_to_keep: set[int] = set()
    remaining_idx = list(range(n_before))
    remaining_w = list(keep_weights)
    for _ in range(min(target_n, n_before)):
        total_w = sum(remaining_w)
        r = rng.random() * total_w
        cumsum = 0.0
        picked = len(remaining_idx) - 1
        for i, w in enumerate(remaining_w):
            cumsum += w
            if r <= cumsum:
                picked = i
                break
        indices_to_keep.add(remaining_idx[picked])
        remaining_w.pop(picked)
        remaining_idx.pop(picked)

    drop_orders = {idx for idx in range(n_before) if idx not in indices_to_keep}
    if not drop_orders:
        return annotation_xml, n_before, n_before

    rebuilt = rebuild_annotation_nested_dropout(
        logical_document=logical_document,
        annotation_xml=annotation_xml,
        drop_nested_orders=drop_orders,
    )
    diag = format_diagnostics(output=rebuilt, document=logical_document)
    if not diag["ok"]:
        return annotation_xml, n_before, n_before  # fallback: keep original
    return rebuilt, n_before, n_before - len(drop_orders)
