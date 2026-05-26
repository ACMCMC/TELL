# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "matplotlib",
#   "numpy",
#   "tinker",
#   "transformers",
# ]
# ///
"""One-doc trace: greedy sample (max 10 new tokens), teacher logprobs, top-K ``beam`` grid, PDF.

``matplotlib`` is a normal ``[project]`` dependency; Tinker stack lives in the ``training``
uv group. Install with:

  uv sync

Full trace (calls Tinker):

  uv run python experiments/tinker_why_logprob_trace.py \\
    --checkpoint tinker://.../weights/best-step-N \\
    --out-dir experiments/tmp/why_trace_one

Plots only from an existing ``why_logprob_trace.json`` (no API):

  uv run python experiments/tinker_why_logprob_trace.py \\
    --pdf-only --out-dir experiments/tmp/why_trace_one

Document text defaults to ``ORIGINAL_DOCUMENT`` unless ``--document`` / ``--document-file``.
By default we **append** ``ANNOTATED_PREFIX`` (tokenized) **after** the analysis stub
(``<|channel|>final<|message|>``), so greedy + top-K decoding **continues inside** the
partial annotation (e.g. right after ``why="``). Requires ``force_stub_sampling: true``
in config; use ``--no-append-annotated-prefix`` to match plain eval (prefix not in prompt).
JSON includes ``sampler_prompt_token_ids`` and ``sampler_prompt_decode_exact``. Writes
``out-dir``: ``plot_01_context.pdf``, ``plot_02_surprisal.pdf``,
``plot_03_local_topk_bars.pdf``, ``plot_04_decoding_tree.pdf`` (greedy-prefix local tree).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import re
import textwrap
from pathlib import Path

import numpy as np

from rl_detector.config import CFG
from rl_detector.prompt_utils import (
    format_prompt_for_model,
    get_think_already_open,
    load_tokenizer,
)
from rl_detector.rollouts import (
    _get_analysis_stub_tokens,
    decode_response_text,
)

logger = logging.getLogger(__name__)

_WHY_RE = re.compile(r'why\s*=\s*"((?:\\.|[^"])*)"', re.IGNORECASE)

# Max new tokens for this experiment (overrides Hydra ``sampling.max_tokens`` here only).
_MAX_COMPLETION_TOKENS = 10

# Plain source passed to ``format_prompt_for_model`` (same logical doc as in the prefix below).
ORIGINAL_DOCUMENT = (
    "NFS is an abbreviation for Network File System. It allows you to mount remote directories as if they were locally stored drives so that all connected machines can access them. This method uses two servers running Ubuntu 18.04 LTS with one acting as both the server and the other as the client."
)

# Same passage in TELL markup, cut right before the verdict type value (new wire format).
ANNOTATED_PREFIX = (
    # '<text>NFS is an abbreviation for Network File System. It allows you to mount remote directories as if they were locally stored drives so that all connected machines can access them. This method uses two servers running Ubuntu 18.04 LTS with one acting as both the server and the other as the client.<verdict type="AI" why="'
    '<text>'
)


def completion_token_char_spans(tokenizer, completion_tokens: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for i in range(len(completion_tokens)):
        s = len(
            tokenizer.decode(
                completion_tokens[:i],
                skip_special_tokens=False,
            )
        )
        e = len(
            tokenizer.decode(
                completion_tokens[: i + 1],
                skip_special_tokens=False,
            )
        )
        spans.append((s, e))
    return spans


def first_why_value_char_span(completion_text: str) -> tuple[int, int] | None:
    m = _WHY_RE.search(completion_text)
    if m is None:
        return None
    return (int(m.start(1)), int(m.end(1)))


def token_indices_overlapping_char_span(
    char_spans: list[tuple[int, int]],
    lo: int,
    hi: int,
) -> list[int]:
    out: list[int] = []
    for i, span in enumerate(char_spans):
        s, e = span
        if s < hi and e > lo:
            out.append(i)
    return out


def topk_rows_for_completion(
    topk_nested: list[list[tuple[int, float]] | None] | None,
    prompt_len: int,
    n_completion: int,
) -> list[list[dict[str, float | int | str]] | None]:
    if topk_nested is None:
        return [None for _ in range(n_completion)]
    out: list[list[dict[str, float | int | str]] | None] = []
    for j in range(n_completion):
        pos = prompt_len + j
        if pos >= len(topk_nested):
            out.append(None)
            continue
        cell = topk_nested[pos]
        if cell is None:
            out.append(None)
            continue
        row: list[dict[str, float | int | str]] = []
        for tid, lp in cell:
            row.append({"token_id": int(tid), "logprob": float(lp)})
        out.append(row)
    return out


def _softmax_row(logprobs: list[float]) -> list[float]:
    m = max(logprobs)
    ex = [math.exp(lp - m) for lp in logprobs]
    s = sum(ex)
    return [e / s for e in ex]


def _token_label(tokenizer, token_id: int, max_len: int) -> str:
    piece = tokenizer.decode([int(token_id)], skip_special_tokens=False)
    piece_one = piece.replace("\n", "\\n")
    if len(piece_one) > max_len:
        piece_one = piece_one[: max_len - 3] + "..."
    return repr(piece_one)


def _ibm_plex_sans_family() -> str:
    from matplotlib import font_manager

    rel = Path(__file__).resolve().parent / "tmp" / "fonts" / "IBMPlexSans-Regular.ttf"
    if rel.is_file():
        font_manager.fontManager.addfont(str(rel))
        return str(font_manager.FontProperties(fname=str(rel)).get_name())
    return "IBM Plex Sans"


def _token_display_one_line(tokenizer, token_id: int, max_len: int) -> str:
    s = tokenizer.decode([int(token_id)], skip_special_tokens=False).replace("\n", "↵")
    if len(s) > max_len:
        return s[: max_len - 1] + "…"
    return s


def write_context_pdf(
    path: Path,
    original_document: str,
    annotated_prefix: str,
) -> None:
    from matplotlib import pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8.5, 11.0))
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.set_title("documents (embedded in script)", fontsize=11, pad=12, loc="left")
    body_a = textwrap.fill(
        "Original (sampling input):\n\n" + original_document,
        width=92,
        break_long_words=False,
        replace_whitespace=False,
    )
    body_b = textwrap.fill(
        "\n\nAnnotated prefix (reference, chopped before why text closes):\n\n" + annotated_prefix,
        width=92,
        break_long_words=False,
        replace_whitespace=False,
    )
    ax.text(
        0.02,
        0.98,
        body_a + body_b,
        transform=ax.transAxes,
        fontsize=7,
        family="monospace",
        verticalalignment="top",
        wrap=False,
    )
    fig.savefig(fname=path, format="pdf", bbox_inches="tight")
    plt.close(fig=fig)


def write_surprisal_pdf(
    path: Path,
    per_tok: list[dict[str, object]],
    why_token_indices: list[int] | None,
    max_completion_tokens: int,
) -> None:
    from matplotlib import pyplot as plt

    sur = [
        float(-t["teacher_logprob"]) if t["teacher_logprob"] == t["teacher_logprob"] else 0.0
        for t in per_tok
    ]
    fig, ax = plt.subplots(figsize=(11.0, 4.0))
    x = np.arange(len(sur))
    ax.plot(x, sur, color="#1f77b4", linewidth=1.0, marker="o", markersize=3)
    ax.set_xlabel("completion token index")
    ax.set_ylabel("teacher surprisal (-logprob)")
    ax.set_title(
        f"greedy path (max_tokens={max_completion_tokens}); shaded = first why= span",
        fontsize=10,
    )
    if why_token_indices:
        lo, hi = min(why_token_indices), max(why_token_indices)
        ax.axvspan(lo - 0.45, hi + 0.45, color="#ffbb78", alpha=0.35)
    fig.savefig(fname=path, format="pdf", bbox_inches="tight")
    plt.close(fig=fig)


def write_local_topk_bars_pdf(
    path: Path,
    tokenizer,
    per_tok: list[dict[str, object]],
    beam_width: int,
) -> None:
    from matplotlib import pyplot as plt

    n_steps = len(per_tok)
    if n_steps == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6.0, 2.0))
        ax.text(0.5, 0.5, "no completion tokens", ha="center", va="center")
        ax.axis("off")
        fig.savefig(fname=path, format="pdf", bbox_inches="tight")
        plt.close(fig=fig)
        return
    B = int(beam_width)
    row_h = 0.42
    fig_h = min(11.0, 1.2 + row_h * n_steps)
    fig, axes = plt.subplots(
        nrows=n_steps,
        ncols=1,
        figsize=(11.0, max(fig_h, 0.55 * n_steps + 1.5)),
        squeeze=False,
    )
    for t in range(n_steps):
        axb = axes[t, 0]
        cell = per_tok[t].get("topk")
        greedy_id = int(per_tok[t]["token_id"])
        if not cell:
            axb.axis("off")
            axb.set_title(f"step {t}: (no topk)", loc="left", fontsize=8)
            continue
        rows = sorted(cell, key=lambda r: -float(r["logprob"]))[: min(B, len(cell))]
        lps = [float(r["logprob"]) for r in rows]
        probs = _softmax_row(logprobs=lps)
        labels: list[str] = []
        colors: list[tuple[float, float, float]] = []
        widths: list[float] = []
        for r, pr in zip(rows, probs, strict=True):
            tid = int(r["token_id"])
            piece_one = _token_label(tokenizer=tokenizer, token_id=tid, max_len=36)
            mark = "*" if tid == greedy_id else " "
            labels.append(f"{mark} {piece_one}  lp={float(r['logprob']):.3f}")
            colors.append((0.2, 0.55, 0.85) if tid == greedy_id else (0.75, 0.78, 0.82))
            widths.append(max(pr, 1e-6))
        y = np.arange(len(rows))
        axb.barh(y, width=widths, color=colors, edgecolor="#333333", linewidth=0.3, height=0.75)
        axb.set_yticks(y)
        axb.set_yticklabels(labels, fontsize=6, family="monospace")
        axb.set_xlabel("renormalized mass over top-K at this step", fontsize=7)
        axb.set_title(f"step {t}  ( * = greedy token )", loc="left", fontsize=8)
        axb.set_xlim(0.0, min(1.05, max(widths) * 1.15 + 0.02))
    fig.suptitle(
        "local top-K at each greedy prefix (not a joint beam search tree)",
        fontsize=9,
    )
    fig.subplots_adjust(hspace=0.65, top=0.96)
    fig.savefig(fname=path, format="pdf", bbox_inches="tight")
    plt.close(fig=fig)


def _decoding_tree_label_bbox(
    ax,
    fig,
    renderer,
    *,
    ff: str,
    label_text: str,
    fontsize: float,
    is_greedy: bool,
) -> tuple[float, float]:
    lw = 1.1 if is_greedy else 0.52
    tb = ax.text(
        0.0,
        0.0,
        label_text,
        ha="center",
        va="center",
        fontsize=fontsize,
        family=ff,
        color="#007ACC",
        alpha=0.0,
        bbox={
            "boxstyle": "square,pad=0.18",
            "facecolor": "#ffffff",
            "edgecolor": "#007ACC",
            "linewidth": lw,
        },
    )
    fig.canvas.draw()
    bb = tb.get_window_extent(renderer=renderer).transformed(ax.transData.inverted())
    tb.remove()
    return float(bb.width) * 0.5, float(bb.height) * 0.5


def _column_y_centers(half_heights: list[float], row_gap: float) -> np.ndarray:
    n = len(half_heights)
    if n == 1:
        return np.array([0.0])
    total = sum(2.0 * h for h in half_heights) + row_gap * (n - 1)
    y = total * 0.5 - half_heights[0]
    ys: list[float] = []
    for h in half_heights:
        ys.append(y)
        y -= 2.0 * h + row_gap
    return np.array(ys, dtype=float)


def write_decoding_tree_pdf(
    path: Path,
    tokenizer,
    per_tok: list[dict[str, object]],
    beam_width: int,
) -> None:
    from matplotlib import pyplot as plt

    ACCENT = "#007ACC"
    T = len(per_tok)
    if T == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6.0, 2.0))
        ax.text(0.5, 0.5, "no completion tokens", ha="center", va="center")
        ax.axis("off")
        fig.savefig(fname=path, format="pdf", bbox_inches="tight", pad_inches=0.0)
        plt.close(fig=fig)
        return

    ff = _ibm_plex_sans_family()
    B = int(beam_width)
    col_gap = 0.36
    row_gap = 0.035
    root_font = 6.0
    node_font = 5.8
    fig_h_in = 0.92

    step_rows: list[list[dict[str, float | int]]] = []
    for t in range(T):
        cell = per_tok[t].get("topk")
        greedy_id = int(per_tok[t]["token_id"])
        if not cell:
            step_rows.append([{"token_id": greedy_id, "logprob": 0.0}])
            continue
        rows = sorted(cell, key=lambda r: -float(r["logprob"]))[: min(B, len(cell))]
        step_rows.append(rows)

    fig_w_in = 6.0
    fig = None
    ax = None
    renderer = None
    root_half = 0.0
    col_half: list[float] = []
    col_x: list[float] = [0.0]
    step_y_centers: list[np.ndarray] = []
    for _ in range(2):
        if fig is not None:
            plt.close(fig=fig)
        fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in))
        ax.axis("off")
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        root_half, _ = _decoding_tree_label_bbox(
            ax=ax,
            fig=fig,
            renderer=renderer,
            ff=ff,
            label_text="…",
            fontsize=root_font,
            is_greedy=True,
        )
        col_half = []
        step_y_centers = []
        for t in range(T):
            greedy_id = int(per_tok[t]["token_id"])
            rows = step_rows[t]
            lps = [float(r["logprob"]) for r in rows]
            probs = _softmax_row(logprobs=lps)
            max_hw = 0.0
            half_hs: list[float] = []
            for r, pr in zip(rows, probs, strict=True):
                tid = int(r["token_id"])
                disp = _token_display_one_line(tokenizer=tokenizer, token_id=tid, max_len=14)
                pct_i = int(round(100.0 * float(pr)))
                pct_i = max(0, min(100, pct_i))
                label = f"{disp} ({pct_i}%)"
                hw, hh = _decoding_tree_label_bbox(
                    ax=ax,
                    fig=fig,
                    renderer=renderer,
                    ff=ff,
                    label_text=label,
                    fontsize=node_font,
                    is_greedy=tid == greedy_id,
                )
                max_hw = max(max_hw, hw)
                half_hs.append(hh)
            col_half.append(max_hw)
            step_y_centers.append(_column_y_centers(half_heights=half_hs, row_gap=row_gap))
        col_x = [0.0]
        x_cursor = 0.0
        for t in range(T):
            left_hw = root_half if t == 0 else col_half[t - 1]
            x_cursor = x_cursor + left_hw + col_gap + col_half[t]
            col_x.append(float(x_cursor))
        span_x = float(col_x[-1] + col_half[-1] + root_half + 0.20)
        fig_w_next = max(6.0, span_x * 0.52)
        if abs(fig_w_next - fig_w_in) < 0.05:
            break
        fig_w_in = fig_w_next

    node_xy: list[list[tuple[float, float, int, float]]] = []
    for t in range(T):
        x_col = float(col_x[t + 1])
        rows = step_rows[t]
        lps = [float(r["logprob"]) for r in rows]
        probs = _softmax_row(logprobs=lps)
        ys = step_y_centers[t]
        step_nodes: list[tuple[float, float, int, float]] = []
        for r, pr, y in zip(rows, probs, ys, strict=True):
            tid = int(r["token_id"])
            step_nodes.append((x_col, float(y), tid, float(pr)))
        node_xy.append(step_nodes)

    ax.set_xlim(-root_half - 0.04, col_x[-1] + col_half[-1] + 0.04)

    def add_label(
        x: float,
        y: float,
        label_text: str,
        fontsize: float,
        *,
        is_greedy: bool,
    ):
        lw = 1.1 if is_greedy else 0.52
        return ax.text(
            x,
            y,
            label_text,
            ha="center",
            va="center",
            fontsize=fontsize,
            family=ff,
            color=ACCENT,
            zorder=5,
            bbox={
                "boxstyle": "square,pad=0.18",
                "facecolor": "#ffffff",
                "edgecolor": ACCENT,
                "linewidth": lw,
            },
        )

    def anchor(tt, kind: str, renderer):
        bb = tt.get_window_extent(renderer=renderer).transformed(ax.transData.inverted())
        cx = bb.x0 + bb.width * 0.5
        if kind == "bottom":
            return cx, bb.y0
        return cx, bb.y1

    drawn_artists: list[object] = []
    root_t = add_label(x=col_x[0], y=0.0, label_text="…", fontsize=root_font, is_greedy=True)
    drawn_artists.append(root_t)
    parent_t = root_t
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    for t in range(T):
        step_nodes = node_xy[t]
        greedy_id = int(per_tok[t]["token_id"])
        child_texts: list[tuple[object, float]] = []
        for x, y, tid, pr in step_nodes:
            pr = max(float(pr), 1e-12)
            disp = _token_display_one_line(tokenizer=tokenizer, token_id=tid, max_len=14)
            pct_i = int(round(100.0 * pr))
            pct_i = max(0, min(100, pct_i))
            label = f"{disp} ({pct_i}%)"
            is_g = tid == greedy_id
            tb = add_label(
                x=x,
                y=y,
                label_text=label,
                fontsize=node_font,
                is_greedy=is_g,
            )
            drawn_artists.append(tb)
            child_texts.append((tb, float(pr)))
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        x1, y1 = anchor(tt=parent_t, kind="bottom", renderer=renderer)
        for tb, pr in child_texts:
            x2, y2 = anchor(tt=tb, kind="top", renderer=renderer)
            lw = 0.3 + 2.0 * pr
            alpha = 0.28 + 0.72 * pr
            ln = ax.plot(
                [x1, x2],
                [y1, y2],
                color=ACCENT,
                linewidth=lw,
                alpha=min(alpha, 1.0),
                solid_capstyle="round",
                zorder=2,
            )
            drawn_artists.extend(ln)
        greedy_tb = None
        for (x, y, tid, pr), (tb, _) in zip(step_nodes, child_texts, strict=True):
            if int(tid) == greedy_id:
                greedy_tb = tb
                break
        if greedy_tb is None:
            greedy_tb = child_texts[0][0]
        parent_t = greedy_tb

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    x_lo = float("inf")
    x_hi = float("-inf")
    y_lo = float("inf")
    y_hi = float("-inf")
    for art in drawn_artists:
        bb = art.get_window_extent(renderer=renderer).transformed(ax.transData.inverted())
        x_lo = min(x_lo, float(bb.x0))
        x_hi = max(x_hi, float(bb.x1))
        y_lo = min(y_lo, float(bb.y0))
        y_hi = max(y_hi, float(bb.y1))
    edge_pad = 0.02
    ax.set_xlim(x_lo - edge_pad, x_hi + edge_pad)
    ax.set_ylim(y_lo - edge_pad, y_hi + edge_pad)
    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
    fig.savefig(fname=path, format="pdf", bbox_inches="tight", pad_inches=0.0)
    plt.close(fig=fig)


def write_all_plot_pdfs(
    out_dir: Path,
    original_document: str,
    annotated_prefix: str,
    tokenizer,
    per_tok: list[dict[str, object]],
    why_token_indices: list[int] | None,
    beam_width: int,
    max_completion_tokens: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_context_pdf(
        path=out_dir / "plot_01_context.pdf",
        original_document=original_document,
        annotated_prefix=annotated_prefix,
    )
    logger.info("wrote %s", (out_dir / "plot_01_context.pdf").resolve())
    write_surprisal_pdf(
        path=out_dir / "plot_02_surprisal.pdf",
        per_tok=per_tok,
        why_token_indices=why_token_indices,
        max_completion_tokens=max_completion_tokens,
    )
    logger.info("wrote %s", (out_dir / "plot_02_surprisal.pdf").resolve())
    write_local_topk_bars_pdf(
        path=out_dir / "plot_03_local_topk_bars.pdf",
        tokenizer=tokenizer,
        per_tok=per_tok,
        beam_width=beam_width,
    )
    logger.info("wrote %s", (out_dir / "plot_03_local_topk_bars.pdf").resolve())
    write_decoding_tree_pdf(
        path=out_dir / "plot_04_decoding_tree.pdf",
        tokenizer=tokenizer,
        per_tok=per_tok,
        beam_width=beam_width,
    )
    logger.info("wrote %s", (out_dir / "plot_04_decoding_tree.pdf").resolve())


def render_pdf_from_record(
    record: dict[str, object],
    out_dir: Path,
    tokenizer,
    beam_width: int,
) -> None:
    per_tok = record["per_completion_token"]
    assert isinstance(per_tok, list)
    why_raw = record.get("why_token_indices_completion_only", record.get("why_token_indices"))
    why_token_indices: list[int] | None = None
    if isinstance(why_raw, list):
        why_token_indices = [int(x) for x in why_raw]
    mct = record.get("max_completion_tokens")
    max_completion_tokens = int(mct) if mct is not None else _MAX_COMPLETION_TOKENS
    od = record.get("original_document")
    original_document = str(od) if isinstance(od, str) else ORIGINAL_DOCUMENT
    ap = record.get("annotated_prefix")
    annotated_prefix = str(ap) if isinstance(ap, str) else ANNOTATED_PREFIX
    write_all_plot_pdfs(
        out_dir=out_dir,
        original_document=original_document,
        annotated_prefix=annotated_prefix,
        tokenizer=tokenizer,
        per_tok=per_tok,
        why_token_indices=why_token_indices,
        beam_width=beam_width,
        max_completion_tokens=max_completion_tokens,
    )


def pdf_only_from_out_dir(out_dir: Path, beam_width: int) -> None:
    json_path = out_dir / "why_logprob_trace.json"
    raw = json_path.read_text(encoding="utf-8")
    record = json.loads(raw)
    assert isinstance(record, dict)
    bw_raw = record.get("beam_width_plot")
    beam_use = int(bw_raw) if bw_raw is not None else int(beam_width)
    tokenizer = load_tokenizer()
    render_pdf_from_record(
        record=record,
        out_dir=out_dir,
        tokenizer=tokenizer,
        beam_width=beam_use,
    )


async def run_trace(
    checkpoint: str,
    document: str,
    annotated_prefix: str,
    out_dir: Path,
    topk: int,
    eval_seed: int,
    beam_width: int,
    max_completion_tokens: int,
    append_annotated_prefix: bool,
    injected_label: str = "",
) -> dict:
    import tinker

    out_dir.mkdir(parents=True, exist_ok=True)
    service_client = tinker.ServiceClient()
    training_client = await service_client.create_training_client_from_state_with_optimizer_async(
        path=checkpoint,
    )
    sampling_client = await training_client.save_weights_and_get_sampling_client_async()
    tokenizer = load_tokenizer()

    force_stub = bool(CFG.sampling.force_stub_sampling)
    logical_user_prompt_text, chat_templated_prompt_string = format_prompt_for_model(
        tokenizer=tokenizer,
        text=document,
        add_generation_prompt=True,
    )
    neutral_prompt_tokens = list(tokenizer.encode(chat_templated_prompt_string))
    stub_open_ids: list[int] = []
    stub_close_ids: list[int] = []
    if force_stub:
        think_already_open = get_think_already_open(tokenizer=tokenizer)
        stub_open, stub_close = _get_analysis_stub_tokens(
            tokenizer=tokenizer,
            think_already_open=think_already_open,
        )
        stub_open_ids = [int(x) for x in stub_open]
        stub_close_ids = [int(x) for x in stub_close]
        label_token_ids: list[int] = []
        if injected_label.strip():
            label_token_ids = list(
                tokenizer.encode(injected_label, add_special_tokens=False)
            )
        prompt_tokens = neutral_prompt_tokens + stub_open + label_token_ids + stub_close
    else:
        label_token_ids = []
        prompt_tokens = list(neutral_prompt_tokens)

    annotated_prefix_token_ids: list[int] = []
    if append_annotated_prefix:
        if not annotated_prefix.strip():
            raise ValueError("append_annotated_prefix is true but annotated_prefix is empty")
        if not force_stub:
            raise RuntimeError(
                "append_annotated_prefix requires sampling.force_stub_sampling=true "
                "(annotated XML must sit after the <|channel|>final<|message|> stub).",
            )
        annotated_prefix_token_ids = list(
            tokenizer.encode(annotated_prefix, add_special_tokens=False),
        )
        prompt_tokens = list(prompt_tokens) + annotated_prefix_token_ids

    sampler_prompt_token_ids = [int(x) for x in prompt_tokens]
    sampler_prompt_decode_exact = tokenizer.decode(
        sampler_prompt_token_ids,
        skip_special_tokens=False,
    )

    sp_sample = tinker.SamplingParams(
        max_tokens=int(max_completion_tokens),
        seed=int(eval_seed),
        temperature=0.0,
        top_p=1.0,
        reasoning_effort=CFG.sampling.reasoning_effort,
    )
    sampled = await sampling_client.sample_async(
        prompt=tinker.ModelInput.from_ints(prompt_tokens),
        num_samples=1,
        sampling_params=sp_sample,
    )
    seq = sampled.sequences[0]
    completion_tokens = list(seq.tokens)
    sample_lps = list(seq.logprobs) if seq.logprobs is not None else None

    full_input = tinker.ModelInput.from_ints(prompt_tokens + completion_tokens)
    teacher_lps_raw = await sampling_client.compute_logprobs_async(prompt=full_input)
    p = len(prompt_tokens)
    teacher_completion = [
        float(teacher_lps_raw[i]) if teacher_lps_raw[i] is not None else float("nan")
        for i in range(p, len(teacher_lps_raw))
    ]

    sp_topk = tinker.SamplingParams(
        max_tokens=1,
        seed=int(eval_seed),
        temperature=0.0,
        top_p=1.0,
        reasoning_effort=CFG.sampling.reasoning_effort,
    )
    topk_res = await sampling_client.sample_async(
        prompt=full_input,
        num_samples=1,
        sampling_params=sp_topk,
        include_prompt_logprobs=True,
        topk_prompt_logprobs=int(topk),
    )
    topk_nested = topk_res.topk_prompt_logprobs
    topk_per_completion = topk_rows_for_completion(
        topk_nested=topk_nested,
        prompt_len=p,
        n_completion=len(completion_tokens),
    )

    completion_text = tokenizer.decode(
        completion_tokens,
        skip_special_tokens=False,
    )
    response_text = decode_response_text(
        tokenizer=tokenizer,
        completion_tokens=completion_tokens,
        completion_text=completion_text,
        force_stub_sampling=force_stub,
    )
    if append_annotated_prefix:
        full_for_why = annotated_prefix + completion_text
        off = len(annotated_prefix)
    else:
        full_for_why = completion_text
        off = 0
    why_span_full = first_why_value_char_span(completion_text=full_for_why)
    char_spans = completion_token_char_spans(
        tokenizer=tokenizer,
        completion_tokens=completion_tokens,
    )
    why_token_indices: list[int] | None = None
    if why_span_full is not None:
        lo = why_span_full[0] - off
        hi = why_span_full[1] - off
        if hi > 0 and lo < len(completion_text):
            lo = max(0, lo)
            hi = min(hi, len(completion_text))
            idxs = token_indices_overlapping_char_span(
                char_spans=char_spans,
                lo=lo,
                hi=hi,
            )
            why_token_indices = idxs if idxs else None

    per_tok: list[dict[str, object]] = []
    for i in range(len(completion_tokens)):
        tid = int(completion_tokens[i])
        piece = tokenizer.decode([tid], skip_special_tokens=False)
        tl = teacher_completion[i] if i < len(teacher_completion) else float("nan")
        sl = float(sample_lps[i]) if sample_lps is not None and i < len(sample_lps) else float("nan")
        per_tok.append({
            "i": i,
            "token_id": tid,
            "piece": piece,
            "teacher_logprob": tl,
            "sampled_logprob": sl,
            "surprisal_teacher": float(-tl) if tl == tl else float("nan"),
            "topk": topk_per_completion[i],
        })

    record: dict[str, object] = {
        "checkpoint": checkpoint,
        "force_stub_sampling": force_stub,
        "eval_seed": int(eval_seed),
        "max_completion_tokens": int(max_completion_tokens),
        "topk_requested": int(topk),
        "beam_width_plot": int(beam_width),
        "n_prompt_tokens": p,
        "n_completion_tokens": len(completion_tokens),
        "why_value_char_span_in_full_final_text": list(why_span_full) if why_span_full else None,
        "why_token_indices_completion_only": why_token_indices,
        "response_text_head": full_for_why[:800],
        "original_document": document,
        "annotated_prefix": annotated_prefix,
        "annotated_prefix_embedded": annotated_prefix[:500],
        "append_annotated_prefix_to_prompt": bool(append_annotated_prefix),
        "annotated_prefix_token_ids": annotated_prefix_token_ids,
        "annotated_prefix_token_count": len(annotated_prefix_token_ids),
        "logical_user_prompt_text": logical_user_prompt_text,
        "chat_templated_prompt_string": chat_templated_prompt_string,
        "neutral_prompt_token_ids": neutral_prompt_tokens,
        "stub_open_token_ids": stub_open_ids if force_stub else None,
        "injected_label": injected_label,
        "injected_label_token_ids": label_token_ids if force_stub else None,
        "stub_close_token_ids": stub_close_ids if force_stub else None,
        "sampler_prompt_token_ids": sampler_prompt_token_ids,
        "sampler_prompt_decode_exact": sampler_prompt_decode_exact,
        "completion_token_ids": [int(x) for x in completion_tokens],
        "full_sequence_token_ids": sampler_prompt_token_ids + [int(x) for x in completion_tokens],
        "per_completion_token": per_tok,
    }
    out_json = out_dir / "why_logprob_trace.json"
    out_json.write_text(
        data=json.dumps(obj=record, indent=2),
        encoding="utf-8",
    )
    logger.info("wrote %s", out_json.resolve())

    write_all_plot_pdfs(
        out_dir=out_dir,
        original_document=document,
        annotated_prefix=annotated_prefix,
        tokenizer=tokenizer,
        per_tok=per_tok,
        why_token_indices=why_token_indices,
        beam_width=beam_width,
        max_completion_tokens=max_completion_tokens,
    )

    return record


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tinker trace + PDF beam-style top-K (max 10 tokens).")
    p.add_argument("--checkpoint", type=str, default="")
    p.add_argument("--document", type=str, default="")
    p.add_argument("--document-file", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--beam-width", type=int, default=8)
    p.add_argument("--eval-seed", type=int, default=2242)
    p.add_argument("--max-completion-tokens", type=int, default=_MAX_COMPLETION_TOKENS)
    p.add_argument(
        "--pdf-only",
        action="store_true",
        help="Only read why_logprob_trace.json under --out-dir and write plot_*.pdf files (no Tinker).",
    )
    p.add_argument(
        "--no-append-annotated-prefix",
        action="store_true",
        help="Do not append ANNOTATED_PREFIX tokens after the stub (plain eval-style prompt).",
    )
    p.add_argument(
        "--annotated-prefix-file",
        type=Path,
        default=None,
        help="Optional UTF-8 file overriding the in-script ANNOTATED_PREFIX string.",
    )
    p.add_argument(
        "--label",
        type=str,
        default="",
        help=(
            "Text injected into the reasoning prompt between stub_open and stub_close, "
            "e.g. 'Text origin is human.' Requires force_stub_sampling=true in config."
        ),
    )
    return p.parse_args()


async def async_main(args: argparse.Namespace) -> None:
    if not str(args.checkpoint).strip():
        raise SystemExit("--checkpoint is required unless --pdf-only")
    doc_path = args.document_file
    if doc_path is not None:
        document = doc_path.read_text(encoding="utf-8")
    elif str(args.document).strip():
        document = str(args.document)
    else:
        document = ORIGINAL_DOCUMENT
    ap = ANNOTATED_PREFIX
    if args.annotated_prefix_file is not None:
        ap = args.annotated_prefix_file.read_text(encoding="utf-8")
    await run_trace(
        checkpoint=str(args.checkpoint),
        document=document,
        annotated_prefix=ap,
        out_dir=args.out_dir,
        topk=int(args.topk),
        eval_seed=int(args.eval_seed),
        beam_width=int(args.beam_width),
        max_completion_tokens=int(args.max_completion_tokens),
        append_annotated_prefix=not bool(args.no_append_annotated_prefix),
        injected_label=str(args.label),
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if bool(args.pdf_only):
        pdf_only_from_out_dir(out_dir=args.out_dir, beam_width=int(args.beam_width))
    else:
        asyncio.run(async_main(args=args))


if __name__ == "__main__":
    main()
