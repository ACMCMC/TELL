# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "matplotlib",
# ]
# ///
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

TEX_STUBS = Path(__file__).resolve().parent / "tex_stubs"

plt.rcParams["pgf.texsystem"] = "pdflatex"
plt.rcParams["pgf.preamble"] = r"\usepackage[T1]{fontenc}\usepackage[utf8]{inputenc}"

JSON_PATH = Path(__file__).resolve().parent / "detectors_and_reported_scores.json"
OUT_PDF = Path(__file__).resolve().parent / "detectors_and_reported_scores.pdf"
OUT_PGF = Path(__file__).resolve().parent / "detectors_and_reported_scores.pgf"
OUT_AUROC_PDF = Path(__file__).resolve().parent / "detectors_and_reported_scores_auroc.pdf"
OUT_AUROC_PGF = Path(__file__).resolve().parent / "detectors_and_reported_scores_auroc.pgf"
OUT_GAPS_JSON = Path(__file__).resolve().parent / "detectors_and_reported_scores_gaps.json"

AUROC_KEYS = ["AUROC"]
TPR_KEYS = ["TPR@1%FPR", "TPR@0.01"]
MARKERS = ["o", "s", "^", "D", "v", "*", "P", "X", "h", "8", "<", ">"]
SKIP_PHRASES = (
    "not reported",
    "not applicable",
    "n/a",
    "significant drop",
    "varies by",
    "considerable unreliability",
    "evaluated as",
    "evaluated across",
    "near-zero error",
    "trade-off possible",
    "reduced in real-world",
    "average improvement",
    "low reliability",
)


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(fp=f)


def get_score(scores: dict, keys: list[str]) -> object:
    for key in keys:
        if key in scores:
            return scores[key]
    return None


def parse_score(raw: object) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        value = float(raw)
    else:
        text = str(raw).strip().lower()
        if not text:
            return None
        if any(phrase in text for phrase in SKIP_PHRASES):
            return None
        nums = re.findall(r"\d+\.\d+|\d+", text)
        if not nums:
            return None
        if " from " in text and " to " in text:
            value = float(nums[-1])
        else:
            value = float(nums[0])
    if value > 1.0 and value <= 100.0:
        value = value / 100.0
    return value


def score_rows(block: dict) -> list[dict]:
    metrics = block.get("metrics")
    if metrics is not None:
        return metrics
    scores = block.get("scores")
    if scores is not None:
        nested = scores.get("metrics")
        if nested is not None:
            return nested
        return [scores]
    return [block]


def metric_label(row: dict) -> str | None:
    for key in ("condition", "dataset", "model"):
        value = row.get(key)
        if value is not None:
            return str(value)
    return None


def evaluation_paper(evaluation: dict) -> str | None:
    paper = evaluation.get("paper_title")
    if paper is not None:
        return str(paper)
    source = evaluation.get("source")
    if source is not None:
        return str(source)
    return None


def collect_points(data: dict) -> list[dict]:
    points = []
    for detector_idx, detector in enumerate(data["detectors"]):
        name = detector["name"]
        marker = MARKERS[detector_idx % len(MARKERS)]
        for row in score_rows(block=detector["original_scores"]):
            auroc = parse_score(raw=get_score(scores=row, keys=AUROC_KEYS))
            tpr = parse_score(raw=get_score(scores=row, keys=TPR_KEYS))
            if auroc is not None and tpr is not None:
                points.append(
                    {
                        "detector": name,
                        "marker": marker,
                        "auroc": auroc,
                        "tpr": tpr,
                        "source": "original",
                    }
                )
        for evaluation in detector["independent_evaluations"]:
            for row in score_rows(block=evaluation):
                auroc = parse_score(raw=get_score(scores=row, keys=AUROC_KEYS))
                tpr = parse_score(raw=get_score(scores=row, keys=TPR_KEYS))
                if auroc is not None and tpr is not None:
                    points.append(
                        {
                            "detector": name,
                            "marker": marker,
                            "auroc": auroc,
                            "tpr": tpr,
                            "source": "independent",
                        }
                    )
    return points


def collect_auroc_points(data: dict) -> list[dict]:
    points = []
    for detector_idx, detector in enumerate(data["detectors"]):
        name = detector["name"]
        marker = MARKERS[detector_idx % len(MARKERS)]
        for row in score_rows(block=detector["original_scores"]):
            auroc = parse_score(raw=get_score(scores=row, keys=AUROC_KEYS))
            if auroc is not None:
                points.append(
                    {
                        "detector": name,
                        "marker": marker,
                        "auroc": auroc,
                        "source": "original",
                    }
                )
        for evaluation in detector["independent_evaluations"]:
            for row in score_rows(block=evaluation):
                auroc = parse_score(raw=get_score(scores=row, keys=AUROC_KEYS))
                if auroc is not None:
                    points.append(
                        {
                            "detector": name,
                            "marker": marker,
                            "auroc": auroc,
                            "source": "independent",
                        }
                    )
    return points


def collect_gaps(data: dict) -> dict:
    auroc_pairs = []
    tpr_pairs = []
    for detector in data["detectors"]:
        name = detector["name"]
        orig_rows = score_rows(block=detector["original_scores"])
        orig_aurocs = []
        orig_tprs = []
        for row in orig_rows:
            auroc = parse_score(raw=get_score(scores=row, keys=AUROC_KEYS))
            tpr = parse_score(raw=get_score(scores=row, keys=TPR_KEYS))
            if auroc is not None:
                orig_aurocs.append((auroc, metric_label(row=row)))
            if tpr is not None:
                orig_tprs.append((tpr, metric_label(row=row)))
        for evaluation in detector["independent_evaluations"]:
            paper = evaluation_paper(evaluation=evaluation)
            for row in score_rows(block=evaluation):
                indep_auroc = parse_score(raw=get_score(scores=row, keys=AUROC_KEYS))
                indep_tpr = parse_score(raw=get_score(scores=row, keys=TPR_KEYS))
                indep_condition = metric_label(row=row)
                if indep_auroc is not None:
                    for orig_auroc, orig_condition in orig_aurocs:
                        auroc_pairs.append(
                            {
                                "detector": name,
                                "original": orig_auroc,
                                "independent": indep_auroc,
                                "gap": orig_auroc - indep_auroc,
                                "original_condition": orig_condition,
                                "independent_condition": indep_condition,
                                "paper": paper,
                            }
                        )
                if indep_tpr is not None:
                    for orig_tpr, orig_condition in orig_tprs:
                        tpr_pairs.append(
                            {
                                "detector": name,
                                "original": orig_tpr,
                                "independent": indep_tpr,
                                "gap": orig_tpr - indep_tpr,
                                "original_condition": orig_condition,
                                "independent_condition": indep_condition,
                                "paper": paper,
                            }
                        )
    auroc_gaps = [pair["gap"] for pair in auroc_pairs]
    tpr_gaps = [pair["gap"] for pair in tpr_pairs]
    return {
        "AUROC": {
            "pairs": auroc_pairs,
            "mean_gap": sum(auroc_gaps) / len(auroc_gaps) if auroc_gaps else None,
            "n_pairs": len(auroc_gaps),
        },
        "TPR@1%FPR": {
            "pairs": tpr_pairs,
            "mean_gap": sum(tpr_gaps) / len(tpr_gaps) if tpr_gaps else None,
            "n_pairs": len(tpr_gaps),
        },
    }


def write_gaps(path: Path, gaps: dict) -> None:
    with path.open(mode="w", encoding="utf-8") as f:
        json.dump(gaps, fp=f, indent=4)
        f.write("\n")


def print_gaps(gaps: dict) -> None:
    for metric in ("AUROC", "TPR@1%FPR"):
        summary = gaps[metric]
        mean_gap = summary["mean_gap"]
        if mean_gap is None:
            print(f"{metric}: no comparable pairs")
            continue
        print(f"{metric} mean gap (original - independent): {mean_gap:.4f} over {summary['n_pairs']} pairs")
        for pair in summary["pairs"]:
            print(
                f"  {pair['detector']}: {pair['original']:.4f} - {pair['independent']:.4f} = {pair['gap']:.4f}"
                f" ({pair['paper']})"
            )


def save_figure(fig: plt.Figure, out_pdf: Path, out_pgf: Path) -> None:
    fig.savefig(fname=out_pdf, format="pdf", bbox_inches="tight")
    if TEX_STUBS.is_dir():
        texinputs = os.environ.get("TEXINPUTS", "")
        os.environ["TEXINPUTS"] = f"{TEX_STUBS}//:{texinputs}"
        try:
            fig.savefig(fname=out_pgf, format="pgf", bbox_inches="tight")
        except Exception:
            pass


def source_color(source: str) -> str:
    if source == "original":
        return "#d62728"
    return "#1f77b4"


def legend_handles(detector_markers: dict[str, str]) -> list[Line2D]:
    handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            color="black",
            linestyle="None",
            markersize=7,
            label=name,
        )
        for name, marker in detector_markers.items()
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#d62728",
            markeredgecolor="black",
            markeredgewidth=0.6,
            markersize=7,
            label="original paper",
        )
    )
    handles.append(
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#1f77b4",
            markeredgecolor="black",
            markeredgewidth=0.6,
            markersize=7,
            label="independent work",
        )
    )
    return handles


def plot_points(points: list[dict], out_pdf: Path, out_pgf: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    detector_markers = {}
    for point in points:
        ax.scatter(
            point["tpr"],
            point["auroc"],
            marker=point["marker"],
            s=70,
            c=source_color(source=point["source"]),
            edgecolors="black",
            linewidths=0.6,
            alpha=0.4,
            zorder=3,
        )
        if point["detector"] not in detector_markers:
            detector_markers[point["detector"]] = point["marker"]
    ax.set_xlabel("TPR@1%FPR")
    ax.set_ylabel("AUROC")
    ax.set_xlim(left=0.0, right=1.0)
    ax.set_ylim(bottom=0.5, top=1.0)
    ax.legend(handles=legend_handles(detector_markers=detector_markers), loc="lower left", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    save_figure(fig=fig, out_pdf=out_pdf, out_pgf=out_pgf)
    plt.close(fig=fig)


def plot_auroc_only(points: list[dict], out_pdf: Path, out_pgf: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 2.0))
    detector_markers = {}
    ax.axhline(y=0.0, color="#cccccc", linewidth=0.8, zorder=1)
    for point in points:
        ax.scatter(
            point["auroc"],
            0.0,
            marker=point["marker"],
            s=70,
            c=source_color(source=point["source"]),
            edgecolors="black",
            linewidths=0.6,
            alpha=0.4,
            zorder=3,
        )
        if point["detector"] not in detector_markers:
            detector_markers[point["detector"]] = point["marker"]
    ax.set_xlabel("AUROC")
    ax.set_xlim(left=0.5, right=1.0)
    ax.set_ylim(bottom=-0.4, top=0.4)
    ax.set_yticks([])
    ax.legend(
        handles=legend_handles(detector_markers=detector_markers),
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=8,
        framealpha=0.9,
    )
    fig.tight_layout(rect=[0.0, 0.0, 0.78, 1.0])
    save_figure(fig=fig, out_pdf=out_pdf, out_pgf=out_pgf)
    plt.close(fig=fig)


def main() -> None:
    data = load_json(path=JSON_PATH)
    points = collect_points(data=data)
    auroc_points = collect_auroc_points(data=data)
    gaps = collect_gaps(data=data)
    plot_points(points=points, out_pdf=OUT_PDF, out_pgf=OUT_PGF)
    plot_auroc_only(points=auroc_points, out_pdf=OUT_AUROC_PDF, out_pgf=OUT_AUROC_PGF)
    write_gaps(path=OUT_GAPS_JSON, gaps=gaps)
    print_gaps(gaps=gaps)
    print(f"wrote {OUT_GAPS_JSON}")
    print(f"wrote {OUT_PDF}")
    if OUT_PGF.is_file():
        print(f"wrote {OUT_PGF}")
    print(f"plotted {len(points)} scatter points")
    print(f"wrote {OUT_AUROC_PDF}")
    if OUT_AUROC_PGF.is_file():
        print(f"wrote {OUT_AUROC_PGF}")
    print(f"plotted {len(auroc_points)} auroc-only points")


if __name__ == "__main__":
    main()
