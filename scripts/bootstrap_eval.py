"""
Bootstrap ranking stability + DeLong pairwise significance for the HF5k benchmark.

Produces:
  results/bootstrap_eval/summary.json        — point estimates, AUROC + TPR@1%FPR CIs, pairwise p-values
  results/bootstrap_eval/ranking_table.md   — main paper table (markdown)
  results/bootstrap_eval/domain_table.md    — per-domain AUROC table (markdown)
  results/bootstrap_eval/pairwise_table.md  — BH-corrected significance matrix (markdown)
  results/bootstrap_eval/ranking_table.tex  — main paper table (LaTeX)
  results/bootstrap_eval/ranking_table_short.tex — compact detector comparison (LaTeX)
  results/bootstrap_eval/domain_table.tex   — per-domain AUROC table (LaTeX)
  results/bootstrap_eval/pairwise_table.tex — BH-corrected significance matrix (LaTeX)

Usage:
  python scripts/bootstrap_eval.py [--bootstrap 10000] [--seed 42]
"""

import argparse
import json
import os
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from statsmodels.stats.multitest import multipletests

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASELINE_DIR = Path("detectors/results/multi_domain_hf5k_benchmark/merged_predictions")
ACMC_HF_FIVE_JSON = Path("results/acmc_hf_five_detector_scores.json")
OUR_RESULTS = Path("results/eval_v6_best235_acmc_test_5k.jsonl")
OUT_DIR = Path("results/bootstrap_eval")
PAPER_TABLES_DIR = Path("tables")

FAILURE_SCORE = 0.0

DETECTOR_DISPLAY = {
    "rl_detector": "TELL (ours)",
    "openai_roberta": "OpenAI RoBERTa",
    "radar": "RADAR",
    "mage_d": "MAGE",
    "chatgpt_d": "ChatGPT-D",
    "argugpt": "ArguGPT",
    "dnagpt": "DNA-GPT",
    "fast_detectgpt": "Fast-DetectGPT",
    "detectllm_lrr": "DetectLLM-LRR",
    "detectllm_npr": "DetectLLM-NPR",
    "pangram_editlens_llama": "Pangram EditLens",
    "t5_sentinel": "T5Sentinel",
    "aigc_mpu_env3": "AIGC MPU",
    "logrank_gpt2_medium": "LogRank GPT-2-medium",
    "binoculars": "Binoculars",
    "phd_roberta": "PHD RoBERTa",
}

SHORT_DISPLAY = {
    "rl_detector": r"\ourmodel~(ours)",
    "mage_d": "MAGE",
    "pangram_editlens_llama": "Pangram-EditLens",
    "fast_detectgpt": "Fast-DetectGPT",
    "argugpt": "ArguGPT",
    "t5_sentinel": "T5Sentinel",
    "detectllm_npr": "DetectLLM-NPR",
    "openai_roberta": "OpenAI RoBERTa",
    "aigc_mpu_env3": "AIGC MPU",
    "detectllm_lrr": "DetectLLM-LRR",
    "logrank_gpt2_medium": "LogRank",
    "radar": "RADAR",
    "chatgpt_d": "ChatGPT-D",
    "binoculars": "Binoculars",
    "dnagpt": "DNA-GPT",
    "phd_roberta": "PHD RoBERTa",
}


def fmt_short_auroc_cell(auroc: float, lo: float, hi: float, bold: bool) -> str:
    body = rf"\num{{{auroc:.3f}}} [\num{{{lo:.3f}}}, \num{{{hi:.3f}}}]"
    if bold:
        return rf"\textbf{{{body}}}"
    return body


def fmt_short_tpr_cell(tpr: float, bold: bool) -> str:
    body = rf"\num{{{tpr * 100:.1f}}}"
    if bold:
        return rf"\textbf{{{body}}}"
    return body


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def score_for_analysis(raw_score) -> float:
    if raw_score is None:
        return FAILURE_SCORE
    return float(raw_score)


def load_baselines(baseline_dir: Path) -> tuple[dict[str, dict[str, float]], dict[str, int]]:
    """Returns ({detector: {doc_id: analysis_score}}, {detector: n_failures})."""
    detectors: dict[str, dict[str, float]] = {}
    n_failures: dict[str, int] = {}
    for path in sorted(baseline_dir.glob("*.predictions.jsonl")):
        name = path.name.split(".predictions.jsonl")[0]
        scores: dict[str, float] = {}
        fails = 0
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                raw = row.get("score_ai")
                if raw is None or row.get("error"):
                    fails += 1
                scores[row["id"]] = score_for_analysis(raw_score=raw)
        detectors[name] = scores
        n_failures[name] = fails
    return detectors, n_failures


def load_acmc_hf_five_baselines(path: Path) -> tuple[dict[str, dict[str, float]], dict[str, int]]:
    """Colleague five-detector addendum on acmc/multi_domain_ai_human_text test split."""
    text = path.read_text()
    if not text.lstrip().startswith("{"):
        text = "{" + text
    payload = json.loads(text)
    rows = payload["splits"]["test"]["rows"]
    detectors = {name: {} for name in payload["detectors"]}
    n_failures = {name: 0 for name in payload["detectors"]}
    for row in rows:
        doc_id = row["id"]
        for name, det in row["scores"].items():
            raw = det.get("score_ai")
            if raw is None or det.get("error"):
                n_failures[name] += 1
            detectors[name][doc_id] = score_for_analysis(raw_score=raw)
    return detectors, n_failures


def load_our_system(path: Path) -> tuple[dict[str, float], dict[str, int], int]:
    """Returns ({doc_id: score_ai_normalized}, {doc_id: label}, n_failures)."""
    with open(path) as f:
        data = json.load(f)
    scores: dict[str, float] = {}
    labels: dict[str, int] = {}
    n_failures = 0
    for doc in data["docs"]:
        doc_id = doc["doc_id"]
        labels[doc_id] = doc["label"]
        raw = doc["agg_score"]
        if raw is None:
            n_failures += 1
            scores[doc_id] = FAILURE_SCORE
        else:
            scores[doc_id] = (raw + 1.0) / 2.0
    return scores, labels, n_failures


def build_matrix(
    baselines: dict[str, dict[str, float]],
    our_scores: dict[str, float],
    our_labels: dict[str, int],
) -> tuple[list[str], np.ndarray, np.ndarray, list[str]]:
    doc_ids = sorted(our_labels.keys())
    M = len(doc_ids)
    label_vec = np.array([our_labels[doc_id] for doc_id in doc_ids], dtype=np.int32)
    detector_names = ["rl_detector"] + sorted(baselines.keys())
    N = len(detector_names)
    score_matrix = np.zeros((N, M), dtype=np.float64)
    for j, doc_id in enumerate(doc_ids):
        score_matrix[0, j] = our_scores[doc_id]
    for i, name in enumerate(detector_names[1:], start=1):
        det_scores = baselines[name]
        for j, doc_id in enumerate(doc_ids):
            score_matrix[i, j] = det_scores.get(doc_id, FAILURE_SCORE)
    return detector_names, score_matrix, label_vec, doc_ids


# ---------------------------------------------------------------------------
# DeLong variance (for pairwise z-test)
# ---------------------------------------------------------------------------

def delong_variance(scores: np.ndarray, labels: np.ndarray) -> float:
    """Returns DeLong variance for a single detector's AUROC estimate."""
    pos_mask = labels == 1
    neg_mask = labels == 0
    n_pos = pos_mask.sum()
    n_neg = neg_mask.sum()
    scores_pos = scores[pos_mask]
    scores_neg = scores[neg_mask]

    # Placement values (kernel)
    # V10[i] = P(score_pos[i] > score_neg) — placement value for positive
    # V01[j] = P(score_pos > score_neg[j]) — placement value for negative
    V10 = np.mean(
        (scores_pos[:, None] > scores_neg[None, :]).astype(float)
        + 0.5 * (scores_pos[:, None] == scores_neg[None, :]).astype(float),
        axis=1,
    )
    V01 = np.mean(
        (scores_pos[None, :] > scores_neg[:, None]).astype(float)
        + 0.5 * (scores_pos[None, :] == scores_neg[:, None]).astype(float),
        axis=1,
    )

    auc = V10.mean()
    s10 = np.var(V10, ddof=1)
    s01 = np.var(V01, ddof=1)
    var = s10 / n_pos + s01 / n_neg
    return var, auc


def delong_pvalue(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    labels: np.ndarray,
) -> float:
    """
    Two-sided p-value for H0: AUROC_a == AUROC_b using DeLong's correlated test.
    """
    pos_mask = labels == 1
    neg_mask = labels == 0
    n_pos = pos_mask.sum()
    n_neg = neg_mask.sum()
    sp = scores_a[pos_mask]
    sn = scores_a[neg_mask]
    tp = scores_b[pos_mask]
    tn = scores_b[neg_mask]

    def placements(pos, neg):
        V10 = np.mean(
            (pos[:, None] > neg[None, :]).astype(float)
            + 0.5 * (pos[:, None] == neg[None, :]).astype(float),
            axis=1,
        )
        V01 = np.mean(
            (pos[None, :] > neg[:, None]).astype(float)
            + 0.5 * (pos[None, :] == neg[:, None]).astype(float),
            axis=1,
        )
        return V10, V01

    V10_a, V01_a = placements(sp, sn)
    V10_b, V01_b = placements(tp, tn)

    auc_a = V10_a.mean()
    auc_b = V10_b.mean()

    s10_aa = np.var(V10_a, ddof=1)
    s01_aa = np.var(V01_a, ddof=1)
    s10_bb = np.var(V10_b, ddof=1)
    s01_bb = np.var(V01_b, ddof=1)
    s10_ab = np.cov(V10_a, V10_b, ddof=1)[0, 1]
    s01_ab = np.cov(V01_a, V01_b, ddof=1)[0, 1]

    var_diff = (s10_aa + s01_aa) / n_pos / n_neg  # wrong, redo properly
    # Correct formula:
    var_diff = (
        s10_aa / n_pos + s01_aa / n_neg
        + s10_bb / n_pos + s01_bb / n_neg
        - 2 * s10_ab / n_pos - 2 * s01_ab / n_neg
    )

    if var_diff <= 0:
        return 1.0

    z = (auc_a - auc_b) / np.sqrt(var_diff)
    from scipy import stats
    p = 2 * stats.norm.sf(abs(z))
    return float(p)


# ---------------------------------------------------------------------------
# TPR @ fixed FPR
# ---------------------------------------------------------------------------

def tpr_at_fpr_001(labels: np.ndarray, scores: np.ndarray) -> float:
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float("nan")
    fpr, tpr, _ = roc_curve(labels, scores)
    vals = tpr[fpr <= 0.01]
    return float(np.max(vals)) if len(vals) else 0.0


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def bootstrap_aurocs(
    score_matrix: np.ndarray,
    label_vec: np.ndarray,
    B: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Returns boot_aurocs of shape (B, N_detectors).
    Vectorized: resample indices, then compute AUROC for each detector on each resample.
    """
    N, M = score_matrix.shape
    boot_aurocs = np.zeros((B, N), dtype=np.float64)

    for b in range(B):
        idx = rng.integers(0, M, size=M)
        labels_b = label_vec[idx]
        # Skip degenerate resamples (all same class)
        if labels_b.sum() == 0 or labels_b.sum() == M:
            boot_aurocs[b] = np.nan
            continue
        for i in range(N):
            boot_aurocs[b, i] = roc_auc_score(labels_b, score_matrix[i, idx])

    return boot_aurocs


def bootstrap_tprs(
    score_matrix: np.ndarray,
    label_vec: np.ndarray,
    B: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Returns boot_tprs of shape (B, N_detectors)."""
    N, M = score_matrix.shape
    boot_tprs = np.zeros((B, N), dtype=np.float64)

    for b in range(B):
        idx = rng.integers(0, M, size=M)
        labels_b = label_vec[idx]
        if labels_b.sum() == 0 or labels_b.sum() == M:
            boot_tprs[b] = np.nan
            continue
        for i in range(N):
            boot_tprs[b, i] = tpr_at_fpr_001(labels_b, score_matrix[i, idx])

    return boot_tprs


def kendall_tau_from_rankings(rank_a: np.ndarray, rank_b: np.ndarray) -> float:
    """Kendall τ between two rank arrays (concordant − discordant) / total pairs."""
    N = len(rank_a)
    concordant = 0
    discordant = 0
    for i in range(N):
        for j in range(i + 1, N):
            diff_a = rank_a[i] - rank_a[j]
            diff_b = rank_b[i] - rank_b[j]
            if diff_a * diff_b > 0:
                concordant += 1
            elif diff_a * diff_b < 0:
                discordant += 1
    total = N * (N - 1) // 2
    return (concordant - discordant) / total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    baselines, n_failures = load_baselines(baseline_dir=BASELINE_DIR)
    acmc_names = {
        "t5_sentinel", "aigc_mpu_env3", "logrank_gpt2_medium", "binoculars", "phd_roberta"
    }
    missing_acmc = sorted(acmc_names - set(baselines.keys()))
    if missing_acmc and ACMC_HF_FIVE_JSON.exists():
        acmc_five, acmc_fails = load_acmc_hf_five_baselines(path=ACMC_HF_FIVE_JSON)
        baselines.update({k: acmc_five[k] for k in missing_acmc})
        n_failures.update({k: acmc_fails[k] for k in missing_acmc})
        print(f"  Added {len(missing_acmc)} detectors from {ACMC_HF_FIVE_JSON}")
    elif missing_acmc:
        raise RuntimeError(
            f"Missing colleague detectors {missing_acmc}: add merged_predictions "
            f"or place {ACMC_HF_FIVE_JSON}"
        )
    our_scores, our_labels, tell_failures = load_our_system(path=OUR_RESULTS)
    n_failures["rl_detector"] = tell_failures

    detector_names, score_matrix, label_vec, doc_ids = build_matrix(
        baselines=baselines,
        our_scores=our_scores,
        our_labels=our_labels,
    )
    N, M = score_matrix.shape
    print(f"  {N} detectors × {M} docs (failed scores imputed as {FAILURE_SCORE})")
    for name in detector_names:
        if n_failures.get(name, 0):
            print(f"    {name}: {n_failures[name]} row failures → score 0")
    print(f"  Label distribution: {label_vec.sum()} AI / {(1-label_vec).sum()} human")

    # ----- Point-estimate AUROCs -----
    print("Computing point-estimate AUROCs...")
    point_aurocs = np.array([
        roc_auc_score(label_vec, score_matrix[i]) for i in range(N)
    ])
    point_tprs = np.array([
        tpr_at_fpr_001(label_vec, score_matrix[i]) for i in range(N)
    ])

    # Original ranking (0 = best)
    orig_order = np.argsort(-point_aurocs)  # descending by AUROC
    orig_ranks = np.argsort(orig_order)  # rank of each detector (0-based)

    # ----- Bootstrap -----
    print(f"Running bootstrap (B={args.bootstrap:,})...")
    rng = np.random.default_rng(args.seed)
    boot_aurocs = bootstrap_aurocs(
        score_matrix=score_matrix,
        label_vec=label_vec,
        B=args.bootstrap,
        rng=rng,
    )
    # Drop degenerate resamples
    valid_mask = ~np.any(np.isnan(boot_aurocs), axis=1)
    boot_aurocs = boot_aurocs[valid_mask]
    B_valid = boot_aurocs.shape[0]
    print(f"  {B_valid:,} valid resamples (dropped {args.bootstrap - B_valid} degenerate)")

    ci_lo = np.percentile(boot_aurocs, 2.5, axis=0)
    ci_hi = np.percentile(boot_aurocs, 97.5, axis=0)

    print(f"Running bootstrap for TPR@1%FPR (B={args.bootstrap:,})...")
    boot_tprs = bootstrap_tprs(
        score_matrix=score_matrix,
        label_vec=label_vec,
        B=args.bootstrap,
        rng=rng,
    )
    valid_tpr_mask = ~np.any(np.isnan(boot_tprs), axis=1)
    boot_tprs = boot_tprs[valid_tpr_mask]
    B_tpr_valid = boot_tprs.shape[0]
    print(f"  {B_tpr_valid:,} valid resamples (dropped {args.bootstrap - B_tpr_valid} degenerate)")

    tpr_ci_lo = np.percentile(boot_tprs, 2.5, axis=0)
    tpr_ci_hi = np.percentile(boot_tprs, 97.5, axis=0)

    # Bootstrap rankings (0 = best per resample)
    boot_ranks = np.argsort(np.argsort(-boot_aurocs, axis=1), axis=1)  # (B, N)

    # Kendall τ per bootstrap sample vs original
    taus = []
    for b in range(B_valid):
        tau = kendall_tau_from_rankings(orig_ranks, boot_ranks[b])
        taus.append(tau)
    mean_tau = float(np.mean(taus))
    print(f"  Mean Kendall τ (bootstrap vs original): {mean_tau:.4f}")

    # P(rank_i < rank_j) for adjacent pairs in original ranking
    adjacent_probs = {}
    for pos in range(N - 1):
        i = orig_order[pos]      # detector ranked pos
        j = orig_order[pos + 1]  # detector ranked pos+1
        p_holds = float(np.mean(boot_ranks[:, i] < boot_ranks[:, j]))
        adjacent_probs[f"{detector_names[i]} > {detector_names[j]}"] = p_holds

    # ----- DeLong pairwise + BH -----
    print("Running DeLong pairwise tests...")
    pairs = list(combinations(range(N), 2))
    raw_pvals = []
    for i, j in pairs:
        p = delong_pvalue(score_matrix[i], score_matrix[j], label_vec)
        raw_pvals.append(p)

    reject, pvals_corrected, _, _ = multipletests(raw_pvals, alpha=0.05, method="fdr_bh")

    pairwise = {}
    for k, (i, j) in enumerate(pairs):
        key = f"{detector_names[i]} vs {detector_names[j]}"
        pairwise[key] = {
            "p_raw": float(raw_pvals[k]),
            "p_bh": float(pvals_corrected[k]),
            "significant": bool(reject[k]),
            "delta_auroc": float(point_aurocs[i] - point_aurocs[j]),
        }

    # ----- Per-domain AUROC -----
    print("Computing per-domain AUROCs...")
    # Load domain labels for each doc_id from baseline file
    id_to_domain = {}
    with open(BASELINE_DIR / "openai_roberta.predictions.jsonl") as f:
        for line in f:
            row = json.loads(line)
            id_to_domain[row["id"]] = row["domain"]

    domain_of_doc = np.array([id_to_domain[doc_id] for doc_id in doc_ids])
    domains = sorted(set(domain_of_doc))

    domain_aurocs: dict[str, dict[str, float]] = {}
    for dom in domains:
        mask = domain_of_doc == dom
        n_dom = mask.sum()
        labels_dom = label_vec[mask]
        if labels_dom.sum() == 0 or labels_dom.sum() == n_dom:
            continue
        domain_aurocs[dom] = {"n": int(n_dom)}
        for i, name in enumerate(detector_names):
            domain_aurocs[dom][name] = float(
                roc_auc_score(label_vec[mask], score_matrix[i, mask])
            )

    # ----- Save JSON summary -----
    summary = {
        "n_docs": M,
        "failure_score_imputation": FAILURE_SCORE,
        "n_failures": {name: n_failures.get(name, 0) for name in detector_names},
        "n_detectors": N,
        "bootstrap_B": B_valid,
        "seed": args.seed,
        "mean_kendall_tau": mean_tau,
        "detectors": {},
        "adjacent_rank_stability": adjacent_probs,
        "pairwise": pairwise,
        "domain_aurocs": domain_aurocs,
    }
    for i, name in enumerate(detector_names):
        summary["detectors"][name] = {
            "auroc": float(point_aurocs[i]),
            "ci_lo": float(ci_lo[i]),
            "ci_hi": float(ci_hi[i]),
            "tpr_at_fpr_0.01": float(point_tprs[i]),
            "tpr_ci_lo": float(tpr_ci_lo[i]),
            "tpr_ci_hi": float(tpr_ci_hi[i]),
            "rank": int(orig_ranks[i]) + 1,  # 1-indexed
            "n_failures": n_failures.get(name, 0),
            "display_name": DETECTOR_DISPLAY.get(name, name),
        }

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved {OUT_DIR}/summary.json")

    n_pairwise = len(pairs)

    # ----- Ranking table (markdown) -----
    lines = [
        f"# Benchmark Ranking — HF5k Test Set (n={M:,})",
        "",
        f"Failed detector rows are scored as **{FAILURE_SCORE}** for AUROC/TPR. "
        f"Bootstrap B={B_valid:,}, seed={args.seed}. "
        f"Mean Kendall τ (ranking stability) = **{mean_tau:.4f}**.",
        "",
        "| Rank | Detector | AUROC | 95% CI | TPR@1%FPR | P(rank holds vs. next) |",
        "|---:|---|---:|---|---:|---:|",
    ]
    for pos in range(N):
        i = orig_order[pos]
        name = detector_names[i]
        display = DETECTOR_DISPLAY.get(name, name)
        auroc = point_aurocs[i]
        lo = ci_lo[i]
        hi = ci_hi[i]
        tpr = point_tprs[i]
        if pos < N - 1:
            j = orig_order[pos + 1]
            adj_key = f"{name} > {detector_names[j]}"
            p_holds = adjacent_probs.get(adj_key, float("nan"))
            p_str = f"{p_holds:.3f}"
        else:
            p_str = "—"
        lines.append(
            f"| {pos+1} | {display} | {auroc:.4f} | [{lo:.4f}, {hi:.4f}] | "
            f"{tpr:.4f} | {p_str} |"
        )

    lines += [
        "",
        "**P(rank holds vs. next)**: fraction of bootstrap resamples where this "
        "detector ranked strictly above the next-lower detector.",
        "",
        f"**Failures**: row-level detector errors or missing scores are imputed as "
        f"{FAILURE_SCORE} before AUROC/TPR (n={M:,} for every detector).",
    ]

    with open(OUT_DIR / "ranking_table.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Saved {OUT_DIR}/ranking_table.md")

    # ----- Domain table (markdown) -----
    det_order = [detector_names[orig_order[pos]] for pos in range(N)]
    det_display = [DETECTOR_DISPLAY.get(n, n) for n in det_order]

    dom_lines = [
        "# Per-Domain AUROC",
        "",
        "| Domain | n | " + " | ".join(det_display) + " |",
        "|---|---:|" + "|".join("---:" for _ in det_order) + "|",
    ]
    for dom in domains:
        if dom not in domain_aurocs:
            continue
        n_dom = domain_aurocs[dom]["n"]
        scores_str = []
        for name in det_order:
            val = domain_aurocs[dom].get(name, float("nan"))
            scores_str.append(f"{val:.3f}")
        dom_lines.append(f"| {dom} | {n_dom} | " + " | ".join(scores_str) + " |")

    with open(OUT_DIR / "domain_table.md", "w") as f:
        f.write("\n".join(dom_lines) + "\n")
    print(f"  Saved {OUT_DIR}/domain_table.md")

    # ----- Pairwise significance table (markdown) -----
    pw_lines = [
        "# Pairwise Significance (BH FDR q=0.05, DeLong test)",
        "",
        "Detectors ordered by rank (best first). "
        "**bold** = BH-significant. Values are Δ AUROC (row − col).",
        "",
        "| | " + " | ".join(det_display) + " |",
        "|---|" + "|".join("---" for _ in det_order) + "|",
    ]
    for pos_i, ni in enumerate(det_order):
        row_cells = [det_display[pos_i]]
        for pos_j, nj in enumerate(det_order):
            if pos_i == pos_j:
                row_cells.append("—")
                continue
            key = f"{ni} vs {nj}" if ni < nj else f"{nj} vs {ni}"
            if key not in pairwise:
                key = f"{nj} vs {ni}" if nj < ni else f"{ni} vs {nj}"
            # find it regardless of order
            found = pairwise.get(f"{ni} vs {nj}") or pairwise.get(f"{nj} vs {ni}")
            if found is None:
                row_cells.append("?")
                continue
            delta = point_aurocs[detector_names.index(ni)] - point_aurocs[detector_names.index(nj)]
            sig = found["significant"]
            cell = f"{delta:+.3f}"
            if sig:
                cell = f"**{cell}**"
            row_cells.append(cell)
        pw_lines.append("| " + " | ".join(row_cells) + " |")

    with open(OUT_DIR / "pairwise_table.md", "w") as f:
        f.write("\n".join(pw_lines) + "\n")
    print(f"  Saved {OUT_DIR}/pairwise_table.md")

    # ----- LaTeX: ranking table -----
    def tex_escape(s: str) -> str:
        return s.replace("&", r"\&").replace("_", r"\_").replace("%", r"\%")

    # S column: table-format=1.4, detect-weight picks up \bfseries for TELL row,
    # mode=text keeps numbers in text mode so font weight is inherited correctly.
    tex_rank = [
        r"% Requires: \usepackage{booktabs,siunitx}",
        r"% siunitx setup: \sisetup{mode=text,detect-weight}",
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{Detector ranking on the TELL benchmark test set ($n=\num{{{M}}}$). "
        r"Failed rows imputed to score $0$. "
        r"AUROC 95\% CIs from bootstrap resampling ($B=\num{10000}$). "
        rf"Mean Kendall $\tau=\num{{{mean_tau:.4f}}}$. "
        r"$\dagger$~gap not significant vs.\ adjacent rank below "
        r"(DeLong, BH FDR $q=0.05$).}",
        r"\label{tab:ranking}",
        r"\begin{tabular}{r l S[table-format=1.4,detect-weight,mode=text] c "
        r"S[table-format=1.4,detect-weight,mode=text] r}",
        r"\toprule",
        r"{Rank} & {Detector} & {AUROC} & {95\% CI} & {TPR@1\%FPR} & "
        r"{$P(\text{rank holds})$} \\",
        r"\midrule",
    ]
    for pos in range(N):
        i = orig_order[pos]
        name = detector_names[i]
        display = tex_escape(DETECTOR_DISPLAY.get(name, name))
        auroc = point_aurocs[i]
        lo, hi = ci_lo[i], ci_hi[i]
        tpr = point_tprs[i]
        is_tell = name == "rl_detector"
        if pos < N - 1:
            j = orig_order[pos + 1]
            adj_key = f"{name} > {detector_names[j]}"
            p_holds = adjacent_probs.get(adj_key, float("nan"))
            pair_info = pairwise.get(f"{name} vs {detector_names[j]}") or \
                        pairwise.get(f"{detector_names[j]} vs {name}")
            dagger = r"$\dagger$" if pair_info and not pair_info["significant"] else ""
            p_cell = rf"\num{{{p_holds:.3f}}}{dagger}"
        else:
            p_cell = r"\multicolumn{1}{c}{---}"
        # S column: bare number; \bfseries triggers detect-weight bold for TELL
        auroc_cell = rf"\bfseries {auroc:.4f}" if is_tell else f"{auroc:.4f}"
        tpr_cell = rf"\bfseries {tpr:.4f}" if is_tell else f"{tpr:.4f}"
        name_cell = rf"\bfseries {display}" if is_tell else display
        ci_cell = rf"[\num{{{lo:.4f}}}, \num{{{hi:.4f}}}]"
        tex_rank.append(
            f"  {pos+1} & {name_cell} & {auroc_cell} & {ci_cell} & {tpr_cell} & "
            f"{p_cell} \\\\"
        )
    tex_rank += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    with open(OUT_DIR / "ranking_table.tex", "w") as f:
        f.write("\n".join(tex_rank) + "\n")
    print(f"  Saved {OUT_DIR}/ranking_table.tex")

    # ----- LaTeX: short ranking table (appendix / main compact) -----
    tex_short = [
        r"\begin{table}[]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4.2pt}",
        r"\begin{tabular}{@{}l l r@{}}",
        r"\toprule",
        r"Detector & AUROC (95\% CI) & TPR@1\%FPR \\",
        r"\midrule",
    ]
    for pos in range(N):
        i = orig_order[pos]
        name = detector_names[i]
        is_tell = name == "rl_detector"
        display = SHORT_DISPLAY.get(name, tex_escape(DETECTOR_DISPLAY.get(name, name)))
        auroc_cell = fmt_short_auroc_cell(
            auroc=point_aurocs[i],
            lo=ci_lo[i],
            hi=ci_hi[i],
            bold=is_tell,
        )
        tpr_cell = fmt_short_tpr_cell(tpr=point_tprs[i], bold=is_tell)
        tex_short.append(f"{display} & {auroc_cell} & {tpr_cell} \\\\")
    tex_short += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Comparison of detection methods. \ourmodel~achieves the best "
        r"scores (all metrics: higher is better), with MAGE "
        r"\cite{liMAGEMachinegeneratedText2023} closely behind on AUROC.}",
        r"\label{tab:main_detector_benchmark}",
        r"\end{table}",
    ]
    with open(OUT_DIR / "ranking_table_short.tex", "w") as f:
        f.write("\n".join(tex_short) + "\n")
    print(f"  Saved {OUT_DIR}/ranking_table_short.tex")

    # ----- LaTeX: domain table -----
    DET_ABBREV = {
        "rl_detector": r"\textbf{TELL}",
        "mage_d": "MAGE",
        "pangram_editlens_llama": "Pangram",
        "fast_detectgpt": "F-DGT",
        "argugpt": "ArguGPT",
        "detectllm_npr": "DL-NPR",
        "openai_roberta": "OAI-RB",
        "detectllm_lrr": "DL-LRR",
        "radar": "RADAR",
        "chatgpt_d": "CGPT-D",
        "dnagpt": "DNA-GPT",
        "t5_sentinel": "T5Sent.",
        "aigc_mpu_env3": "AIGC MPU",
        "logrank_gpt2_medium": "LogRank",
        "binoculars": "Binoc.",
        "phd_roberta": "PHD",
    }
    abbrevs = [DET_ABBREV.get(n, tex_escape(n)) for n in det_order]

    # S[detect-weight,mode=text]: \bfseries value bolds the best-per-domain cell.
    # Headers in S columns must be wrapped in {braces}.
    s_col = "S[table-format=1.3,detect-weight,mode=text]"
    col_spec_dom = "l " + " ".join([s_col] * N)

    tex_dom = [
        r"% Requires: \usepackage{booktabs,siunitx}",
        r"% siunitx setup: \sisetup{mode=text,detect-weight}",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Per-domain AUROC on the TELL benchmark test set. "
        r"Best result per domain in \textbf{bold}. "
        r"Detectors ordered by overall rank (left = best).}",
        r"\label{tab:domain}",
        r"\resizebox{\linewidth}{!}{%",
        rf"\begin{{tabular}}{{{col_spec_dom}}}",
        r"\toprule",
        r"{Domain} & " + " & ".join(f"{{{a}}}" for a in abbrevs) + r" \\",
        r"\midrule",
    ]
    for dom in domains:
        if dom not in domain_aurocs:
            continue
        dom_vals = [domain_aurocs[dom].get(n, float("nan")) for n in det_order]
        best_val = max(v for v in dom_vals if not np.isnan(v))
        cells = []
        for v in dom_vals:
            # \bfseries prefix triggers detect-weight inside S column
            cell = rf"\bfseries {v:.3f}" if abs(v - best_val) < 1e-9 else f"{v:.3f}"
            cells.append(cell)
        dom_display = tex_escape(dom)
        tex_dom.append(f"  {{{dom_display}}} & " + " & ".join(cells) + r" \\")
    tex_dom += [
        r"\bottomrule",
        r"\end{tabular}}",
        r"\end{table}",
    ]
    with open(OUT_DIR / "domain_table.tex", "w") as f:
        f.write("\n".join(tex_dom) + "\n")
    print(f"  Saved {OUT_DIR}/domain_table.tex")

    # ----- LaTeX: pairwise significance table -----
    # S[table-format=+1.3]: aligns sign + 1 integer digit + 3 decimal digits.
    # Diagonal {---} braces tell siunitx the cell is not a number.
    # \bfseries prefix bolds significant pairs via detect-weight.
    s_col_pw = "S[table-format=+1.3,detect-weight,mode=text]"
    col_spec_pw = "l " + " ".join([s_col_pw] * N)

    tex_pw = [
        r"% Requires: \usepackage{booktabs,siunitx}",
        r"% siunitx setup: \sisetup{mode=text,detect-weight}",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Pairwise $\Delta$AUROC (row $-$ col). "
        r"\textbf{Bold} = BH-significant (DeLong test, FDR $q=0.05$, "
        rf"\num{{{n_pairwise}}} comparisons). "
        r"Detectors ordered by rank (left/top = best).}",
        r"\label{tab:pairwise}",
        r"\resizebox{\linewidth}{!}{%",
        rf"\begin{{tabular}}{{{col_spec_pw}}}",
        r"\toprule",
        r"{} & " + " & ".join(f"{{{a}}}" for a in abbrevs) + r" \\",
        r"\midrule",
    ]
    for pos_i, ni in enumerate(det_order):
        row_cells = [f"{{{abbrevs[pos_i]}}}"]  # text entry — needs braces in S col
        for pos_j, nj in enumerate(det_order):
            if pos_i == pos_j:
                row_cells.append("{---}")
                continue
            found = pairwise.get(f"{ni} vs {nj}") or pairwise.get(f"{nj} vs {ni}")
            if found is None:
                row_cells.append("{?}")
                continue
            delta = point_aurocs[detector_names.index(ni)] - point_aurocs[detector_names.index(nj)]
            # bare signed number — siunitx S column formats sign alignment
            cell = rf"\bfseries {delta:+.3f}" if found["significant"] else f"{delta:+.3f}"
            row_cells.append(cell)
        tex_pw.append("  " + " & ".join(row_cells) + r" \\")
    tex_pw += [
        r"\bottomrule",
        r"\end{tabular}}",
        r"\end{table}",
    ]
    with open(OUT_DIR / "pairwise_table.tex", "w") as f:
        f.write("\n".join(tex_pw) + "\n")
    print(f"  Saved {OUT_DIR}/pairwise_table.tex")

    # ----- Console summary -----
    print()
    print("=" * 70)
    print("RANKING SUMMARY")
    print("=" * 70)
    print(f"{'Rank':<5} {'Detector':<28} {'AUROC':<8} {'95% CI':<22} "
          f"{'TPR@1%':<8} {'P(holds)'}")
    print("-" * 80)
    for pos in range(N):
        i = orig_order[pos]
        name = detector_names[i]
        display = DETECTOR_DISPLAY.get(name, name)
        auroc = point_aurocs[i]
        lo, hi = ci_lo[i], ci_hi[i]
        tpr = point_tprs[i]
        if pos < N - 1:
            j = orig_order[pos + 1]
            adj_key = f"{name} > {detector_names[j]}"
            p_holds = adjacent_probs.get(adj_key, float("nan"))
            p_str = f"{p_holds:.3f}"
        else:
            p_str = "—"
        print(
            f"{pos+1:<5} {display:<28} {auroc:.4f}   [{lo:.4f}, {hi:.4f}]   "
            f"{tpr:.4f}   {p_str}"
        )

    print()
    print(f"Mean Kendall τ: {mean_tau:.4f}")
    print()
    n_sig = sum(1 for v in pairwise.values() if v["significant"])
    print(f"BH-significant pairs: {n_sig} / {len(pairs)}")
    print()
    print("Adjacent pair stability:")
    for k, v in adjacent_probs.items():
        a, b = k.split(" > ")
        print(f"  P({DETECTOR_DISPLAY.get(a,a)} > {DETECTOR_DISPLAY.get(b,b)}) = {v:.3f}")

    PAPER_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("ranking_table.tex", "pairwise_table.tex", "domain_table.tex"):
        src = OUT_DIR / name
        dst = PAPER_TABLES_DIR / name
        dst.write_text(src.read_text())
        print(f"  Copied {dst}")


if __name__ == "__main__":
    main()
