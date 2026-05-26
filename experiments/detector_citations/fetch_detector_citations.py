#!/usr/bin/env python3
"""
Fetch AUROC and TPR@1%FPR scores from papers citing 8 AI text detectors.
Uses OpenAlex API (free, no key needed, 10 req/s).

Strategy:
  1. Find each detector's OpenAlex work ID via title search.
  2. Paginate cites:{oa_id} to collect all citing papers with abstracts.
  3. Regex-extract AUROC / TPR@1%FPR from each abstract.
"""

import sys
import time
import re
import json
import requests

sys.stdout.reconfigure(line_buffering=True)

OA_BASE = "https://api.openalex.org"
REQUEST_DELAY = 1.0  # OA politely asks for ≤10 req/s; 1s is safe

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "ArxivCitationExtractor/1.0 (acreo@ucsd.edu)",
})


# ── OpenAlex helpers ──────────────────────────────────────────────────────────

def _get(url: str, params: dict = None, retries: int = 5) -> dict | None:
    for attempt in range(retries):
        try:
            resp = SESSION.get(url, params=params, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 429:
                wait = 60 * (attempt + 1)
                print(f"    OA rate-limited; waiting {wait}s …")
                time.sleep(wait)
                continue
            print(f"    OA HTTP {resp.status_code}: {resp.text[:120]}")
            time.sleep(10)
        except Exception as exc:
            wait = 20 * (attempt + 1)
            print(f"    Request error: {exc}; retrying in {wait}s …")
            time.sleep(wait)
    return None


def find_paper_by_title(title_query: str) -> dict | None:
    """Return first OpenAlex work matching the title query."""
    data = _get(f"{OA_BASE}/works", {
        "search": title_query,
        "per_page": 1,
        "select": "id,title,ids,publication_year,authorships,abstract_inverted_index,cited_by_count",
    })
    time.sleep(REQUEST_DELAY)
    if not data:
        return None
    results = data.get("results", [])
    return results[0] if results else None


def get_citing_works(oa_id: str, per_page: int = 200) -> list[dict]:
    """Paginate all works citing oa_id (short ID like W4402667057)."""
    results = []
    cursor = "*"
    fields = "id,title,ids,publication_year,authorships,abstract_inverted_index,primary_location"
    while True:
        data = _get(f"{OA_BASE}/works", {
            "filter": f"cites:{oa_id}",
            "per_page": per_page,
            "cursor": cursor,
            "select": fields,
        })
        time.sleep(REQUEST_DELAY)
        if not data:
            break
        batch = data.get("results", [])
        results.extend(batch)
        total = data.get("meta", {}).get("count", len(results))
        print(f"    … fetched {len(results)} / {total}")
        next_cursor = data.get("meta", {}).get("next_cursor")
        if not next_cursor or len(batch) < per_page:
            break
        cursor = next_cursor
    return results


def reconstruct_abstract(inv_index: dict | None) -> str | None:
    """Convert OpenAlex inverted-index abstract to plain text."""
    if not inv_index:
        return None
    tokens: dict[int, str] = {}
    for word, positions in inv_index.items():
        for pos in positions:
            tokens[pos] = word
    return " ".join(tokens[i] for i in sorted(tokens))


def arxiv_id_from_work(work: dict) -> str | None:
    ids = work.get("ids") or {}
    arxiv = ids.get("arxiv")  # e.g. "https://arxiv.org/abs/2305.13242"
    if arxiv:
        m = re.search(r"abs/([^\s/v]+)", arxiv)
        if m:
            return m.group(1)
    # Fall back to checking primary_location URL
    loc = (work.get("primary_location") or {}).get("landing_page_url") or ""
    m = re.search(r"arxiv\.org/abs/([^\s/v]+)", loc)
    return m.group(1) if m else None


def authors_from_work(work: dict) -> list[str]:
    return [
        a.get("author", {}).get("display_name", "")
        for a in (work.get("authorships") or [])
    ]


# ── Score extraction ──────────────────────────────────────────────────────────

AUROC_PATTERNS = [
    r'AUROC\s*(?:score|value)?\s*[=:of]\s*(\d+\.?\d*)\s*%?',
    r'AUC-ROC\s*[=:]\s*(\d+\.?\d*)',
    r'AUC\s*[=:]\s*(\d+\.?\d*)\s*%',
    r'AUC\s+(?:score\s+)?of\s+(\d+\.?\d*)',
    r'area under the ROC curve\s*[=:]\s*(\d+\.?\d*)',
    r'(\d{1,3}\.?\d*)\s*%\s*AUROC',
    r'AUROC\s+of\s+(\d+\.?\d*)',
    r'AUROC\s+(\d+\.?\d*)',
]

TPR_PATTERNS = [
    r'TPR\s*@\s*1\s*%\s*FPR\s*[=:of]\s*(\d+\.?\d*)',
    r'TPR@1%FPR\s*[=:]\s*(\d+\.?\d*)',
    r'TPR\s+at\s+1\s*%\s+FPR\s*[=:]\s*(\d+\.?\d*)',
    r'true\s+positive\s+rate\s+at\s+1\s*%\s+false\s+positive\s+rate\s*[=:]\s*(\d+\.?\d*)',
    r'TPR@0\.01\s*[=:]\s*(\d+\.?\d*)',
]


def _first_match(patterns: list[str], text: str) -> float | None:
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            val = float(m.group(1))
            if val > 1:
                val /= 100.0
            return round(val, 4)
    return None


def extract_auroc(abstract: str | None) -> float | None:
    return _first_match(AUROC_PATTERNS, abstract) if abstract else None


def extract_tpr(abstract: str | None) -> float | None:
    return _first_match(TPR_PATTERNS, abstract) if abstract else None


def has_metric_keywords(abstract: str | None) -> bool:
    if not abstract:
        return False
    low = abstract.lower()
    return any(kw in low for kw in [
        "auroc", "auc-roc", "auc_roc", "area under the roc", " auc ",
        "tpr@1%", "tpr at 1%", "tpr@0.01", "true positive rate at 1%",
    ])


# ── Per-detector citation filtering ──────────────────────────────────────────

DETECTOR_FILTERS = {
    "OpenAI RoBERTa": lambda a: bool(a and (
        "roberta" in a.lower() or "openai detector" in a.lower()
    )),
    "ChatGPT": lambda a: bool(a and (
        "hc3" in a.lower() or "chatgpt-d" in a.lower() or "chatgpt detector" in a.lower()
    )),
}

# ── Detectors ─────────────────────────────────────────────────────────────────
# title_query should uniquely identify the paper in OpenAlex
DETECTORS = [
    {"name": "MAGE",             "arxiv_id": "2305.13242",
     "title_query": "MAGE machine-generated text detection wild"},
    {"name": "RADAR",            "arxiv_id": "2307.03838",
     "title_query": "RADAR robust AI-text detection adversarial learning"},
    {"name": "DetectLLM",        "arxiv_id": "2306.05540",
     "title_query": "DetectLLM log rank zero-shot detection machine-generated"},
    {"name": "Fast-DetectGPT",   "arxiv_id": "2310.05130",
     "title_query": "Fast-DetectGPT efficient zero-shot detection machine-generated"},
    {"name": "Pangram EditLens", "arxiv_id": "2510.03154",
     "title_query": "EditLens quantifying extent AI editing text"},
    {"name": "OpenAI RoBERTa",   "arxiv_id": "1908.09203",
     "title_query": "Release Strategies Social Impacts Language Models"},
    {"name": "ArguGPT",          "arxiv_id": "2304.07666",
     "title_query": "ArguGPT evaluating understanding identifying argumentative essays"},
    {"name": "ChatGPT",          "arxiv_id": "2301.07597",
     "title_query": "How close ChatGPT human experts comparison corpus HC3"},
]

# ── Main ──────────────────────────────────────────────────────────────────────

output = {
    "metadata": {
        "search_date": "2026-05-25",
        "method": "OpenAlex citation API (cites filter) + abstract reconstruction",
        "limitations": (
            "Abstracts extracted from OpenAlex inverted-index format. "
            "AUROC/TPR extracted via regex from abstracts only; full text not searched. "
            "Some papers may mention metrics without being direct evaluations."
        ),
    },
    "detectors": [],
}

report_lines = []

for det in DETECTORS:
    name      = det["name"]
    arxiv_id  = det["arxiv_id"]
    tq        = det["title_query"]

    print(f"\n{'='*60}")
    print(f"Processing: {name}  (arXiv:{arxiv_id})")

    # ── Find original paper on OpenAlex ─────────────────────────
    print(f"  Looking up paper …")
    orig_work = find_paper_by_title(tq)
    if not orig_work:
        print(f"  ERROR: not found on OpenAlex")
        report_lines.append(f"  {name}: NOT FOUND on OpenAlex.")
        continue

    oa_id     = orig_work["id"].split("/")[-1]
    orig_abs  = reconstruct_abstract(orig_work.get("abstract_inverted_index"))
    orig_auroc = extract_auroc(orig_abs)
    orig_tpr   = extract_tpr(orig_abs)

    print(f"  OA ID       : {oa_id}")
    print(f"  Title       : {orig_work['title'][:70]}")
    print(f"  Cited by    : {orig_work.get('cited_by_count', '?')}")
    print(f"  Orig AUROC  : {orig_auroc}  |  Orig TPR@1%FPR : {orig_tpr}")

    orig_paper = {
        "arxiv_id": arxiv_id,
        "title":    orig_work.get("title"),
        "authors":  authors_from_work(orig_work),
        "year":     orig_work.get("publication_year"),
        "url":      f"https://arxiv.org/abs/{arxiv_id}",
    }

    # ── Get citing papers ────────────────────────────────────────
    print(f"  Fetching citing works …")
    raw = get_citing_works(oa_id)
    print(f"  Total citing works: {len(raw)}")

    extra_filter = DETECTOR_FILTERS.get(name)
    citing_papers = []

    for w in raw:
        abstract = reconstruct_abstract(w.get("abstract_inverted_index"))
        if extra_filter and not extra_filter(abstract):
            continue
        if not has_metric_keywords(abstract):
            continue
        auroc = extract_auroc(abstract)
        tpr   = extract_tpr(abstract)
        if auroc is None and tpr is None:
            continue

        cit_arxiv_id = arxiv_id_from_work(w)
        url = (f"https://arxiv.org/abs/{cit_arxiv_id}" if cit_arxiv_id
               else (w.get("primary_location") or {}).get("landing_page_url"))

        doi = (w.get("ids") or {}).get("doi")

        citing_papers.append({
            "title":    w.get("title"),
            "authors":  authors_from_work(w),
            "year":     w.get("publication_year"),
            "arxiv_id": cit_arxiv_id,
            "url":      url,
            "doi":      doi,
            "scores":   {"AUROC": auroc, "TPR@1%FPR": tpr},
            "note":     None,
        })

    print(f"  Citing papers with extracted scores: {len(citing_papers)}")

    orig_note = None if (orig_auroc or orig_tpr) else "No AUROC or TPR@1%FPR in original abstract."
    if orig_auroc is None and orig_tpr is None:
        report_lines.append(
            f"  {name}: No metrics in original abstract. "
            f"{len(citing_papers)} citing paper(s) with extracted scores."
        )
    else:
        report_lines.append(
            f"  {name}: AUROC={orig_auroc}, TPR@1%FPR={orig_tpr} in original. "
            f"{len(citing_papers)} citing paper(s) with extracted scores."
        )
    if not citing_papers:
        report_lines.append(f"    ⚠  No citing papers with AUROC/TPR found for {name}.")

    output["detectors"].append({
        "name": name,
        "original_paper":  orig_paper,
        "original_scores": {"AUROC": orig_auroc, "TPR@1%FPR": orig_tpr, "note": orig_note},
        "citing_papers":   citing_papers,
    })

# ── Write output ──────────────────────────────────────────────────────────────

out_path = "detector_citation_scores.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"\n{'='*60}")
print(f"Output written to: {out_path}")
print("\n=== SHORT REPORT ===")
for line in report_lines:
    print(line)
