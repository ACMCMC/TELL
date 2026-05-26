"""Tell quality analysis: diversity, common phrases, clustering, intra-doc overlap.

Usage:
    python -m rl_detector.tell_analysis \\
        --checkpoint tinker://RUN_ID:train:0/weights/best-step-30 \\
        [--n-docs 250] \\
        [--output tells.json]

Metrics reported:
  - Lexical diversity (type-token ratio, vocab size, explanation lengths)
  - Most frequent n-grams in explanations (catches "typical of X" patterns)
  - TF-IDF clustering of all explanations (shows recurring tell archetypes)
  - Intra-document tell overlap (how redundant are the tells for one doc)

Separate breakdowns are shown for AI-type tells vs human-type tells.
"""

import argparse
import asyncio
import json
import logging
import random

import numpy as np
import tinker
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer

from rl_detector.config import CFG
from rl_detector.data import load_docs
from rl_detector.prompt_utils import format_prompt_for_model, load_tokenizer
from rl_detector.rewards import parse_indicators
from rl_detector.rollouts import extract_response_text

logger = logging.getLogger(__name__)

_SEED = 42


# ── Data collection ───────────────────────────────────────────────────────────

def _select_docs(docs: list[dict], n: int) -> list[dict]:
    rng = random.Random(_SEED)
    ai = [d for d in docs if d["label"] == 1]
    human = [d for d in docs if d["label"] == 0]
    rng.shuffle(ai)
    rng.shuffle(human)
    n_ai = min(n // 2, len(ai))
    n_human = min(n - n_ai, len(human))
    chosen = ai[:n_ai] + human[:n_human]
    rng.shuffle(chosen)
    return chosen[:n]


async def _rollout_one(sampling_client, tokenizer, doc: dict) -> dict | None:
    _, prompt = format_prompt_for_model(tokenizer=tokenizer, text=doc["text"])
    model_input = tinker.ModelInput.from_ints(tokenizer.encode(prompt))
    try:
        sampled = await sampling_client.sample_async(
            prompt=model_input,
            num_samples=1,
            sampling_params=tinker.SamplingParams(
                max_tokens=CFG.sampling.max_tokens,
                seed=_SEED,
                temperature=CFG.sampling.temperature,
                top_p=CFG.sampling.top_p,
                reasoning_effort=CFG.sampling.reasoning_effort,
            ),
        )
    except Exception as e:
        logger.warning("sample failed: %s", e)
        return None

    completion_text = tokenizer.decode(list(sampled.sequences[0].tokens))
    response_text = extract_response_text(completion_text)
    tells = parse_indicators(response_text)
    if not tells:
        return None
    return {
        "doc_id": str(doc.get("id", "")),
        "label": int(doc["label"]),
        "n_tells": len(tells),
        "tells": [
            {"span_text": t["span_text"], "explanation": t["explanation"], "type": t.get("type")}
            for t in tells
        ],
    }


async def collect_tells(checkpoint: str, n_docs: int) -> list[dict]:
    service_client = tinker.ServiceClient()
    loop = asyncio.get_event_loop()

    # Load training client from the weights checkpoint (same path format used during training),
    # then derive a sampling client — mirroring how train.py does eval.
    training_client, tokenizer, all_docs = await asyncio.gather(
        service_client.create_training_client_from_state_with_optimizer_async(path=checkpoint),
        loop.run_in_executor(None, load_tokenizer),
        loop.run_in_executor(None, lambda: load_docs(None, use_eval_split=True)),
    )
    sampling_client = await training_client.save_weights_and_get_sampling_client_async()

    docs = _select_docs(all_docs, n_docs)
    logger.info("Sampling %d docs (1 rollout each) …", len(docs))

    sem = asyncio.Semaphore(32)

    async def _bounded(doc):
        async with sem:
            return await _rollout_one(sampling_client, tokenizer, doc)

    results = await asyncio.gather(*[_bounded(d) for d in docs])
    records = [r for r in results if r is not None]
    logger.info("Got tells from %d / %d docs (%d failed/empty)", len(records), len(docs), len(docs) - len(records))
    return records


# ── Metrics ───────────────────────────────────────────────────────────────────

def _lexical_diversity(explanations: list[str]) -> dict:
    if not explanations:
        return {}
    tokens = []
    for e in explanations:
        tokens.extend(e.lower().split())
    if not tokens:
        return {}
    lengths = sorted(len(e.split()) for e in explanations)
    n = len(lengths)
    return {
        "n_explanations": len(explanations),
        "vocab_size": len(set(tokens)),
        "total_tokens": len(tokens),
        "ttr": round(len(set(tokens)) / len(tokens), 4),
        "mean_length_words": round(sum(lengths) / n, 1),
        "median_length_words": lengths[n // 2],
        "p95_length_words": lengths[min(n - 1, int(0.95 * n))],
    }


def _common_ngrams(explanations: list[str], top_n: int = 25) -> dict:
    results = {}
    for ngram_range, key in [
        ((1, 1), "top_unigrams"),
        ((2, 3), "top_bigrams_trigrams"),
        ((3, 5), "top_long_phrases"),
    ]:
        # No stop-word removal for multi-word n-grams so we catch "typical of", "common in", etc.
        stop = "english" if ngram_range == (1, 1) else None
        vec = CountVectorizer(ngram_range=ngram_range, max_features=2000, stop_words=stop)
        try:
            X = vec.fit_transform(explanations)
            counts = X.sum(axis=0).A1
            top_idx = counts.argsort()[::-1][:top_n]
            vocab = vec.get_feature_names_out()
            results[key] = [
                {
                    "phrase": vocab[i],
                    "count": int(counts[i]),
                    "pct": round(100 * counts[i] / len(explanations), 1),
                }
                for i in top_idx
            ]
        except ValueError:
            results[key] = []
    return results


def _tfidf_clusters(explanations: list[str], n_clusters: int = 10) -> dict:
    if len(explanations) < n_clusters * 2:
        return {"skipped": f"too few explanations ({len(explanations)})"}
    vec = TfidfVectorizer(max_features=2000, ngram_range=(1, 2), sublinear_tf=True)
    X = vec.fit_transform(explanations).toarray()
    km = KMeans(n_clusters=n_clusters, random_state=_SEED, n_init=10)
    labels = km.fit_predict(X)
    sil = float(silhouette_score(X, labels, metric="cosine")) if len(set(labels)) > 1 else 0.0

    feature_names = vec.get_feature_names_out()
    clusters = []
    for c in range(n_clusters):
        mask = labels == c
        size = int(mask.sum())
        center = km.cluster_centers_[c]
        top_terms = [feature_names[i] for i in center.argsort()[::-1][:8]]
        members_idx = np.where(mask)[0]
        dists = np.linalg.norm(X[mask] - center, axis=1)
        rep_idx = members_idx[dists.argmin()]
        clusters.append({
            "id": c,
            "size": size,
            "pct": round(100 * size / len(explanations), 1),
            "top_terms": top_terms,
            "representative": explanations[rep_idx],
        })
    clusters.sort(key=lambda c: c["size"], reverse=True)
    return {"n_clusters": n_clusters, "silhouette": round(sil, 4), "clusters": clusters}


def _intra_doc_overlap(records: list[dict]) -> dict:
    per_doc = []
    single_tell = 0
    for rec in records:
        exps = [t["explanation"] for t in rec["tells"]]
        if len(exps) < 2:
            single_tell += 1
            continue
        vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        try:
            X = vec.fit_transform(exps).toarray()
        except ValueError:
            continue
        sims = cosine_similarity(X)
        n = len(exps)
        vals = [sims[i][j] for i in range(n) for j in range(i + 1, n)]
        per_doc.append(sum(vals) / len(vals))

    if not per_doc:
        return {"error": "no multi-tell docs"}
    arr = sorted(per_doc)
    n = len(arr)
    return {
        "n_docs_with_multi_tells": n,
        "single_tell_docs": single_tell,
        "mean_intra_sim": round(float(np.mean(per_doc)), 4),
        "median_intra_sim": round(arr[n // 2], 4),
        "p75_intra_sim": round(arr[min(n - 1, int(0.75 * n))], 4),
        "p95_intra_sim": round(arr[min(n - 1, int(0.95 * n))], 4),
    }


def analyse(records: list[dict]) -> dict:
    all_exp = [t["explanation"] for r in records for t in r["tells"]]
    ai_exp = [t["explanation"] for r in records for t in r["tells"] if t.get("type") == "AI"]
    human_exp = [t["explanation"] for r in records for t in r["tells"] if t.get("type") == "human"]
    tpd = [r["n_tells"] for r in records]
    return {
        "summary": {
            "n_docs": len(records),
            "total_tells": len(all_exp),
            "mean_tells_per_doc": round(sum(tpd) / len(tpd), 2) if tpd else 0,
            "n_ai_tells": len(ai_exp),
            "n_human_tells": len(human_exp),
        },
        "lexical_diversity": {
            "all": _lexical_diversity(all_exp),
            "ai_tells": _lexical_diversity(ai_exp),
            "human_tells": _lexical_diversity(human_exp),
        },
        "common_ngrams": {
            "all": _common_ngrams(all_exp),
            "ai_tells": _common_ngrams(ai_exp),
            "human_tells": _common_ngrams(human_exp),
        },
        "clustering": {
            "all": _tfidf_clusters(all_exp),
            "ai_tells": _tfidf_clusters(ai_exp),
            "human_tells": _tfidf_clusters(human_exp),
        },
        "intra_doc_overlap": _intra_doc_overlap(records),
    }


# ── Report printing ───────────────────────────────────────────────────────────

def _div_section(title: str, ld: dict) -> None:
    if not ld:
        return
    print(f"\n  {title}")
    print(f"    vocab {ld.get('vocab_size')}  |  TTR {ld.get('ttr')}  |  "
          f"mean {ld.get('mean_length_words')}w  p95 {ld.get('p95_length_words')}w  "
          f"({ld.get('n_explanations')} explanations)")


def _ngram_section(title: str, ngrams: dict, key: str, top: int = 20) -> None:
    items = ngrams.get(key, [])[:top]
    if not items:
        return
    print(f"\n── {title} ──")
    for item in items:
        bar = "█" * max(1, int(item["pct"] / 2))
        print(f"  {item['pct']:5.1f}%  {bar:<18}  {item['phrase']}")


def print_report(report: dict) -> None:
    s = report["summary"]
    w = 72
    print(f"\n{'=' * w}")
    print(f"  TELL QUALITY ANALYSIS")
    print(f"  {s['n_docs']} docs · {s['total_tells']} tells · {s['mean_tells_per_doc']} tells/doc")
    print(f"  AI tells: {s['n_ai_tells']}  |  human tells: {s['n_human_tells']}")
    print(f"{'=' * w}")

    # Lexical diversity
    print(f"\n── LEXICAL DIVERSITY ──")
    _div_section("all tells  ", report["lexical_diversity"]["all"])
    _div_section("AI tells   ", report["lexical_diversity"]["ai_tells"])
    _div_section("human tells", report["lexical_diversity"]["human_tells"])
    print("  (TTR: 1.0 = every word unique; lower = more repetition)")

    # Common n-grams — all, then split by type
    _ngram_section("TOP BIGRAMS / TRIGRAMS  (all tells)", report["common_ngrams"]["all"], "top_bigrams_trigrams")
    _ngram_section("TOP LONG PHRASES  (all tells)", report["common_ngrams"]["all"], "top_long_phrases", top=15)
    _ngram_section("TOP BIGRAMS / TRIGRAMS  (AI tells)", report["common_ngrams"]["ai_tells"], "top_bigrams_trigrams")
    _ngram_section("TOP BIGRAMS / TRIGRAMS  (human tells)", report["common_ngrams"]["human_tells"], "top_bigrams_trigrams")

    # Clustering
    for scope in ("all", "ai_tells", "human_tells"):
        cl = report["clustering"][scope]
        if "skipped" in cl:
            continue
        label = {"all": "ALL TELLS", "ai_tells": "AI TELLS", "human_tells": "HUMAN TELLS"}[scope]
        print(f"\n── CLUSTERS — {label}  (k={cl['n_clusters']}, silhouette={cl['silhouette']}) ──")
        for c in cl["clusters"]:
            print(f"\n  [{c['id']}] {c['size']} tells ({c['pct']}%)  terms: {', '.join(c['top_terms'][:6])}")
            print(f"       rep: {c['representative'][:110]}")

    # Intra-doc overlap
    ov = report["intra_doc_overlap"]
    print(f"\n── INTRA-DOCUMENT TELL OVERLAP ──")
    if "error" in ov:
        print(f"  {ov['error']}")
    else:
        print(f"  docs with ≥2 tells: {ov['n_docs_with_multi_tells']}  |  single-tell: {ov['single_tell_docs']}")
        print(f"  mean cosine sim:  {ov['mean_intra_sim']}  (0 = fully diverse, 1 = identical)")
        print(f"  median: {ov['median_intra_sim']}   p75: {ov['p75_intra_sim']}   p95: {ov['p95_intra_sim']}")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

async def _main(checkpoint: str, n_docs: int, output: str | None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    records = await collect_tells(checkpoint, n_docs)
    if not records:
        raise SystemExit("No tells collected — check checkpoint and dataset config.")
    report = analyse(records)
    print_report(report)
    if output:
        with open(output, "w") as f:
            json.dump({"records": records, "report": report}, f, indent=2)
        print(f"Raw data written to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse tell quality for a trained checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Tinker weights path, e.g. tinker://…/weights/best-step-30")
    parser.add_argument("--n-docs", type=int, default=250, help="Number of eval docs (default: 250)")
    parser.add_argument("--output", help="Write raw records + report to this JSON file")
    args = parser.parse_args()
    asyncio.run(_main(args.checkpoint, args.n_docs, args.output))
