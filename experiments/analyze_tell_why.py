# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "adjustText",
#   "datasets",
#   "fastembed",
#   "matplotlib",
#   "numpy",
#   "scikit-learn",
# ]
# ///
from __future__ import annotations

import argparse
import csv
import html
import json
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from adjustText import adjust_text
from matplotlib import font_manager
import numpy as np
from datasets import Dataset, DatasetDict, load_dataset
from fastembed import TextEmbedding
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE


DATASET_NAME = "acmc/TELL"
SEED = 2242
STOPWORDS = {
    "a", "again", "also", "an", "and", "are", "as", "at", "be", "because", "but", "by", "for",
    "feel", "feels", "from", "has", "have", "i", "in", "is", "it", "like", "more", "not", "of", "often",
    "on", "or", "same", "so", "sound", "sounds", "that", "the", "think", "this", "to", "while",
    "with", "would", "writer",
}
LABEL_STOPWORDS = STOPWORDS | {"ai", "human"}
ATTR_BLOCK_RE = re.compile(r"\]\{([^{}]*why=\"(?:\\.|[^\"])*\"[^{}]*)\}")
ATTR_RE = re.compile(r"(\w+)=\"((?:\\.|[^\"])*)\"")
NEW_ANNOTATION_RE = re.compile(r"<annotation\b([^>]*)/\s*>", re.IGNORECASE)
WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")


def normalize_span_type(raw: str) -> str:
    t = str(raw).strip().lower()
    if t == "ai":
        return "AI"
    if t == "human":
        return "human"
    return str(raw).strip()


def parse_annotation_tag_attrs(fragment: str) -> dict[str, str]:
    attrs = {}
    for key in ("type", "why", "score"):
        dq = re.search(rf'{re.escape(key)}\s*=\s*"((?:\\.|[^"])*)"', fragment)
        sq = re.search(rf"{re.escape(key)}\s*=\s*'((?:\\.|[^'])*)'", fragment)
        if dq:
            attrs[key] = dq.group(1).replace(r"\"", '"')
        elif sq:
            attrs[key] = sq.group(1).replace(r"\'", "'")
    return attrs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze common TELL why= explanations.")
    parser.add_argument("--dataset", type=str, default=DATASET_NAME)
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/tell_why_analysis"))
    parser.add_argument("--top-k", type=int, default=80)
    parser.add_argument("--max-ngram", type=int, default=5)
    parser.add_argument("--min-count", type=int, default=3)
    parser.add_argument("--tfidf-ngram-max", type=int, default=4)
    parser.add_argument("--tfidf-top-k", type=int, default=200)
    parser.add_argument("--embedding-model", type=str, default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--max-embeddings", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--projections", nargs="+", choices=["pca", "tsne"], default=["pca", "tsne"])
    parser.add_argument("--plot-labels", type=int, default=20)
    parser.add_argument("--label-scores", nargs="+", choices=["count", "tfidf"], default=["count", "tfidf"])
    parser.add_argument("--label-font-size", type=float, default=6.5)
    parser.add_argument("--label-min-ngram", type=int, default=2)
    parser.add_argument("--label-max-ngram", type=int, default=5)
    parser.add_argument("--font-family", type=str, default="IBM Plex Sans")
    parser.add_argument("--font-path", type=Path, default=Path("experiments/tmp/fonts/IBMPlexSans-Regular.ttf"))
    parser.add_argument("--skip-embeddings", action="store_true")
    parser.add_argument("--skip-ngram-plots", action="store_true")
    parser.add_argument("--ngram-plot-ns", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--ngram-plot-top-k", type=int, default=40)
    parser.add_argument("--local-jsonl", type=Path, default=None,
                        help="Load annotations from a local JSONL file instead of HuggingFace.")
    parser.add_argument("--max-rows", type=int, default=0,
                        help="Randomly subsample this many rows for all analysis (0 = all).")
    return parser.parse_args()


def normalized_text(text: str) -> str:
    text = text.lower()
    text = text.replace("ai-written", "ai written")
    text = text.replace("ai-like", "ai like")
    table = str.maketrans({char: " " for char in string.punctuation})
    text = text.translate(table)
    return " ".join(text.split())


def words_for(text: str, remove_stopwords: bool) -> list[str]:
    words = WORD_RE.findall(normalized_text(text=text))
    if remove_stopwords:
        words = [word for word in words if word not in STOPWORDS and len(word) > 1]
    return words


def label_words_for(text: str) -> list[str]:
    words = WORD_RE.findall(normalized_text(text=text))
    return [word for word in words if word not in LABEL_STOPWORDS and len(word) > 1]


def load_rows_from_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            why = str(item.get("explanation") or "").strip()
            if not why:
                continue
            rows.append({
                "split": "local",
                "example_id": str(item.get("doc_id") or ""),
                "source": str(item.get("doc_id") or ""),
                "source_dataset": str(item.get("dataset_id") or ""),
                "text_type": str(item.get("domain") or ""),
                "label": int(item.get("label", -1)),
                "span_idx": int(item.get("ann_idx", 0)),
                "span_type": normalize_span_type(raw=item.get("type", "")),
                "score": str(item.get("model_score", "")),
                "why": why,
                "why_norm": normalized_text(text=why),
            })
    return rows


def load_splits(dataset_name: str, split_names: list[str]) -> DatasetDict:
    loaded = load_dataset(path=dataset_name)
    return DatasetDict({split: loaded[split] for split in split_names})


def parse_why_rows(dataset: DatasetDict) -> list[dict[str, Any]]:
    rows = []
    for split, split_dataset in dataset.items():
        for item in split_dataset:
            annotation = str(item["annotation"])
            if "<annotation" in annotation.lower():
                for span_idx, match in enumerate(NEW_ANNOTATION_RE.finditer(string=annotation)):
                    attrs = parse_annotation_tag_attrs(fragment=match.group(1))
                    why = html.unescape(attrs.get("why", "").replace(r"\"", '"').strip())
                    if not why:
                        continue
                    rows.append({
                        "split": split,
                        "example_id": str(item.get("example_id") or ""),
                        "source": str(item.get("source") or ""),
                        "source_dataset": str(item.get("source_dataset") or ""),
                        "text_type": str(item.get("text_type") or ""),
                        "label": int(item.get("label", -1)),
                        "span_idx": span_idx,
                        "span_type": normalize_span_type(raw=attrs.get("type", "")),
                        "score": attrs.get("score", ""),
                        "why": why,
                        "why_norm": normalized_text(text=why),
                    })
            else:
                for span_idx, match in enumerate(ATTR_BLOCK_RE.finditer(string=annotation)):
                    attrs = dict(ATTR_RE.findall(string=match.group(1)))
                    why = html.unescape(attrs.get("why", "").replace(r"\"", '"').strip())
                    if not why:
                        continue
                    rows.append({
                        "split": split,
                        "example_id": str(item.get("example_id") or ""),
                        "source": str(item.get("source") or ""),
                        "source_dataset": str(item.get("source_dataset") or ""),
                        "text_type": str(item.get("text_type") or ""),
                        "label": int(item.get("label", -1)),
                        "span_idx": span_idx,
                        "span_type": normalize_span_type(raw=attrs.get("type", "")),
                        "score": attrs.get("score", ""),
                        "why": why,
                        "why_norm": normalized_text(text=why),
                    })
    return rows


def ngram_counts(rows: list[dict[str, Any]], max_ngram: int, remove_stopwords: bool) -> list[dict[str, Any]]:
    counts: Counter[tuple[int, str]] = Counter()
    type_counts: dict[tuple[int, str], Counter[str]] = {}
    for row in rows:
        words = words_for(text=row["why"], remove_stopwords=remove_stopwords)
        seen = set()
        for n in range(1, max_ngram + 1):
            for start in range(0, len(words) - n + 1):
                phrase = " ".join(words[start:start + n])
                seen.add((n, phrase))
        for key in seen:
            counts[key] += 1
            type_counts.setdefault(key, Counter())[str(row["span_type"])] += 1
    total = len(rows)
    out_rows = []
    for (n, phrase), count in counts.most_common():
        out_rows.append({
            "n": n,
            "phrase": phrase,
            "count": count,
            "support_pct": round(100.0 * count / total, 3),
            "ai_count": type_counts[(n, phrase)].get("AI", 0),
            "human_count": type_counts[(n, phrase)].get("human", 0),
        })
    return out_rows


def char_substring_counts(rows: list[dict[str, Any]], lengths: list[int]) -> list[dict[str, Any]]:
    counts: Counter[tuple[int, str]] = Counter()
    for row in rows:
        text = normalized_text(text=row["why"])
        seen = set()
        for length in lengths:
            for start in range(0, len(text) - length + 1):
                chunk = text[start:start + length].strip()
                if len(chunk) == length and " " in chunk:
                    seen.add((length, chunk))
        counts.update(seen)
    total = len(rows)
    return [
        {
            "length": length,
            "substring": substring,
            "count": count,
            "support_pct": round(100.0 * count / total, 3),
        }
        for (length, substring), count in counts.most_common()
    ]


def label_ngram_counts(rows: list[dict[str, Any]], max_ngram: int) -> list[dict[str, Any]]:
    counts: Counter[tuple[int, str]] = Counter()
    for row in rows:
        words = label_words_for(text=row["why"])
        seen = set()
        for n in range(1, max_ngram + 1):
            for start in range(0, len(words) - n + 1):
                seen.add((n, " ".join(words[start:start + n])))
        counts.update(seen)
    total = len(rows)
    return [
        {
            "n": n,
            "phrase": phrase,
            "count": count,
            "support_pct": round(100.0 * count / total, 3),
        }
        for (n, phrase), count in counts.most_common()
    ]


def idf_ngram_rows(rows: list[dict[str, Any]], max_ngram: int) -> list[dict[str, Any]]:
    doc_freq: Counter[tuple[int, str]] = Counter()
    span_freq: Counter[tuple[int, str]] = Counter()
    for row in rows:
        words = label_words_for(text=row["why"])
        seen_docs = set()
        for n in range(1, max_ngram + 1):
            for start in range(0, len(words) - n + 1):
                key = (n, " ".join(words[start:start + n]))
                span_freq[key] += 1
                seen_docs.add(key)
        doc_freq.update(seen_docs)
    n_docs = len(rows)
    out_rows = []
    for (n, phrase), df in doc_freq.items():
        idf = float(np.log((n_docs + 1) / (df + 1)) + 1.0)
        out_rows.append({
            "n": n,
            "phrase": phrase,
            "count": int(span_freq[(n, phrase)]),
            "doc_count": int(df),
            "doc_pct": round(100.0 * df / n_docs, 3),
            "idf": round(idf, 6),
        })
    out_rows.sort(key=lambda row: (-row["idf"], -row["count"]))
    return out_rows


def ngram_rows_for_plot(
    rows: list[dict[str, Any]],
    n: int,
    score: str,
    max_ngram: int,
    min_count: int,
    top_k: int,
) -> list[dict[str, Any]]:
    if score == "count":
        raw = label_ngram_counts(rows=rows, max_ngram=max_ngram)
        key = "count"
    else:
        raw = idf_ngram_rows(rows=rows, max_ngram=max_ngram)
        key = "idf"
    filtered = [row for row in raw if row["n"] == n and row["count"] >= min_count]
    filtered.sort(key=lambda row: -row[key])
    return filtered[:top_k]


def plot_ngram_bars(
    plot_rows: list[dict[str, Any]],
    path: Path,
    title: str,
    score: str,
    font_family: str,
    font_path: Path,
) -> None:
    configure_font(font_family=font_family, font_path=font_path)
    score_key = "count" if score == "count" else "idf"
    score_label = "span count" if score == "count" else "IDF (rare = creative)"
    phrases = [row["phrase"] for row in plot_rows][::-1]
    values = [row[score_key] for row in plot_rows][::-1]
    height = max(4.0, 0.28 * len(phrases) + 1.2)
    fig, ax = plt.subplots(figsize=(8.8, height))
    bars = ax.barh(y=range(len(phrases)), width=values, color="#4c72b0", height=0.72)
    ax.set_yticks(range(len(phrases)))
    ax.set_yticklabels(phrases, fontsize=8 if max(len(p) for p in phrases) > 28 else 9)
    ax.set_xlabel(score_label)
    if title:
        ax.set_title(title)
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_width() + max(values) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}" if score == "idf" else str(int(value)),
            va="center",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(fname=path)
    plt.close(fig=fig)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text(data="", encoding="utf-8")
        return
    with path.open(mode="w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(f=fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def tfidf_rows(rows: list[dict[str, Any]], ngram_max: int, top_k: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    docs = [" ".join(label_words_for(text=row["why"])) for row in rows]
    if not rows or not any(doc.strip() for doc in docs):
        return [], [], []
    vectorizer = TfidfVectorizer(
        lowercase=False,
        token_pattern=r"(?u)\b\w[\w']+\b",
        ngram_range=(1, ngram_max),
        min_df=2,
        norm="l2",
    )
    try:
        matrix = vectorizer.fit_transform(raw_documents=docs)
    except ValueError:
        vectorizer = TfidfVectorizer(
            lowercase=False,
            token_pattern=r"(?u)\b\w[\w']+\b",
            ngram_range=(1, ngram_max),
            min_df=1,
            norm="l2",
        )
        matrix = vectorizer.fit_transform(raw_documents=docs)
    features = np.asarray(vectorizer.get_feature_names_out())
    mean_scores = np.asarray(matrix.mean(axis=0)).ravel()
    max_scores = np.asarray(matrix.max(axis=0).toarray()).ravel()
    doc_counts = np.asarray((matrix > 0).sum(axis=0)).ravel()
    term_order = np.argsort(-max_scores)[:top_k]
    term_rows = [
        {
            "term": str(features[idx]),
            "max_tfidf": round(float(max_scores[idx]), 6),
            "mean_tfidf": round(float(mean_scores[idx]), 6),
            "doc_count": int(doc_counts[idx]),
            "doc_pct": round(100.0 * float(doc_counts[idx]) / len(rows), 3),
        }
        for idx in term_order
    ]

    explanation_rows = []
    dedup_rows = []
    seen_why = set()
    for row_idx, row in enumerate(rows):
        row_matrix = matrix.getrow(row_idx)
        if row_matrix.nnz == 0:
            continue
        order = np.argsort(-row_matrix.data)[:5]
        top_terms = [
            f"{features[row_matrix.indices[i]]}:{row_matrix.data[i]:.3f}"
            for i in order
        ]
        score = float(np.mean(row_matrix.data[order]))
        explanation_rows.append({
            "unique_score": round(score, 6),
            "top_tfidf_terms": "; ".join(top_terms),
            "span_type": row["span_type"],
            "source": row["source"],
            "text_type": row["text_type"],
            "example_id": row["example_id"],
            "span_idx": row["span_idx"],
            "why": row["why"],
        })
        if row["why_norm"] not in seen_why:
            seen_why.add(row["why_norm"])
            dedup_rows.append(explanation_rows[-1])
    explanation_rows.sort(key=lambda row: row["unique_score"], reverse=True)
    dedup_rows.sort(key=lambda row: row["unique_score"], reverse=True)
    return term_rows, explanation_rows[:top_k], dedup_rows[:top_k]


def embed_texts(texts: list[str], model_name: str, batch_size: int) -> np.ndarray:
    model = TextEmbedding(model_name=model_name)
    embeddings = np.asarray(list(model.embed(documents=texts, batch_size=batch_size)))
    norms = np.linalg.norm(x=embeddings, axis=1, keepdims=True)
    return embeddings / norms


def project_embeddings(embeddings: np.ndarray, projection: str) -> np.ndarray:
    if projection == "pca":
        return PCA(n_components=2, random_state=SEED).fit_transform(X=embeddings)
    first_dims = min(50, embeddings.shape[1], embeddings.shape[0] - 1)
    reduced = PCA(n_components=first_dims, random_state=SEED).fit_transform(X=embeddings)
    perplexity = min(30, max(5, (embeddings.shape[0] - 1) // 3))
    return TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=SEED,
    ).fit_transform(X=reduced)


def configure_font(font_family: str, font_path: Path) -> None:
    if font_path.exists():
        font_manager.fontManager.addfont(path=font_path)
    plt.rcParams.update({
        "font.family": font_family,
        "font.sans-serif": [font_family],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def phrase_matches(row: dict[str, Any], phrase: str) -> bool:
    label_text = " ".join(label_words_for(text=row["why"]))
    return f" {phrase} " in f" {label_text} "


def tfidf_label_candidates(rows: list[dict[str, Any]], max_ngram: int) -> list[dict[str, Any]]:
    docs = [" ".join(label_words_for(text=row["why"])) for row in rows]
    if not rows or not any(doc.strip() for doc in docs):
        return []
    vectorizer = TfidfVectorizer(
        lowercase=False,
        token_pattern=r"(?u)\b\w[\w']+\b",
        ngram_range=(1, max_ngram),
        min_df=2,
        norm="l2",
    )
    try:
        matrix = vectorizer.fit_transform(raw_documents=docs)
    except ValueError:
        vectorizer = TfidfVectorizer(
            lowercase=False,
            token_pattern=r"(?u)\b\w[\w']+\b",
            ngram_range=(1, max_ngram),
            min_df=1,
            norm="l2",
        )
        matrix = vectorizer.fit_transform(raw_documents=docs)
    features = np.asarray(vectorizer.get_feature_names_out())
    max_scores = np.asarray(matrix.max(axis=0).toarray()).ravel()
    doc_counts = np.asarray((matrix > 0).sum(axis=0)).ravel()
    order = np.argsort(-max_scores)
    return [
        {
            "n": len(str(features[idx]).split()),
            "phrase": str(features[idx]),
            "count": int(doc_counts[idx]),
            "tfidf": round(float(max_scores[idx]), 6),
        }
        for idx in order
    ]


def label_ngrams(
    rows: list[dict[str, Any]],
    points: np.ndarray,
    max_ngram: int,
    min_ngram: int,
    label_max_ngram: int,
    label_score: str,
    limit: int,
) -> list[dict[str, Any]]:
    if label_score == "tfidf":
        raw_candidates = tfidf_label_candidates(rows=rows, max_ngram=max_ngram)
    else:
        raw_candidates = label_ngram_counts(rows=rows, max_ngram=max_ngram)
    candidates = [row for row in raw_candidates if min_ngram <= row["n"] <= label_max_ngram]
    labels = []
    for candidate in candidates:
        idx = [i for i, row in enumerate(rows) if phrase_matches(row=row, phrase=candidate["phrase"])]
        if len(idx) < 3:
            continue
        xy = points[idx]
        center = np.mean(a=xy, axis=0)
        labels.append({
            "phrase": candidate["phrase"],
            "count": candidate["count"],
            "tfidf": candidate.get("tfidf", ""),
            "x": float(center[0]),
            "y": float(center[1]),
        })
        if len(labels) == limit:
            break
    return labels


def plot_projection(
    points: np.ndarray,
    rows: list[dict[str, Any]],
    path: Path,
    title: str,
    labels: list[dict[str, Any]],
    label_font_size: float,
) -> None:
    colors = {"AI": "#d62728", "human": "#2ca02c"}
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    for span_type in ["AI", "human"]:
        idx = [i for i, row in enumerate(rows) if row["span_type"] == span_type]
        if idx:
            ax.scatter(
                points[idx, 0],
                points[idx, 1],
                s=6,
                alpha=0.45,
                c=colors[span_type],
                label=f"{span_type} why",
                linewidths=0,
            )
    texts = []
    for label in labels:
        texts.append(ax.text(
            label["x"],
            label["y"],
            label["phrase"],
            fontsize=label_font_size,
            ha="center",
            va="center",
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": "white",
                "edgecolor": "#bbbbbb",
                "alpha": 0.82,
                "lw": 0.35,
            },
        ))
    adjust_text(
        texts,
        ax=ax,
        expand=(1.15, 1.35),
        force_text=(0.25, 0.45),
        force_static=(0.15, 0.25),
        arrowprops={"arrowstyle": "-", "color": "#555555", "lw": 0.45, "alpha": 0.75},
    )
    if title:
        ax.set_title(title)
    ax.set_xlabel("component 1")
    ax.set_ylabel("component 2")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(fname=path)
    plt.close(fig=fig)


def sampled_rows(rows: list[dict[str, Any]], max_rows: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed=SEED)
    if max_rows <= 0 or len(rows) <= max_rows:
        return rows
    idx = rng.choice(a=len(rows), size=max_rows, replace=False)
    return [rows[i] for i in sorted(idx.tolist())]


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.local_jsonl is not None:
        rows = load_rows_from_jsonl(path=args.local_jsonl)
    else:
        dataset = load_splits(dataset_name=args.dataset, split_names=args.splits)
        rows = parse_why_rows(dataset=dataset)
    if args.max_rows > 0 and len(rows) > args.max_rows:
        rng = np.random.default_rng(seed=SEED)
        idx = rng.choice(a=len(rows), size=args.max_rows, replace=False)
        rows = [rows[i] for i in sorted(idx.tolist())]
        print(f"Subsampled to {len(rows):,} rows (--max-rows {args.max_rows})")

    write_csv(path=args.out_dir / "why_spans.csv", rows=rows)
    write_csv(
        path=args.out_dir / "top_ngrams_raw.csv",
        rows=[
            row for row in ngram_counts(rows=rows, max_ngram=args.max_ngram, remove_stopwords=False)
            if row["count"] >= args.min_count
        ][:args.top_k],
    )
    content_ngrams = [
        row for row in ngram_counts(rows=rows, max_ngram=args.max_ngram, remove_stopwords=True)
        if row["count"] >= args.min_count
    ][:args.top_k]
    write_csv(path=args.out_dir / "top_ngrams_content.csv", rows=content_ngrams)
    idf_ngrams = [
        row for row in idf_ngram_rows(rows=rows, max_ngram=args.max_ngram)
        if row["count"] >= args.min_count
    ][:args.top_k]
    write_csv(path=args.out_dir / "top_ngrams_idf.csv", rows=idf_ngrams)
    write_csv(
        path=args.out_dir / "top_char_substrings.csv",
        rows=[
            row for row in char_substring_counts(rows=rows, lengths=[16, 24, 32])
            if row["count"] >= args.min_count
        ][:args.top_k],
    )
    terms, explanations, dedup_explanations = tfidf_rows(rows=rows, ngram_max=args.tfidf_ngram_max, top_k=args.tfidf_top_k)
    write_csv(path=args.out_dir / "tfidf_top_terms.csv", rows=terms)
    write_csv(path=args.out_dir / "tfidf_unique_explanations.csv", rows=explanations)
    write_csv(path=args.out_dir / "tfidf_unique_explanations_dedup.csv", rows=dedup_explanations)

    metadata = {
        "dataset": args.dataset,
        "splits": args.splits,
        "seed": SEED,
        "num_why_spans": len(rows),
        "tfidf_ngram_max": args.tfidf_ngram_max,
        "tfidf_top_k": args.tfidf_top_k,
        "embedding_model": args.embedding_model,
        "projections": args.projections,
        "max_embeddings": args.max_embeddings,
        "plot_labels": args.plot_labels,
        "label_scores": args.label_scores,
        "label_font_size": args.label_font_size,
        "label_min_ngram": args.label_min_ngram,
        "label_max_ngram": args.label_max_ngram,
        "font_family": args.font_family,
        "font_path": str(args.font_path),
        "ngram_plot_ns": args.ngram_plot_ns,
        "ngram_plot_top_k": args.ngram_plot_top_k,
    }
    (args.out_dir / "run_metadata.json").write_text(data=json.dumps(obj=metadata, indent=2), encoding="utf-8")

    if not args.skip_ngram_plots:
        configure_font(font_family=args.font_family, font_path=args.font_path)
        for n in args.ngram_plot_ns:
            for score in ("count", "idf"):
                plot_rows = ngram_rows_for_plot(
                    rows=rows,
                    n=n,
                    score=score,
                    max_ngram=args.max_ngram,
                    min_count=args.min_count,
                    top_k=args.ngram_plot_top_k,
                )
                write_csv(path=args.out_dir / f"ngram_{n}_{score}_plot.csv", rows=plot_rows)
                plot_ngram_bars(
                    plot_rows=plot_rows,
                    path=args.out_dir / f"ngram_{n}_{score}.pdf",
                    title="",
                    score=score,
                    font_family=args.font_family,
                    font_path=args.font_path,
                )

    if not args.skip_embeddings:
        configure_font(font_family=args.font_family, font_path=args.font_path)
        plot_rows = sampled_rows(rows=rows, max_rows=args.max_embeddings)
        embeddings = embed_texts(
            texts=[row["why"] for row in plot_rows],
            model_name=args.embedding_model,
            batch_size=args.batch_size,
        )
        for projection in args.projections:
            points = project_embeddings(embeddings=embeddings, projection=projection)
            proj_rows = []
            for row, point in zip(plot_rows, points, strict=True):
                proj_row = dict(row)
                proj_row["projection"] = projection
                proj_row["x"] = float(point[0])
                proj_row["y"] = float(point[1])
                proj_rows.append(proj_row)
            write_csv(path=args.out_dir / f"embedding_projection_{projection}.csv", rows=proj_rows)
            for label_score in args.label_scores:
                labels = label_ngrams(
                    rows=plot_rows,
                    points=points,
                    max_ngram=args.max_ngram,
                    min_ngram=args.label_min_ngram,
                    label_max_ngram=args.label_max_ngram,
                    label_score=label_score,
                    limit=args.plot_labels,
                )
                write_csv(
                    path=args.out_dir / f"plot_ngram_labels_{projection}_{label_score}.csv",
                    rows=labels,
                )
                plot_projection(
                    points=points,
                    rows=plot_rows,
                    path=args.out_dir / f"why_embeddings_{projection}_{label_score}.pdf",
                    title="",
                    labels=labels,
                    label_font_size=args.label_font_size,
                )

    print(f"Wrote {len(rows):,} why spans to {args.out_dir}")


if __name__ == "__main__":
    main()
