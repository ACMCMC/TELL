"""spaCy lemma sets for span–why Jaccard: English blank pipeline + rule lemmatizer (no sm download).

Cite spaCy (Honnibal and Montani, 2017).  We use ``English()`` + rule lemmatizer so ``uv sync``
does not pull ``en_core_web_sm`` (slow or fragile on some stacks); lemmas are still standard spaCy.
spaCy 3.8+ needs ``spacy-lookups-data`` for ``lemma_rules`` when ``mode`` is ``rule``.
"""

from __future__ import annotations

import html
import string

from spacy.lang.en import English


def normalize_fragment(text: str) -> str:
    t = html.unescape(text).strip().lower()
    t = t.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    return " ".join(t.split())


_EN_LEMMA_NLP: English | None = None


def _en_rule_lemma_nlp() -> English:
    global _EN_LEMMA_NLP
    if _EN_LEMMA_NLP is None:
        nlp = English()
        nlp.add_pipe("lemmatizer", config={"mode": "rule"})
        nlp.initialize()
        _EN_LEMMA_NLP = nlp
    return _EN_LEMMA_NLP


def build_span_str_to_lemma_frozenset(
    unique_spans: list[str],
    *,
    pipe_batch_size: int,
    exclude_stopwords: bool,
) -> dict[str, frozenset[str]]:
    nlp = _en_rule_lemma_nlp()
    out: dict[str, frozenset[str]] = {}
    for text, doc in zip(unique_spans, nlp.pipe(unique_spans, batch_size=int(pipe_batch_size))):
        lemmas: list[str] = []
        for tok in doc:
            if tok.is_space or tok.is_punct:
                continue
            if exclude_stopwords and tok.is_stop:
                continue
            lem = str(tok.lemma_).strip().lower()
            if lem and lem != "-pron-":
                lemmas.append(lem)
        out[text] = frozenset(lemmas)
    return out
