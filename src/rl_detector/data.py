"""Dataset loading and sampling helpers."""

import hashlib
import json
import math
import os
import random
import re
from pathlib import Path

from omegaconf import ListConfig

from rl_detector.config import CFG

_HTML_TAG_RE = re.compile(r"<[^>\s][^>]*>")
_WHITESPACE_RE = re.compile(r"\s+")


def clean_document_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text)
    text = _HTML_TAG_RE.sub(" ", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _manifest_path(use_eval_split: bool) -> str:
    attr = "eval_docs_path" if use_eval_split else "train_docs_path"
    value = getattr(CFG.data, attr, None)
    if not value:
        split = "eval" if use_eval_split else "train"
        raise ValueError(
            f"data.{attr} is not set. Use hf://acmc/multi_domain_ai_human_text/{split} "
            f"or upload splits with scripts/upload_to_hf.py."
        )
    return str(value)


def _doc_from_row(row: dict) -> dict | None:
    text = clean_document_text(row.get("text", ""))
    if not text:
        return None
    doc = {k: v for k, v in row.items() if k not in {"text", "label"}}
    doc.update({"text": text, "label": int(row["label"])})
    return doc


def _normalize_filter_raw(raw) -> set[str] | None:
    """Parse a filter value (null / str / list) into a set of strings, or None if unset."""
    if raw is None:
        return None
    if isinstance(raw, (ListConfig, list, tuple)):
        seq = list(raw)
    else:
        seq = [raw]
    out = {str(x).strip() for x in seq if str(x).strip()}
    return out or None


def _dataset_id_allowlist(use_eval_split: bool) -> set[str] | None:
    if use_eval_split:
        ev = _normalize_filter_raw(getattr(CFG.data, "eval_dataset_ids_filter", None))
        if ev is not None:
            return ev
        return _normalize_filter_raw(getattr(CFG.data, "train_dataset_ids_filter", None))
    return _normalize_filter_raw(getattr(CFG.data, "train_dataset_ids_filter", None))


def _strata_allowlist(use_eval_split: bool) -> set[str] | None:
    """Return the active strata filter as a set of 'dataset_id|domain|label' keys, or None."""
    if use_eval_split:
        ev = _normalize_filter_raw(getattr(CFG.data, "eval_strata_filter", None))
        if ev is not None:
            return ev
        return _normalize_filter_raw(getattr(CFG.data, "train_strata_filter", None))
    return _normalize_filter_raw(getattr(CFG.data, "train_strata_filter", None))


def _doc_stratum_key(doc: dict) -> str:
    """Return 'dataset_id|domain|label' for a doc — matches the StratumSampler key format."""
    return f"{doc.get('dataset_id', '')}|{doc.get('domain', 'unknown')}|{int(doc.get('label', 0))}"


def _doc_passes_filter(
    doc: dict,
    dataset_allowlist: set[str] | None,
    strata_allowlist: set[str] | None = None,
) -> bool:
    """Return True iff the doc passes both the dataset-level and stratum-level allowlists."""
    if dataset_allowlist and "*" not in dataset_allowlist:
        if str(doc.get("dataset_id", "")) not in dataset_allowlist:
            return False
    if strata_allowlist and "*" not in strata_allowlist:
        if _doc_stratum_key(doc) not in strata_allowlist:
            return False
    return True


def _parse_hf_path(path: str) -> tuple[str, str]:
    spec = path.removeprefix("hf://").strip("/")
    parts = spec.split("/")
    if len(parts) < 3:
        raise ValueError(f"HF dataset path must be hf://owner/repo/split, got {path!r}")
    repo_id = "/".join(parts[:2])
    split = "/".join(parts[2:])
    return repo_id, split


def _stream_rows(path: str):
    """Yield raw row dicts from an HF parquet path or a local JSONL file."""
    if path.startswith("hf://"):
        from datasets import load_dataset
        repo_id, split = _parse_hf_path(path)
        data_files = {split: f"data/{split}-*.parquet"}
        for row in load_dataset(repo_id, split=split, streaming=True, data_files=data_files):
            yield dict(row)
    else:
        with Path(path).expanduser().open() as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def _stream_filtered_docs(
    path: str,
    dataset_allowlist: set[str] | None,
    strata_allowlist: set[str] | None,
):
    """Yield ``_doc_from_row``-decoded docs that pass the active filters."""
    for raw in _stream_rows(path):
        doc = _doc_from_row(raw)
        if doc is not None and _doc_passes_filter(doc, dataset_allowlist, strata_allowlist):
            yield doc


def _load_docs_simple(
    path: str,
    max_docs: int | None,
    dataset_allowlist: set[str] | None = None,
    strata_allowlist: set[str] | None = None,
) -> list[dict]:
    """Stream up to ``max_docs`` filtered docs in source order."""
    docs: list[dict] = []
    for doc in _stream_filtered_docs(path, dataset_allowlist, strata_allowlist):
        if max_docs is not None and len(docs) >= max_docs:
            break
        docs.append(doc)
    return docs


def _load_docs_balanced(
    path: str,
    max_docs: int,
    seed: int,
    dataset_allowlist: set[str] | None = None,
    strata_allowlist: set[str] | None = None,
    proportional: bool = False,
) -> list[dict]:
    """Build a label-balanced pool (50/50 AI/human).

    proportional=False (default): round-robin across strata so each stratum gets
    equal representation regardless of size.

    proportional=True: pool all docs per label then sample randomly, so each stratum
    is represented in proportion to its natural frequency in the dataset. Use this for
    eval pools that should mirror the test distribution.
    """
    half = (int(max_docs) + 1) // 2
    by_label_stratum: dict[int, dict[tuple[str, str, int], list[dict]]] = {0: {}, 1: {}}
    for doc in _stream_filtered_docs(path, dataset_allowlist, strata_allowlist):
        lbl = int(doc.get("label", 0))
        if lbl not in (0, 1):
            continue
        k = _doc_stratum_key(doc)
        by_label_stratum[lbl].setdefault(k, []).append(doc)

    rng = random.Random(int(seed))

    def _draw_proportional(label: int, target_n: int) -> list[dict]:
        pool = [d for bucket in by_label_stratum[label].values() for d in bucket]
        rng.shuffle(pool)
        return pool[:target_n]

    def _draw_diverse(label: int, target_n: int) -> list[dict]:
        buckets = by_label_stratum[label]
        keys = list(buckets.keys())
        for k in keys:
            rng.shuffle(buckets[k])
        chosen: list[dict] = []
        while len(chosen) < target_n and keys:
            rng.shuffle(keys)
            next_keys: list[tuple[str, str, int]] = []
            for k in keys:
                bucket = buckets[k]
                if not bucket:
                    continue
                chosen.append(bucket.pop())
                if bucket:
                    next_keys.append(k)
                if len(chosen) >= target_n:
                    break
            keys = next_keys
        return chosen

    draw = _draw_proportional if proportional else _draw_diverse
    ai_docs = draw(label=1, target_n=half)
    hum_docs = draw(label=0, target_n=half)
    merged = ai_docs + hum_docs
    rng.shuffle(merged)
    return merged


def load_docs(
    _dataset_ids: list[str] | None = None,
    use_eval_split: bool = False,
    max_docs: int | None = None,
    balance_eval_pool: bool | None = None,
) -> list[dict]:
    """Load docs from the configured HF split (train_docs_path / eval_docs_path).

    dataset_ids is legacy API compatibility; use CFG.data.train_dataset_ids_filter /
    eval_dataset_ids_filter for parquet dataset_id subsets.
    """
    path = _manifest_path(use_eval_split=use_eval_split)
    if balance_eval_pool is None:
        balance_eval_pool = bool(use_eval_split and getattr(CFG.data, "balance_eval_pool", False))
    seed = int(getattr(CFG.frozen, "seed", 2262))
    allow = _dataset_id_allowlist(use_eval_split)
    strata = _strata_allowlist(use_eval_split)

    if balance_eval_pool and max_docs is not None and max_docs > 0:
        proportional = bool(getattr(CFG.data, "proportional_eval_pool", False))
        docs = _load_docs_balanced(path, max_docs, seed, dataset_allowlist=allow, strata_allowlist=strata, proportional=proportional)
    else:
        docs = _load_docs_simple(path, max_docs=max_docs, dataset_allowlist=allow, strata_allowlist=strata)
    if not use_eval_split and docs:
        random.Random(seed).shuffle(docs)
    return docs


_PREPROCESS_CACHE_SCHEMA = "v2"


def _upstream_fingerprint(path: str) -> str:
    """Fingerprint source data so cache auto-invalidates when upstream changes."""
    if path.startswith("hf://"):
        try:
            from huggingface_hub import HfApi
            repo_id, _split = _parse_hf_path(path)
            info = HfApi().dataset_info(repo_id=repo_id)
            sha = str(getattr(info, "sha", "") or "")
            updated = str(getattr(info, "last_modified", "") or "")
            return f"hf:{repo_id}:{sha}:{updated}"
        except Exception:
            # Fall back to path-only fingerprint if metadata lookup fails.
            return f"hf:{path}"
    p = Path(path).expanduser()
    st = p.stat()
    return f"file:{p.resolve()}:{st.st_size}:{st.st_mtime_ns}"


def _cache_base_dir() -> Path:
    raw = getattr(getattr(CFG.data, "cache", {}), "base_dir", ".cache/preprocessed_docs")
    p = Path(str(raw)).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    return p


def _preprocessed_cache_path(
    *,
    use_eval_split: bool,
    max_docs: int | None,
    balance_eval_pool: bool,
    max_doc_tokens: int,
    min_doc_tokens: int,
) -> Path:
    path = _manifest_path(use_eval_split=use_eval_split)
    dataset_allow = sorted(_dataset_id_allowlist(use_eval_split) or [])
    strata_allow = sorted(_strata_allowlist(use_eval_split) or [])
    tokenizer_id = str(getattr(CFG.model, "base_model", "unknown"))
    payload = {
        "schema": _PREPROCESS_CACHE_SCHEMA,
        "source_path": path,
        "source_fp": _upstream_fingerprint(path),
        "split": "eval" if use_eval_split else "train",
        "max_docs": max_docs,
        "balance_eval_pool": bool(balance_eval_pool),
        "max_doc_tokens": int(max_doc_tokens),
        "min_doc_tokens": int(min_doc_tokens),
        "dataset_allowlist": dataset_allow,
        "strata_allowlist": strata_allow,
        "tokenizer_id": tokenizer_id,
        "seed": int(getattr(CFG.frozen, "seed", 2262)),
    }
    key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:24]
    return _cache_base_dir() / f"{'eval' if use_eval_split else 'train'}_{key}.jsonl"


def _read_docs_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def _write_docs_jsonl(path: Path, docs: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def load_docs_preprocessed(
    *,
    tokenizer,
    use_eval_split: bool,
    max_docs: int | None,
    max_doc_tokens: int,
    min_doc_tokens: int,
    balance_eval_pool: bool | None = None,
) -> tuple[list[dict], dict[str, object]]:
    """Load docs with local preprocessed cache; cache key includes upstream dataset fingerprint."""
    if balance_eval_pool is None:
        balance_eval_pool = bool(use_eval_split and getattr(CFG.data, "balance_eval_pool", False))
    cache_cfg = getattr(CFG.data, "cache", {})
    cache_enabled = bool(getattr(cache_cfg, "enabled", True))
    cache_path = _preprocessed_cache_path(
        use_eval_split=use_eval_split,
        max_docs=max_docs,
        balance_eval_pool=bool(balance_eval_pool),
        max_doc_tokens=int(max_doc_tokens),
        min_doc_tokens=int(min_doc_tokens),
    )
    if cache_enabled and cache_path.exists():
        docs = _read_docs_jsonl(cache_path)
        return docs, {"cache_hit": True, "cache_path": str(cache_path), "docs": len(docs)}

    # Oversample when a min-token filter is active so we end up with max_docs after dropping.
    _load_max = (max_docs * 2) if (max_docs is not None and int(min_doc_tokens) > 0) else max_docs
    docs = load_docs(
        use_eval_split=use_eval_split,
        max_docs=_load_max,
        balance_eval_pool=bool(balance_eval_pool),
    )
    n_short = truncate_documents_in_place(tokenizer=tokenizer, docs=docs, max_doc_tokens=int(max_doc_tokens))
    n_drop = drop_documents_shorter_than_min_tokens_in_place(tokenizer=tokenizer, docs=docs, min_doc_tokens=int(min_doc_tokens))
    if max_docs is not None and len(docs) > max_docs:
        docs = docs[:max_docs]

    if cache_enabled:
        _write_docs_jsonl(cache_path, docs)
    return docs, {
        "cache_hit": False,
        "cache_path": str(cache_path),
        "docs": len(docs),
        "n_shortened": int(n_short),
        "n_dropped_short": int(n_drop),
    }


def truncate_document_text(*, tokenizer, text: str, max_doc_tokens: int, sentence_lookback_words: int = 25) -> str:
    ids = tokenizer.encode(text=text, add_special_tokens=False)
    if len(ids) <= max_doc_tokens:
        return text
    truncated = tokenizer.decode(token_ids=ids[:max_doc_tokens], skip_special_tokens=True)
    # Try to end on a sentence boundary by scanning back up to `sentence_lookback_words` words.
    # Split on whitespace, scan backwards for a token ending in . ! or ?
    words = truncated.split()
    lookback = min(sentence_lookback_words, len(words) - 1)  # keep at least one word
    for i in range(len(words) - 1, len(words) - 1 - lookback, -1):
        if words[i][-1] in ".!?":
            return " ".join(words[: i + 1])
    return truncated


def truncate_documents_in_place(*, tokenizer, docs: list[dict], max_doc_tokens: int) -> int:
    n_shortened = 0
    for d in docs:
        before = d["text"]
        after = truncate_document_text(tokenizer=tokenizer, text=before, max_doc_tokens=max_doc_tokens)
        if after != before:
            n_shortened += 1
            d["text"] = after
    return n_shortened


def drop_documents_shorter_than_min_tokens_in_place(*, tokenizer, docs: list[dict], min_doc_tokens: int) -> int:
    """Drop docs whose tokenized text is shorter than min_doc_tokens."""
    mt = int(min_doc_tokens)
    if mt <= 0:
        return 0
    kept: list[dict] = []
    dropped = 0
    for d in docs:
        n_tok = len(tokenizer.encode(text=d["text"], add_special_tokens=False))
        if n_tok < mt:
            dropped += 1
            continue
        kept.append(d)
    docs[:] = kept
    return dropped


def iter_balanced_steps(docs: list[dict], docs_per_step: int = 4, seed: int | None = None):
    """Yield balanced AI/human batches with uniform sampling."""
    rng = random.Random(seed)
    ai_docs = [d for d in docs if d["label"] == 1]
    human_docs = [d for d in docs if d["label"] == 0]
    if not ai_docs or not human_docs:
        return

    n_ai = docs_per_step // 2
    n_human = docs_per_step - n_ai
    while True:
        batch_ai = rng.choices(ai_docs, k=n_ai)
        batch_human = rng.choices(human_docs, k=n_human)

        yield batch_ai + batch_human


def doc_stratum_key(doc: dict) -> tuple[str, str, int]:
    return (str(doc.get("dataset_id", "unknown")), str(doc.get("domain", "unknown")), int(doc["label"]))


def doc_ease_uid(doc: dict) -> str:
    """Stable id for rolling ease stats (format pass rate proxy)."""
    s = "|".join(
        (
            str(doc.get("dataset_id", "")),
            str(doc.get("domain", "")),
            str(int(doc["label"])),
            str(doc.get("text", "")[:4000]),
        )
    )
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()[:26]


def pick_stratum_probe_docs(docs: list[dict], samples_per_stratum: int, seed: int) -> list[dict]:
    """Pick up to N docs per stratum for startup probing."""
    rng = random.Random(seed + 19)
    pools: dict[tuple[str, str, int], list[dict]] = {}
    for d in docs:
        pools.setdefault(doc_stratum_key(d), []).append(d)
    keys = list(pools.keys())
    for k in keys:
        rng.shuffle(pools[k])
    if not keys or samples_per_stratum <= 0:
        return []
    picked: list[dict] = []
    n_each = int(samples_per_stratum)
    for k in keys:
        pool = pools[k]
        take = min(n_each, len(pool))
        picked.extend(pool[:take])
    rng.shuffle(picked)
    return picked


class StratumSampler:
    """UCB per (dataset_id, domain, label); optional linear blend from probe reward prior (continuous, per stratum)."""

    def __init__(
        self,
        docs: list[dict],
        docs_per_step: int = 4,
        seed: int | None = None,
        alpha: float = 0.05,
        beta: float = 2.0,
        ucb_c: float = 0.5,
        floor: float = 0.1,
        hardness_weight: float = 0.0,
        max_stratum_weight: float = 1.0,
        global_batch_offset: int = 0,
        initial_ema: dict[tuple[str, str, int], float] | None = None,
        initial_n_visits: dict[tuple[str, str, int], int] | None = None,
        probe_stratum_reward: dict[tuple[str, str, int], float] | None = None,
        probe_blend_ramp_steps: int = 0,
        probe_prior_beta: float = 6.0,
        curriculum_gaussian_gamma: float = 0.0,
        curriculum_reward_ema_alpha: float = 0.12,
        curriculum_tau_start: float = 0.25,
        curriculum_tau_end: float = 0.85,
        curriculum_ramp_steps: int = 1000,
    ):
        self.docs_per_step = docs_per_step
        self.alpha = alpha
        self.beta = beta
        self.ucb_c = ucb_c
        self.floor = floor
        self.hardness_weight = float(hardness_weight)
        self.max_stratum_weight = float(max_stratum_weight)
        self._rng = random.Random(seed)
        self._yield_count = 0
        self._global_batch_offset = int(global_batch_offset)
        self._last_probe_blend_lambda: float | None = None
        self._last_step_stratum_diag: dict[str, float] = {}
        self._last_batch_counts: dict[tuple[str, str, int], int] = {}
        self._sample_counts_total: dict[tuple[str, str, int], int] = {}
        self._probe_blend_ramp = max(0, int(probe_blend_ramp_steps))
        self._probe_prior_beta = float(probe_prior_beta)
        pr = probe_stratum_reward or {}
        self._probe_reward: dict[tuple[str, str, int], float] = {k: float(v) for k, v in pr.items()}
        self._probe_blend_active = self._probe_blend_ramp > 0 and bool(self._probe_reward)

        self._curriculum_gamma = max(0.0, float(curriculum_gaussian_gamma))
        self._reward_ema_alpha = float(curriculum_reward_ema_alpha)
        self._tau0 = float(curriculum_tau_start)
        self._tau1 = float(curriculum_tau_end)
        self._curriculum_ramp = max(1, int(curriculum_ramp_steps))
        self._reward_ema: dict[str, float] = {}
        self._curriculum_sample_diag: dict[str, float] = {}
        self._curriculum_reward_diag: dict[str, float] = {}

        self._ai_docs = [d for d in docs if d["label"] == 1]
        self._human_docs = [d for d in docs if d["label"] == 0]
        if not self._ai_docs or not self._human_docs:
            raise ValueError(
                f"StratumSampler needs both AI and human docs (n_ai={len(self._ai_docs)} n_human={len(self._human_docs)}); "
                "empty train pool or single-class filter."
            )

        self._ema: dict[tuple[str, str, int], float] = {}
        self._ema_var: dict[str, float] = {}  # per-stratum EMA of reward variance (keyed by uid string)
        self._n_visits: dict[tuple[str, str, int], int] = {}
        for d in docs:
            key = (str(d.get("dataset_id", "unknown")), str(d.get("domain", "unknown")), int(d["label"]))
            if key not in self._ema:
                self._ema[key] = 0.5
                self._n_visits[key] = 0
                self._sample_counts_total[key] = 0
                # init variance at 0.25 = max variance for [0,1] rewards (p=0.5); decays as we observe
                self._ema_var["|".join(str(x) for x in key)] = 0.25
        if initial_ema:
            for kk, vv in initial_ema.items():
                if kk in self._ema:
                    self._ema[kk] = float(vv)
                # also seed the curriculum EMA so the Gaussian has real difficulty estimates from step 0
                str_key = "|".join(str(x) for x in kk)
                self._reward_ema[str_key] = min(1.0, max(0.0, float(vv)))
        if initial_n_visits:
            for kk, vv in initial_n_visits.items():
                if kk in self._n_visits:
                    self._n_visits[kk] = int(vv)

    def _doc_key(self, doc: dict) -> tuple[str, str, int]:
        return doc_stratum_key(doc)

    def _softmax_floor(self, raw: dict[tuple[str, str, int], float], keys: list[tuple[str, str, int]]) -> dict[tuple[str, str, int], float]:
        n = len(keys)
        max_r = max(raw[k] for k in keys)
        exp_r = {k: math.exp(self.beta * (raw[k] - max_r)) for k in keys}
        total_exp = sum(exp_r.values())
        softmax_w = {k: exp_r[k] / total_exp for k in keys}
        floor_per = self.floor / n
        weights = {k: floor_per + (1.0 - self.floor) * softmax_w[k] for k in keys}
        if self.max_stratum_weight < 1.0:
            weights = {k: min(w, self.max_stratum_weight) for k, w in weights.items()}
            total = sum(weights.values())
            weights = {k: v / total for k, v in weights.items()}
        return weights

    def _ucb_raw(self, keys: list[tuple[str, str, int]]) -> dict[tuple[str, str, int], float]:
        # Variance-based UCB: favor strata where reward varies most (sometimes right, sometimes wrong)
        # = maximum learning potential. UCB exploration term covers under-visited strata.
        # ema_var initialized at 0.25 (max variance) so new strata are explored first.
        # hardness_weight bonus: max(0, 0.5 - ema) gives failing strata extra priority
        # (ema < 0.5 = model failing; ema > 0.5 = model passing → no bonus for easy strata)
        return {
            k: self._ema_var.get("|".join(str(x) for x in k), 0.25)
               + self.ucb_c / math.sqrt(self._n_visits.get(k, 0) + 1)
               + self.hardness_weight * max(0.0, 0.5 - self._ema.get(k, 0.5))
            for k in keys
        }

    def _probe_prior_softmax(self, keys: list[tuple[str, str, int]]) -> dict[tuple[str, str, int], float]:
        # upweight strata with higher measured training reward, continuous (no dataset bucketing)
        scores = {k: float(self._probe_reward.get(k, 0.5)) for k in keys}
        pb = self._probe_prior_beta
        max_s = max(scores.values())
        exp_m = {k: math.exp(pb * (scores[k] - max_s)) for k in keys}
        tot = sum(exp_m.values())
        return {k: exp_m[k] / tot for k in keys}

    def _reward_difficulty(self, uid: str) -> float:
        # high when historically low reward trust region [0,1]
        rr = float(self._reward_ema.get(uid, 0.5))
        rr = min(1.0, max(0.0, rr))
        return 1.0 - rr

    def _curriculum_tau(self, global_step: int) -> float:
        p = min(1.0, float(global_step) / float(self._curriculum_ramp))
        return self._tau0 + (self._tau1 - self._tau0) * p

    def _weights_ucb_then_gaussian_curriculum(
        self, pool: list[dict], base_w: list[float], global_step: int
    ) -> tuple[list[float], dict[str, float]]:
        if self._curriculum_gamma <= 0.0 or not pool:
            return base_w, {"curriculum_gaussian_disabled": 1.0}

        tau = self._curriculum_tau(global_step)
        gm = float(self._curriculum_gamma)
        gfs: list[float] = []
        dvs: list[float] = []
        for i, d in enumerate(pool):
            # use stratum-level UID (dataset|domain|label) so Gaussian curriculum
            # reads the same seeded EMA keys produced by the probe / sampler init
            str_key = "|".join(str(x) for x in doc_stratum_key(d))
            dv = self._reward_difficulty(str_key)
            dvs.append(dv)
            gfs.append(math.exp(-gm * (dv - tau) ** 2) * max(1e-9, float(base_w[i])))
        ts = sum(gfs)
        if ts <= 1e-12:
            return base_w, {"curriculum_tau": float(tau), "curriculum_fuse_degenerate": 1.0}
        out = [g / ts for g in gfs]
        m_d = sum(dvs) / len(dvs) if dvs else 0.0
        m_g_raw = math.exp(-gm * (m_d - tau) ** 2)
        diag = {
            "curriculum_tau": float(tau),
            "curriculum_gamma": float(gm),
            "curriculum_ramp_frac": float(min(1.0, global_step / float(self._curriculum_ramp))),
            "curriculum_pool_mean_d_historical": float(m_d),
            "curriculum_gauss_at_mean_d_vs_tau": float(m_g_raw),
        }
        return out, diag

    def ingest_curriculum_reward_rows(self, rows: list[dict]) -> None:
        """After step: u=stratum_key string, rw=doc mean curriculum signal (tier2 or total); EMA for Gaussian d=1-rw."""
        if not rows:
            self._curriculum_reward_diag = {}
            return
        a = float(self._reward_ema_alpha)
        # Group rewards by stratum to compute per-stratum step variance.
        from collections import defaultdict
        stratum_rws: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            uid = str(row["u"])
            rwcl = min(1.0, max(0.0, float(row["rw"])))
            stratum_rws[uid].append(rwcl)
            ol = float(self._reward_ema.get(uid, rwcl))
            self._reward_ema[uid] = min(1.0, max(0.0, (1.0 - a) * ol + a * rwcl))
        # Update variance EMA: step variance from this step's observations.
        for uid, rws in stratum_rws.items():
            mean_rw = sum(rws) / len(rws)
            if len(rws) > 1:
                step_var = sum((r - mean_rw) ** 2 for r in rws) / len(rws)
            else:
                # Single observation: use deviation from running mean as variance proxy.
                prev_mean = float(self._reward_ema.get(uid, mean_rw))
                step_var = (rws[0] - prev_mean) ** 2
            old_var = float(self._ema_var.get(uid, 0.25))
            self._ema_var[uid] = (1.0 - a) * old_var + a * step_var
        vs = list(self._reward_ema.values())
        ds = [1.0 - min(1.0, max(0.0, v)) for v in vs]
        self._curriculum_reward_diag = {
            "reward_ema_rows_n": float(len(rows)),
            "reward_ema_tracked_docs": float(len(self._reward_ema)),
            "reward_ema_mean": sum(vs) / len(vs) if vs else 0.0,
            "curriculum_hist_mean_difficulty_proxy": sum(ds) / len(ds) if ds else 0.0,
        }

    def last_curriculum_diag(self) -> dict[str, float]:
        return {**self._curriculum_sample_diag, **self._curriculum_reward_diag}

    def _weights(self, docs: list[dict], global_step: int) -> list[float]:
        if not docs:
            return []
        present_keys = list({self._doc_key(d) for d in docs})
        ucb_w = self._softmax_floor(self._ucb_raw(present_keys), present_keys)
        if not self._probe_blend_active:
            self._last_probe_blend_lambda = None
            return [ucb_w[self._doc_key(d)] for d in docs]
        lam = min(1.0, float(global_step) / float(self._probe_blend_ramp))
        self._last_probe_blend_lambda = lam
        pref = self._probe_prior_softmax(present_keys)
        blended = {k: (1.0 - lam) * pref[k] + lam * ucb_w[k] for k in present_keys}
        s_b = sum(blended.values())
        blended = {k: blended[k] / s_b for k in present_keys}
        return [blended[self._doc_key(d)] for d in docs]

    def sample_batch(self) -> list[dict]:
        gstep = self._global_batch_offset + self._yield_count
        n_ai = self.docs_per_step // 2
        n_human = self.docs_per_step - n_ai
        wt_ai = self._weights(self._ai_docs, gstep)
        wt_hum = self._weights(self._human_docs, gstep)

        fus_ai, d_ai = self._weights_ucb_then_gaussian_curriculum(self._ai_docs, wt_ai, gstep)
        fus_hum, d_hum = self._weights_ucb_then_gaussian_curriculum(self._human_docs, wt_hum, gstep)

        merged: dict[str, float] = {}
        merged.update({f"curriculum_ai_{k.removeprefix('curriculum_')}": float(v) for k, v in d_ai.items()})
        merged.update({f"curriculum_human_{k.removeprefix('curriculum_')}": float(v) for k, v in d_hum.items()})
        self._curriculum_sample_diag = merged

        batch_ai = self._rng.choices(self._ai_docs, weights=fus_ai, k=n_ai)
        batch_human = self._rng.choices(self._human_docs, weights=fus_hum, k=n_human)

        batch = batch_ai + batch_human

        step_counts: dict[tuple[str, str, int], int] = {}
        for d in batch:
            k = self._doc_key(d)
            step_counts[k] = step_counts.get(k, 0) + 1
            self._sample_counts_total[k] = self._sample_counts_total.get(k, 0) + 1
        self._last_batch_counts = step_counts
        self._yield_count += 1
        return batch

    def update(self, stratum_mean_rewards: dict[tuple[str, str, int], float]) -> None:
        deltas: list[float] = []
        for key, mean_reward in stratum_mean_rewards.items():
            if key not in self._ema:
                continue
            old = self._ema[key]
            self._ema[key] = (1.0 - self.alpha) * old + self.alpha * float(mean_reward)
            self._n_visits[key] += 1
            deltas.append(self._ema[key] - old)
        vals = list(self._ema.values())
        var_vals = list(self._ema_var.values())
        n_d = len(deltas)
        mean_delta = sum(deltas) / n_d if n_d else 0.0
        self._last_step_stratum_diag = {
            "stratum_min_ema": min(vals) if vals else 0.0,
            "stratum_max_ema": max(vals) if vals else 0.0,
            "stratum_mean_ema": sum(vals) / len(vals) if vals else 0.0,
            "stratum_mean_ema_delta": mean_delta,
            "stratum_min_ema_delta": min(deltas) if deltas else 0.0,
            "stratum_learning_index": mean_delta,
            "stratum_mean_var_ema": sum(var_vals) / len(var_vals) if var_vals else 0.0,
            "stratum_max_var_ema": max(var_vals) if var_vals else 0.0,
        }
        if self._last_probe_blend_lambda is not None:
            self._last_step_stratum_diag["stratum_probe_blend_lambda"] = float(self._last_probe_blend_lambda)

    def last_stratum_diag(self) -> dict[str, float]:
        return dict(self._last_step_stratum_diag)

    def stratum_emas(self) -> dict[tuple[str, str, int], float]:
        return dict(self._ema)

    def stratum_emas_flat_str(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for (ds, dom, lab), v in self._ema.items():
            out[f"{ds}::{dom}::lab{lab}"] = float(v)
        return out

    def stratum_weights(self) -> dict[tuple[str, str, int], float]:
        keys = list(self._ema.keys())
        if not keys:
            return {}
        return self._softmax_floor(self._ucb_raw(keys), keys)

    def last_batch_counts(self) -> dict[tuple[str, str, int], int]:
        return dict(self._last_batch_counts)

    def cumulative_sample_counts(self) -> dict[tuple[str, str, int], int]:
        return dict(self._sample_counts_total)

    def get_ucb_state(self) -> dict:
        """Return serializable UCB state (ema, n_visits, ema_var, reward_ema)."""
        return {
            "ema": {"|".join(str(x) for x in k): float(v) for k, v in self._ema.items()},
            "n_visits": {"|".join(str(x) for x in k): int(v) for k, v in self._n_visits.items()},
            "ema_var": dict(self._ema_var),
            "reward_ema": dict(self._reward_ema),
        }

    def set_ucb_state(self, state: dict) -> None:
        """Restore UCB state from a dict produced by get_ucb_state()."""
        def _parse_key(s: str) -> tuple[str, str, int]:
            parts = s.split("|")
            return (parts[0], parts[1], int(parts[2]))
        for sk, v in state.get("ema", {}).items():
            k = _parse_key(sk)
            if k in self._ema:
                self._ema[k] = float(v)
        for sk, v in state.get("n_visits", {}).items():
            k = _parse_key(sk)
            if k in self._n_visits:
                self._n_visits[k] = int(v)
        for sk, v in state.get("ema_var", {}).items():
            if sk in self._ema_var:
                self._ema_var[sk] = float(v)
        for sk, v in state.get("reward_ema", {}).items():
            self._reward_ema[sk] = float(v)

    def __iter__(self):
        while True:
            yield self.sample_batch()


class StratumReplaySampler:
    """UCB-over-strata fresh sampling combined with a prioritized replay pool and rollout caching.

    Fresh slots are filled by first selecting a stratum via UCB-weighted softmax (driven by
    per-stratum variance EMA and visit counts), then drawing a random doc from that stratum.
    Replay slots are drawn from the replay pool weighted by per-doc var_ema + reward_ema
    priority, identical to UniformReplaySampler.

    Stratum UCB state is updated via update() + ingest_curriculum_reward_rows() after each step.
    Doc-level replay state is updated via observe_docs() after each step.
    Both are active simultaneously — this is the "all in one" design.
    """

    def __init__(
        self,
        docs: list[dict],
        docs_per_step: int,
        seed: int,
        # UCB / stratum params
        alpha: float = 0.12,
        beta: float = 3.5,
        ucb_c: float = 0.5,
        floor: float = 0.1,
        hardness_weight: float = 0.0,
        max_stratum_weight: float = 1.0,
        global_batch_offset: int = 0,
        initial_ema: dict[tuple[str, str, int], float] | None = None,
        initial_n_visits: dict[tuple[str, str, int], int] | None = None,
        probe_stratum_reward: dict[tuple[str, str, int], float] | None = None,
        probe_blend_ramp_steps: int = 0,
        probe_prior_beta: float = 6.0,
        curriculum_gaussian_gamma: float = 0.0,
        curriculum_reward_ema_alpha: float = 0.12,
        curriculum_tau_start: float = 0.35,
        curriculum_tau_end: float = 0.55,
        curriculum_ramp_steps: int = 500,
        # Replay pool params
        replay_fraction_start: float = 0.35,
        replay_fraction_end: float = 0.60,
        replay_fraction_ramp_steps: int = 50,
        replay_min_count: int = 64,
        replay_pool_max_size: int = 6000,
        priority_var_weight: float = 0.80,
        priority_hard_weight: float = 0.20,
        priority_format_weight: float = 0.00,
        reward_ema_alpha: float = 0.30,
        monitor_top_k: int = 10,
        cache_rollouts: bool = False,
        max_rollout_reuses: int = 3,
        cache_pool_max_size: int = 800,
    ):
        self.docs_per_step = int(docs_per_step)
        self._rng = random.Random(int(seed))
        self._yield_count = 0
        self._global_batch_offset = int(global_batch_offset)
        self._last_batch_counts: dict[tuple[str, str, int], int] = {}
        self._sample_counts_total: dict[tuple[str, str, int], int] = {}

        # UCB params
        self.alpha = alpha
        self.beta = beta
        self._ucb_c = float(ucb_c)
        self.floor = floor
        self._hardness_weight = float(hardness_weight)
        self._max_stratum_weight = float(max_stratum_weight)
        self._probe_blend_ramp = max(0, int(probe_blend_ramp_steps))
        self._probe_prior_beta = float(probe_prior_beta)
        pr = probe_stratum_reward or {}
        self._probe_reward: dict[tuple[str, str, int], float] = {k: float(v) for k, v in pr.items()}
        self._probe_blend_active = self._probe_blend_ramp > 0 and bool(self._probe_reward)
        self._last_probe_blend_lambda: float | None = None

        # Curriculum params
        self._curriculum_gamma = max(0.0, float(curriculum_gaussian_gamma))
        self._curriculum_reward_ema_alpha = float(curriculum_reward_ema_alpha)
        self._tau0 = float(curriculum_tau_start)
        self._tau1 = float(curriculum_tau_end)
        self._curriculum_ramp = max(1, int(curriculum_ramp_steps))
        self._curriculum_sample_diag: dict[str, float] = {}
        self._curriculum_reward_diag: dict[str, float] = {}

        # Replay params
        self._replay_fraction_start = float(replay_fraction_start)
        self._replay_fraction_end = float(replay_fraction_end)
        self._replay_fraction_ramp_steps = max(1, int(replay_fraction_ramp_steps))
        self._replay_min_count = max(0, int(replay_min_count))
        self._replay_pool_max_size = max(1, int(replay_pool_max_size))
        self._priority_var_weight = float(priority_var_weight)
        self._priority_hard_weight = float(priority_hard_weight)
        self._priority_format_weight = float(priority_format_weight)
        self._reward_ema_alpha = float(reward_ema_alpha)
        self._monitor_top_k = max(1, int(monitor_top_k))
        self._cache_rollouts = bool(cache_rollouts)
        self._max_rollout_reuses = max(1, int(max_rollout_reuses))
        self._cache_pool_max_size = max(1, int(cache_pool_max_size))

        # Per-stratum UCB state (keyed by (dataset_id, domain, label) tuple)
        self._ema: dict[tuple[str, str, int], float] = {}
        self._ema_var: dict[str, float] = {}  # keyed by "|".join(str(x) for x in key)
        self._n_visits: dict[tuple[str, str, int], int] = {}
        self._reward_ema: dict[str, float] = {}  # for Gaussian curriculum, keyed by stratum str

        # Per-doc replay state
        self._replay_entries: dict[str, dict] = {}
        self._replay_order: list[str] = []
        self._cache_order: list[str] = []
        self._last_replay_uids: list[str] = []
        self._last_cached_rollouts: list[list[dict] | None] = []
        self._last_diag: dict[str, float] = {}
        self._last_step_stratum_diag: dict[str, float] = {}
        self._cache_hits_total: int = 0
        self._cache_evictions_total: int = 0

        self._ai_docs = [d for d in docs if int(d["label"]) == 1]
        self._human_docs = [d for d in docs if int(d["label"]) == 0]
        if not self._ai_docs or not self._human_docs:
            raise ValueError("StratumReplaySampler needs both AI and human docs")

        self._strata_ai: dict[tuple[str, str, int], list[dict]] = {}
        self._strata_human: dict[tuple[str, str, int], list[dict]] = {}
        for d in docs:
            k = doc_stratum_key(d)
            if int(d["label"]) == 1:
                self._strata_ai.setdefault(k, []).append(d)
            else:
                self._strata_human.setdefault(k, []).append(d)
            self._sample_counts_total[k] = 0
            if k not in self._ema:
                self._ema[k] = 0.5
                self._n_visits[k] = 0
                self._ema_var["|".join(str(x) for x in k)] = 0.25

        if initial_ema:
            for kk, vv in initial_ema.items():
                if kk in self._ema:
                    self._ema[kk] = float(vv)
                str_key = "|".join(str(x) for x in kk)
                self._reward_ema[str_key] = min(1.0, max(0.0, float(vv)))
        if initial_n_visits:
            for kk, vv in initial_n_visits.items():
                if kk in self._n_visits:
                    self._n_visits[kk] = int(vv)

    # --- UCB helpers ---

    def _softmax_floor(self, raw: dict[tuple[str, str, int], float], keys: list[tuple[str, str, int]]) -> dict[tuple[str, str, int], float]:
        n = len(keys)
        max_r = max(raw[k] for k in keys)
        exp_r = {k: math.exp(self.beta * (raw[k] - max_r)) for k in keys}
        total_exp = sum(exp_r.values())
        softmax_w = {k: exp_r[k] / total_exp for k in keys}
        floor_per = self.floor / n
        weights = {k: floor_per + (1.0 - self.floor) * softmax_w[k] for k in keys}
        if self._max_stratum_weight < 1.0:
            weights = {k: min(w, self._max_stratum_weight) for k, w in weights.items()}
            total = sum(weights.values())
            weights = {k: v / total for k, v in weights.items()}
        return weights

    def _ucb_raw(self, keys: list[tuple[str, str, int]]) -> dict[tuple[str, str, int], float]:
        return {
            k: self._ema_var.get("|".join(str(x) for x in k), 0.25)
               + self._ucb_c / math.sqrt(self._n_visits.get(k, 0) + 1)
               + self._hardness_weight * max(0.0, 0.5 - self._ema.get(k, 0.5))
            for k in keys
        }

    def _probe_prior_softmax(self, keys: list[tuple[str, str, int]]) -> dict[tuple[str, str, int], float]:
        scores = {k: float(self._probe_reward.get(k, 0.5)) for k in keys}
        pb = self._probe_prior_beta
        max_s = max(scores.values())
        exp_m = {k: math.exp(pb * (scores[k] - max_s)) for k in keys}
        tot = sum(exp_m.values())
        return {k: exp_m[k] / tot for k in keys}

    def _weights(self, strata_keys: list[tuple[str, str, int]], global_step: int) -> dict[tuple[str, str, int], float]:
        ucb_w = self._softmax_floor(self._ucb_raw(strata_keys), strata_keys)
        if not self._probe_blend_active:
            self._last_probe_blend_lambda = None
            return ucb_w
        lam = min(1.0, float(global_step) / float(self._probe_blend_ramp))
        self._last_probe_blend_lambda = lam
        pref = self._probe_prior_softmax(strata_keys)
        blended = {k: (1.0 - lam) * pref[k] + lam * ucb_w[k] for k in strata_keys}
        s_b = sum(blended.values())
        return {k: blended[k] / s_b for k in strata_keys}

    # --- Replay helpers ---

    def _replay_fraction(self, step: int) -> float:
        p = min(1.0, float(step) / float(self._replay_fraction_ramp_steps))
        return self._replay_fraction_start + (self._replay_fraction_end - self._replay_fraction_start) * p

    def _replay_priority(self, e: dict) -> float:
        mean_reward = float(e.get("reward_ema", 0.5))
        var_reward = float(e.get("var_ema", 0.0))
        fmt_pre = float(e.get("format_pre_ema", 1.0))
        hard = 1.0 - min(1.0, max(0.0, mean_reward))
        fmt_hard = 1.0 - min(1.0, max(0.0, fmt_pre))
        pri = (
            self._priority_var_weight * var_reward
            + self._priority_hard_weight * hard
            + self._priority_format_weight * fmt_hard
        )
        return max(1e-6, float(pri))

    def _consume_cached_rollouts(self, uid: str) -> list[dict] | None:
        if not self._cache_rollouts:
            return None
        e = self._replay_entries.get(uid)
        if not e:
            return None
        cached = e.get("cached_rollouts") or None
        if not cached:
            return None
        e["cached_use_count"] = int(e.get("cached_use_count", 0)) + 1
        self._cache_hits_total += 1
        if e["cached_use_count"] >= self._max_rollout_reuses:
            e["cached_rollouts"] = []
            e["cached_use_count"] = 0
            try:
                self._cache_order.remove(uid)
            except ValueError:
                pass
        return cached

    def _sample_replay(self, label: int, k: int) -> tuple[list[dict], list[str], list[list[dict] | None]]:
        if k <= 0:
            return [], [], []
        candidates = [e for e in self._replay_entries.values() if int(e.get("label", -1)) == int(label)]
        if len(candidates) < self._replay_min_count:
            return [], [], []
        weights = [self._replay_priority(e) for e in candidates]
        picked = self._rng.choices(candidates, weights=weights, k=k)
        docs_out = [e["doc"] for e in picked]
        uids_out = [str(e["uid"]) for e in picked]
        cached_out = [self._consume_cached_rollouts(u) for u in uids_out]
        return docs_out, uids_out, cached_out

    def _sample_fresh_ucb(self, label: int, k: int, global_step: int) -> list[dict]:
        if k <= 0:
            return []
        strata_dict = self._strata_ai if label == 1 else self._strata_human
        strata_keys = list(strata_dict.keys())
        if len(strata_keys) == 1:
            pool = strata_dict[strata_keys[0]]
            return [self._rng.choice(pool) for _ in range(k)]
        weights = self._weights(strata_keys, global_step)
        w_list = [weights[sk] for sk in strata_keys]
        out = []
        for _ in range(k):
            sk = self._rng.choices(strata_keys, weights=w_list)[0]
            out.append(self._rng.choice(strata_dict[sk]))
        return out

    # --- Public interface ---

    def sample_batch(self) -> list[dict]:
        step = self._yield_count
        gstep = self._global_batch_offset + step
        rf = self._replay_fraction(step)
        n_ai = self.docs_per_step // 2
        n_human = self.docs_per_step - n_ai

        n_ai_replay = int(round(n_ai * rf))
        n_hu_replay = int(round(n_human * rf))
        ai_replay, ai_replay_uids, ai_replay_cached = self._sample_replay(label=1, k=n_ai_replay)
        hu_replay, hu_replay_uids, hu_replay_cached = self._sample_replay(label=0, k=n_hu_replay)
        ai_fresh = self._sample_fresh_ucb(label=1, k=n_ai - len(ai_replay), global_step=gstep)
        hu_fresh = self._sample_fresh_ucb(label=0, k=n_human - len(hu_replay), global_step=gstep)

        batch = ai_replay + ai_fresh + hu_replay + hu_fresh
        self._last_replay_uids = ai_replay_uids + hu_replay_uids
        self._last_cached_rollouts = (
            ai_replay_cached + [None] * len(ai_fresh)
            + hu_replay_cached + [None] * len(hu_fresh)
        )

        step_counts: dict[tuple[str, str, int], int] = {}
        for d in batch:
            sk = doc_stratum_key(d)
            step_counts[sk] = step_counts.get(sk, 0) + 1
            self._sample_counts_total[sk] = self._sample_counts_total.get(sk, 0) + 1
        self._last_batch_counts = step_counts

        n_cache_hits = sum(1 for c in self._last_cached_rollouts if c)
        self._last_diag = {
            "sampler_replay_fraction_target": float(rf),
            "sampler_replay_pool_size": float(len(self._replay_entries)),
            "sampler_ai_replay_n": float(len(ai_replay)),
            "sampler_human_replay_n": float(len(hu_replay)),
            "sampler_ai_fresh_n": float(len(ai_fresh)),
            "sampler_human_fresh_n": float(len(hu_fresh)),
            "sampler_cached_rollout_hits": float(n_cache_hits),
            "sampler_cache_pool_size": float(len(self._cache_order)),
        }
        self._yield_count += 1
        return batch

    def last_cached_rollouts(self) -> list[list[dict] | None]:
        return list(self._last_cached_rollouts)

    def observe_docs(self, replay_rows: list[dict]) -> None:
        a = self._reward_ema_alpha
        for row in replay_rows:
            uid = str(row["uid"])
            rw = float(row.get("reward_mean", 0.0))
            fmt = float(row.get("format_rate_before_fixing", 0.0))
            doc = row["doc"]
            cached_rollouts = row.get("cached_rollouts") or []
            if uid not in self._replay_entries:
                self._replay_entries[uid] = {
                    "uid": uid,
                    "doc": doc,
                    "label": int(doc["label"]),
                    "reward_ema": rw,
                    "var_ema": 0.25,
                    "format_pre_ema": fmt,
                    "n": 0,
                    "cached_rollouts": [],
                    "cached_use_count": 0,
                }
                self._replay_order.append(uid)
            e = self._replay_entries[uid]
            old = float(e["reward_ema"])
            e["reward_ema"] = (1.0 - a) * old + a * rw
            if "reward_var" in row:
                within_step_var = float(row["reward_var"])
            else:
                within_step_var = (rw - old) ** 2
            e["var_ema"] = (1.0 - a) * float(e["var_ema"]) + a * within_step_var
            e["format_pre_ema"] = (1.0 - a) * float(e["format_pre_ema"]) + a * fmt
            e["doc"] = doc
            e["n"] = int(e.get("n", 0)) + 1
            if self._cache_rollouts and cached_rollouts:
                had_cache = bool(e.get("cached_rollouts"))
                e["cached_rollouts"] = list(cached_rollouts)
                e["cached_use_count"] = 0
                if not had_cache:
                    self._cache_order.append(uid)
                else:
                    try:
                        self._cache_order.remove(uid)
                    except ValueError:
                        pass
                    self._cache_order.append(uid)

        while len(self._replay_order) > self._replay_pool_max_size:
            old_uid = self._replay_order.pop(0)
            ent = self._replay_entries.pop(old_uid, None)
            if ent and ent.get("cached_rollouts"):
                try:
                    self._cache_order.remove(old_uid)
                except ValueError:
                    pass

        while len(self._cache_order) > self._cache_pool_max_size:
            evict_uid = self._cache_order.pop(0)
            self._cache_evictions_total += 1
            ev = self._replay_entries.get(evict_uid)
            if ev:
                ev["cached_rollouts"] = []
                ev["cached_use_count"] = 0

    def update(self, stratum_mean_rewards: dict[tuple[str, str, int], float]) -> None:
        """Update per-stratum reward EMA and visit counts — drives UCB weights for fresh sampling."""
        deltas: list[float] = []
        for key, mean_reward in stratum_mean_rewards.items():
            if key not in self._ema:
                continue
            old = self._ema[key]
            self._ema[key] = (1.0 - self.alpha) * old + self.alpha * float(mean_reward)
            self._n_visits[key] += 1
            deltas.append(self._ema[key] - old)
        vals = list(self._ema.values())
        var_vals = list(self._ema_var.values())
        n_d = len(deltas)
        mean_delta = sum(deltas) / n_d if n_d else 0.0
        self._last_step_stratum_diag = {
            "stratum_min_ema": min(vals) if vals else 0.0,
            "stratum_max_ema": max(vals) if vals else 0.0,
            "stratum_mean_ema": sum(vals) / len(vals) if vals else 0.0,
            "stratum_mean_ema_delta": mean_delta,
            "stratum_min_ema_delta": min(deltas) if deltas else 0.0,
            "stratum_learning_index": mean_delta,
            "stratum_mean_var_ema": sum(var_vals) / len(var_vals) if var_vals else 0.0,
            "stratum_max_var_ema": max(var_vals) if var_vals else 0.0,
        }
        if self._last_probe_blend_lambda is not None:
            self._last_step_stratum_diag["stratum_probe_blend_lambda"] = float(self._last_probe_blend_lambda)

    def ingest_curriculum_reward_rows(self, rows: list[dict]) -> None:
        """Update per-stratum variance EMA from step reward distribution — feeds UCB numerator."""
        if not rows:
            self._curriculum_reward_diag = {}
            return
        from collections import defaultdict
        a = float(self._curriculum_reward_ema_alpha)
        stratum_rws: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            uid = str(row["u"])
            rwcl = min(1.0, max(0.0, float(row["rw"])))
            stratum_rws[uid].append(rwcl)
            ol = float(self._reward_ema.get(uid, rwcl))
            self._reward_ema[uid] = min(1.0, max(0.0, (1.0 - a) * ol + a * rwcl))
        for uid, rws in stratum_rws.items():
            mean_rw = sum(rws) / len(rws)
            if len(rws) > 1:
                step_var = sum((r - mean_rw) ** 2 for r in rws) / len(rws)
            else:
                prev_mean = float(self._reward_ema.get(uid, mean_rw))
                step_var = (rws[0] - prev_mean) ** 2
            old_var = float(self._ema_var.get(uid, 0.25))
            self._ema_var[uid] = (1.0 - a) * old_var + a * step_var
        vs = list(self._reward_ema.values())
        ds = [1.0 - min(1.0, max(0.0, v)) for v in vs]
        self._curriculum_reward_diag = {
            "reward_ema_rows_n": float(len(rows)),
            "reward_ema_tracked_docs": float(len(self._reward_ema)),
            "reward_ema_mean": sum(vs) / len(vs) if vs else 0.0,
            "curriculum_hist_mean_difficulty_proxy": sum(ds) / len(ds) if ds else 0.0,
        }

    def replay_snapshot(self) -> dict:
        entries = list(self._replay_entries.values())
        ranked = sorted(entries, key=self._replay_priority, reverse=True)
        top = ranked[: self._monitor_top_k]
        return {
            "pool_size": len(entries),
            "selected_replay_uids": list(self._last_replay_uids),
            "top_priority": [
                {
                    "uid": str(e["uid"]),
                    "label": int(e["label"]),
                    "dataset_id": str(e["doc"].get("dataset_id", "unknown")),
                    "domain": str(e["doc"].get("domain", "unknown")),
                    "reward_ema": float(e["reward_ema"]),
                    "var_ema": float(e["var_ema"]),
                    "format_pre_ema": float(e["format_pre_ema"]),
                    "priority": float(self._replay_priority(e)),
                    "n_updates": int(e.get("n", 0)),
                }
                for e in top
            ],
        }

    def stratum_emas(self) -> dict[tuple[str, str, int], float]:
        return dict(self._ema)

    def stratum_emas_flat_str(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for (ds, dom, lab), v in self._ema.items():
            out[f"{ds}::{dom}::lab{lab}"] = float(v)
        return out

    def stratum_weights(self) -> dict[tuple[str, str, int], float]:
        keys = list(self._ema.keys())
        if not keys:
            return {}
        return self._softmax_floor(self._ucb_raw(keys), keys)

    def last_stratum_diag(self) -> dict[str, float]:
        return {**self._last_step_stratum_diag, **self._last_diag}

    def last_curriculum_diag(self) -> dict[str, float]:
        return {**self._curriculum_sample_diag, **self._curriculum_reward_diag}

    def last_batch_counts(self) -> dict[tuple[str, str, int], int]:
        return dict(self._last_batch_counts)

    def cumulative_sample_counts(self) -> dict[tuple[str, str, int], int]:
        return dict(self._sample_counts_total)

    def get_ucb_state(self) -> dict:
        """Return serializable UCB state (ema, n_visits, ema_var, reward_ema)."""
        return {
            "ema": {"|".join(str(x) for x in k): float(v) for k, v in self._ema.items()},
            "n_visits": {"|".join(str(x) for x in k): int(v) for k, v in self._n_visits.items()},
            "ema_var": dict(self._ema_var),
            "reward_ema": dict(self._reward_ema),
        }

    def set_ucb_state(self, state: dict) -> None:
        """Restore UCB state from a dict produced by get_ucb_state()."""
        def _parse_key(s: str) -> tuple[str, str, int]:
            parts = s.split("|")
            return (parts[0], parts[1], int(parts[2]))
        for sk, v in state.get("ema", {}).items():
            k = _parse_key(sk)
            if k in self._ema:
                self._ema[k] = float(v)
        for sk, v in state.get("n_visits", {}).items():
            k = _parse_key(sk)
            if k in self._n_visits:
                self._n_visits[k] = int(v)
        for sk, v in state.get("ema_var", {}).items():
            if sk in self._ema_var:
                self._ema_var[sk] = float(v)
        for sk, v in state.get("reward_ema", {}).items():
            self._reward_ema[sk] = float(v)

    def __iter__(self):
        while True:
            yield self.sample_batch()


class UniformReplaySampler:
    """Uniform-over-strata fresh sampling + prioritized replay mixture.

    Optionally caches the rollouts (tokens + sampler logprobs + advantages) of past
    docs so a replay pick can skip resampling and scoring entirely. PPO clip
    handles the policy drift via the IS ratio between cached old logprobs and current
    model logprobs (recomputed by Tinker during forward_backward). Cached groups are
    dropped after `max_rollout_reuses` replays.
    """

    def __init__(
        self,
        docs: list[dict],
        docs_per_step: int,
        seed: int,
        replay_fraction_start: float,
        replay_fraction_end: float,
        replay_fraction_ramp_steps: int,
        replay_min_count: int,
        replay_pool_max_size: int,
        priority_var_weight: float,
        priority_hard_weight: float,
        priority_format_weight: float,
        reward_ema_alpha: float,
        monitor_top_k: int = 10,
        cache_rollouts: bool = False,
        max_rollout_reuses: int = 3,
        cache_pool_max_size: int = 800,
    ):
        self.docs_per_step = int(docs_per_step)
        self._rng = random.Random(int(seed))
        self._yield_count = 0
        self._last_batch_counts: dict[tuple[str, str, int], int] = {}
        self._sample_counts_total: dict[tuple[str, str, int], int] = {}

        self._replay_fraction_start = float(replay_fraction_start)
        self._replay_fraction_end = float(replay_fraction_end)
        self._replay_fraction_ramp_steps = max(1, int(replay_fraction_ramp_steps))
        self._replay_min_count = max(0, int(replay_min_count))
        self._replay_pool_max_size = max(1, int(replay_pool_max_size))
        self._priority_var_weight = float(priority_var_weight)
        self._priority_hard_weight = float(priority_hard_weight)
        self._priority_format_weight = float(priority_format_weight)
        self._reward_ema_alpha = float(reward_ema_alpha)
        self._monitor_top_k = max(1, int(monitor_top_k))

        self._cache_rollouts = bool(cache_rollouts)
        self._max_rollout_reuses = max(1, int(max_rollout_reuses))
        self._cache_pool_max_size = max(1, int(cache_pool_max_size))

        self._all_docs = docs
        self._ai_docs = [d for d in docs if int(d["label"]) == 1]
        self._human_docs = [d for d in docs if int(d["label"]) == 0]
        if not self._ai_docs or not self._human_docs:
            raise ValueError("UniformReplaySampler needs both AI and human docs")

        self._strata_ai: dict[tuple[str, str, int], list[dict]] = {}
        self._strata_human: dict[tuple[str, str, int], list[dict]] = {}
        for d in docs:
            k = doc_stratum_key(d)
            if int(d["label"]) == 1:
                self._strata_ai.setdefault(k, []).append(d)
            else:
                self._strata_human.setdefault(k, []).append(d)
            self._sample_counts_total[k] = 0

        self._replay_entries: dict[str, dict] = {}
        self._replay_order: list[str] = []
        # uids whose cached rollouts are populated, in FIFO order — bounded by
        # cache_pool_max_size to cap memory (~hundreds of MB per group)
        self._cache_order: list[str] = []
        self._last_replay_uids: list[str] = []
        # parallel array to last sample_batch(): cached rollouts for each doc, or None.
        # Kept in sample_batch() return order so train_step can zip with docs.
        self._last_cached_rollouts: list[list[dict] | None] = []
        self._last_diag: dict[str, float] = {}
        # diagnostics
        self._cache_hits_total: int = 0
        self._cache_evictions_total: int = 0

    def _replay_fraction(self, step: int) -> float:
        p = min(1.0, float(step) / float(self._replay_fraction_ramp_steps))
        return self._replay_fraction_start + (self._replay_fraction_end - self._replay_fraction_start) * p

    def _sample_fresh_uniform(self, label: int, k: int) -> list[dict]:
        if k <= 0:
            return []
        strata = list(self._strata_ai.keys()) if label == 1 else list(self._strata_human.keys())
        out: list[dict] = []
        for _ in range(k):
            sk = self._rng.choice(strata)
            pool = self._strata_ai[sk] if label == 1 else self._strata_human[sk]
            out.append(self._rng.choice(pool))
        return out

    def _replay_priority(self, e: dict) -> float:
        mean_reward = float(e.get("reward_ema", 0.5))
        var_reward = float(e.get("var_ema", 0.0))
        fmt_pre = float(e.get("format_pre_ema", 1.0))
        hard = 1.0 - min(1.0, max(0.0, mean_reward))
        fmt_hard = 1.0 - min(1.0, max(0.0, fmt_pre))
        pri = (
            self._priority_var_weight * var_reward
            + self._priority_hard_weight * hard
            + self._priority_format_weight * fmt_hard
        )
        return max(1e-6, float(pri))

    def _consume_cached_rollouts(self, uid: str) -> list[dict] | None:
        """Pop and return cached rollouts for `uid` (or None). Increments use counter
        on returned rollouts; drops the entry's cache when max_rollout_reuses is hit."""
        if not self._cache_rollouts:
            return None
        e = self._replay_entries.get(uid)
        if not e:
            return None
        cached = e.get("cached_rollouts") or None
        if not cached:
            return None
        e["cached_use_count"] = int(e.get("cached_use_count", 0)) + 1
        self._cache_hits_total += 1
        if e["cached_use_count"] >= self._max_rollout_reuses:
            e["cached_rollouts"] = []
            e["cached_use_count"] = 0
            try:
                self._cache_order.remove(uid)
            except ValueError:
                pass
        return cached

    def _sample_replay(self, label: int, k: int) -> tuple[list[dict], list[str], list[list[dict] | None]]:
        if k <= 0:
            return [], [], []
        candidates = [e for e in self._replay_entries.values() if int(e.get("label", -1)) == int(label)]
        if len(candidates) < self._replay_min_count:
            return [], [], []
        weights = [self._replay_priority(e) for e in candidates]
        picked = self._rng.choices(candidates, weights=weights, k=k)
        docs_out = [e["doc"] for e in picked]
        uids_out = [str(e["uid"]) for e in picked]
        cached_out = [self._consume_cached_rollouts(u) for u in uids_out]
        return docs_out, uids_out, cached_out

    def sample_batch(self) -> list[dict]:
        step = self._yield_count
        rf = self._replay_fraction(step)
        n_ai = self.docs_per_step // 2
        n_human = self.docs_per_step - n_ai

        n_ai_replay = int(round(n_ai * rf))
        n_hu_replay = int(round(n_human * rf))
        ai_replay, ai_replay_uids, ai_replay_cached = self._sample_replay(label=1, k=n_ai_replay)
        hu_replay, hu_replay_uids, hu_replay_cached = self._sample_replay(label=0, k=n_hu_replay)
        ai_fresh = self._sample_fresh_uniform(label=1, k=n_ai - len(ai_replay))
        hu_fresh = self._sample_fresh_uniform(label=0, k=n_human - len(hu_replay))

        batch = ai_replay + ai_fresh + hu_replay + hu_fresh
        self._last_replay_uids = ai_replay_uids + hu_replay_uids
        # parallel to batch: cached rollouts (or None for fresh slots and replay slots without cache)
        self._last_cached_rollouts = (
            ai_replay_cached + [None] * len(ai_fresh)
            + hu_replay_cached + [None] * len(hu_fresh)
        )
        step_counts: dict[tuple[str, str, int], int] = {}
        for d in batch:
            sk = doc_stratum_key(d)
            step_counts[sk] = step_counts.get(sk, 0) + 1
            self._sample_counts_total[sk] = self._sample_counts_total.get(sk, 0) + 1
        self._last_batch_counts = step_counts
        n_cache_hits = sum(1 for c in self._last_cached_rollouts if c)
        self._last_diag = {
            "sampler_replay_fraction_target": float(rf),
            "sampler_replay_pool_size": float(len(self._replay_entries)),
            "sampler_ai_replay_n": float(len(ai_replay)),
            "sampler_human_replay_n": float(len(hu_replay)),
            "sampler_ai_fresh_n": float(len(ai_fresh)),
            "sampler_human_fresh_n": float(len(hu_fresh)),
            "sampler_cached_rollout_hits": float(n_cache_hits),
            "sampler_cache_pool_size": float(len(self._cache_order)),
        }
        self._yield_count += 1
        return batch

    def last_cached_rollouts(self) -> list[list[dict] | None]:
        """Per-doc cached rollouts for the most recently sampled batch (None where fresh)."""
        return list(self._last_cached_rollouts)

    def observe_docs(self, replay_rows: list[dict]) -> None:
        """Update reward/format EMAs from a batch of doc results. If `cached_rollouts`
        is present on a row and rollout caching is enabled, store them on the entry so
        future replays can skip resampling."""
        a = self._reward_ema_alpha
        for row in replay_rows:
            uid = str(row["uid"])
            rw = float(row.get("reward_mean", 0.0))
            fmt = float(row.get("format_rate_before_fixing", 0.0))
            doc = row["doc"]
            cached_rollouts = row.get("cached_rollouts") or []
            if uid not in self._replay_entries:
                self._replay_entries[uid] = {
                    "uid": uid,
                    "doc": doc,
                    "label": int(doc["label"]),
                    "reward_ema": rw,
                    "var_ema": 0.25,
                    "format_pre_ema": fmt,
                    "n": 0,
                    "cached_rollouts": [],
                    "cached_use_count": 0,
                }
                self._replay_order.append(uid)
            e = self._replay_entries[uid]
            old = float(e["reward_ema"])
            e["reward_ema"] = (1.0 - a) * old + a * rw
            # Use within-step rollout variance (spread across K rollouts) when available.
            # This is the correct signal for GRPO: docs where some rollouts succeed and
            # some fail produce the highest gradient. Between-step delta ((rw-old)^2) is
            # a poor proxy — it's high for transitioning docs but zero for docs stuck at
            # a consistent ~0.5 mean, which are actually the most valuable GRPO examples.
            if "reward_var" in row:
                within_step_var = float(row["reward_var"])
            else:
                within_step_var = (rw - old) ** 2
            e["var_ema"] = (1.0 - a) * float(e["var_ema"]) + a * within_step_var
            e["format_pre_ema"] = (1.0 - a) * float(e["format_pre_ema"]) + a * fmt
            e["doc"] = doc
            e["n"] = int(e.get("n", 0)) + 1
            if self._cache_rollouts and cached_rollouts:
                # Replace any prior cached group: stale logprobs hurt more than they help once
                # the policy has moved. We trust the latest sample to have the freshest IS-ratio
                # baseline; PPO clip protects against the residual drift across reuses.
                had_cache = bool(e.get("cached_rollouts"))
                e["cached_rollouts"] = list(cached_rollouts)
                e["cached_use_count"] = 0
                if not had_cache:
                    self._cache_order.append(uid)
                else:
                    try:
                        self._cache_order.remove(uid)
                    except ValueError:
                        pass
                    self._cache_order.append(uid)

        while len(self._replay_order) > self._replay_pool_max_size:
            old_uid = self._replay_order.pop(0)
            ent = self._replay_entries.pop(old_uid, None)
            if ent and ent.get("cached_rollouts"):
                try:
                    self._cache_order.remove(old_uid)
                except ValueError:
                    pass

        while len(self._cache_order) > self._cache_pool_max_size:
            evict_uid = self._cache_order.pop(0)
            self._cache_evictions_total += 1
            ev = self._replay_entries.get(evict_uid)
            if ev:
                ev["cached_rollouts"] = []
                ev["cached_use_count"] = 0

    def last_stratum_diag(self) -> dict[str, float]:
        return dict(self._last_diag)

    def replay_snapshot(self) -> dict:
        entries = list(self._replay_entries.values())
        ranked = sorted(entries, key=self._replay_priority, reverse=True)
        top = ranked[: self._monitor_top_k]
        return {
            "pool_size": len(entries),
            "selected_replay_uids": list(self._last_replay_uids),
            "top_priority": [
                {
                    "uid": str(e["uid"]),
                    "label": int(e["label"]),
                    "dataset_id": str(e["doc"].get("dataset_id", "unknown")),
                    "domain": str(e["doc"].get("domain", "unknown")),
                    "reward_ema": float(e["reward_ema"]),
                    "var_ema": float(e["var_ema"]),
                    "format_pre_ema": float(e["format_pre_ema"]),
                    "priority": float(self._replay_priority(e)),
                    "n_updates": int(e.get("n", 0)),
                }
                for e in top
            ],
        }

    def last_curriculum_diag(self) -> dict[str, float]:
        return {}

    def last_batch_counts(self) -> dict[tuple[str, str, int], int]:
        return dict(self._last_batch_counts)

    def cumulative_sample_counts(self) -> dict[tuple[str, str, int], int]:
        return dict(self._sample_counts_total)

    def stratum_emas(self) -> dict[tuple[str, str, int], float]:
        return {}

    def stratum_weights(self) -> dict[tuple[str, str, int], float]:
        return {}

    def ingest_curriculum_reward_rows(self, rows: list[dict]) -> None:
        _ = rows

    def update(self, stratum_mean_rewards: dict[tuple[str, str, int], float]) -> None:
        _ = stratum_mean_rewards

    def __iter__(self):
        while True:
            yield self.sample_batch()
