from __future__ import annotations

import os
import sys
import types

import numpy as np

from .base import DetectorWrapper
from detectors_bench.registry import vendor_path
from detectors_bench.schemas import Example, Prediction


class GhostbusterWrapper(DetectorWrapper):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.root = vendor_path(cfg)
        self.openai_key_env = cfg.get("openai_key_env", "OPENAI_API_KEY")
        self.original_ada_model = cfg.get("original_ada_model", "ada")
        self.original_davinci_model = cfg.get("original_davinci_model", "davinci")
        self.compatibility_variant = cfg.get("compatibility_variant", "original")
        self.ada_model = cfg.get("ada_model", "ada")
        self.davinci_model = cfg.get("davinci_model", "davinci")
        self._loaded = False

    def _load_official_components(self) -> None:
        if self._loaded:
            return
        import dill as pickle
        import openai
        import tiktoken

        # Avoid vendor utils/__init__.py because it imports LLaMA helpers that
        # require gated Meta checkpoints. The official classifier itself uses
        # only featurize, symbolic, and n_gram.
        utils_dir = self.root / "utils"
        utils_pkg = types.ModuleType("utils")
        utils_pkg.__path__ = [str(utils_dir)]  # type: ignore[attr-defined]
        sys.modules.setdefault("utils", utils_pkg)
        sys.path.insert(0, str(self.root))

        from utils.featurize import score_ngram, t_featurize_logprobs  # type: ignore
        from utils.symbolic import get_words, scalar_functions, train_trigram, vec_functions  # type: ignore

        self.openai = openai
        self.enc = tiktoken.encoding_for_model("davinci")
        self.score_ngram = score_ngram
        self.t_featurize_logprobs = t_featurize_logprobs
        self.get_words = get_words
        self.scalar_functions = scalar_functions
        self.vec_functions = vec_functions
        self.best_features = (self.root / "model" / "features.txt").read_text(encoding="utf-8").strip().split("\n")
        self.model = pickle.load(open(self.root / "model" / "model", "rb"))
        self.mu = pickle.load(open(self.root / "model" / "mu", "rb"))
        self.sigma = pickle.load(open(self.root / "model" / "sigma", "rb"))
        self.trigram_model = train_trigram(verbose=False)
        self._loaded = True

    @staticmethod
    def _align_feature_streams(
        ada: np.ndarray,
        davinci: np.ndarray,
        subwords: list[str],
        trigram: np.ndarray,
        unigram: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray, np.ndarray, int]:
        # The original Ghostbuster feature extractor used retired OpenAI models
        # whose echoed token streams aligned. The replacement Completion models
        # can tokenize the same prompt differently, so align by the common prefix
        # before applying the official vector feature algebra.
        n = min(len(ada), len(davinci), len(subwords), len(trigram), len(unigram))
        if n == 0:
            raise RuntimeError("Ghostbuster received empty aligned logprob streams.")
        return ada[:n], davinci[:n], subwords[:n], trigram[:n], unigram[:n], n

    def _openai_logprobs(self, model: str, doc: str) -> tuple[np.ndarray, list[str]]:
        response = self.openai.Completion.create(
            model=model,
            prompt="<|endoftext|>" + doc,
            max_tokens=0,
            echo=True,
            logprobs=1,
        )
        choice = response["choices"][0]
        probs = np.array([np.exp(x) for x in choice["logprobs"]["token_logprobs"][1:]])
        tokens = list(choice["logprobs"]["tokens"][1:])
        return probs, tokens

    def predict_one(self, ex: Example) -> Prediction:
        key = os.environ.get(self.openai_key_env, "")
        if not key:
            raise RuntimeError(f"Ghostbuster official classifier requires {self.openai_key_env}.")
        self._load_official_components()
        self.openai.api_key = key
        doc = ex.text.strip()
        tokens = self.enc.encode(doc)[:2047]
        doc = self.enc.decode(tokens).strip()
        trigram = np.array(self.score_ngram(doc, self.trigram_model, self.enc.encode, n=3, strip_first=False))
        unigram = np.array(self.score_ngram(doc, self.trigram_model.base, self.enc.encode, n=1, strip_first=False))
        ada, _ = self._openai_logprobs(self.ada_model, doc)
        davinci, subwords = self._openai_logprobs(self.davinci_model, doc)
        subwords = [token.replace("\n", "Ċ").replace("\t", "ĉ").replace(" ", "Ġ") for token in subwords]
        raw_lengths = {
            "ada": len(ada),
            "davinci": len(davinci),
            "subwords": len(subwords),
            "trigram": len(trigram),
            "unigram": len(unigram),
        }
        ada, davinci, subwords, trigram, unigram, aligned_len = self._align_feature_streams(
            ada,
            davinci,
            subwords,
            trigram,
            unigram,
        )

        t_features = self.t_featurize_logprobs(davinci, ada, subwords)
        vector_map = {
            "davinci-logprobs": davinci,
            "ada-logprobs": ada,
            "trigram-logprobs": trigram,
            "unigram-logprobs": unigram,
        }

        exp_features = []
        for exp in self.best_features:
            exp_tokens = self.get_words(exp)
            curr = vector_map[exp_tokens[0]]
            i = 1
            while i < len(exp_tokens):
                token = exp_tokens[i]
                if token in self.vec_functions:
                    curr = self.vec_functions[token](curr, vector_map[exp_tokens[i + 1]])
                    i += 2
                elif token in self.scalar_functions:
                    exp_features.append(self.scalar_functions[token](curr))
                    break
                else:
                    i += 1

        data = (np.array(t_features + exp_features) - self.mu) / self.sigma
        score = float(self.model.predict_proba(data.reshape(-1, 1).T)[:, 1][0])
        return Prediction(
            id=ex.id,
            detector=self.name,
            score_ai=score,
            raw_score=score,
            raw_label="machine-generated" if score >= 0.5 else "human-written",
            pred_builtin=int(score >= 0.5),
            features={
                "num_tokens": len(tokens),
                "num_symbolic_features": len(exp_features),
                "compatibility_variant": self.compatibility_variant,
                "original_ada_model": self.original_ada_model,
                "original_davinci_model": self.original_davinci_model,
                "ada_model": self.ada_model,
                "davinci_model": self.davinci_model,
                "uses_official_classifier_artifacts": True,
                "aligned_feature_tokens": aligned_len,
                "raw_stream_lengths": raw_lengths,
            },
        )
