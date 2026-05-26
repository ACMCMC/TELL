"""
Smoke test: two training runs with the same seed must produce byte-identical
metrics after 2 steps.

All external services (Tinker SDK, HuggingFace datasets, W&B, Weave,
AutoTokenizer) are replaced with lightweight deterministic mocks so the test
is self-contained and fast.
"""
import asyncio
import io
import random
import sys
import types
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Build mock modules that must exist in sys.modules BEFORE any rl_detector
# import so that top-level `import tinker` etc. don't fail.
# ---------------------------------------------------------------------------

def _make_tinker_module():
    mod = types.ModuleType("tinker")

    class SamplingParams:
        def __init__(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)

    class ModelInput:
        def __init__(self, tokens):
            self.tokens = list(tokens)

        @classmethod
        def from_ints(cls, tokens):
            return cls(tokens)

    class TensorData:
        def __init__(self, tensor):
            self._t = tensor

        @classmethod
        def from_torch(cls, t):
            return cls(t)

    class Datum:
        def __init__(self, model_input, loss_fn_inputs):
            self.model_input = model_input
            self.loss_fn_inputs = loss_fn_inputs

    class AdamParams:
        def __init__(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)

    class ServiceClient:
        pass  # replaced per-test

    mod.SamplingParams = SamplingParams
    mod.ModelInput = ModelInput
    mod.TensorData = TensorData
    mod.Datum = Datum
    mod.AdamParams = AdamParams
    mod.ServiceClient = ServiceClient
    return mod


def _make_weave_module():
    mod = types.ModuleType("weave")
    mod.init = lambda *a, **kw: None
    mod.finish = lambda *a, **kw: None

    def op(fn=None, **kw):
        if fn is not None:
            return fn
        return lambda f: f

    mod.op = op
    return mod


def _make_wandb_module():
    mod = types.ModuleType("wandb")
    mod.init = lambda *a, **kw: None
    mod.log = lambda *a, **kw: None
    mod.finish = lambda *a, **kw: None

    class Histogram:
        def __init__(self, data):
            pass

    mod.Histogram = Histogram
    return mod


def _make_dotenv_module():
    mod = types.ModuleType("dotenv")
    mod.load_dotenv = lambda *a, **kw: None
    return mod


def _make_transformers_module():
    mod = types.ModuleType("transformers")

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(name, *a, **kw):
            return _MockTokenizer()

    mod.AutoTokenizer = AutoTokenizer
    return mod


def _make_openai_module():
    mod = types.ModuleType("openai")

    class AsyncOpenAI:
        def __init__(self, *a, **kw):
            pass

    class APIConnectionError(Exception):
        pass

    class InternalServerError(Exception):
        pass

    class RateLimitError(Exception):
        pass

    class PermissionDeniedError(Exception):
        pass

    mod.AsyncOpenAI = AsyncOpenAI
    mod.APIConnectionError = APIConnectionError
    mod.InternalServerError = InternalServerError
    mod.RateLimitError = RateLimitError
    mod.PermissionDeniedError = PermissionDeniedError
    return mod


# Install mocks for packages that are NOT installed in this environment.
# Never override packages that ARE installed (e.g. tqdm, sklearn, torch).
for _name, _maker in [
    ("tinker", _make_tinker_module),
    ("weave", _make_weave_module),
    ("wandb", _make_wandb_module),
    ("transformers", _make_transformers_module),
    ("openai", _make_openai_module),
]:
    if _name not in sys.modules:
        sys.modules[_name] = _maker()

# ---------------------------------------------------------------------------
# Deterministic mock tokenizer
# ---------------------------------------------------------------------------

class _MockTokenizer:
    """Maps each unique character to a small integer; decodes back to a
    fixed non-matching string so format validation always fails cleanly."""

    def encode(self, text, add_special_tokens=True):
        return [ord(c) % 8000 + 1000 for c in text[:40]] + [1]

    def decode(self, tokens, skip_special_tokens=True):
        # Return something that does NOT match any document so format_ok=False
        # and reward=0.  Two runs with the same seed get the same zero reward.
        seed = sum(t * (i + 1) for i, t in enumerate(tokens[:8])) % (2 ** 31)
        rng = random.Random(seed)
        return f"<mock>{rng.randint(0, 9999)}</mock>"

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True, **kwargs):
        return messages[0]["content"]


# ---------------------------------------------------------------------------
# Deterministic mock Tinker clients
# ---------------------------------------------------------------------------

class _MockFuture:
    def __init__(self, value=None):
        self._v = value

    async def result_async(self):
        return self._v


class _MockSamplingClient:
    """Returns tokens that are purely a function of the sampling seed."""

    async def sample_async(self, prompt, num_samples, sampling_params):
        seed = getattr(sampling_params, "seed", 0) or 0
        rng = random.Random(seed)
        n = 12
        tokens = [rng.randint(100, 49999) for _ in range(n)]
        logprobs = [rng.uniform(-3.5, -0.05) for _ in range(n)]

        seq = types.SimpleNamespace(tokens=tokens, logprobs=logprobs)
        return types.SimpleNamespace(sequences=[seq])

    async def compute_logprobs_async(self, model_input):
        tokens = list(model_input.tokens)
        seed = sum(t * (i + 1) for i, t in enumerate(tokens[:8])) % (2 ** 31)
        rng = random.Random(seed)
        return [rng.uniform(-3.5, -0.05) for _ in range(len(tokens))]


class _MockTrainingClient:
    async def save_weights_and_get_sampling_client_async(self):
        return _MockSamplingClient()

    async def forward_backward_async(self, data, loss_fn, loss_fn_config=None):
        result = types.SimpleNamespace(loss=0.0)
        return _MockFuture(result)

    async def optim_step_async(self, params):
        return _MockFuture()

    async def save_state_async(self, name, ttl_seconds=None):
        return _MockFuture("mock://ckpt/" + name)

    async def create_training_client_from_state_with_optimizer_async(self, path):
        return self


# ---------------------------------------------------------------------------
# Fixed document pool (replaces HuggingFace dataset loading)
# ---------------------------------------------------------------------------

_FIXED_DOCS = (
    [{"text": f"AI sample document number {i} with artificial phrases.", "label": 1}
     for i in range(20)]
    + [{"text": f"Human written text document {i} with natural prose.", "label": 0}
       for i in range(20)]
)


# ---------------------------------------------------------------------------
# Helper: run N training steps, return list of metrics dicts
# ---------------------------------------------------------------------------

async def _run_steps(n_steps: int) -> list[dict]:
    """Import train inside the function so mocks are already in sys.modules."""
    import importlib
    import rl_detector.train as train_mod
    importlib.reload(train_mod)          # reset module-level state between calls

    from rl_detector.config import CFG

    # Seed everything exactly as production code does.
    import os, numpy as np, torch, random as _random
    seed = int(getattr(CFG.frozen, "seed", 2242))
    os.environ["PYTHONHASHSEED"] = str(seed)
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    tokenizer = _MockTokenizer()
    training_client = _MockTrainingClient()

    # Use a small subset of the fixed doc pool for speed.
    docs_per_step = int(getattr(CFG.training, "docs_per_step", 4))
    all_docs = _FIXED_DOCS[:]

    results = []
    with open(os.devnull, "w") as null_log:
        from rl_detector.data import iter_balanced_steps
        steps_iter = iter_balanced_steps(all_docs, docs_per_step=docs_per_step, seed=seed)
        for step in range(n_steps):
            docs = next(steps_iter)
            metrics = await train_mod.train_step(
                training_client=training_client,
                tokenizer=tokenizer,
                docs=docs,
                step=step,
                audit_log=null_log,
            )
            results.append(metrics)
    return results


# ---------------------------------------------------------------------------
# The actual smoke test
# ---------------------------------------------------------------------------

def test_training_is_reproducible():
    """Two identical runs must produce exactly the same metrics for each step."""
    run1 = asyncio.run(_run_steps(2))
    run2 = asyncio.run(_run_steps(2))

    assert len(run1) == len(run2) == 2, "Expected 2 steps each run"

    # Compare all training metrics (excluding wall-clock timings which are inherently variable).
    sample_keys = run1[0].keys()
    COMPARE_KEYS = [
        k for k in sample_keys
        if not k.startswith("timing_") and not k.startswith("_")
    ]
    assert COMPARE_KEYS, "No metric keys found — training pipeline did not produce output"

    for step_idx, (m1, m2) in enumerate(zip(run1, run2)):
        for key in COMPARE_KEYS:
            v1 = m1.get(key)
            v2 = m2.get(key)
            assert v1 == v2, (
                f"Step {step_idx}: metric '{key}' differs between runs: "
                f"run1={v1!r}  run2={v2!r}"
            )
