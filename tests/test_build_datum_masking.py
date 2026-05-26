"""build_datum: per-role advantages scaled by task weights (baked into advantages for PPO)."""

import sys
import math

_TOL = 1e-5


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=_TOL, abs_tol=_TOL)


def _close_list(xs, ys):
    return len(xs) == len(ys) and all(_close(a, b) for a, b in zip(xs, ys))


from test_reproducibility import (
    _make_dotenv_module,
    _make_openai_module,
    _make_tinker_module,
    _make_transformers_module,
    _make_wandb_module,
    _make_weave_module,
)

for _name, _maker in [
    ("tinker", _make_tinker_module),
    ("weave", _make_weave_module),
    ("wandb", _make_wandb_module),
    ("transformers", _make_transformers_module),
    ("openai", _make_openai_module),
    ("dotenv", _make_dotenv_module),
]:
    if _name not in sys.modules:
        sys.modules[_name] = _maker()

from rl_detector.prompt_utils import (
    ANN_SPECIAL_ID_ANN_PREFIX,
    ANN_SPECIAL_ID_CLOSE,
    ANN_SPECIAL_ID_SPAN_OPEN,
)
from rl_detector.rollouts import compute_task_loss_weights
from rl_detector.train import build_datum

_SPAN_OPEN = ANN_SPECIAL_ID_SPAN_OPEN
_ANN_OPEN = ANN_SPECIAL_ID_ANN_PREFIX
_ANN_CLOSE = ANN_SPECIAL_ID_CLOSE

_UNIT_PTOK_LOSS_SCALES = {
    "ann_type": 1.0,
    "ann_why": 1.0,
    "ann_score": 1.0,
    "verdict_type": 1.0,
    "verdict_why": 1.0,
    "verdict_score": 1.0,
    "span_open": 1.0,
    "structural": 1.0,
}


def _td_to_list(td):
    if hasattr(td, "tolist"):
        return list(td.tolist())
    if hasattr(td, "_t"):
        return td._t.tolist()
    if hasattr(td, "data"):
        return list(td.data)
    raise TypeError(type(td))


def _datum_arrays(datum):
    inputs = datum.loss_fn_inputs
    return (
        _td_to_list(inputs["advantages"]),
        _td_to_list(inputs["mask"]),
    )


def test_build_datum_scales_advantages_by_task_weights():
    doc = [1, 2, 3]
    completion_tokens = [_SPAN_OPEN] + doc + [_ANN_OPEN, 10, _ANN_CLOSE]
    completion_logprobs = [-0.1] * len(completion_tokens)
    response_advs = [0.5] * len(completion_tokens)
    response_weights = compute_task_loss_weights(
        tokenizer=None,
        completion_tokens=completion_tokens,
        n_reasoning_tokens=0,
        span_open_loss_mass=0.15,
        span_ann_mass=1.0,
        ptok_loss_scales=_UNIT_PTOK_LOSS_SCALES,
    )
    datum = build_datum(
        prompt_tokens=[20, 21],
        completion_tokens=completion_tokens,
        completion_logprobs=completion_logprobs,
        response_advantages=response_advs,
        response_task_weights=response_weights,
        n_reasoning_tokens=0,
    )
    advs, mask = _datum_arrays(datum)
    assert len(advs) == len(mask)
    # prompt has 2 tokens -> 1 supervised position; first completion token at index 1
    assert _close(advs[1], 0.5 * response_weights[0])
    assert _close_list(mask[1:], [1.0] * len(completion_tokens))


def test_task_weights_sum_to_one_per_task_group():
    """Doc sums to 1; all span tokens (both tells) sum to span_ann_mass."""
    completion_tokens = (
        [1, 2]
        + [_SPAN_OPEN, 30, _ANN_OPEN, 31, _ANN_CLOSE]
        + [_SPAN_OPEN, 40, 41, _ANN_OPEN, 42, _ANN_CLOSE]
    )
    w = compute_task_loss_weights(
        tokenizer=None,
        completion_tokens=completion_tokens,
        n_reasoning_tokens=0,
        span_open_loss_mass=0.15,
        span_ann_mass=1.0,
        ptok_loss_scales=_UNIT_PTOK_LOSS_SCALES,
    )
    assert _close(sum(w[0:2]), 1.0)
    assert _close(sum(w[2:]), 1.0)
    assert _close(w[2], 0.15)
    assert _close(w[7], 0.15)
