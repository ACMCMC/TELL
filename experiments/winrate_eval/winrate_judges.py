"""Multi-judge panel for win-rate evaluation. All artifacts under data/winrate_eval/."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import tinker
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from rl_detector.config import CFG
from rl_detector.rollouts import _get_analysis_stub_tokens, extract_response_text

JUDGE_PROMPT_VERSION = "v4_content_focus"
DEFAULT_JUDGE_MAX_TOKENS = 512
JUDGE_REASONING_EFFORT = "low"

JUDGE_PANEL: list[dict] = [
    {
        "judge_id": "openai_gpt54mini_flex",
        "backend": "openai",
        "model": "gpt-5.4-mini",
        "api_key_env": "OPENAI_API_KEY",
        "base_url": "",
        "service_tier": "flex",
        "structured_parse": True,
        "use_temperature": False,
    },
    {
        "judge_id": "ucsd_gemma4_26b",
        "backend": "openai",
        "model": "api-gemma-4-26b",
        "api_key_env": "TRITONAI_API_KEY",
        "base_url": "https://tritonai-api.ucsd.edu/v1",
        "service_tier": "",
        "structured_parse": True,
        "disable_thinking": True,
    },
    {
        "judge_id": "deepinfra_deepseek_v4_flash",
        "backend": "openai",
        "model": "deepseek-ai/DeepSeek-V4-Flash",
        "api_key_env": "DEEPINFRA_API_KEY",
        "base_url": "https://api.deepinfra.com/v1/openai",
        "service_tier": "",
        "structured_parse": True,
        "disable_thinking": True,
    },
    {
        "judge_id": "deepinfra_nemotron_super",
        "backend": "openai",
        "model": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B",
        "api_key_env": "DEEPINFRA_API_KEY",
        "base_url": "https://api.deepinfra.com/v1/openai",
        "service_tier": "",
        "structured_parse": False,
        "disable_thinking": True,
    },
    {
        "judge_id": "tinker_gpt_oss_120b",
        "backend": "tinker",
        "model": CFG.model.base_model,
        "api_key_env": "",
        "base_url": "",
        "service_tier": "",
        "structured_parse": False,
    },
]


class JudgeRankingItem(BaseModel):
    item_id: str = Field(description="Blind candidate id from the prompt, e.g. A1")
    rank: int = Field(description="1 is most convincing, N is least convincing")
    quality_score: float = Field(ge=0.0, le=1.0, description="Higher is better")


class JudgeListwiseOutput(BaseModel):
    ranking: list[JudgeRankingItem] = Field(description="All candidates ranked exactly once")


def judge_panel_by_id(judge_ids: list[str] | None) -> list[dict]:
    if not judge_ids:
        return list(JUDGE_PANEL)
    wanted = set(judge_ids)
    return [row for row in JUDGE_PANEL if row["judge_id"] in wanted]


def judge_spec_by_id(judge_id: str) -> dict | None:
    for row in JUDGE_PANEL:
        if row["judge_id"] == judge_id:
            return row
    return None


def judge_spec_extra_body(judge_id: str) -> dict | None:
    spec = judge_spec_by_id(judge_id=judge_id)
    if spec is None:
        return None
    return _openai_extra_body(judge_spec=spec)


def judge_cache_key(config: dict, judge_id: str) -> str:
    rollout_key = config.get("rollout_key", "")
    payload = {
        "judge_prompt_version": JUDGE_PROMPT_VERSION,
        "judge_id": judge_id,
        "rollout_key": rollout_key,
        "judge_max_tokens": config["judge_max_tokens"],
        "blind_seed_offset": 500,
        "reasoning_effort": JUDGE_REASONING_EFFORT,
        "extra_body": judge_spec_extra_body(judge_id=judge_id),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return digest[:12]


def judge_cache_path(cache_layout: dict, judge_id: str, doc_id: str) -> Path:
    return cache_layout["judges_dir"] / judge_id / f"{doc_id}.json"


def load_judge_cache(
    path: Path,
    config: dict,
    judge_id: str,
    doc_id: str,
    invalidate: bool,
) -> dict | None:
    if not path.is_file():
        return None
    cached = json.loads(path.read_text(encoding="utf-8"))
    expected_key = judge_cache_key(config=config, judge_id=judge_id)
    if cached.get("doc_id") != doc_id:
        raise RuntimeError(
            f"doc_id mismatch in judges cache for doc_id={doc_id} at {path} "
            f"(found {cached.get('doc_id')!r})"
        )
    if cached.get("cache_key") != expected_key:
        if invalidate:
            return None
        raise RuntimeError(
            f"stale judges cache for doc_id={doc_id} at {path} "
            f"(expected cache_key={expected_key!r}, found {cached.get('cache_key')!r}); "
            f"pass --invalidate-judges to refresh"
        )
    return cached


def write_judge_cache(path: Path, payload: dict, overwrite: bool) -> None:
    if path.is_file() and not overwrite:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def build_openai_judge_client(judge_spec: dict) -> AsyncOpenAI:
    api_key = os.environ[judge_spec["api_key_env"]]
    base_url = judge_spec["base_url"]
    if base_url:
        return AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=3600.0)
    return AsyncOpenAI(api_key=api_key, timeout=3600.0)


def encode_judge_prompt_tokens(tokenizer, prompt_text: str, think_already_open: bool) -> list[int]:
    messages = [{"role": "user", "content": prompt_text}]
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    base_tokens = tokenizer.encode(formatted, add_special_tokens=False)
    stub_open, stub_close = _get_analysis_stub_tokens(tokenizer=tokenizer, think_already_open=think_already_open)
    return base_tokens + stub_open + stub_close


def parse_judge_json(text: str) -> dict:
    body = extract_response_text(text=text)
    body = body.strip()
    if body.startswith("```"):
        lines = body.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        body = "\n".join(lines).strip()
    start = body.find("{")
    end = body.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"judge JSON not found in response (len={len(body)})")
    return json.loads(body[start : end + 1])


def judge_spec_max_tokens(judge_spec: dict, judge_max_tokens: int) -> int:
    override = judge_spec.get("judge_max_tokens_override")
    if override is not None:
        return int(override)
    return judge_max_tokens


def _openai_token_extra(judge_spec: dict, judge_max_tokens: int) -> dict:
    extra: dict = {}
    tokens = judge_spec_max_tokens(judge_spec=judge_spec, judge_max_tokens=judge_max_tokens)
    if judge_spec["service_tier"]:
        extra["service_tier"] = judge_spec["service_tier"]
    if judge_spec["base_url"] == "https://api.deepinfra.com/v1/openai":
        extra["max_tokens"] = tokens
    else:
        extra["max_completion_tokens"] = tokens
    return extra


def _openai_extra_body(judge_spec: dict) -> dict:
    extra_body: dict = {}
    if judge_spec.get("disable_thinking", False):
        extra_body["chat_template_kwargs"] = {"enable_thinking": False}
    spec_body = judge_spec.get("extra_body")
    if spec_body:
        for key, value in spec_body.items():
            if key == "chat_template_kwargs" and isinstance(value, dict):
                extra_body.setdefault("chat_template_kwargs", {}).update(value)
            else:
                extra_body[key] = value
    return extra_body


def _openai_judge_api_extra(judge_spec: dict, judge_max_tokens: int) -> dict:
    extra = _openai_token_extra(judge_spec=judge_spec, judge_max_tokens=judge_max_tokens)
    if not judge_spec.get("disable_reasoning_effort", False):
        extra["reasoning_effort"] = JUDGE_REASONING_EFFORT
    return extra


def _openai_temperature_kwargs(judge_spec: dict) -> dict:
    if judge_spec.get("use_temperature", True):
        return {"temperature": 0.0}
    return {}


async def run_listwise_judge_openai_parse(
    client: AsyncOpenAI,
    judge_spec: dict,
    judge_max_tokens: int,
    prompt: str,
    seed: int,
) -> dict:
    extra = _openai_judge_api_extra(judge_spec=judge_spec, judge_max_tokens=judge_max_tokens)
    extra_body = _openai_extra_body(judge_spec=judge_spec)
    response = await client.beta.chat.completions.parse(
        model=judge_spec["model"],
        messages=[{"role": "user", "content": prompt}],
        seed=seed,
        response_format=JudgeListwiseOutput,
        extra_body=extra_body,
        **_openai_temperature_kwargs(judge_spec=judge_spec),
        **extra,
    )
    message = response.choices[0].message
    parsed_obj = message.parsed
    raw_content = message.content or ""
    if parsed_obj is None:
        if not raw_content.strip():
            reasoning_content = getattr(message, "reasoning_content", "") or ""
            if reasoning_content.strip():
                raise ValueError(
                    f"judge parse returned no content but reasoning_content len={len(reasoning_content)}; "
                    f"reasoning_effort={JUDGE_REASONING_EFFORT!r}"
                )
            raise ValueError(f"judge parse returned no structured object: refusal={message.refusal!r}")
        raise ValueError(
            f"judge structured parse returned no parsed object (raw_content len={len(raw_content)})"
        )
    parsed = parsed_obj.model_dump()
    if not raw_content:
        raw_content = json.dumps(parsed, ensure_ascii=False)
    reasoning_content = getattr(message, "reasoning_content", "") or ""
    return {
        "prompt": prompt,
        "raw_content": raw_content,
        "reasoning_content": reasoning_content,
        "parsed": parsed,
        "structured_parse": True,
    }


async def run_listwise_judge_openai_json(
    client: AsyncOpenAI,
    judge_spec: dict,
    judge_max_tokens: int,
    prompt: str,
    seed: int,
) -> dict:
    extra = _openai_judge_api_extra(judge_spec=judge_spec, judge_max_tokens=judge_max_tokens)
    extra_body = _openai_extra_body(judge_spec=judge_spec)
    response = await client.chat.completions.create(
        model=judge_spec["model"],
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        seed=seed,
        extra_body=extra_body,
        **_openai_temperature_kwargs(judge_spec=judge_spec),
        **extra,
    )
    raw_content = response.choices[0].message.content or ""
    reasoning_content = getattr(response.choices[0].message, "reasoning_content", "") or ""
    if not raw_content.strip():
        if reasoning_content.strip():
            raise ValueError(
                f"judge returned empty content but reasoning_content len={len(reasoning_content)}; "
                f"reasoning_effort={JUDGE_REASONING_EFFORT!r}"
            )
        raise ValueError("judge returned empty content")
    parsed = parse_judge_json(text=raw_content)
    return {
        "prompt": prompt,
        "raw_content": raw_content,
        "reasoning_content": reasoning_content,
        "parsed": parsed,
        "structured_parse": False,
    }


async def run_listwise_judge_openai(
    client: AsyncOpenAI,
    judge_spec: dict,
    judge_max_tokens: int,
    prompt: str,
    seed: int,
) -> dict:
    if judge_spec.get("structured_parse", False):
        return await run_listwise_judge_openai_parse(
            client=client,
            judge_spec=judge_spec,
            judge_max_tokens=judge_max_tokens,
            prompt=prompt,
            seed=seed,
        )
    return await run_listwise_judge_openai_json(
        client=client,
        judge_spec=judge_spec,
        judge_max_tokens=judge_max_tokens,
        prompt=prompt,
        seed=seed,
    )


async def run_listwise_judge_tinker(
    judge_spec: dict,
    tinker_runtime: dict,
    judge_max_tokens: int,
    prompt: str,
    seed: int,
) -> dict:
    tokenizer = tinker_runtime["tokenizer"]
    sampling_client = tinker_runtime["sampling_client"]
    think_already_open = tinker_runtime["think_already_open"]
    prompt_tokens = encode_judge_prompt_tokens(
        tokenizer=tokenizer,
        prompt_text=prompt,
        think_already_open=think_already_open,
    )
    sampled = await sampling_client.sample_async(
        prompt=tinker.ModelInput.from_ints(prompt_tokens),
        num_samples=1,
        sampling_params=tinker.SamplingParams(
            max_tokens=judge_spec_max_tokens(judge_spec=judge_spec, judge_max_tokens=judge_max_tokens),
            temperature=0.0,
            top_p=1.0,
            seed=seed,
            reasoning_effort=JUDGE_REASONING_EFFORT,
        ),
    )
    completion_tokens = list(sampled.sequences[0].tokens)
    raw_content = tokenizer.decode(completion_tokens, skip_special_tokens=False).strip()
    parsed = parse_judge_json(text=raw_content)
    if isinstance(parsed, dict):
        parsed.pop("short_rationale", None)
    return {
        "prompt": prompt,
        "raw_content": raw_content,
        "reasoning_content": "",
        "parsed": parsed,
        "structured_parse": False,
    }


async def run_listwise_judge_for_spec(
    judge_spec: dict,
    judge_max_tokens: int,
    prompt: str,
    seed: int,
    openai_clients: dict[str, AsyncOpenAI],
    tinker_runtime: dict | None,
) -> dict:
    if judge_spec["backend"] == "tinker":
        if tinker_runtime is None:
            raise RuntimeError("tinker_runtime is required for tinker judge")
        return await run_listwise_judge_tinker(
            judge_spec=judge_spec,
            tinker_runtime=tinker_runtime,
            judge_max_tokens=judge_max_tokens,
            prompt=prompt,
            seed=seed,
        )
    client = openai_clients[judge_spec["judge_id"]]
    return await run_listwise_judge_openai(
        client=client,
        judge_spec=judge_spec,
        judge_max_tokens=judge_max_tokens,
        prompt=prompt,
        seed=seed,
    )


def build_judge_openai_clients(judge_specs: list[dict]) -> dict[str, AsyncOpenAI]:
    clients: dict[str, AsyncOpenAI] = {}
    for spec in judge_specs:
        if spec["backend"] == "openai":
            clients[spec["judge_id"]] = build_openai_judge_client(judge_spec=spec)
    return clients
