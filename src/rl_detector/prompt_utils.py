"""Shared model-prompt formatting entrypoint for all runtime paths.

GRPO, SFT, eval, annotate, detectors: tokenizer prompts MUST go through
``format_prompt_for_model(tokenizer=..., text=logical_document, ...)``.
``logical_document`` is the raw dataset string; escaping for the fence and for span payloads is
centralized as ``tell_xml.escape_document_piece`` (see ``build_prompt``).

"""

import logging
import re

from transformers import AutoTokenizer

from rl_detector.config import CFG
from rl_detector.prompts import MODEL_IDENTITY, build_prompt
from rl_detector.tell_xml import escape_document_piece

logger = logging.getLogger(__name__)

_THINK_ALREADY_OPEN: bool | None = None


def get_think_already_open(tokenizer) -> bool:
    """Detect once whether the chat template already opens the analysis channel
    in the assistant generation prefix.  Cached after first call."""
    global _THINK_ALREADY_OPEN
    if _THINK_ALREADY_OPEN is None:
        suffix = detect_assistant_generation_suffix(tokenizer)
        _THINK_ALREADY_OPEN = "<|channel|>analysis<|message|>" in suffix
        logger.info(
            "startup | assistant generation suffix=%r analysis_channel_already_open=%s",
            suffix[:80], _THINK_ALREADY_OPEN,
        )
    return _THINK_ALREADY_OPEN


def quantile(values: list[float], q: float) -> float:
    """Discrete linear-index quantile of ``values`` at ``q`` ∈ [0, 1]; 0.0 for empty input."""
    if not values:
        return 0.0
    ordered = sorted(values)
    qq = max(0.0, min(1.0, q))
    idx = int(qq * (len(ordered) - 1))
    return ordered[idx]


# Full added-token map for openai/gpt-oss-120b (200000–200018), verified 2026-05-13:
#
#   200000  <|reserved_200000|>   free
#   200001  <|reserved_200001|>   free
#   200002  <|return|>            TAKEN — do not use
#   200003  <|constrain|>         TAKEN — do not use
#   200004  <|reserved_200004|>   free
#   200005  <|channel|>           TAKEN — do not use
#   200006  <|start|>             TAKEN — do not use
#   200007  <|end|>               TAKEN — do not use
#   200008  <|message|>           TAKEN — do not use
#   200009  <|reserved_200009|>   → ANN_SPECIAL_ID_TEXT_OPEN      "<text>"
#   200010  <|reserved_200010|>   → ANN_SPECIAL_ID_VERDICT_PREFIX  '<verdict type="'
#   200011  <|reserved_200011|>   → ANN_SPECIAL_ID_TEXT_CLOSE      '" /></text>'
#   200012  <|call|>              TAKEN — do not use
#   200013  <|reserved_200013|>   → ANN_SPECIAL_ID_SPAN_OPEN       "<span>"
#   200014  <|reserved_200014|>   → ANN_SPECIAL_ID_ANN_PREFIX      '<annotation type="'
#   200015  <|reserved_200015|>   → ANN_SPECIAL_ID_WHY_Q           '" why="'
#   200016  <|reserved_200016|>   → ANN_SPECIAL_ID_SCORE_Q         '" score="'
#   200017  <|reserved_200017|>   → ANN_SPECIAL_ID_CLOSE           '" /></span>'
#   200018  <|endofprompt|>       TAKEN — do not use
#
# ``load_tokenizer`` refuses remap unless each id is still ``<|reserved_{id}|>`` in that checkpoint's added_tokens
# (so we never overwrite <|channel|>, <|call|>, …).  Single source for remap rows and mask logic in rollouts/train.
ANN_SPECIAL_ID_TEXT_OPEN = 200009
ANN_SPECIAL_ID_VERDICT_PREFIX = 200010
ANN_SPECIAL_ID_TEXT_CLOSE = 200011
ANN_SPECIAL_ID_SPAN_OPEN = 200013
ANN_SPECIAL_ID_ANN_PREFIX = 200014
ANN_SPECIAL_ID_WHY_Q = 200015
ANN_SPECIAL_ID_SCORE_Q = 200016
ANN_SPECIAL_ID_CLOSE = 200017

# Remapping makes these strings atomic tokens, preventing BPE merges like
# ``<span>Hello`` -> ``['<span', '>Hello']``.
ANNOTATION_TOKEN_REMAP: dict[int, str] = {
    ANN_SPECIAL_ID_TEXT_OPEN: "<text>",
    ANN_SPECIAL_ID_VERDICT_PREFIX: '<verdict type="',
    ANN_SPECIAL_ID_TEXT_CLOSE: '" /></text>',
    ANN_SPECIAL_ID_SPAN_OPEN: "<span>",
    ANN_SPECIAL_ID_ANN_PREFIX: '<annotation type="',
    ANN_SPECIAL_ID_WHY_Q: '" why="',
    ANN_SPECIAL_ID_SCORE_Q: '" score="',
    ANN_SPECIAL_ID_CLOSE: '" /></span>',
}

_RESERVED_ADDED_TOKEN_RE = re.compile(r"^<\|reserved_([0-9]+)\|>$")


def assert_annotation_remap_ids_are_reserved_placeholders(tok: AutoTokenizer, remap: dict[int, str]) -> None:
    """Fail before remap if any id is not a ``<|reserved_N|>`` added token for this checkpoint.

    We only repurpose HF ``<|reserved_…|>`` slots (never ``<|channel|>``, ``<|call|>``, ``<|endoftext|>``, …).
    """
    for tok_id, new_str in remap.items():
        at = tok.added_tokens_decoder.get(tok_id)
        if at is None:
            raise ValueError(
                f"annotation remap id {tok_id} ({new_str!r}) is missing from this tokenizer's added_tokens_decoder; "
                "pick ids from that model's reserved added-token block only."
            )
        raw = at.content
        m = _RESERVED_ADDED_TOKEN_RE.match(raw)
        if m is None or int(m.group(1)) != tok_id:
            raise ValueError(
                f"annotation remap id {tok_id} maps to {raw!r}, not <|reserved_{tok_id}|>; "
                "refusing to overwrite a non-reserved special."
            )


_tokenizer_cache: AutoTokenizer | None = None


def load_tokenizer() -> AutoTokenizer:
    """Return the process-wide tokenizer singleton, loading it on first call.

    Applies ANNOTATION_TOKEN_REMAP in-memory when model.use_special_annotation_tokens
    is true — no tokenizer files to generate or ship.
    Call reset_tokenizer() if you need to force a reload (e.g. after rebinding CFG).
    """
    global _tokenizer_cache
    if _tokenizer_cache is None:
        logger.info("loading tokenizer from %s", CFG.model.base_model)
        tok = AutoTokenizer.from_pretrained(CFG.model.base_model)
        if getattr(CFG.model, "use_special_annotation_tokens", False):
            assert_annotation_remap_ids_are_reserved_placeholders(tok=tok, remap=ANNOTATION_TOKEN_REMAP)
            _apply_token_remap(tok=tok, remap=ANNOTATION_TOKEN_REMAP)
            logger.info("applied annotation token remap: %s", ANNOTATION_TOKEN_REMAP)
        _tokenizer_cache = tok
    return _tokenizer_cache


def reset_tokenizer() -> None:
    """Invalidate the tokenizer cache so the next load_tokenizer() call reloads from disk."""
    global _tokenizer_cache
    _tokenizer_cache = None


def _apply_token_remap(tok: AutoTokenizer, remap: dict[int, str]) -> None:
    """Patch a fast tokenizer in-memory to remap reserved token IDs to new strings.

    Modifies both the Rust-level tokenizer (for actual encode/decode behaviour)
    and the Python-level added_tokens_encoder/decoder dicts so all lookup paths
    stay consistent.
    """
    import json
    from tokenizers import Tokenizer as RustTokenizer

    # Patch the Rust-level tokenizer via its JSON representation.
    tj = json.loads(tok._tokenizer.to_str())
    for entry in tj.get("added_tokens", []):
        if entry["id"] in remap:
            entry["content"] = remap[entry["id"]]
    tok._tokenizer = RustTokenizer.from_str(json.dumps(tj))

    # Sync the Python-level str->id mapping.
    for old_str, tok_id in list(tok.added_tokens_encoder.items()):
        if tok_id in remap:
            del tok.added_tokens_encoder[old_str]
            tok.added_tokens_encoder[remap[tok_id]] = tok_id

    # Sync the Python-level id->AddedToken mapping.
    for tok_id, added_token in tok.added_tokens_decoder.items():
        if tok_id in remap:
            added_token.content = remap[tok_id]


def detect_assistant_generation_suffix(tokenizer) -> str:
    """Return template suffix appended after user content for generation."""
    sentinel = "\x00SENTINEL\x00"
    fmt = tokenizer.apply_chat_template(
        [{"role": "user", "content": sentinel}],
        tokenize=False,
        add_generation_prompt=True,
        reasoning_effort=CFG.sampling.reasoning_effort,
        model_identity=MODEL_IDENTITY,
    )
    return fmt.split(sentinel, 1)[-1]


def format_prompt_for_model(tokenizer, text: str, add_generation_prompt: bool = True) -> tuple[str, str]:
    """Escaped user prompt (``build_prompt``) + chat‑templated prompt for sampling / training.

    ``text``: LOGICAL document string (HF row); preserve source text exactly, then XML-escape for fence.
    """
    text = str(text)
    prompt_text = build_prompt(text=text)
    formatted = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_text}],
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        reasoning_effort=CFG.sampling.reasoning_effort,
        model_identity=MODEL_IDENTITY,
    )
    # enforce escaping invariant at final prompt boundary
    start = prompt_text.find("<<<\n")
    end = prompt_text.rfind("\n>>>")
    assert start >= 0 and end > start, "prompt text is missing <<< >>> block"
    escaped_payload = prompt_text[start + 4 : end]
    assert escaped_payload == escape_document_piece(text), "prompt escaped payload mismatched slice"
    return prompt_text, formatted
