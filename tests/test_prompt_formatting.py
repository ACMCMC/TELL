from rl_detector.annotation_utils import strip_all_bracket_annotations
from rl_detector.tell_xml import escape_document_piece, wrap_span_piece
from rl_detector.prompt_utils import format_prompt_for_model
from rl_detector.prompts import build_prompt


class _ToyTokenizer:
    def apply_chat_template(
        self,
        conversation,
        tokenize=False,
        add_generation_prompt=True,
        reasoning_effort=None,
        model_identity=None,
    ):
        assert tokenize is False
        text = conversation[0]["content"]
        return text + ("<GEN>" if add_generation_prompt else "")


def _extract_payload(prompt_text: str) -> str:
    start = prompt_text.find("<<<\n")
    end = prompt_text.rfind("\n>>>")
    assert start >= 0 and end > start
    return prompt_text[start + 4 : end]


def test_build_prompt_fence_payload_matches_escape_document_piece():
    doc = 'TEXT [[edge]] corner meta chars \n "\\" END'
    prompt_text = build_prompt(text=doc)
    assert _extract_payload(prompt_text) == escape_document_piece(doc)


def test_format_prompt_for_model_single_entrypoint():
    doc = 'alpha beta and braces {cfg:[1,2]}'
    tok = _ToyTokenizer()
    prompt_text, formatted = format_prompt_for_model(tokenizer=tok, text=doc)
    assert _extract_payload(prompt_text) == escape_document_piece(doc)
    assert formatted.endswith("<GEN>")


def test_mixed_bare_amp_and_literal_entity_text_on_wire():
    # source: bare ``&`` must become ``&amp;``; literal ``&amp;`` (five chars) must become ``&amp;amp;``
    doc = "x & y &amp; z"
    wire = "x &amp; y &amp;amp; z"
    assert escape_document_piece(doc) == wire
    prompt_text = build_prompt(text=doc)
    assert _extract_payload(prompt_text) == wire
    tok = _ToyTokenizer()
    prompt_text2, _formatted = format_prompt_for_model(tokenizer=tok, text=doc)
    assert _extract_payload(prompt_text2) == wire


def test_strip_annotation_inner_is_plaintext():
    nested = wrap_span_piece(
        escape_document_piece("innerx"),
        {"type": "AI", "why": "n", "score": "0.51"},
    )
    ann = wrap_span_piece(
        escape_document_piece("literal ") + nested + escape_document_piece(" tail"),
        {"type": "human", "why": "o", "score": "0.77"},
    )
    assert strip_all_bracket_annotations(ann) == "literal innerx tail"
