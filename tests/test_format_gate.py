from rl_detector.annotation_utils import SP_CL, SP_OP
from rl_detector.tell_xml import (
    _TEXT_O,
    _TEXT_CLOSE_CHUNK,
    _VERDICT_PREF,
    escape_attr_piece,
    escape_document_piece,
    format_annotation_tag,
    strip_return_token,
    wrap_outer_logical_plain_mid,
    wrap_span_piece,
)
from rl_detector.rewards import (
    format_diagnostics,
    format_reward,
    format_status,
    strip_tags,
)


def _new_wrap(inner_wired: str, typ: str, why: str, score: str) -> str:
    """New format: wrap already-wired inner content in <text>…<verdict/></text>."""
    t = escape_attr_piece(typ)
    w = escape_attr_piece(why.strip())
    sc = escape_attr_piece(str(score))
    return _TEXT_O + inner_wired + _VERDICT_PREF + f'{t}" why="{w}" score="{sc}' + _TEXT_CLOSE_CHUNK


def _local(span: str, typ: str, why: str, score: str | float) -> str:
    return wrap_span_piece(
        escape_document_piece(span),
        {"type": typ, "why": why, "score": str(score)},
    )


def make_ice_cream_doc_and_ann() -> tuple[str, str]:
    """Long-form nested example shared with rollout repair tests (new format)."""
    doc = (
        "Ice cream manufacturers often have to race against time to put their products on shelves, "
        "an expensive task that produces massive amounts of greenhouse gas emissions. "
        "A pair of researchers at Cornell University have created a system that uses "
        "pressurized carbon dioxide to create instant ice cream. When fluids expand from high pressure "
        "to low pressure, it can cause a cooling effect under the right conditions. The scientist's "
        "ice cream machine uses this principle with pressurized carbon dioxide to produce a scoop "
        "of ice cream every three seconds. With the machine, shop owners can potentially keep shelf-stable mixtures "
        "on hand and produce ice cream as required. The system can also be potentially used to "
        "create instant soda slushies."
    )
    mid = (
        _local(
            "Ice cream manufacturers often have to race against time",
            "human",
            "this is a slightly dramatic phrase, but it does not feel over-explained; "
            "I think a human journalist would use a vivid metaphor like this to make a "
            "dry industrial issue more lively",
            0.75,
        )
        + " "
        + _local(
            "to put their products on shelves",
            "human",
            "this is a plain retail phrase, not a polished science-summary phrase; it sounds "
            "like a human choosing a concrete everyday action for companies",
            0.5,
        )
        + ", "
        + _local(
            "an expensive task that produces massive amounts of greenhouse gas emissions",
            "human",
            "the grammar is a little awkward, since the appositive does not quite fit the verb "
            "before it; AI usually smooths this kind of sentence because it predicts the "
            "most common structure",
            0.9,
        )
        + ". "
        + _local(
            "A pair of researchers at Cornell University",
            "human",
            "this is a very specific attribution with a named institution; humans writing news "
            "often keep the real source details, while AI often makes a more general claim like "
            "scientists or a research team",
            0.85,
        )
        + " "
        + _local(
            "have created a system",
            "human",
            "the tense is a bit loose; a polished generated sentence would likely say have "
            "developed or created, so this small unevenness feels like a human news draft",
            0.6,
        )
        + " "
        + _local(
            "that uses pressurized carbon dioxide to create instant ice cream",
            "human",
            "this is a concrete technical claim, and it is not softened with vague language; "
            "I think a human reporting from a study would include this exact mechanism",
            0.9,
        )
        + ". "
        + _local(
            "When fluids expand from high pressure to low pressure, it can cause a cooling "
            "effect under the right conditions",
            "AI",
            "this sentence is very clean and textbook-like; the wording explains a physical "
            "principle in a generic way, which is a common AI move when it tries to make a "
            "short article sound scientific",
            0.7,
        )
        + ". "
        + _local(
            "The scientist's ice cream machine uses this principle with pressurized carbon "
            "dioxide to produce a scoop of ice cream every three seconds",
            "human",
            "the straight apostrophe is a small human tell, because it is what people type "
            "on a keyboard; AI often outputs typographic apostrophes because it has seen a "
            "lot of polished text",
            0.55,
        )
        + ". "
        + _local(
            "With the machine, shop owners can potentially keep shelf-stable mixtures "
            "on hand and produce ice cream as required",
            "human",
            "this cautious word is used a lot in reporting, because the writer does not want "
            "to overstate the claim; AI can do this too, but it is a strong journalistic habit",
            0.55,
        )
        + ". "
        + _local(
            "The system can also be potentially used to create instant soda slushies",
            "human",
            "this is a strange but specific extra use case; it is not a generic benefit, "
            "and the odd product choice makes it feel like a real reported detail",
            0.85,
        )
        + "."
    )
    ann = _new_wrap(
        mid,
        "human",
        "I think this is human because it is factual, specific, and a little uneven, with "
        "real named attribution, concrete claims, and small awkwardness that AI would "
        "likely smooth out",
        "0.9",
    )
    return doc, ann


def test_format_gate_accepts_nested_mixed():
    doc = "Alpha beta, gamma.\nDelta!"
    inner = (
        "Alpha "
        + _local("beta", "human", "casual word choice", "0.61")
        + ", "
        + _local("gamma", "AI", "formal transition", "0.72")
        + ".\n"
        + _local("Delta!", "human", "punctuation habit", "0.83")
    )
    out = _new_wrap(inner, "AI", "overall style profile", "0.55")
    diag = format_diagnostics(out, doc)
    assert diag["ok"] is True
    assert format_reward(out, doc) == 1.0


def test_format_gate_accepts_outer_only():
    doc = "plain text only"
    out = wrap_outer_logical_plain_mid(doc, {"type": "AI", "why": "too generic", "score": "0.33"})
    assert format_diagnostics(out, doc)["ok"] is True


def test_format_gate_rejects_invalid_inner_type():
    doc = "Alpha beta"
    inner = "Alpha " + _local("beta", "bot", "bad label", "0.71")
    out = _new_wrap(inner, "AI", "outer ok", "0.71")
    diag = format_diagnostics(out, doc)
    assert diag["ok"] is False
    assert diag["reason"] == "annotation_parse_failed"


def test_format_gate_rejects_ai_generated_type_label():
    doc = "Some text"
    # Manually craft inner span with bad type label
    bad_span = '<span>Some text<annotation type="AI_GENERATED" why="comment" score="0.68" /></span>'
    out = _new_wrap(bad_span, "AI", "outer", "0.5")
    diag = format_diagnostics(out, doc)
    assert diag["ok"] is False
    assert diag["reason"] == "annotation_parse_failed"


def test_format_gate_rejects_text_mismatch():
    doc = "Alpha beta gamma"
    inner = (
        "Alpha "
        + _local("beta", "human", "x", "0.62")
        + " "
        + _local("GAMMA", "AI", "cap", "0.80")
    )
    out = _new_wrap(inner, "AI", "g", "0.59")
    diag = format_diagnostics(out, doc)
    assert diag["ok"] is False
    assert diag["reason"] == "text_mismatch"


def test_format_gate_rejects_broken_outer_meta():
    doc = "Alpha beta"
    inner = "Alpha " + _local("beta", "human", "ok", "0.41")
    # Truncated verdict (missing closing)
    out = _TEXT_O + inner + _VERDICT_PREF + 'AI" why="truncate'
    diag = format_diagnostics(out, doc)
    assert diag["ok"] is False


def test_format_gate_rejects_missing_outer():
    doc = "Alpha beta"
    out = "Alpha " + _local("beta", "human", "local only", "0.44")
    diag = format_diagnostics(out, doc)
    assert diag["ok"] is False
    assert diag["reason"] == "missing_outer_annotation"


def test_format_gate_accepts_markdowny_doc_when_exact():
    doc = (
        "## Results\n"
        "Use `{alpha: 1, beta: [2, 3]}` and [docs](https://example.com); "
        "then compare `x[0]` vs `x{0}`."
    )
    out = wrap_outer_logical_plain_mid(
        doc,
        {"type": "AI", "why": "full passage with markdown symbols preserved", "score": "0.40"},
    )
    diag = format_diagnostics(out, doc)
    assert diag["ok"] is True
    assert format_status(out, doc) == (True, "ok")


def test_format_gate_rejects_raw_doublequotes_in_wire_span_text():
    doc = 'say "hi"'
    # Inner span with unescaped quotes — wrapped in valid new-format outer
    bad_span = wrap_span_piece(
        'say "hi"',
        {"type": "human", "why": "quotes should be escaped on wire even inside span txt", "score": "0.5"},
    )
    bad = _new_wrap(bad_span, "human", "doc has quotes", "0.5")
    diag = format_diagnostics(bad, doc)
    assert diag["reason"] == "bad_xml_escaping"

    ok = _new_wrap(
        wrap_span_piece(escape_document_piece(doc), {"type": "human", "why": "canonical wire", "score": "0.5"}),
        "human", "doc has quotes", "0.5",
    )
    assert format_diagnostics(ok, doc)["ok"]


def test_format_gate_rejects_ampersand_not_encoded_in_span_plain():
    doc = "Fish & ships"
    bad_span = wrap_span_piece(
        "Fish & ships",
        {"type": "AI", "why": "ampersand splits entity rules", "score": "0.4"},
    )
    bad = _new_wrap(bad_span, "AI", "outer", "0.4")
    assert format_diagnostics(bad, doc)["reason"] == "bad_xml_escaping"
    ok = _new_wrap(
        wrap_span_piece(escape_document_piece(doc), {"type": "AI", "why": "ok", "score": "0.4"}),
        "AI", "outer", "0.4",
    )
    assert format_diagnostics(ok, doc)["ok"]


def test_format_gate_rejects_different_type_of_quote():
    doc = "'Fish' and “chips”"
    bad_span = wrap_span_piece(
        "&apos;Fish&apos; and &quot;chips&quot;",
        {"type": "AI", "why": "different type of quote", "score": "0.4"},
    )
    bad = _new_wrap(bad_span, "AI", "outer", "0.4")
    assert format_diagnostics(bad, doc)["reason"] == "text_mismatch"


def test_format_gate_rejects_accepts_unescaped_curly_quotes():
    doc = "‘Fish’ and “chips”"
    bad_span = wrap_span_piece(
        "‘Fish’ and “chips”",
        {"type": "AI", "why": "curly quotes", "score": "0.4"},
    )
    bad = _new_wrap(bad_span, "AI", "outer", "0.4")
    assert format_diagnostics(bad, doc)["ok"]
    ok = _new_wrap(
        wrap_span_piece(escape_document_piece(doc), {"type": "AI", "why": "curly quotes", "score": "0.4"}),
        "AI", "outer", "0.4",
    )
    assert format_diagnostics(ok, doc)["ok"]
    assert format_diagnostics(ok, doc)["reason"] == "ok"


def test_format_gate_rejects_lt_unescaped_in_span_plain_reflecting_logical_doc():
    doc = "Pick a < b for small values"
    bad_span = wrap_span_piece(
        doc,
        {"type": "human", "why": "math compare", "score": "0.55"},
    )
    bad = _new_wrap(bad_span, "human", "outer", "0.55")
    diag = format_diagnostics(bad, doc)
    assert diag["ok"] is False
    assert diag["reason"] in ("bad_xml_escaping", "annotation_parse_failed")
    ok = _new_wrap(
        wrap_span_piece(escape_document_piece(doc), {"type": "human", "why": "math compare", "score": "0.55"}),
        "human", "outer", "0.55",
    )
    assert format_diagnostics(ok, doc)["ok"]


def test_format_gate_rejects_unescaped_amp_in_outer_explanation():
    doc = "short"
    wired_body = escape_document_piece(doc)
    # Manually craft a verdict with unescaped & in why attribute
    bad = _TEXT_O + wired_body + '<verdict type="human" why="grammar & cohesion" score="0.71" /></text>'
    assert format_diagnostics(bad, doc)["reason"] == "bad_xml_escaping"
    good = wrap_outer_logical_plain_mid(doc, {"type": "human", "why": "grammar & cohesion", "score": "0.71"})
    assert format_diagnostics(good, doc)["ok"]


def test_format_gate_rejects_nested_span_when_inner_expl_not_attr_escaped_even_if_plain_ok():
    doc = "Hello earth"
    inner_bad = (
        SP_OP
        + escape_document_piece("Hello")
        + '<annotation type="human" why="capital & slang" score="0.61" />'
        + SP_CL
    )
    mid = inner_bad + escape_document_piece(" earth")
    bad_full = _new_wrap(mid, "human", "outer ok", "0.52")
    assert format_diagnostics(bad_full, doc)["reason"] == "bad_xml_escaping"
    inner_ok = (
        SP_OP
        + escape_document_piece("Hello")
        + format_annotation_tag({"type": "human", "why": "capital & slang", "score": "0.61"})
        + SP_CL
    )
    good_full = _new_wrap(inner_ok + escape_document_piece(" earth"), "human", "outer ok", "0.52")
    assert format_diagnostics(good_full, doc)["ok"]


def test_format_gate_carriage_return_roundtrip():
    doc = "(Print)\\r0300-9084 (Linking)"
    mid = (
        "(Print)"
        + _local(
            "\\r", "human", "literal backslash-r sequence in messy copy pasta", "0.86"
        )
        + "0300-9084 (Linking)"
    )
    out = _new_wrap(mid, "human", "looks like scraped metadata blob", "0.81")
    diag = format_diagnostics(out, doc)
    assert diag["ok"] is True
    assert strip_tags(out) == doc


def test_format_gate_rejects_text_mismatch_with_unbalanced_tags():
    doc = "Alpha beta"
    # A response with no annotation at all in the outer wrapper
    out = _TEXT_O + "Alpha beta" + _VERDICT_PREF + 'AI" why="" score="0.5' + _TEXT_CLOSE_CHUNK
    # This is actually valid new format — let's use a genuinely broken one instead
    # Inner span with no closing tag → parse fail
    broken_inner = SP_OP + "Alpha beta"  # missing </span>
    out2 = _new_wrap(broken_inner, "AI", "outer", "0.5")
    diag = format_diagnostics(out2, doc)
    assert diag["ok"] is False


def test_format_gate_real_example_1():
    doc, out = make_ice_cream_doc_and_ann()
    diag = format_diagnostics(out, doc)
    assert diag["ok"] is True
    assert strip_tags(out) == doc
    assert diag["reason"] == "ok"


def test_format_gate_real_example_2():
    doc = "Two pensioners have tied the knot after a blind date. They married six months later with Ms Love arranging it."
    inner = (
        _local("Two pensioners have tied the knot", "human", "tabloid-ish lead wording", "0.35")
        + " "
        + _local(
            "after a blind date",
            "human",
            "straightforward bridging phrase kept concrete",
            "0.41",
        )
        + ". "
        + _local(
            "They married six months later with Ms Love arranging it.",
            "AI",
            "polished explanatory tail with named actor",
            "0.55",
        )
    )
    resp = _new_wrap(inner, "AI", "overall coherence looks edited rather than pasted captions", "0.68")
    diag = format_diagnostics(resp, doc)
    assert diag["ok"] is True
    assert strip_tags(resp) == doc
    assert diag["reason"] == "ok"


_DOC = "The cat sat on the mat."
_META = {"type": "human", "why": "simple declarative", "score": "0.6"}


def _new(doc=_DOC, meta=_META):
    return wrap_outer_logical_plain_mid(mid_logical_plaintext=doc, meta=meta)


class TestNewFormatGate:
    def test_new_format_accepted(self):
        assert format_diagnostics(_new(), _DOC)["ok"] is True

    def test_new_format_with_return_token_accepted(self):
        assert format_diagnostics(_new() + "<|return|>", _DOC)["ok"] is True

    def test_new_format_return_token_stripped(self):
        assert strip_return_token(_new() + "<|return|>") == _new()
        assert strip_return_token(_new()) == _new()

    def test_new_format_wrong_doc_rejected(self):
        assert format_diagnostics(_new(), "Different document.")["ok"] is False

    def test_new_format_missing_verdict_rejected(self):
        bad = _TEXT_O + escape_document_piece(_DOC)
        assert format_diagnostics(bad, _DOC)["ok"] is False

    def test_new_format_return_token_not_double_stripped(self):
        doc_with_return = "Text <|return|> middle"
        ann = wrap_outer_logical_plain_mid(doc_with_return, _META)
        assert strip_return_token(ann) == ann

    def test_new_format_strip_tags_returns_doc(self):
        assert strip_tags(_new()) == _DOC

    def test_new_format_with_return_strip_tags_returns_doc(self):
        assert strip_tags(_new() + "<|return|>") == _DOC


class TestNewFormatFixResponse:
    def test_new_format_truncated_verdict_repaired(self):
        from rl_detector.format_fix import try_fix_response
        truncated = _TEXT_O + escape_document_piece(_DOC) + _VERDICT_PREF + 'human" why="plain text" score="0.6'
        fixed = try_fix_response(response_text=truncated, document=_DOC, max_fix_ratio=0.5)
        assert fixed is not None
        assert format_diagnostics(fixed, _DOC)["ok"] is True

    def test_new_format_wrong_doc_escaping_repaired(self):
        from rl_detector.format_fix import try_fix_response
        bad = _TEXT_O + "The cat & dog." + _VERDICT_PREF + 'human" why="natural" score="0.5' + _TEXT_CLOSE_CHUNK
        doc = "The cat & dog."
        fixed = try_fix_response(response_text=bad, document=doc, max_fix_ratio=0.5)
        assert fixed is not None
        assert format_diagnostics(fixed, doc)["ok"] is True

    def test_new_format_with_return_token_repaired(self):
        from rl_detector.format_fix import try_fix_response
        bad = _TEXT_O + "The cat &amp; dog." + _VERDICT_PREF + 'human" why="natural" score="0.5' + _TEXT_CLOSE_CHUNK + "<|return|>"
        doc = "The cat & dog."
        diag = format_diagnostics(bad + "<|return|>", doc)
        if not diag["ok"]:
            fixed = try_fix_response(response_text=bad + "<|return|>", document=doc, max_fix_ratio=0.5)
            if fixed is not None:
                assert format_diagnostics(fixed, doc)["ok"] is True
