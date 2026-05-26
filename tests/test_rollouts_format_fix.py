from test_format_gate import make_ice_cream_doc_and_ann

from rl_detector.annotation_utils import SP_CL, SP_OP, strip_all_bracket_annotations_raw as _strip_raw
from rl_detector.tell_xml import (
    _TEXT_CLOSE_CHUNK,
    _TEXT_O,
    _VERDICT_PREF,
    escape_attr_piece,
    escape_document_piece,
    get_outer_meta_dict,
    wrap_outer_logical_plain_mid,
    wrap_span_piece,
)
from rl_detector.rewards import format_diagnostics, strip_tags
from rl_detector.format_fix import _apply_format_fix_to_text_fields, try_fix_response


def _tell(span: str, typ: str, why: str, sc: str) -> str:
    return wrap_span_piece(
        escape_document_piece(span),
        {"type": typ, "why": why, "score": sc},
    )


def _outer(inner: str, typ: str, why: str, sc: str) -> str:
    if "<span>" in inner:
        mid = inner
    else:
        mid = escape_document_piece(inner)
    t = escape_attr_piece(typ)
    w = escape_attr_piece(why.strip())
    sc_esc = escape_attr_piece(str(sc))
    return _TEXT_O + mid + _VERDICT_PREF + f'{t}" why="{w}" score="{sc_esc}' + _TEXT_CLOSE_CHUNK


def _fixed_outer_only(document: str, typ: str, why: str, sc: str) -> str:
    return wrap_outer_logical_plain_mid(
        mid_logical_plaintext=document,
        meta={"type": typ, "why": why, "score": sc},
    )


class _ToyTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False):
        return [ord(ch) for ch in text]

    def decode(self, tokens, skip_special_tokens=False):
        return "".join(chr(t) if t < 128 else "?" for t in tokens)


def test_try_fix_preserves_literal_entity_text_in_document():
    # if source text literally has "&amp;", keep it as document text; wire will escape again as needed
    wrapped_ok = _outer("Coffee &amp; tea", "human", "note", "0.52")
    assert try_fix_response(wrapped_ok, "Coffee &amp; tea", max_fix_ratio=0.5) is None


def test_try_fix_outer_only_when_doc_logical_but_model_span_body_unescaped_gets_fixed():
    doc = "Fish & chips are good"
    bad_inner = wrap_span_piece(
        doc,
        {"type": "human", "why": "ampersand wire missing", "score": "0.55"},
    )
    bad_wire = _outer(
        bad_inner,
        "human",
        "ampersand wire missing",
        "0.55",
    )
    diag = format_diagnostics(bad_wire, doc)
    assert diag["ok"] is False
    assert diag["reason"] == "bad_xml_escaping"
    fixed = try_fix_response(bad_wire, doc, max_fix_ratio=0.3)
    assert fixed is not None
    aft = format_diagnostics(fixed, doc)
    assert aft["ok"] is True
    assert strip_tags(fixed) == doc


def test_apply_format_fix_handles_entity_encoded_document_like_loader():
    doc_ent = "M&amp;Ms"
    bad = _outer(
        wrap_span_piece(escape_document_piece("x"), {"type": "AI", "why": "q", "score": "0.3"}),
        "AI",
        "q",
        "0.3",
    )
    tok = _ToyTokenizer()
    _apply_format_fix_to_text_fields(
        response_text=bad,
        completion_text=bad,
        completion_tokens=tok.encode(bad),
        completion_logprobs=[0.01] * len(tok.encode(bad)),
        document=doc_ent,
        tokenizer=tok,
    )


def test_try_fix_returns_none_when_already_valid():
    document = "Alpha beta"
    response = _outer(
        "Alpha " + _tell("beta", "human", "w", "0.63"),
        "AI",
        "outer",
        "0.41",
    )
    assert try_fix_response(response, document, max_fix_ratio=0.05) is None


def test_try_fix_none_when_far_mismatch():
    document = "Alpha beta"
    response = _outer("This is a completely different text", "AI", "x", "0.27")
    assert try_fix_response(response, document, max_fix_ratio=0.05) is None


def test_try_fix_one_word_substitution():
    document = "Alpha beta"
    response = _outer("Alpha gamma", "AI", "wrong token", "0.27")
    fixed = try_fix_response(response, document, max_fix_ratio=0.50)
    assert fixed is not None
    diag = format_diagnostics(fixed, document)
    assert diag["ok"] is True
    assert strip_tags(fixed) == document
    assert "wrong token" in fixed


def test_apply_format_fix_toy_tokenizer_roundtrip():
    document = "{x: 1}"
    broken = _outer("{x: 2}", "human", "oops", "0.44")
    tok = _ToyTokenizer()
    (
        fixed_response,
        fixed_completion,
        fixed_tokens,
        fixed_logprobs,
        was_fixed,
        wrong_response,
    ) = _apply_format_fix_to_text_fields(
        response_text=broken,
        completion_text=broken,
        completion_tokens=tok.encode(broken),
        completion_logprobs=[0.11] * len(tok.encode(broken)),
        document=document,
        tokenizer=tok,
    )
    assert was_fixed is True
    assert wrong_response == broken
    diag = format_diagnostics(fixed_response, document)
    assert diag["ok"] is True
    assert diag["reason"] == "ok"


def test_try_fix_preserves_markup_tokens_in_literal_doc_payload():
    # keep doc plaintexty; stray angle brackets inside doc would still be xml-escaped downstream
    document = "literal {tok} only ascii fence here"
    wrapped = _outer(document, "human", "ok", "0.55")
    assert try_fix_response(wrapped, document, max_fix_ratio=0.2) is None
    diag = format_diagnostics(wrapped, document)
    assert diag["ok"] is True


def test_strip_raw_matches_strip_tags_for_xml_spans():
    doc = "a"
    ann = _outer(doc, "AI", "r", "0.3")
    assert _strip_raw(ann) == doc


def test_try_fix_cannot_repair_annotation_parse_fail_with_no_annotation():
    # Inner is a lone root <span>…</span> with no nested <annotation/>; format fails. Verdict score is
    # invalid so try_fix cannot read metadata and must bail (no silent snap to document).
    doc = "Alpha beta"
    out = _outer("<span>Alpha beta</span>", "human", "x", "not-a-score")
    diag = format_diagnostics(out, doc)
    assert diag["ok"] is False
    assert diag["reason"] == "annotation_parse_failed"

    fixed = try_fix_response(out, doc, max_fix_ratio=0.2)
    assert fixed is None


def test_fix_real_example_long_valid_nested_matches_doc():
    doc, ann = make_ice_cream_doc_and_ann()
    assert try_fix_response(ann, doc, max_fix_ratio=0.2) is None


def test_fix_ice_cream_small_doc_repairs_inner_typos():
    doc = "Ice cream manufacturers often have to race against time to put their products on shelves."
    bad_inner = (
        _tell(
            "Ice cream manufacturers  often have to race against time",
            "human",
            "slightly dramatic phrase; extra spaces in span",
            "0.75",
        )
        + "  "
        + _tell(
            "put their products on shelves",
            "human",
            "missing leading to token vs doc",
            "0.5",
        )
        + "]]."
    )
    ann = _outer(
        bad_inner,
        "human",
        "outer comment",
        "0.9",
    )
    fixed = try_fix_response(ann, doc, max_fix_ratio=0.2)
    assert fixed is not None
    diag = format_diagnostics(fixed, doc)
    assert diag["ok"] is True
    assert strip_tags(fixed) == doc
    assert "slightly dramatic" in fixed


def test_fix_ice_cream_small_doc_repairs_spacing_and_noise():
    doc = "Ice cream manufacturers often have to race against time to put their products on shelves."
    bad_inner = (
        _tell(
            "Ice cream manufacturers  often have to race against time",
            "human",
            "slightly dramatic phrase",
            "0.75",
        )
        + "  "
        + _tell("put their products on shelves", "human", "plain retail phrase", "0.5")
        + "]]."
    )
    ann = _outer(bad_inner, "human", "I think this is factual", "0.9")
    fixed = try_fix_response(ann, doc, max_fix_ratio=0.2)
    assert fixed is not None
    diag = format_diagnostics(fixed, doc)
    assert diag["ok"] is True
    assert strip_tags(fixed) == doc
    assert "slightly dramatic phrase" in fixed


def test_try_fix_rebuilds_doc_when_inner_garbage_but_outer_meta_ok():
    # repair path can snap inner to the logical document if diff budget allows
    doc = "Just add the secondary symbol using the Comparison button and set the date range using the calendar button."
    resp = _outer(
        _tell(
            "Just make up totally different wording that is not substring aligned",
            "AI",
            "polished",
            "0.32",
        ),
        "human",
        "casual reviewer tone referencing charts etc",
        "0.80",
    )
    fixed = try_fix_response(resp, doc, max_fix_ratio=0.5)
    assert fixed is not None
    assert format_diagnostics(fixed, doc)["ok"] is True
    assert strip_tags(fixed) == doc


def test_try_fix_should_not_patch_doc_response_mismatch():
    doc = "C Am G short lyric line here."
    resp = _outer(
        "The text looks mostly human because it has a rough song-like shape and several small slips.",
        "human",
        "meta commentary not the lyric",
        "0.38",
    )
    assert try_fix_response(resp, doc, max_fix_ratio=0.2) is None


def test_try_fix_on_outer_missing_annotation():
    doc = "Apologies are best articulated when the parties involved are face to face."
    # legacy-shaped inner: root <span>…</span> with copied text but no nested <annotation/>; invalid verdict score so repair bails.
    resp = _outer(
        SP_OP + "Apologies are best articulated" + SP_CL + " when the parties involved are face to face.",
        "human",
        "x",
        "0.7",
    )
    assert format_diagnostics(resp, doc)["reason"] == "annotation_parse_failed"
    fixed = try_fix_response(resp, doc, max_fix_ratio=0.2)
    assert fixed is not None
    diag = format_diagnostics(fixed, doc)
    assert diag["ok"] is True
    assert strip_tags(fixed) == doc


def test_try_fix_minor_punctuation_spacing_from_doc():
    doc = "For what it 's worth , hello ."
    resp = _outer("For what it 's worth, hello.", "human", "overall", "0.9")
    fixed = try_fix_response(resp, doc, max_fix_ratio=0.2)
    assert fixed is not None
    diag = format_diagnostics(fixed, doc)
    assert diag["ok"] is True
    assert strip_tags(fixed) == doc


def test_apply_format_fix_does_not_repair_budget_truncation_rollout():
    from rl_detector.config import CFG

    doc = "The AI model generated this text with typical phrasing."
    inner = _tell("typical phrasing", "AI", "AI cliche", "0.8")
    truncated = _TEXT_O + SP_OP + "The AI model generated this text with " + inner + " but cut off here"

    tok = _ToyTokenizer()
    tokens = tok.encode(truncated)
    logprobs = [0.01] * len(tokens)

    old_max = CFG.sampling.max_tokens
    CFG.sampling.max_tokens = len(tokens)
    try:
        resp, comp, new_tokens, new_logprobs, was_fixed, wrong = _apply_format_fix_to_text_fields(
            response_text=truncated,
            completion_text=truncated,
            completion_tokens=tokens,
            completion_logprobs=logprobs,
            document=doc,
            tokenizer=tok,
        )
    finally:
        CFG.sampling.max_tokens = old_max

    assert was_fixed is False, "budget-truncated rollout must not be synthesized for training"
    assert resp == truncated
    assert wrong is None


def test_try_fix_does_not_get_moved_around():
    doc = "The staff was extremely friendly accommodating. I received my room upgrades both times that i stayed there. And it was absolutely clean - much cleaner than other hotels I've been staying in."
    resp = _outer(
        'The staff was extremely friendly accommodating. I received my room upgrades both times that <span>i<annotation type="human" why="EXPLANATION" score="0.86" /></span> stayed there. And it was absolutely clean - much cleaner than other hotels I&apos;ve been staying in..',
        "human",
        "misplaced",
        "0.7",
    )
    correct_fix = _outer(
        'The staff was extremely friendly accommodating. I received my room upgrades both times that <span>i<annotation type="human" why="EXPLANATION" score="0.86" /></span> stayed there. And it was absolutely clean - much cleaner than other hotels I&apos;ve been staying in.',
        "human",
        "misplaced",
        "0.7",
    )
    fixed = try_fix_response(resp, doc, max_fix_ratio=0.2)
    assert fixed is not None
    diag = format_diagnostics(fixed, doc)
    assert diag["ok"] is True
    assert strip_tags(fixed) == doc
    assert fixed == correct_fix


def test_try_fix_strips_extra_trailing_text_beyond_doc():
    doc = "The quick brown fox jumps over the lazy dog."
    # response has the doc text plus spurious extra words appended
    resp = _outer(
        "The quick brown fox jumps over the lazy dog. Some extra hallucinated words.",
        "human",
        "summary",
        "0.6",
    )
    fixed = try_fix_response(resp, doc, max_fix_ratio=0.5)
    assert fixed is not None
    assert format_diagnostics(fixed, doc)["ok"] is True
    assert strip_tags(fixed) == doc


def test_try_fix_banks_text_repairs_extra_period():
    doc = "I believe you'd be good to go. Many banks simply won't do it."
    resp = _outer(
        'I believe you&apos;d be good to go. <span>Many banks simply won&apos;t do it.<annotation type="human" why="EXPLANATION" score="0.36" /></span>.'
        , "human", "extra period", "0.7")
    fixed = try_fix_response(resp, doc, max_fix_ratio=0.2)
    assert fixed is not None
    diag = format_diagnostics(fixed, doc)
    assert diag["ok"] is True
    assert strip_tags(fixed) == doc


def test_try_fix_recovers_misplaced_outer_annotation():
    # The model omitted <span> for the last nested tell. The annotation landed directly in the
    # outer span content, causing </span> to close the outer span early and the real outer
    # annotation to appear outside it. The last <annotation in the response is always the outer.
    doc = "Alpha beta gamma delta."
    nested_tell = _tell("beta", "human", "casual phrase", "0.3")
    # model wrote a nested annotation without its <span> opener
    misplaced_ann = '<annotation type="human" why="EXPLANATION" score="0.4" />'
    real_outer_ann = '<annotation type="AI" why="EXPLANATION" score="0.72" />'

    # structure: <span>Alpha [nested]... gamma delta.<misplaced/></span><outer/></span> (legacy inner), then <verdict/>.
    malformed_inner = SP_OP + "Alpha " + nested_tell + " gamma delta." + misplaced_ann + SP_CL + real_outer_ann + SP_CL
    malformed = _outer(malformed_inner, "AI", "EXPLANATION", "0.72")

    diag = format_diagnostics(malformed, doc)
    assert diag["reason"] == "annotation_parse_failed"

    fixed = try_fix_response(malformed, doc, max_fix_ratio=0.3)
    assert fixed is not None
    assert format_diagnostics(fixed, doc)["ok"] is True
    assert strip_tags(fixed) == doc

    # outer annotation must come from the real outer (last <annotation), not the misplaced nested one
    meta = get_outer_meta_dict(fixed)
    assert meta is not None
    assert meta["type"] == "AI"
    assert abs(meta["score_magnitude"] - 0.72) < 1e-6
    assert "EXPLANATION" in meta["explanation"]


def test_try_fix_recovers_misplaced_outer_annotation_exact_text_match():
    # Same failure mode but inner text matches the document exactly (exercises the guard path).
    doc = "Short exact text here."
    nested_tell = _tell("exact text", "human", "specific phrase", "0.55")
    misplaced_ann = '<annotation type="human" why="EXPLANATION" score="0.35" />'
    real_outer_ann = '<annotation type="AI" why="EXPLANATION" score="0.68" />'

    malformed_inner = SP_OP + "Short " + nested_tell + " here." + misplaced_ann + SP_CL + real_outer_ann + SP_CL
    malformed = _outer(malformed_inner, "AI", "EXPLANATION", "0.68")

    assert format_diagnostics(malformed, doc)["reason"] == "annotation_parse_failed"

    fixed = try_fix_response(malformed, doc, max_fix_ratio=0.3)
    assert fixed is not None
    assert format_diagnostics(fixed, doc)["ok"] is True
    assert strip_tags(fixed) == doc

    meta = get_outer_meta_dict(fixed)
    assert meta is not None
    assert meta["type"] == "AI"
    assert abs(meta["score_magnitude"] - 0.68) < 1e-6


def test_try_fix_does_not_overly_remove_annotations_when_mismatched():
    # if the doc and response are very different, we don't want to strip all the annotations off and return a bare doc string, which would be a false positive pass. The fix should fail instead.
    doc = '**Gameplay Excellence** The core "one more turn" addiction remains as potent as ever. **Technical and Artistic Triumph** The shift to full 3D graphics was executed beautifully, with a clean, readable interface that never sacrifices functionality for form.'
    resp = _outer(
        '<span>**<annotation type="AI" why="EXPLANATION" score="0.6" /></span>Gameplay Excellence<annotation type="AI" why="EXPLANATION" score="0.55" /></span><span>**<annotation type="AI" why="EXPLANATION" score="0.55" /></span> The core <span>&quot;<annotation type="human" why="EXPLANATION" score="0.2" /></span>one more turn<span>&quot;<annotation type="human" why="EXPLANATION" score="0.2" /></span> addiction remains as potent as ever. <span>**<annotation type="AI" why="EXPLANATION" score="0.65" /></span>Technical and Artistic Triumph<annotation type="AI" why="EXPLANATION" score="0.6" /></span><span>**<annotation type="AI" why="EXPLANATION" score="0.55" /></span> The shift to full 3D graphics was <span>executed beautifully<annotation type="AI" why="EXPLANATION" score="0.55" /></span>, with a clean, readable interface that never sacrifices functionality for form.',
        "AI",
        "mismatched",
        "0.5",
    )
    fixed = try_fix_response(resp, doc, max_fix_ratio=0.5)
    correct_fix = _outer(
        '<span>**<annotation type="AI" why="EXPLANATION" score="0.6" /></span>Gameplay Excellence<span>**<annotation type="AI" why="EXPLANATION" score="0.55" /></span> The core <span>&quot;<annotation type="human" why="EXPLANATION" score="0.2" /></span>one more turn<span>&quot;<annotation type="human" why="EXPLANATION" score="0.2" /></span> addiction remains as potent as ever. <span>**<annotation type="AI" why="EXPLANATION" score="0.65" /></span>Technical and Artistic Triumph<span>**<annotation type="AI" why="EXPLANATION" score="0.55" /></span> The shift to full 3D graphics was <span>executed beautifully<annotation type="AI" why="EXPLANATION" score="0.55" /></span>, with a clean, readable interface that never sacrifices functionality for form.',
        "AI",
        "mismatched",
        "0.5",
    )
    assert fixed is not None
    diag = format_diagnostics(fixed, doc)
    assert diag["ok"] is True
    assert strip_tags(fixed) == doc
    assert fixed == correct_fix


def test_fix_real_example_not_overcorrecting():
    """
    This tests a real example where the model originally removed a span annotation unnecessarily, when the only problem was an extra comma that the model added.
    """
    doc = "Some suggestions include: Bath bombs Bath salts Meditation CD, or relaxing music Scented candles A loofah A body brush Soaps - gourmet versions, glycerin, and scented, some castile Moisturizer for hands, face, and body Bath and/or shower gel Hand mitts Sleeping mask for relaxing Facial mask/ingredients, body mask Face washer, hand towel, body towel, etc. Lotions and creams Bubble bath, bath oil Perfume/scent/essential oils Shampoo/conditioner/hair oil, etc. Massage tools Cute stuffed toy A book or magazine to read in the bath Shower cap, bathrobe Lavender bags Any other items that you think would make the basket awesome; , That way you'll know"
    resp = _outer(
        'Some suggestions include: Bath bombs Bath salts Meditation CD, or relaxing music Scented candles A loofah A body brush Soaps - <span>gourmet versions, glycerin, and scented, some castile<annotation type="human" why="EXPLANATION" score="0.62" /></span>, Moisturizer for hands, face, and body Bath and/or shower gel Hand mitts Sleeping mask for relaxing <span>Facial mask/ingredients, body mask<annotation type="human" why="EXPLANATION" score="0.58" /></span> Face washer, hand towel, body towel, etc. Lotions and creams Bubble bath, bath oil Perfume/scent/essential oils Shampoo/conditioner/hair oil, etc. <span>Massage tools Cute stuffed toy A book or magazine to read in the bath<annotation type="human" why="EXPLANATION" score="0.78" /></span> Shower cap, bathrobe Lavender bags Any other items that you think would make the basket awesome; <span>,<annotation type="human" why="EXPLANATION" score="0.72" /></span> That way you&apos;ll know',
        "human",
        "misplaced",
        "0.7",
    )
    correct_fix = _outer(
        'Some suggestions include: Bath bombs Bath salts Meditation CD, or relaxing music Scented candles A loofah A body brush Soaps - <span>gourmet versions, glycerin, and scented, some castile<annotation type="human" why="EXPLANATION" score="0.62" /></span> Moisturizer for hands, face, and body Bath and/or shower gel Hand mitts Sleeping mask for relaxing <span>Facial mask/ingredients, body mask<annotation type="human" why="EXPLANATION" score="0.58" /></span> Face washer, hand towel, body towel, etc. Lotions and creams Bubble bath, bath oil Perfume/scent/essential oils Shampoo/conditioner/hair oil, etc. <span>Massage tools Cute stuffed toy A book or magazine to read in the bath<annotation type="human" why="EXPLANATION" score="0.78" /></span> Shower cap, bathrobe Lavender bags Any other items that you think would make the basket awesome; <span>,<annotation type="human" why="EXPLANATION" score="0.72" /></span> That way you&apos;ll know',
        "human",
        "misplaced",
        "0.7",
    )
    fixed = try_fix_response(resp, doc, max_fix_ratio=0.2)
    assert fixed is not None
    diag = format_diagnostics(fixed, doc)
    assert diag["ok"] is True
    assert strip_tags(fixed) == doc
    assert fixed == correct_fix


def test_fix_real_example_final_annotation_misplacement():
    """
    Originally, the model was misplacing the final annotation on the last word of the document by moving it earlier. We wanna ensure that it remains at the end.
    I think this might be a special case cause the model emitted an empty span annotation at the end of the text - so there's nothing to anchor to.
    I.e. here the text should NOT be fixed
    """
    doc = "Remove from pan and set aside. 4. Pour the vegetable mixture from the bowl into the skillet and stir it around until the vegetables"
    resp = _outer(
        '<span>Remove from pan and set aside.<annotation type="AI" why="EXPLANATION" score="0.35" /></span> 4. Pour the vegetable mixture from the bowl into the skillet and stir it around until the vegetables<span><annotation type="AI" why="EXPLANATION" score="0.86" /></span>',
        "human",
        "misplaced",
        "0.7",
    )
    fixed = try_fix_response(resp, doc, max_fix_ratio=0.2)
    assert fixed is None
    assert strip_tags(resp) == doc


def test_fix_real_example_final_annotation_misplacement_2():
    """
    This is a variant of the test above where we have a final word that the model didn't output and instead just replaced with an empty span annotation.
    """
    doc = "Remove from pan and set aside. 4. Pour the vegetable mixture from the bowl into the skillet and stir it around until the vegetables are"
    resp = _outer(
        '<span>Remove from pan and set aside.<annotation type="AI" why="EXPLANATION" score="0.35" /></span> 4. Pour the vegetable mixture from the bowl into the skillet and stir it around until the vegetables <span><annotation type="AI" why="EXPLANATION" score="0.86" /></span>',
        "human",
        "misplaced",
        "0.7",
    )
    correct_fix = _outer(
        '<span>Remove from pan and set aside.<annotation type="AI" why="EXPLANATION" score="0.35" /></span> 4. Pour the vegetable mixture from the bowl into the skillet and stir it around until the vegetables are<span><annotation type="AI" why="EXPLANATION" score="0.86" /></span>',
        "human",
        "misplaced",
        "0.7",
    )
    fixed = try_fix_response(resp, doc, max_fix_ratio=0.2)
    assert fixed is not None
    diag = format_diagnostics(fixed, doc)
    assert diag["ok"] is True
    assert strip_tags(fixed) == doc
    assert fixed == correct_fix


def test_fix_real_example_annotation_misplacement_3():
    """
    This is a variant of the test above where we have a final word that the model didn't output and instead just replaced with an empty span annotation.
    """
    doc = "Remove from pan and set aside. 4. Pour the vegetable mixture from the bowl into the skillet and stir it around until the vegetables are"
    resp = _outer(
        'Remove from pan and set aside.<span><annotation type="AI" why="EXPLANATION" score="0.35" /></span> 4. Pour the vegetable mixture from the bowl into the skillet and stir it around until the vegetables <span>are<annotation type="AI" why="EXPLANATION" score="0.86" /></span>',
        "human",
        "misplaced",
        "0.7",
    )
    fixed = try_fix_response(resp, doc, max_fix_ratio=0.2)
    assert fixed is None
    assert strip_tags(resp) == doc


def test_fix_real_example_redundant_text():
    """
    Redundant text can be tricky - let's make sure the model doesn't move annotations around unnecessarily.
    """
    doc = "**What's Included:** • Professional satellite dish installation and setup **Why Choose Dish Network:** • Access to 290+ channels with our top packages"
    resp = _outer(
        '<span>**What&apos;s Included:**<annotation type="AI" why="EXPLANATION" score="0.68" /></span> <span>•<annotation type="AI" why="EXPLANATION" score="0.58" /></span> Professional satellite dish installation and setup <span>**What&apos;s Included:**<annotation type="AI" why="EXPLANATION" score="0.5" /></span> <span>•<annotation type="AI" why="EXPLANATION" score="0.5" /></span> Access to <span>290+ channels<annotation type="human" why="EXPLANATION" score="0.22" /></span> with our top packages <span>•<annotation type="AI" why="EXPLANATION" score="0.45" /></span> Award-winning Hopper DVR technology <span>**Why Choose Dish Network:**<annotation type="AI" why="EXPLANATION" score="0.66" /></span> <span>•<annotation type="AI" why="EXPLANATION" score="0.4" /></span> Access to <span>290+ channels<annotation type="AI" why="EXPLANATION" score="0.78" /></span> with our top packages',
        "human",
        "redundant",
        "0.7",
    )
    correct_fix = _outer(
        '<span>**What&apos;s Included:**<annotation type="AI" why="EXPLANATION" score="0.68" /></span> <span>•<annotation type="AI" why="EXPLANATION" score="0.58" /></span> Professional satellite dish installation and setup <span>**Why Choose Dish Network:**<annotation type="AI" why="EXPLANATION" score="0.66" /></span> <span>•<annotation type="AI" why="EXPLANATION" score="0.4" /></span> Access to <span>290+ channels<annotation type="AI" why="EXPLANATION" score="0.78" /></span> with our top packages',
        "human",
        "redundant",
        "0.7",
    )
    fixed = try_fix_response(resp, doc, max_fix_ratio=0.2)
    assert fixed is not None
    diag = format_diagnostics(fixed, doc)
    assert diag["ok"] is True
    assert strip_tags(fixed) == doc
    assert fixed == correct_fix


def test_real_example_unclosed_span():
    doc = "We're thrilled that so many people have gone on to sign this petition and that we now have close to 100,"
    resp = _outer(
        'We&apos;re thrilled that so many people have gone on to sign this petition and that we now have close to 100,<span><span><span><annotation type="AI" why="EXPLANATION" score="0.96" /></span><annotation type="AI" why="EXPLANATION" score="0.82" /></span>',
        "human",
        "unclosed_span",
        "0.7",
    )
    correct_fix = _outer(
        'We&apos;re thrilled that so many people have gone on to sign this petition and that we now have close to 100,<span><span><annotation type="AI" why="EXPLANATION" score="0.96" /></span><annotation type="AI" why="EXPLANATION" score="0.82" /></span>',
        "human",
        "unclosed_span",
        "0.7",
    )
    fixed = try_fix_response(resp, doc, max_fix_ratio=0.2)
    assert fixed is not None
    diag = format_diagnostics(fixed, doc)
    assert diag["ok"] is True
    assert strip_tags(fixed) == doc
    assert fixed == correct_fix


def test_format_fix_does_not_double_escape_apos_in_why():
    doc = "Hi there"
    inner = wrap_span_piece(
        escape_document_piece("Hi"),
        {"type": "human", "why": "feels like a real reviewer&apos;s simple opinion", "score": "0.4"},
    )
    resp = _outer(inner, "human", "outer", "0.5")
    fixed = try_fix_response(resp, doc, max_fix_ratio=0.5)
    assert fixed is not None
    assert 'why="feels like a real reviewer&apos;s simple opinion"' in fixed
    assert "reviewer&amp;apos;s" not in fixed


def test_format_fix_keeps_annotations():
    doc = 'The ion sword finally charged. The negative impulses streamed across the blade creating a layer of plasma that could cut through granite as easily as air. I brought the sword high up above my head... There is no sport, just slaughter." "Oh?"Jeb fumed."I suppose you\'d slay a dragon buck naked, with your bare hands?'
    resp = _outer(
        'The ion sword finally charged. The negative impulses streamed across the blade creating a layer of plasma that could cut through granite as easily as air. I brought the sword high up above my head<span>...<annotation type="human" why="EXPLANATION" score="0.42" /></span> There is no no sport, just slaughter.&quot; &quot;Oh?&quot;<span>Jeb<annotation type="human" why="EXPLANATION" score="0.61" /></span> fumed.&quot;I suppose you&apos;d slay a dragon buck naked, with your bare hands?',
        "human",
        "overall good",
        "0.9",
    )
    correct_fix = _outer(
        'The ion sword finally charged. The negative impulses streamed across the blade creating a layer of plasma that could cut through granite as easily as air. I brought the sword high up above my head<span>...<annotation type="human" why="EXPLANATION" score="0.42" /></span> There is no sport, just slaughter.&quot; &quot;Oh?&quot;<span>Jeb<annotation type="human" why="EXPLANATION" score="0.61" /></span> fumed.&quot;I suppose you&apos;d slay a dragon buck naked, with your bare hands?',
        "human",
        "overall good",
        "0.9",
    )
    fixed = try_fix_response(resp, doc, max_fix_ratio=0.2)
    assert fixed is not None
    diag = format_diagnostics(fixed, doc)
    assert diag["ok"] is True
    assert strip_tags(fixed) == doc
    assert fixed == correct_fix


def test_nested_span_1():
    # Outer <span> is never closed (budget hit after writing two inner tells).
    # </span> can only appear as part of the " /></span>" special token, so a closed
    # outer span without an annotation is impossible — this is the realistic failure mode.
    doc = "The X-ray luminosity is = 1.3 x 10 28 erg/s in the keV band."
    resp = _outer(
        'The X-ray luminosity is <span>= 1.3 x 10 <span>28<annotation type="human" why="compact power-of-ten notation" score="0.34" /></span> <span>erg/s<annotation type="human" why="standard astrophysics unit abbreviation" score="0.30" /></span> in the keV band.',
        "human",
        "dense technical notation reads like an actual observation report not a summary",
        "0.6",
    )
    correct_fix = _outer(
        'The X-ray luminosity is = 1.3 x 10 <span>28<annotation type="human" why="compact power-of-ten notation" score="0.34" /></span> <span>erg/s<annotation type="human" why="standard astrophysics unit abbreviation" score="0.30" /></span> in the keV band.',
        "human",
        "dense technical notation reads like an actual observation report not a summary",
        "0.6",
    )
    fixed = try_fix_response(resp, doc, max_fix_ratio=0.2)
    assert fixed == correct_fix


def test_nested_span_2_not_fixed():
    """
    This example is okay - it should not be fixed because the nested spans are properly closed and the text matches the document.
    """
    doc = "Police have charged a Norwegian man over the attacks."
    resp = _outer(
        '<span>Police have <span>charged a Norwegian man<annotation type="AI" why="formal legal phrasing" score="0.38" /></span> over the attacks.<annotation type="AI" why="compressed one-sentence news lead" score="0.45" /></span>',
        "AI",
        "clean compressed AI-style news opening",
        "0.6",
    )
    correct_fix = None
    fixed = try_fix_response(resp, doc, max_fix_ratio=0.2)
    assert fixed == correct_fix


def test_nested_span_3_not_fixed():
    """
    This should not be fixed because the nested spans are properly closed and the text matches the document.
    """
    doc = "Significant improvement was observed in recovery metrics."
    resp = _outer(
        '<span>Significant <span><span>improvement<annotation type="AI" why="vague positive noun" score="0.38" /></span> was observed<annotation type="AI" why="passive-voice hedged claim" score="0.52" /></span> in recovery metrics.<annotation type="AI" why="full sentence reads like an AI-generated clinical summary" score="0.65" /></span>',
        "AI",
        "clinical-report AI phrasing throughout",
        "0.7",
    )
    correct_fix = None
    fixed = try_fix_response(resp, doc, max_fix_ratio=0.2)
    assert fixed == correct_fix


def test_nested_span_4():
    # Outer <span> opened, inner leaf annotated, outer never closed (budget hit).
    doc = "The study showed significant improvements in patient outcomes."
    resp = _outer(
        '<span>The study showed <span>significant<annotation type="AI" why="vague intensifier" score="0.35" /></span> improvements in patient outcomes.',
        "AI",
        "passive clinical summary style",
        "0.7",
    )
    correct_fix = _outer(
        'The study showed <span>significant<annotation type="AI" why="vague intensifier" score="0.35" /></span> improvements in patient outcomes.',
        "AI",
        "passive clinical summary style",
        "0.7",
    )
    fixed = try_fix_response(resp, doc, max_fix_ratio=0.2)
    assert fixed == correct_fix


def test_nested_span_5():
    doc = "We are thrilled to announce our new partnership with the team."
    resp = _outer(
        'We are <span>thrilled to <span>announce our new partnership with the team.<annotation type="AI" why="PR boilerplate opener" score="0.65" /></span>',
        "AI",
        "corporate press-release phrasing",
        "0.6",
    )
    correct_fix = _outer(
        'We are thrilled to <span>announce our new partnership with the team.<annotation type="AI" why="PR boilerplate opener" score="0.65" /></span>',
        "AI",
        "corporate press-release phrasing",
        "0.6",
    )
    fixed = try_fix_response(resp, doc, max_fix_ratio=0.2)
    assert fixed == correct_fix


def test_format_fix_fixes_unstripped_type_or_score_in_annotations():
    doc = "Sample text for testing."
    resp = _outer(
        'Sample text for testing.<span><annotation type=" human  " why="EXPLANATION" score=" 0.5 " /></span>',
        "  AI  ",
        "unstripped",
        " 0.5",
    )
    correct_fix = _outer(
        'Sample text for testing.<span><annotation type="human" why="EXPLANATION" score="0.5" /></span>',
        "AI",
        "unstripped",
        "0.5",
    )
    fixed = try_fix_response(resp, doc, max_fix_ratio=0.2)
    assert fixed is not None
    diag = format_diagnostics(fixed, doc)
    assert diag["ok"] is True
    assert strip_tags(fixed) == doc
    assert fixed == correct_fix


def test_fix_text_not_closed_properly():
    doc = "abc def"
    resp = '<text>abc def<verdict type="AI" why="EXPLANATION" score="0.18" /></text>" score="0.37" /></text>'
    correct_fix = '<text>abc def<verdict type="AI" why="EXPLANATION" score="0.18" /></text>'
    fixed = try_fix_response(resp, doc, max_fix_ratio=0.2)
    assert fixed is not None
    diag = format_diagnostics(fixed, doc)
    assert diag["ok"] is True
    assert strip_tags(fixed) == doc
    assert fixed == correct_fix


def test_fix_score_out_of_bounds_higher():
    doc = "abc def"
    resp = _outer(
        'abc def',
        "AI",
        "score out of bounds",
        "1.5",
    )
    correct_fix = _outer(
        'abc def',
        "AI",
        "score out of bounds",
        "1.0",
    )
    fixed = try_fix_response(resp, doc, max_fix_ratio=0.2)
    assert fixed is not None
    diag = format_diagnostics(fixed, doc)
    assert diag["ok"] is True
    assert strip_tags(fixed) == doc
    assert fixed == correct_fix


def test_fix_score_out_of_bounds_lower():
    doc = "abc def"
    resp = _outer(
        'abc def',
        "AI",
        "score out of bounds",
        "-1.5",
    )
    correct_fix = _outer(
        'abc def',
        "AI",
        "score out of bounds",
        "0.0",
    )
    fixed = try_fix_response(resp, doc, max_fix_ratio=0.2)
    assert fixed is not None
    diag = format_diagnostics(fixed, doc)
    assert diag["ok"] is True
    assert strip_tags(fixed) == doc
    assert fixed == correct_fix


def test_fix_multiple_annotations():
    doc = "Singer Shania Twain became a feminist country classic, and famous."
    resp = _outer(
        '<span>Singer<annotation type="AI" why="EXPLANATION1" score="0.58" /></span> Shania Twain became a <span>feminist country classic<annotation type="AI" why="EXPLANATION2" score="0.55" /></span>, and famous.<annotation type="AI" why="EXPLANATION3" score="0.76" /></span>',
        "AI",
        "multiple annotations",
        "0.5",
    )
    correct_fix = _outer(
        '<span>Singer<annotation type="AI" why="EXPLANATION1" score="0.58" /></span> Shania Twain became a <span>feminist country classic<annotation type="AI" why="EXPLANATION2" score="0.55" /></span>, and famous.',
        "AI",
        "multiple annotations",
        "0.5",
    )
    fixed = try_fix_response(resp, doc, max_fix_ratio=0.2)
    assert fixed is not None
    diag = format_diagnostics(fixed, doc)
    assert diag["ok"] is True
    assert strip_tags(fixed) == doc
    assert fixed == correct_fix


# def test_fix_empty_spans():
#     doc = "Singer Shania Twain became a feminist country classic."
#     resp = _outer(
#         'Singer<span><annotation type="AI" why="EXPLANATION1" score="0.58" /></span> Shania Twain became a <span>feminist country classic<span><annotation type="AI" why="EXPLANATION2" score="0.55" /></span>.',
#         "AI",
#         "empty spans",
#         "0.5",
#     )
#     correct_fix = _outer(
#         'Singer Shania Twain became a <span>feminist country classic<annotation type="AI" why="EXPLANATION2" score="0.55" /></span>.',
#         "AI",
#         "empty spans",
#         "0.5",
#     )
#     fixed = try_fix_response(resp, doc, max_fix_ratio=0.2)
#     assert fixed is not None
#     diag = format_diagnostics(fixed, doc)
#     assert diag["ok"] is True
#     assert strip_tags(fixed) == doc
#     assert fixed == correct_fix


def test_fix_annotations_with_some_wrong_tokens():
    doc = "In addition, the paper defines a basic architecture that defines the interaction between the main components."
    resp = _outer(
        'In addition, <span>the paper<span><span> defines<annotation type="human" why="typo for defines" score="0.78" /></span><annotation type="human" why="duplicated word" score="0.79" /></span> the interaction<annotation type="human" why="repeated define" score="0.76" /></span> between the main components.',
        "AI",
        "some wrong tokens",
        "0.5",
    )
    correct_fix = _outer(
        'In addition, <span>the paper<span><span> defines<annotation type="human" why="typo for defines" score="0.78" /></span><annotation type="human" why="duplicated word" score="0.79" /></span> a basic architecture that defines the interaction<annotation type="human" why="repeated define" score="0.76" /></span> between the main components.',
        "AI",
        "some wrong tokens",
        "0.5",
    )
    fixed = try_fix_response(resp, doc, max_fix_ratio=0.2)
    assert fixed is not None
    diag = format_diagnostics(fixed, doc)
    assert diag["ok"] is True
    assert strip_tags(fixed) == doc
    assert fixed == correct_fix


def test_dangling_tokens_after_text_is_finished():
    doc = "abc def"
    resp = '<text><span>abc<annotation type="human" why="EXPLANATION" score="0.30" /></span>) def<verdict type="AI" why="EXPLANATION" score="0.55" /></span>." score="0.45" /></text>'
    # Dangling `) ` before `def` is not in the document so surgical repair drops it;
    # the garbage tokens after the verdict self-close (</span>." score="0.45" />) are stripped.
    correct_fix = '<text><span>abc<annotation type="human" why="EXPLANATION" score="0.30" /></span> def<verdict type="AI" why="EXPLANATION" score="0.55" /></text>'
    fixed = try_fix_response(resp, doc, max_fix_ratio=0.2)
    assert fixed is not None
    diag = format_diagnostics(fixed, doc)
    assert diag["ok"] is True
    assert strip_tags(fixed) == doc
    assert fixed == correct_fix


def test_dangling_tokens_after_text_is_finished_2():
    doc = "abc def"
    resp = '<text>abc def<verdict type="AI" why="EXPLANATION1" score="0.82" /></span><verdict type="AI" why="EXPLANATION2" score="0.58" /></text>'
    correct_fix = '<text>abc def<verdict type="AI" why="EXPLANATION2" score="0.58" /></text>'
    fixed = try_fix_response(resp, doc, max_fix_ratio=0.2)
    assert fixed is not None
    diag = format_diagnostics(fixed, doc)
    assert diag["ok"] is True
    assert strip_tags(fixed) == doc
    assert fixed == correct_fix
