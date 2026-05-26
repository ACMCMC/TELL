"""Annotation surface tests: special-token remap, wire encode/decode, masks.

Layout
------
``wire_samples.py``
    Curated XML wires only (no pytest).

``conftest.py``
    Shared ``remapped_tok`` fixture (``openai/gpt-oss-120b`` + in-memory remap).

``test_remap_and_ids.py``
    Remap rows, single-id encode, added-token tables, ``rollouts`` / SFT id alignment.

``test_encode_decode_and_parse.py``
    Roundtrip corpus, ``tell_xml`` parse, checkpoint goldens, ``wrap_outer_logical_plain_mid``.

``test_masks.py``
    Synthetic id state machine for ``compute_annotation_token_mask``, real-wire mask
    properties, ``find_attr_value_spans``.
"""
