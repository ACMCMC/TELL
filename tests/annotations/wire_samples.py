"""Realistic annotation wire strings for tokenizer and mask tests.

These are logical XML snippets in the same shape the model emits (``tell_xml`` conventions).
"""

# Tiny sanity checks (also used for checkpoint golden id lists).
WIRE_MINIMAL = '<span>x<annotation type="AI" why="ok" score="0.50" /></span>'

WIRE_MINIMAL_OUTER = (
    '<text><span>x<annotation type="AI" why="ok" score="0.50" /></span>'
    '<verdict type="AI" why="ok" score="0.50" /></text>'
)

WIRE_NESTED_TWO_TELLS = (
    '<span>a<span>b<annotation type="AI" why="n" score="0.4" />'
    '</span><annotation type="human" why="o" score="0.5" /></span>'
)

# Review-style: nested span + outer human verdict.
WIRE_REVIEW_NESTED = (
    '<span>The room was <span>spotless<annotation type="human" why="specific praise; '
    'AI often writes generic cleanliness" score="0.61" /></span> and quiet at night.'
    '<annotation type="human" why="two concrete observations in one sentence" score="0.55" /></span>'
)

# Apostrophe entity inside span (dataset-style escaping in the wild).
WIRE_APOS_IN_SPAN = (
    '<span>We&apos;d stay again.<annotation type="AI" why="contraction in casual review" '
    'score="0.38" /></span>'
)

# Long ``why`` value (many subwords) + decimal score.
WIRE_LONG_WHY = (
    '<span>Policy<annotation type="AI" why="'
    "This paragraph stacks abstract policy nouns without a concrete anchor; "
    "human memos usually name one department, one date, or one exception path."
    '" score="0.44" /></span>'
)

# Mixed nested tells (style similar to format-fix regression corpus).
WIRE_TRIPLE_SPAN = (
    '<span>Intro <span>middle<annotation type="AI" why="short nested claim" score="0.51" /></span>'
    ' tail.<annotation type="human" why="outer framing" score="0.47" /></span>'
)

# Unicode in span and in ``why`` (escaped attrs flow through ``wrap_outer`` elsewhere; here raw UTF-8).
WIRE_UNICODE = '<span>café<annotation type="human" why="diacritics in body and why" score="0.29" /></span>'

ALL_ROUNDTRIP_WIRES = [
    WIRE_MINIMAL,
    WIRE_MINIMAL_OUTER,
    WIRE_NESTED_TWO_TELLS,
    WIRE_REVIEW_NESTED,
    WIRE_APOS_IN_SPAN,
    WIRE_LONG_WHY,
    WIRE_TRIPLE_SPAN,
    WIRE_UNICODE,
]
