"""All prompt templates for training and evaluation."""

from rl_detector.fewshots import pick_fewshot
from rl_detector.config import CFG
from rl_detector.tell_xml import escape_document_piece, strip_score_attrs, get_outer_meta_dict, strip_text_wrapper


MODEL_IDENTITY = (
    "You are a smart and perspicacious forensic text analyst of AI and human tells."
)

FOCUS_HINT_CATEGORIES = [
    "I'll focus on vocabulary and word choice: unusual words, field-specific jargon, or statistically atypical phrasing.",
    "I'll focus on sentence structure: length variation, syntactic complexity, and grammatical patterns.",
    "I'll focus on factual claims and specificity: precision of details, named entities, and verifiable statements.",
    "I'll focus on formatting and punctuation: em dashes, Oxford commas, capitalization, and whitespace.",
    "I'll focus on tone and register: formality level, emotional affect, hedging, and directness.",
    "I'll focus on discourse markers and transitions: how ideas connect and flow between sentences.",
    "I'll focus on content patterns: topic handling, depth of knowledge, and breadth vs. specificity.",
    "I'll focus on rhetorical style: argumentation, persuasion techniques, and voice consistency.",
]


def get_focus_hint(rollout_index: int) -> str | None:
    """Return the focus hint for this rollout if focus_hints_enabled, else None."""
    if not getattr(CFG.training, "focus_hints_enabled", False):
        return None
    return FOCUS_HINT_CATEGORIES[rollout_index % len(FOCUS_HINT_CATEGORIES)]


def build_instructions() -> str:
    """Build the output rules."""
    score_rule = "\n- SCORE is a float 0.0..1.0: 0.0-0.25 weak, 0.25-0.75 moderate, 0.75-1.0 only for undeniable evidence."
    output_fmt = '<text>doc text...<span>TELL<annotation type="AI|human" why="EXPLANATION" score="0.0..1.0" /></span>...more doc...<verdict type="AI|human" why="VERDICT" score="0.0..1.0" /></text>'
    return f"""\
Rules:
- Reproduce the ENTIRE document character by character — no omissions.
- EXPLANATION: specific mechanism-based reason why SPAN is a tell; not generic or vague.{score_rule}
- Add >=1 tell; nested spans allowed. Think like a detective: style, content, formatting, semantics, grammar, vocabulary, inconsistencies.
- Maximize granularity: prefer small focused spans.
- OUTPUT ONLY: {output_fmt}"""

def label_think_prefix(main_label_hint: int) -> str:
    """Text appended after the assistant generation prompt to force the reasoning chain to
    start with a known label.  The model then generates the free continuation from there.

    Used when the chat template does NOT already open the analysis channel in the assistant
    prefix (add_generation_prompt emits <|start|>assistant only).
    Call detect_assistant_generation_suffix() once at startup to check.
    """
    origin = "AI" if main_label_hint == 1 else "human"
    return f"<|channel|>analysis<|message|>Text origin is {origin}."


def label_think_continuation(main_label_hint: int) -> str:
    """Like label_think_prefix but without the channel opener — for use when the
    chat template already emits <|channel|>analysis<|message|> as part of add_generation_prompt=True."""
    origin = "AI" if main_label_hint == 1 else "human"
    return f"Text origin is {origin}."


def _fewshot_block(main_text: str) -> str:
    """Build the few-shot block, or return empty string if disabled in config."""
    if not getattr(CFG.training, "use_fewshot_examples", False):
        return ""
    example = pick_fewshot(main_text=main_text)
    return (
        "Here is one example of how to annotate a document, with explanations for each tell:\n"
        f"Example:\n{example}\n"
    )


def build_prompt(text: str) -> str:
    """Build the user prompt bundle. ``text`` is LOGICAL document text only; escapes via ``escape_document_piece``."""

    fewshot = _fewshot_block(main_text=text)
    instructions = build_instructions()
    fewshot_section = (fewshot + "\n") if fewshot else ""
    return f"""\
{instructions}

{fewshot_section}Text:
<<<
{escape_document_piece(text)}
>>>"""


def build_rubric_prompt(tagged_text: str, n: int = 0) -> str:
    """Build the rubric evaluation prompt for the rubric model."""
    outer_meta = get_outer_meta_dict(tagged_text)
    shell = strip_text_wrapper(tx=tagged_text)
    clean_text = strip_score_attrs(shell if shell is not None else tagged_text)
    general_comment_line = ""
    if outer_meta is not None:
        outer_type = outer_meta.get("type", "")
        outer_why = outer_meta.get("explanation", "")
        general_comment_line = f'\nOverall verdict (type="{outer_type}"): {outer_why}\n'
    ann_count_line = f"\nJSON fields: ann must contain exactly {n} float{'s' if n != 1 else ''} (one per annotation, in order) — do NOT include the Overall verdict in ann. overall must contain exactly 1 float for the Overall verdict."
    return f"""\
You are a critical evaluator of authorship-detection annotations.

The annotated document uses <span>text<annotation type="AI|human" why="explanation" /></span> to mark evidence spans. Rate each annotation and the Overall verdict for credibility.

Credibility (0.0..1.0): how well does the why= explanation identify a specific, mechanistic reason the span (or the overall verdict) is a tell for the stated type? A mechanism is the underlying cause that would make an AI or human produce that exact text. Use the full range: 0.0 for vague, generic, or incorrect mechanisms; 1.0 only for explanations that state an undeniable mechanism. Also reward explanations that feel like a human reviewer would write over polished explanations. The overall verdict is also punished if it quotes or mentions text that wasn’t annotated previously. If there are no <span> annotations, you should still score the overall verdict. Assume 0.5 is the default, and adjust up or down based on the quality of the explanations.

Example input 1:
<<<
<span>**<annotation type="AI" why="markdown; AI often adds markdown formatting because keyboards make it easy" /></span>Apple to build <span>$1.375<annotation type="human" why="odd exact dollar amount; AI is more likely to fill in a generic amount like $1.234" /></span> billion data center. CEO Tim Cook announced Thursday that the company will build a <span>$1.375 billion<annotation type="human" why="redundant; humans tend to repeat their own text" /></span> data center located on <span>2,000<annotation type="human" why="specific land size" /></span> acres of land in <span>Waukee, Iowa<annotation type="human" why="specific location" /></span>. <span>Would you like me to continue?<annotation type="AI" why="chatbot speak" /></span>.
>>>

Overall verdict (type="AI"): This text is AI — extremely generic and lacking in the specific details and redundancies that would typically be included by a human author.

Example reasoning 1:
1. markdown: it's true that AI often adds markdown formatting, but the mechanism is wrong, AI can't use keyboards since it doesn’t have hands (credibility=0.00)
2. odd exact dollar amount: true, averaged training data makes AIs produce generic numbers (credibility=0.62)
3. redundant exact dollar amount: flipped — repetition artifacts are AI tells, not human (credibility=0.00)
4. specific land size: 2,000 is a round number, not specific — explanation is false (credibility=0.10)
5. specific location: specific detail that grounds the story, medium human tell (credibility=0.53)
6. chatbot speak: undeniable, no human would write this (credibility=1.00)
Overall verdict (baseline=0.50): it doesn't specify the mechanisms, just a vague claim of “generic and lacking in the specific details” (-0.30). And the writing style feels too polished and artificial (-0.10). (credibility=0.10)

Example input 2:
<<<
The <span>mechanism<annotation type="AI" why="classic AI phrase" /></span> of fever is <span>largely caused by the release of endorphins<annotation type="AI" why="this is false, a real doctor would know endorphins reduce stress" /></span> <span>( cytokines ) , which affect the <span>brains<annotation type="human" why="typo; humans can make them easily by typing quickly, but AI is trained specifically to avoid such errors" /></span> temperature <span>centre<annotation type="AI" why="British spelling; typical of AI" /></span> and trigger the <span>bodys respons<annotation type="human" why="typo again" /></span> to cold.
>>>

Overall verdict (type="human"): To me, this is written by AI, with phrases like “The mechanism of fever”. I was thinking that it could be human at first, because there are some typos that could be a human signal, but then I realized that there is a hallucination that no real doctor would make. To me, it's clear, this is AI written.

Example reasoning 2:
1. classic AI phrase: doesn't explain the mechanism (credibility=0.00)
2. false medical claim: undeniable falsehood, no real doctor would say this (credibility=1.00)
3. typo: undeniable, AI is trained to avoid typos (credibility=0.91)
4. British spelling: not a strong signal, many AIs are trained on American text (credibility=0.27)
5. typo again: undeniable, strong human signal (credibility=0.91)
6. punctuation errors: medium human signal, AI is trained to produce polished text (credibility=0.62)
Overall verdict (baseline=0.50): it's detailed, creative, and specific about the mechanisms (+0.40). Plus, the explanation language is a bit chatty and conversational, which feels like a human would write (+0.10). But it mentions “The mechanism of fever” and this is not in any annotation (-0.15). (credibility=0.85)

Input:
<<<
{clean_text}
>>>{general_comment_line}{ann_count_line}"""
