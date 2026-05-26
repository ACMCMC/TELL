import math
import unittest

from rl_detector.causal_tell_metrics import (
    Example,
    Tell,
    contradiction_metrics,
    delete_spans,
    evaluate_causal_tells,
    extract_spans,
    genericity_metrics,
    locate_tells,
    merge_intervals,
    spans_for,
)


class CausalTellMetricsTest(unittest.TestCase):
    def test_merge_intervals(self):
        self.assertEqual(merge_intervals([(5, 8), (1, 3), (2, 6), (10, 11)]), [(1, 8), (10, 11)])

    def test_locate_duplicate_spans_in_order(self):
        text = "I cant believe it. I cant stop laughing."
        tells = locate_tells(
            text,
            [
                {"span_text": "cant", "explanation": "first", "type": "human", "rubric_credibility": 0.9},
                {"span_text": "cant", "explanation": "second", "type": "human", "rubric_credibility": 0.7},
            ],
        )
        self.assertEqual([(t.start, t.end) for t in tells], [(2, 6), (21, 25)])
        self.assertEqual([t.polarity for t in tells], [-1, -1])
        self.assertEqual([t.score for t in tells], [0.9, 0.7])

    def test_extract_and_delete_spans(self):
        text = "alpha beta gamma delta"
        spans = [(6, 10), (17, 22)]
        self.assertEqual(extract_spans(text, spans), "beta [...] delta")
        self.assertEqual(delete_spans(text, spans), "alpha  gamma")
        self.assertEqual(delete_spans(text, spans, mask_token="[MASK]"), "alpha [MASK] gamma [MASK]")

    def test_contradiction_metrics(self):
        ex = Example(
            doc_id="x",
            text="abc",
            y=1,
            tells=[
                Tell(0, 1, 1, 0.9, "a", "good", "AI"),
                Tell(1, 2, -1, 0.8, "b", "bad", "human"),
                Tell(2, 3, -1, 0.2, "c", "weak", "human"),
            ],
        )
        metrics = contradiction_metrics([ex], high_score_threshold=0.5)
        self.assertAlmostEqual(metrics["contradiction_rate_high_score"], 0.5)
        self.assertAlmostEqual(metrics["weighted_contradiction_high_score"], 0.8 / 1.7)

    def test_genericity_requires_low_span_overlap(self):
        text = "Therefore this is concise."
        generic_low_overlap = Example(
            doc_id="a",
            text=text,
            y=1,
            tells=[Tell(0, 9, 1, 0.8, "Therefore", "typical of AI writing", "AI")],
        )
        generic_high_overlap = Example(
            doc_id="b",
            text=text,
            y=1,
            tells=[Tell(0, 9, 1, 0.8, "Therefore", "typical of AI writing because of Therefore", "AI")],
        )
        self.assertEqual(genericity_metrics([generic_low_overlap])["genericity_rate"], 1.0)
        self.assertEqual(genericity_metrics([generic_high_overlap])["genericity_rate"], 0.0)

    def test_evaluate_causal_tells_with_fake_scorer(self):
        ai = Example(
            doc_id="ai",
            text="Therefore this AI generated text is formal. Overall it concludes.",
            y=1,
            tells=[
                Tell(0, 27, 1, 0.9, "Therefore this AI generated", "explicit AI cue", "AI"),
                Tell(44, 51, 1, 0.8, "Overall", "summary cue", "AI"),
            ],
        )
        human = Example(
            doc_id="human",
            text="I cant believe this lol.",
            y=-1,
            tells=[Tell(2, 23, -1, 0.9, "cant believe this lol.", "typo and slang", "human")],
        )

        def score_fn(texts):
            out = []
            for text in texts:
                lower = text.lower()
                val = 0.5
                if "ai generated" in lower or "therefore" in lower or "overall" in lower:
                    val += 0.35
                if "cant" in lower or "lol" in lower:
                    val -= 0.35
                out.append(max(0.0, min(1.0, val)))
            return out

        metrics = evaluate_causal_tells([ai, human], score_fn=score_fn)
        self.assertEqual(metrics["n_examples"], 2)
        self.assertEqual(metrics["full"]["auroc"], 1.0)
        self.assertEqual(metrics["tell_only"]["auroc"], 1.0)
        self.assertGreater(metrics["comprehensiveness_drop_mean"], 0.0)
        self.assertGreater(metrics["signed_deletion_score"], 0.0)
        self.assertFalse(math.isnan(metrics["area_under_budget_curve_auroc"]))


class CausalSpanHelpersTest(unittest.TestCase):
    def test_spans_for_filters(self):
        ex = Example(
            doc_id="x",
            text="abcd",
            y=1,
            tells=[
                Tell(0, 1, 1, 0.4, "a", "", "AI"),
                Tell(1, 2, 1, 0.8, "b", "", "AI"),
                Tell(2, 3, -1, 0.9, "c", "", "human"),
            ],
        )
        self.assertEqual(spans_for(ex, polarity=1, min_score=0.5), [(1, 2)])
        self.assertEqual(spans_for(ex, polarity=-1), [(2, 3)])


if __name__ == "__main__":
    unittest.main()
