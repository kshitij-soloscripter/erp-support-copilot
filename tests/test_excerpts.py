"""The model can select source IDs, but cannot invent answer wording."""

import unittest

from langchain_core.documents import Document

from erp_copilot.generation.answer import build_evidence_units, extract_answer, normalize_excerpt
from erp_copilot.generation.policy import ABSTENTION


class ExcerptTests(unittest.TestCase):
    def setUp(self):
        self.docs = [
            Document(
                page_content="# Attachment limits\nOnly **one** attachment can be the default.\nThe system recalculates the due date using the invoice date."
            )
        ]

    def test_real_excerpt_is_accepted(self):
        self.assertIn("**one** attachment", extract_answer({"units": [1]}, self.docs))

    def test_invented_fact_with_real_citation_is_rejected(self):
        self.assertIn(
            "verified excerpt",
            extract_answer(
                {"units": [1], "text": "Two attachments can be the default."}, self.docs
            ),
        )

    def test_negation_change_is_rejected(self):
        self.assertIn(
            "verified excerpt",
            extract_answer(
                {"units": [2], "text": "The system does not recalculate the due date."}, self.docs
            ),
        )

    def test_wrong_source_is_rejected(self):
        self.assertIn("invalid source", extract_answer({"units": [99]}, self.docs))

    def test_abstention_is_allowed(self):
        self.assertEqual(extract_answer({"units": []}, self.docs), ABSTENTION)

    def test_numbers_are_not_normalized_away(self):
        self.assertNotEqual(normalize_excerpt("Use -2 days."), normalize_excerpt("Use 2 days."))

    def test_lists_stay_together(self):
        docs = [
            Document(
                page_content="### Components\n+ Header data entities\n+ Invoice exception processing\n+ Side-by-side attachment viewer",
                metadata={"subsection": "Components"},
            )
        ]
        units = build_evidence_units(docs)
        self.assertEqual(len(units), 1)
        self.assertIn("exception processing", units[0]["text"])
        self.assertEqual(units[0]["heading"], "Components")

    def test_best_ranked_evidence_is_not_dropped(self):
        units = build_evidence_units(self.docs)
        units[0]["relevance_score"] = 0.99
        units[1]["relevance_score"] = 0.1
        self.assertIn("one** attachment", extract_answer({"units": [2]}, self.docs, units))

    def test_list_markers_are_not_doubled(self):
        docs = [
            Document(
                page_content="- Only one attachment can be the default.\n- Other attachments remain available."
            )
        ]
        self.assertFalse(extract_answer({"units": [1]}, docs).startswith("- -"))

    def test_secondary_example_does_not_replace_primary_rule(self):
        from unittest.mock import Mock

        from erp_copilot.reranking.qwen import QwenReranker

        reranker = QwenReranker()

        def fake_rank(query, documents):
            for d in documents:
                d.metadata["rerank_score"] = 0.9
            return documents

        reranker.rerank = Mock(side_effect=fake_rank)
        units = [
            {"id": 1, "source": 1, "text": "Use a To date adjustment of 2 days."},
            {
                "id": 2,
                "source": 2,
                "text": "A different example uses a minimum date adjustment of -2.",
            },
        ]
        selected = reranker.select_units("To date adjustment", units)
        self.assertTrue(all(u["source"] == 1 for u in selected))

    def test_empty_selection_can_show_labeled_strong_evidence(self):
        units = build_evidence_units(self.docs)
        units[0]["relevance_score"] = 0.999
        self.assertTrue(
            extract_answer({"units": []}, self.docs, units).startswith("Closest source excerpt:")
        )

    def test_yes_no_answer_preserves_explicit_limit(self):
        docs = [
            Document(
                page_content="You can set one attachment as the default.\nYou can set only one attachment as the default."
            )
        ]
        units = build_evidence_units(docs)
        units[0]["relevance_score"] = 0.9999
        units[1]["relevance_score"] = 0.999
        answer = extract_answer({"units": [1]}, docs, units, "Can I set two default attachments?")
        self.assertIn("only one", answer)

    def test_headings_are_not_answer_units(self):
        self.assertFalse(
            any("Attachment limits" in u["text"] for u in build_evidence_units(self.docs))
        )


if __name__ == "__main__":
    unittest.main()
