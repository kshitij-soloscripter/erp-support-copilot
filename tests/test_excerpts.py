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

    def test_headings_are_not_answer_units(self):
        self.assertFalse(
            any("Attachment limits" in u["text"] for u in build_evidence_units(self.docs))
        )


if __name__ == "__main__":
    unittest.main()
