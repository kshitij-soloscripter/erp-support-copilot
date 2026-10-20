"""Regressions for the two answerability failures and citation validation."""

import unittest

from erp_copilot.generation.answer import validate_answer
from erp_copilot.generation.policy import ABSTENTION, question_limitation


class AnswerPolicyTests(unittest.TestCase):
    def test_wrong_product_is_rejected(self):
        self.assertIn(
            ABSTENTION, question_limitation("Which SAP transaction code reverses a vendor invoice?")
        )

    def test_actual_company_numbers_are_not_doc_examples(self):
        self.assertIn(
            "cannot access live",
            question_limitation("What was our company's unpaid vendor balance yesterday?"),
        )

    def test_balance_instructions_stay_allowed(self):
        self.assertIsNone(
            question_limitation("How do I view my company's outstanding vendor balances?")
        )

    def test_supported_product_question_stays_allowed(self):
        self.assertIsNone(
            question_limitation("What is three-way matching in Dynamics 365 Finance?")
        )

    def test_uncited_abstention_is_valid(self):
        self.assertEqual(validate_answer(ABSTENTION, 4), ABSTENTION)

    def test_unknown_citation_is_rejected(self):
        self.assertIn("invalid source", validate_answer("Do this [5].", 4))

    def test_truncated_answer_is_not_shown_as_complete(self):
        self.assertIn("truncated", validate_answer("Do this [1]", 4, "length"))

    def test_cited_answer_keeps_its_content(self):
        self.assertEqual(validate_answer("Compare prices [1].", 1, "stop"), "Compare prices [1].")


if __name__ == "__main__":
    unittest.main()
