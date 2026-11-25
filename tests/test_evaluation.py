"""Metric tests include plausible but wrong answers and broken source references."""

import unittest

from langchain_core.documents import Document

from erp_copilot.evaluation.suite import load_golden, score_case, summarize


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.case = {
            "type": "answer",
            "facts": [["created"], ["delete"]],
            "relevant_chunk_ids": ["a"],
        }
        self.doc = Document(
            page_content="You can delete a payment only if its status is Created.",
            metadata={"chunk_id": "a"},
        )
        self.result = {
            "answer": "- You can delete a payment only if its status is Created. [1]",
            "context": [self.doc],
            "ranked": [self.doc],
        }

    def test_gold_has_twenty_valid_cases(self):
        cases = load_golden()["cases"]
        self.assertEqual(sum(c["type"] == "answer" for c in cases), 16)
        self.assertEqual(sum(c["type"] == "abstain" for c in cases), 4)

    def test_correct_quote_and_retrieval(self):
        scores = score_case(self.case, self.result)
        self.assertTrue(scores["pass"])
        self.assertEqual(scores["mrr_at_k"], 1)
        self.assertEqual(scores["ndcg_at_k"], 1)

    def test_wrong_answer_with_valid_citation_fails_grounding(self):
        self.result["answer"] = "- You can delete a payment in Posted status. [1]"
        scores = score_case(self.case, self.result)
        self.assertTrue(scores["valid_citations"])
        self.assertFalse(scores["grounded_quotes"])
        self.assertEqual(scores["fact_coverage"], 0.5)
        self.assertFalse(scores["pass"])

    def test_bad_or_missing_citation_fails(self):
        for answer in ("Created", "Created [9]", self.result["answer"] + " Invented extra claim."):
            self.result["answer"] = answer
            self.assertFalse(score_case(self.case, self.result)["pass"])

    def test_refusal_with_citation_does_not_pass_scope_case(self):
        case = {"type": "abstain", "facts": [], "relevant_chunk_ids": []}
        self.result["answer"] = "Not covered by this library. [1]"
        self.assertFalse(score_case(case, self.result)["pass"])
        self.result["answer"] = "Not covered by this library."
        self.assertTrue(score_case(case, self.result)["pass"])

    def test_rank_metrics_and_summary_errors(self):
        unrelated = Document(page_content="other", metadata={"chunk_id": "b"})
        self.result["ranked"] = [unrelated, self.doc]
        metrics = score_case(self.case, self.result)
        self.assertEqual(metrics["mrr_at_k"], 0.5)
        self.assertAlmostEqual(metrics["ndcg_at_k"], 0.6309297536)
        summary = summarize(
            [
                {"type": "answer", "metrics": metrics},
                {"type": "answer", "error": "failed", "metrics": {"pass": False}},
            ]
        )
        self.assertEqual((summary["completed"], summary["passed"], summary["errors"]), (2, 1, 1))

    def test_numeric_fact_does_not_match_other_numbers(self):
        self.case["facts"] = [["2"]]
        self.result["answer"] = "- Posted 2026 invoices. [1]"
        self.assertEqual(score_case(self.case, self.result)["fact_coverage"], 0)

    def test_suite_persists_errors_and_checks_version_drift(self):
        import json
        import tempfile
        from pathlib import Path
        from threading import RLock
        from types import SimpleNamespace
        from unittest.mock import patch

        from erp_copilot.evaluation.suite import run_suite

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "source-manifest.json").write_text('{"documents": []}')
            settings = SimpleNamespace(
                data_dir=root,
                embedding_model="fixture",
                embedding_dimensions=2,
                reranker_model="fixture",
                llm_filename="fixture",
            )
            cases = [
                {
                    "id": str(i),
                    "question": str(i),
                    "topic": "All topics",
                    "expected": "decline",
                    "type": "abstain",
                    "facts": [],
                    "relevant_chunk_ids": [],
                }
                for i in range(20)
            ]
            dataset = {
                "cases": cases,
                "source_hashes": {},
                "usage": "fixture",
                "snapshot_commit": "fixture",
            }
            service = SimpleNamespace(
                lock=RLock(), store=SimpleNamespace(documents=[], load=lambda: None)
            )

            def ask(question, *args):
                if question == "0":
                    raise RuntimeError("inference failed")
                return {"answer": "Not covered by this library.", "context": [], "ranked": []}

            service.ask = ask
            updates = []
            with (
                patch("erp_copilot.evaluation.suite.get_settings", return_value=settings),
                patch("erp_copilot.evaluation.suite.load_golden", return_value=dataset),
            ):
                _report, path = run_suite(
                    service, progress=lambda value, **kwargs: updates.append(value)
                )
                saved = json.loads(Path(path).read_text())
                self.assertEqual(saved["summary"]["errors"], 1)
                self.assertEqual(saved["summary"]["passed"], 19)
                self.assertEqual(len(saved["results"]), 20)
                self.assertEqual(updates[-1], 1)
                dataset["source_hashes"] = {"changed.md": "old"}
                with self.assertRaisesRegex(ValueError, "versions differ"):
                    run_suite(service)
