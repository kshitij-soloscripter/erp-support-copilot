"""Parent section expansion must keep split settings together."""

import unittest

from langchain_core.documents import Document

from erp_copilot.retrieval.store import SearchStore, join_passages


def make_chunk(text, identifier, section="Options"):
    return Document(
        page_content=text, metadata={"file": "dates.md", "section": section, "chunk_id": identifier}
    )


class ContextTests(unittest.TestCase):
    def test_split_options_are_kept_together(self):
        store = SearchStore()
        store.documents = [
            make_chunk("Use the system date.", "a"),
            make_chunk("Use the first day of the new open period.", "b"),
        ]
        context = store.expand_context([store.documents[0], store.documents[1]], 4)
        self.assertEqual(len(context), 1)
        self.assertIn("first day", context[0].page_content)
        self.assertEqual(context[0].metadata["expanded_chunk_ids"], ["a", "b"])

    def test_unrelated_section_is_not_merged(self):
        store = SearchStore()
        store.documents = [
            make_chunk("Use the system date.", "a"),
            make_chunk("Check cash discounts.", "b", "Discounts"),
        ]
        context = store.expand_context([store.documents[0]], 4)
        self.assertNotIn("discounts", context[0].page_content)

    def test_overlapping_boundary_is_not_duplicated(self):
        boundary = "This is a shared boundary sentence."
        self.assertEqual(
            join_passages(["First. " + boundary, boundary + " Last."]),
            "First. " + boundary + " Last.",
        )


if __name__ == "__main__":
    unittest.main()
