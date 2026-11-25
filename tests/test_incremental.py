"""Exercise vector deltas and revision ingestion against real local Qdrant."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from erp_copilot.ingestion.chunking import load_chunks
from erp_copilot.ingestion.update import ingest_revision
from erp_copilot.retrieval.store import SearchStore


class Embeddings:
    def __init__(self):
        self.texts = []
        self.fail = False

    def embed_documents(self, texts):
        if self.fail:
            raise RuntimeError("simulated embedding failure")
        self.texts.extend(texts)
        return [
            [1.0, float(int(hashlib.sha256(t.encode()).hexdigest()[:4], 16)) / 65535] for t in texts
        ]

    def embed_query(self, text):
        return [1.0, 0.5]


class IncrementalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.settings = SimpleNamespace(
            data_dir=self.root,
            embedding_model="fixture",
            embedding_dimensions=2,
            collection_name="test",
            qdrant_url=None,
            qdrant_api_key=None,
            retrieval_k=8,
            rerank_candidates=8,
        )
        for module in ("retrieval.store", "ingestion.chunking", "ingestion.update"):
            patcher = patch(f"erp_copilot.{module}.get_settings", return_value=self.settings)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.store = SearchStore()
        self.store.embeddings = Embeddings()
        self.addCleanup(lambda: self.store._client.close() if self.store._client else None)
        (self.root / "raw").mkdir()
        self.first = "# Guide\n\n## Alpha\n" + "Alpha invoice guidance remains stable. " * 5
        self.second = "\n## Beta\n" + "Beta payment policy allows review before posting. " * 5
        self.set_source(self.first + self.second)

    def set_source(self, text, commit="v1"):
        path = self.root / "raw/guide.md"
        path.write_text(text)
        entry = {
            "file": "guide.md",
            "repository_path": "articles/guide.md",
            "source": "https://example.org/guide",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "topic": "Payments",
        }
        (self.root / "source-manifest.json").write_text(
            json.dumps({"commit": commit, "documents": [entry]})
        )

    def test_noop_and_metadata_reuse(self):
        self.assertEqual(self.store.sync()["embedded"], 2)
        self.store.embeddings.texts.clear()
        self.assertEqual(self.store.sync()["embedded"], 0)
        self.set_source(self.first + self.second, "v2")
        stats = self.store.sync()
        self.assertEqual(stats["metadata_updated"], 2)
        self.assertEqual(self.store.embeddings.texts, [])
        self.assertTrue(all(d.metadata["commit"] == "v2" for d in self.store.documents))

    def test_edit_only_reembeds_affected_section(self):
        self.store.sync()
        stable = self.store.documents[0].metadata["chunk_id"]
        self.store.embeddings.texts.clear()
        self.set_source(self.first + self.second.replace("allows", "requires"))
        stats = self.store.sync()
        self.assertEqual((stats["embedded"], stats["reused"], stats["deleted"]), (1, 1, 1))
        self.assertEqual(len(self.store.embeddings.texts), 1)
        self.assertEqual(self.store.documents[0].metadata["chunk_id"], stable)
        self.assertTrue(
            any("requires" in d.page_content for d in self.store.search("requires", "Dense"))
        )

    def test_deleted_and_added_sections(self):
        self.store.sync()
        self.set_source(self.first)
        self.assertEqual(self.store.sync()["deleted"], 1)
        self.set_source(
            self.first + self.second + "\n## Gamma\n" + "Unique gamma invoice policy. " * 5
        )
        stats = self.store.sync()
        self.assertEqual((stats["embedded"], stats["reused"]), (2, 1))
        self.assertEqual(self.store.client.count("test", exact=True).count, 3)

    def test_failure_blocks_search_and_retry_recovers(self):
        self.store.sync()
        self.set_source(self.first + self.second.replace("allows", "requires"))
        self.store.embeddings.fail = True
        with self.assertRaisesRegex(RuntimeError, "simulated"):
            self.store.sync()
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            self.store.search("invoice", "Keyword")
        self.store.embeddings.fail = False
        self.assertEqual(self.store.sync()["embedded"], 1)
        self.assertFalse(self.store.chunks_path.with_suffix(".pending").exists())

    def test_missing_vector_repaired(self):
        self.store.sync()
        from qdrant_client import models

        points, _ = self.store.client.scroll("test")
        self.store.client.delete("test", models.PointIdsList(points=[points[0].id]), wait=True)
        self.assertEqual(self.store.sync()["embedded"], 1)

    def test_revision_archive_and_idempotent_upload(self):
        self.store.sync()
        upload = self.root / "revision.md"
        upload.write_text(self.first + self.second.replace("allows", "requires"))
        before = (self.root / "raw/guide.md").read_bytes()
        revision = ingest_revision("guide.md", str(upload))
        self.assertTrue(revision["source_changed"])
        archived = self.root / "versions" / hashlib.sha256(before).hexdigest() / "guide.md"
        self.assertEqual(archived.read_bytes(), before)
        stats = self.store.sync()
        self.assertEqual((stats["embedded"], stats["reused"]), (1, 1))
        self.assertTrue(
            all(d.metadata["commit"].startswith("local:") for d in self.store.documents)
        )
        self.assertFalse(ingest_revision("guide.md", str(upload))["source_changed"])
        self.assertEqual(self.store.sync()["embedded"], 0)

    def test_invalid_upload_and_tampered_source_do_not_mutate_index(self):
        self.store.sync()
        upload = self.root / "revision.md"
        upload.write_text("# Empty")
        with self.assertRaisesRegex(ValueError, "no usable"):
            ingest_revision("guide.md", str(upload))
        with self.assertRaisesRegex(ValueError, "existing"):
            ingest_revision("../guide.md", str(upload))
        (self.root / "raw/guide.md").write_text("tampered")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            load_chunks()
        self.assertEqual(self.store.client.count("test", exact=True).count, 2)

    def test_embedding_configuration_change_rejected(self):
        self.store.sync()
        self.settings.embedding_model = "different"
        with self.assertRaisesRegex(RuntimeError, "configuration changed"):
            self.store.sync()

    def test_search_blocks_stale_manifest_and_persistent_reload(self):
        self.store.sync()
        self.set_source(self.first + self.second, "v2")
        with self.assertRaisesRegex(RuntimeError, "Sources changed"):
            self.store.search("invoice", "Dense")
        self.store.sync()
        self.store._client.close()
        self.store._client = None
        self.store.documents = []
        self.store.load()
        self.assertEqual(len(self.store.documents), 2)

    def test_lexical_index_refresh_with_positive_bm25_scores(self):
        self.set_source(
            self.first + self.second + "\n## Gamma\n" + "Gamma covers a separate process. " * 5
        )
        self.store.sync()
        self.set_source(
            self.first
            + self.second.replace("allows", "requires")
            + "\n## Gamma\n"
            + "Gamma covers a separate process. " * 5
        )
        self.store.sync()
        self.assertIn("requires", self.store.search("requires", "Keyword")[0].page_content)
        self.assertEqual(self.store.search("allows", "Keyword"), [])

    def test_partial_upsert_retry_reuses_already_written_vector(self):
        self.store.sync()
        self.set_source(self.first + self.second.replace("allows", "requires"))
        with (
            patch.object(
                self.store.client, "delete", side_effect=RuntimeError("delete interrupted")
            ),
            self.assertRaisesRegex(RuntimeError, "delete interrupted"),
        ):
            self.store.sync()
        self.assertEqual(self.store.client.count("test", exact=True).count, 3)
        self.store.embeddings.texts.clear()
        stats = self.store.sync()
        self.assertEqual((stats["embedded"], stats["deleted"]), (0, 1))
        self.assertEqual(self.store.embeddings.texts, [])
        self.assertEqual(self.store.client.count("test", exact=True).count, 2)

    def test_frontmatter_only_change_reuses_all_vectors(self):
        self.store.sync()
        self.set_source("---\ntitle: Updated guide\n---\n" + self.first + self.second)
        stats = self.store.sync()
        self.assertEqual((stats["embedded"], stats["reused"], stats["metadata_updated"]), (0, 2, 2))
        self.assertEqual(self.store.documents[0].metadata["title"], "Updated guide")
