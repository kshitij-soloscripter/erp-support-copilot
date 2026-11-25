"""Validate and archive a document revision before synchronizing its changed chunks."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from erp_copilot.config.settings import get_settings
from erp_copilot.ingestion.chunking import split_document


def ingest_revision(filename: str, upload: str) -> dict:
    settings = get_settings()
    manifest_path = settings.data_dir / "source-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    entry = next((d for d in manifest["documents"] if d["file"] == filename), None)
    if entry is None or Path(filename).name != filename:
        raise ValueError("Select an existing library document.")
    path = Path(upload)
    if path.suffix.lower() != ".md" or path.stat().st_size > 5_000_000:
        raise ValueError("Upload a UTF-8 Markdown file smaller than 5 MB.")
    content = path.read_bytes()
    text = content.decode("utf-8")
    digest = hashlib.sha256(content).hexdigest()
    if not split_document(text, entry, manifest["commit"]):
        raise ValueError("The revision contains no usable documentation.")
    raw = settings.data_dir / "raw" / filename
    if hashlib.sha256(raw.read_bytes()).hexdigest() != entry["sha256"]:
        raise ValueError("The existing source differs from its manifest; restore it first.")
    if digest == entry["sha256"]:
        return {"document": filename, "source_changed": False, "sha256": digest}
    archive = settings.data_dir / "versions" / entry["sha256"]
    archive.mkdir(parents=True, exist_ok=True)
    (archive / filename).write_bytes(raw.read_bytes())
    (archive / f"{filename}.metadata.json").write_text(json.dumps(entry, indent=2))
    entry.update(
        sha256=digest,
        commit=f"local:{digest}",
        revision_note="Uploaded local revision; source link refers to the original public document.",
        ingested_at=datetime.now(UTC).isoformat(),
    )
    # Preserve the archived bytes if the application exits between these two atomic replaces.
    temporary = raw.with_suffix(".tmp")
    temporary.write_bytes(content)
    temporary.replace(raw)
    temporary = manifest_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2))
    temporary.replace(manifest_path)
    return {"document": filename, "source_changed": True, "sha256": digest}
