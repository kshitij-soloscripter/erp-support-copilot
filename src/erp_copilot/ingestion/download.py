"""Download a reproducible, attributed subset of Microsoft's public ERP docs."""

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import PurePosixPath

import httpx

from erp_copilot.config.settings import ROOT, get_settings

API = "https://api.github.com/repos"


def download_corpus(progress=None) -> dict:
    settings = get_settings()
    source = json.loads((ROOT / "config/sources.json").read_text())
    repo, directory = source["repository"], source["directory"]
    raw = settings.data_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    with httpx.Client(
        timeout=60, follow_redirects=True, headers={"User-Agent": "erp-support-copilot"}
    ) as client:

        def get(url):
            response = client.get(url)
            response.raise_for_status()
            return response

        commit = get(f"{API}/{repo}/commits/main").json()["sha"]
        entries = get(f"{API}/{repo}/contents/{directory}?ref={commit}").json()
        entries = [
            e
            for e in entries
            if e["name"].endswith(".md") and any(t in e["name"] for t in source["topics"])
        ]
        entries = sorted(entries, key=lambda e: e["name"])[: settings.document_limit]
        if not entries:
            raise RuntimeError("No documentation matched the configured topics.")
        documents = []
        for i, entry in enumerate(entries):
            text = get(f"https://raw.githubusercontent.com/{repo}/{commit}/{entry['path']}").text

            # Expand shared text includes, never shell commands or remote arbitrary URLs.
            def expand(match, entry=entry):
                include = match.group(1)
                path = PurePosixPath(entry["path"]).parent / include
                import posixpath

                normalized = posixpath.normpath(str(path))
                if not normalized.startswith("articles/") or not normalized.endswith(".md"):
                    return ""
                try:
                    return get(
                        f"https://raw.githubusercontent.com/{repo}/{commit}/{normalized}"
                    ).text
                except httpx.HTTPError:
                    return ""

            text = re.sub(r"\[!INCLUDE[^\]]*?\]\(([^)]+)\)", expand, text, flags=re.IGNORECASE)
            (raw / entry["name"]).write_text(text)
            documents.append(
                {
                    "file": entry["name"],
                    "repository_path": entry["path"],
                    "source": f"https://github.com/{repo}/blob/{commit}/{entry['path']}",
                    "public_url": "https://learn.microsoft.com/en-us/dynamics365/"
                    + entry["path"].removeprefix("articles/").removesuffix(".md"),
                    "sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "product": source["product"],
                    "module": source["module"],
                    "topic": (
                        "Matching"
                        if "matching" in entry["name"].lower()
                        else "Payments"
                        if "payment" in entry["name"].lower()
                        else "Automation"
                        if "automation" in entry["name"].lower()
                        else "Vendor invoices"
                    ),
                }
            )
            if progress:
                progress(
                    (i + 1) / len(entries), desc=f"Downloading {i + 1}/{len(entries)} documents"
                )
        licenses = settings.data_dir / "licenses"
        licenses.mkdir(exist_ok=True)
        for name in ("LICENSE", "LICENSE-CODE", "ThirdPartyNotices"):
            try:
                (licenses / name).write_text(
                    get(f"https://raw.githubusercontent.com/{repo}/{commit}/{name}").text
                )
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code != 404:
                    raise
    manifest = {
        "repository": repo,
        "commit": commit,
        "license": source["license"],
        "downloaded_at": datetime.now(UTC).isoformat(),
        "documents": documents,
    }
    (settings.data_dir / "source-manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
