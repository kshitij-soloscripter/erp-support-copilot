"""Install the official llama.cpp binary locally without Homebrew changes."""

import hashlib
import json
import platform
import tarfile
import zipfile
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]


def main():
    system, machine = platform.system(), platform.machine().lower()
    if system != "Darwin" or machine not in ("arm64", "aarch64"):
        raise SystemExit(
            "Automatic runtime setup currently targets Apple Silicon. Install llama-server manually into .tools/ on other platforms."
        )
    destination = ROOT / ".tools"
    destination.mkdir(exist_ok=True)
    with httpx.Client(follow_redirects=True, timeout=180) as client:
        response = client.get(
            "https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/b11750"
        )
        response.raise_for_status()
        release = response.json()
        assets = [
            a
            for a in release["assets"]
            if ("macos-arm64" in a["name"] or "osx-arm64" in a["name"])
            and a["name"].endswith((".tar.gz", ".zip"))
        ]
        if not assets:
            raise SystemExit(
                "No Apple Silicon runtime asset found. Check the official llama.cpp releases."
            )
        asset = assets[0]
        archive = destination / asset["name"]
        digest = hashlib.sha256()
        with client.stream("GET", asset["browser_download_url"]) as download:
            download.raise_for_status()
            with archive.open("wb") as file:
                for block in download.iter_bytes():
                    file.write(block)
                    digest.update(block)
        expected = asset.get("digest")
        if expected and expected != "sha256:" + digest.hexdigest():
            archive.unlink()
            raise SystemExit("Runtime checksum mismatch.")
        if archive.name.endswith(".zip"):
            with zipfile.ZipFile(archive) as bundle:
                for member in bundle.infolist():
                    if (
                        not (destination / member.filename)
                        .resolve()
                        .is_relative_to(destination.resolve())
                    ):
                        raise ValueError("Unsafe archive path")
                bundle.extractall(destination)
        else:
            with tarfile.open(archive) as bundle:
                bundle.extractall(destination, filter="data")
        archive.unlink()
        for executable in destination.rglob("llama-*"):
            if executable.is_file() and executable.suffix not in (".dylib", ".so"):
                executable.chmod(executable.stat().st_mode | 0o111)
        (destination / "runtime.json").write_text(
            json.dumps(
                {
                    "release": release["tag_name"],
                    "asset": asset["name"],
                    "sha256": digest.hexdigest(),
                },
                indent=2,
            )
        )
    print(f"Installed {release['tag_name']} into .tools/")


if __name__ == "__main__":
    main()
