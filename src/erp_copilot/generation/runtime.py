"""Download Unsloth GGUF and manage a loopback-only llama.cpp server."""

import atexit
import subprocess
import time

import httpx

from erp_copilot.config.settings import ROOT, get_settings

_process = None


def download_model() -> str:
    from huggingface_hub import hf_hub_download

    settings = get_settings()
    return hf_hub_download(
        repo_id=settings.llm_repository,
        filename=settings.llm_filename,
        local_dir=str(ROOT / "models"),
    )


def ready() -> bool:
    try:
        return httpx.get(get_settings().llm_base_url + "/health", timeout=2).is_success
    except httpx.HTTPError:
        return False


def stop():
    global _process
    if _process and _process.poll() is None:
        _process.terminate()
        try:
            _process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _process.kill()
    _process = None


def start() -> str:
    global _process
    if ready():
        return "Local generation server is ready."
    settings = get_settings()
    model = ROOT / "models" / settings.llm_filename
    if not model.exists():
        raise RuntimeError("Download the answer model in the Library tab first.")
    candidates = sorted((ROOT / ".tools").rglob("llama-server"))
    if not candidates:
        raise RuntimeError("llama-server is missing. Run: python scripts/install_runtime.py")
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    common = [
        str(candidates[0]),
        "-m",
        str(model),
        "--host",
        "127.0.0.1",
        "--port",
        str(settings.llm_port),
        "-c",
        str(settings.llm_context_size),
        "--jinja",
        "--chat-template-kwargs",
        '{"enable_thinking":false}',
        "--parallel",
        "1",
        "--fit",
        "off",
        "-t",
        "4",
        "-tb",
        "4",
    ]
    cpu_flags = ["--device", "none", "-ngl", "0", "--no-kv-offload", "--no-op-offload"]
    attempts = [cpu_flags] if settings.torch_device() == "cpu" else [["-ngl", "99"], cpu_flags]
    for flags in attempts:
        with (logs / "llama-server.log").open("a") as log:
            _process = subprocess.Popen(common + flags, stdout=log, stderr=log)
        for _ in range(120):
            if ready():
                return "Local generation server is ready."
            if _process.poll() is not None:
                break
            time.sleep(1)
        if _process.poll() is None:
            stop()
    raise RuntimeError("Model server could not start. See logs/llama-server.log.")


atexit.register(stop)
