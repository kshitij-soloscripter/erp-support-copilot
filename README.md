# ERP Support Copilot

A local Gradio assistant for Dynamics 365 Finance vendor-invoice questions. It retrieves public Microsoft documentation, combines semantic and keyword search, reranks evidence, and selects source-verified answer excerpts with numbered source references.

## Stack

- **Interface:** Gradio, bound to localhost.
- **Orchestration:** LangChain runnables.
- **Generation:** `unsloth/Qwen3-1.7B-GGUF`, `Qwen3-1.7B-Q4_K_M.gguf`, served by llama.cpp. Thinking disabled.
- **Embeddings:** `Qwen/Qwen3-Embedding-0.6B`, normalized **512D** vectors; **256D** configurable.
- **Retrieval:** persistent local Qdrant + BM25 over the same chunks, fused with RRF.
- **Reranking:** `Qwen/Qwen3-Reranker-0.6B`, yes/no relevance scoring for chunks and compressed evidence units.
- **Grounding:** the answer model selects valid evidence IDs; application code copies original text instead of displaying unverified paraphrases.

## Run on Apple Silicon

Python 3.11–3.13, internet access for initial downloads, and several GB of free storage are required. 16 GB RAM is recommended; performance depends on hardware and corpus size. Models are downloaded from Hugging Face without training or an API key. Qwen embeddings and reranking use half-precision weights and PyTorch MPS when available; generation uses llama.cpp Metal, with CPU fallback.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
python scripts/install_runtime.py  # pinned llama.cpp b11750
cp .env.example .env
python app.py
```

Open http://127.0.0.1:7860. In **Library**, download public documents, build the index, download the answer model, and start it. Then use **Ask**. The first index build also downloads the embedding model; first reranking downloads the reranker. Subsequent runs reuse caches. After setup, double-click `Run.command` to start the app. Run only one app process when using local Qdrant.

For a reproducible dependency installation, install `requirements-lock.txt` before `pip install -e . --no-deps`. The lock records the development machine's environment; platform-specific wheels can differ.

### CLI

```bash
erp-copilot download
erp-copilot index
erp-copilot download-model
erp-copilot ask "How does three-way invoice matching work?"
```

The CLI answer command starts the local model server if necessary. The Gradio process keeps its server alive between questions and shuts it down on exit. The app does not stop a server it did not launch.

## Pipeline

```text
Public Markdown → heading-aware chunks → Qwen embeddings → Qdrant
                                  └──────────────────────→ BM25
Question → scope guard → dense + keyword retrieval → RRF → chunk reranking
         → expand related sections → unit compression → quantized Qwen selects IDs
         → verified original excerpts + citations
```

Generation receives a compressed shortlist of evidence units from expanded source sections. Lists remain intact to avoid missing options or components. RRF is rank fusion, separate from model-based reranking. By default, eight chunks are reranked, related parent sections are expanded, and up to twelve evidence units receive a second relevance pass. The strongest parent section supplies the evidence units. Its best six units reach Qwen, which selects at most two under a constrained JSON schema; the strongest reranked unit anchors the answer. Empty selections may show a labeled closest-excerpt fallback when `MIN_EXCERPT_SCORE` is met. This score is not an accuracy probability. The Search details panel exposes both scores, stage timings, and topic filters applied to both retrieval paths. Retrieval modes and reranking can be changed for inspection; no benchmark claims are made.

## Project layout

```text
app.py                     thin launch entry point
config/sources.json        public-corpus selection
src/erp_copilot/
  config/                  settings and local paths
  ingestion/               download, provenance, chunking
  embeddings/              Qwen dense encoder
  retrieval/               Qdrant, BM25, fusion
  reranking/               Qwen relevance scorer
  generation/              runtime management and grounded answers
  ui/                      Gradio interface
  pipeline.py              LangChain composition
  cli.py                   command-line interface
scripts/install_runtime.py local llama.cpp installer
```

## Public dataset and attribution

The corpus is a selected subset of **Microsoft's Dynamics 365 Finance Accounts payable documentation**, from [MicrosoftDocs/dynamics-365-unified-operations-public](https://github.com/MicrosoftDocs/dynamics-365-unified-operations-public/tree/main/articles/finance/accounts-payable). Microsoft and the repository's contributors are the original authors. Documentation is published under [CC BY 4.0](https://github.com/MicrosoftDocs/dynamics-365-unified-operations-public/blob/main/LICENSE); code samples have separate notices. This project downloads original notices and records the source commit, URLs, and hashes in `data/source-manifest.json`. Local transformations include include expansion, metadata extraction, cleanup, and chunking. Cite the original documentation when reusing it.

Corpus downloads, models, runtime binaries, credentials, and generated indexes are ignored by Git. Source downloads happen through the Library tab or CLI. `config/sources.json` controls the selected topics and `.env` controls the document limit.

## Configuration

Copy `.env.example` to `.env`. Use `EMBEDDING_DIMENSIONS=256` or `512`. Rebuild the index after changing dimensions, embedding model, or corpus. Collections and local chunk files are separated by dimension. The app validates the embedding model and point count when loading an index. `DEVICE=cpu` overrides accelerator selection.

The local server has a 4096-token context window. Up to four parent sections are available as evidence, while answer selection focuses on the strongest section. `LLM_MAX_TOKENS=384` limits the generated selection JSON, not the length of source excerpts displayed afterward. A reranker score is a ranking signal, not a calibrated confidence probability.

## Limitations and next steps

- This library covers selected accounts-payable topics, not all ERP functionality.
- Answer selection focuses on the highest-ranked parent section to suit the small local model. Other sections remain visible; full synthesis across several sections is a future extension.
- The assistant cannot inspect a live tenant or confirm an actual issue's cause.
- The output renderer validates source IDs and copies original passages. This prevents invented wording but does not prove that selected evidence is relevant, complete, or applicable to a live tenant.
- Conservative scope rules reject recognized unsupported ERP products and requests for live company balances; they are not an exhaustive intent classifier.
- Each question is retrieved independently; conversation is displayed but not used as implicit evidence.
- Includes that cannot be resolved are omitted. Tables and long procedures can span chunks.
- A ten-question validation report and targeted regression tests are included; these curated cases are not a broad accuracy benchmark.

## Walkthrough and validation

- [Beginner code walkthrough](docs/WALKTHROUGH.md): every project function and the concepts behind it.
- [Ten-question validation report](docs/VALIDATION.md): questions, reviewed outcomes, fixes, and limitations.
- `evaluation/questions.json` and `evaluation/results.json`: reproducible inputs and recorded outputs. Source excerpts in evaluation artifacts retain the Microsoft documentation CC BY 4.0 license and attribution.

With the app already running, execute `.venv/bin/python scripts/check_pipeline.py`. For fast checks without model inference, run `.venv/bin/python -m unittest discover -s tests -v`.

## Development

```bash
ruff check .
python -m compileall -q src app.py scripts
```

The application code is MIT licensed. Model weights, runtime binaries, and documentation retain their respective upstream licenses. This project is independent and is not endorsed by Microsoft, Qwen, or Unsloth.
