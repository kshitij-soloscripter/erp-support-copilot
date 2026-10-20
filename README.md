# ERP Support Copilot

A local Gradio assistant for Dynamics 365 Finance vendor-invoice questions. It retrieves public Microsoft documentation, combines semantic and keyword search, reranks evidence, and generates answers with numbered source references.

## Stack

- **Interface:** Gradio, bound to localhost.
- **Orchestration:** LangChain runnables.
- **Generation:** `unsloth/Qwen3-1.7B-GGUF`, `Qwen3-1.7B-Q4_K_M.gguf`, served by llama.cpp. Thinking disabled.
- **Embeddings:** `Qwen/Qwen3-Embedding-0.6B`, normalized **512D** vectors; **256D** configurable.
- **Retrieval:** persistent local Qdrant + BM25 over the same chunks, fused with RRF.
- **Reranking:** `Qwen/Qwen3-Reranker-0.6B`, yes/no relevance scoring.

## Run on Apple Silicon

Python 3.11–3.13, internet access for initial downloads, and several GB of free storage are required. 16 GB RAM is recommended; performance depends on hardware and corpus size. Models are downloaded from Hugging Face without training or an API key. Qwen embeddings and reranking use PyTorch MPS when available; generation uses llama.cpp Metal.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
python scripts/install_runtime.py
cp .env.example .env
python app.py
```

Open http://127.0.0.1:7860. In **Library**, download public documents, build the index, download the answer model, and start it. Then use **Ask**. The first index build also downloads the embedding model; first reranking downloads the reranker. Subsequent runs reuse caches. Run only one app process when using local Qdrant.

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
Question → dense + keyword retrieval → RRF → Qwen reranker
         → top evidence → quantized Qwen → cited answer + excerpts
```

Generation receives the selected source chunks. RRF is rank fusion, separate from model-based reranking. The Search details panel exposes both scores and stage timings. Retrieval modes and reranking can be changed for inspection; no benchmark claims are made.

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

The default context window is 4096 tokens, with four evidence chunks and up to 640 answer tokens. Reduce chunk count or increase the context window if experimenting with longer evidence. A reranker relevance score is a ranking signal, not a calibrated confidence probability.

## Limitations and next steps

- This library covers selected accounts-payable topics, not all ERP functionality.
- The assistant cannot inspect a live tenant or confirm an actual issue's cause.
- Numeric citation checks detect unknown source identifiers, not unsupported claims.
- Each question is retrieved independently; conversation is displayed but not used as implicit evidence.
- Includes that cannot be resolved are omitted. Tables and long procedures can span chunks.
- Formal retrieval and answer-quality evaluations are deferred until the application is stable.

## Development

```bash
ruff check .
python -m compileall -q src app.py scripts
```

The application code is MIT licensed. Model weights, runtime binaries, and documentation retain their respective upstream licenses. This project is independent and is not endorsed by Microsoft, Qwen, or Unsloth.
