# ERP Support Copilot code walkthrough

This guide explains the application from a beginner's perspective. Read the flow first, then the file descriptions. Each function gets a short explanation; library internals are outside this guide.

## 1 The whole application

**Prepare once:** public Microsoft documents → clean text → small chunks → Qwen embeddings → Qdrant and BM25.

**For each question:** check scope → retrieve relevant chunks → combine rankings → rerank → ask Qwen to select answer units → copy verified source text and citations → display the answer and sources.

The app reads documentation. It cannot inspect an ERP tenant, make transactions, or know your company's actual balances. To keep the small local model focused, answer selection uses the strongest parent section; other retrieved sections remain visible. Full synthesis across multiple sections is a future extension.

## 2 Concepts you need

| Term | Meaning in this project |
|---|---|
| ERP | Software connecting business processes such as purchasing, inventory, and accounting. The library covers Dynamics 365 Finance Accounts payable. |
| RAG | Retrieval Augmented Generation: give a language model relevant documents before it answers. This does not train or change its weights. |
| Chunk | A small piece of a document. We split around headings, then limit pieces to about 1,100 characters, with 150-character overlap to preserve nearby context. |
| Metadata | Labels attached to a chunk: document title, topic, source URL, repository version, and identifier. |
| Embedding | A numerical representation of meaning. Here each chunk becomes a normalized list of 512 numbers; 256 is configurable. |
| Model size versus dimensions | The embedding model has about 600 million parameters. Its output has 512 dimensions. These measure different things. |
| Qdrant and cosine similarity | Qdrant stores vectors and metadata. Cosine similarity compares vector direction to find semantically related text. |
| BM25 | Keyword ranking. It helps with exact terms that semantic search might overlook. BM25 runs in Python over the same chunks as Qdrant. |
| Hybrid search and RRF | Hybrid search combines semantic and keyword retrieval. Reciprocal Rank Fusion adds `1 / (60 + rank)` from each list, avoiding direct comparison of incompatible raw scores. |
| Reranking | A second model reads each question and candidate passage together. Qwen scores the relative likelihood of “yes” versus “no,” then sorts passages. This is not calibrated confidence or proof of answerability. |
| Evidence unit | A paragraph or list item inside retrieved context. Qwen returns its ID; application code copies the original wording, preventing invented paraphrases. |
| Contextual compression | A second relevance pass keeps the best evidence units before the answer model sees them, reducing distracting background. |
| Context and tokens | Context is the evidence passed to the answer model. Tokens are model text units, not characters. The local server has a 4,096-token context window. |
| Quantization and GGUF | Quantization reduces weight precision to save memory. The answer model is an Unsloth Qwen3 1.7B model in Q4_K_M GGUF format, run by llama.cpp. Embeddings and reranking use separate half-precision models. |
| LangChain runnable | A composable processing step. The `|` operator connects retrieval, reranking, and answer generation. |
| Abstention and citations | Abstention means declining an unsupported question. A valid citation number identifies a retrieved passage; it does not automatically prove the claim is correct. |

## 3 Read the code in this order

Paths below are relative to the project folder. `__init__` is Python's object constructor. A `@property` method behaves like an attribute. “Lazy loading” means loading an expensive model only when first needed.

### Entry points

| File | What it does |
|---|---|
| `app.py` | Imports and calls `launch()` when run directly; contains no custom functions. |
| `Run.command` | Changes to the project folder and starts `app.py` with its virtual environment. No Python functions. |
| `src/erp_copilot/__init__.py` | Imports settings early so model libraries use the project cache. No functions. |
| Other package `__init__.py` files | Mark `config`, `ingestion`, `embeddings`, `retrieval`, `reranking`, `generation`, and `ui` as packages. They contain no functions. |

### Configuration

**`src/erp_copilot/config/settings.py`** defines a `Settings` object. Environment variables and `.env` can override defaults. Cache paths are set before model imports.

| Function | What happens |
|---|---|
| `Settings.model_cache_dir()` | Returns the project-local Hugging Face model cache path. |
| `Settings.data_dir()` | Returns the folder containing documents, chunk metadata, and vector storage. |
| `Settings.collection_name()` | Builds a Qdrant collection name containing the embedding dimension, keeping 256D and 512D indexes separate. |
| `Settings.torch_device()` | Uses the configured device or selects CUDA, then Apple MPS, then CPU. |
| `get_settings()` | Loads and caches one validated settings object; only 256 or 512 dimensions are accepted. |

### Downloading and chunking

**`src/erp_copilot/ingestion/download.py`** gets public documentation and records where it came from.

| Function | What happens |
|---|---|
| `download_corpus(progress)` | Selects documents using `config/sources.json`, downloads a fixed repository snapshot, saves license notices, and writes a source manifest with hashes. Returns that manifest. |
| Nested `get(url)` | Makes an HTTP request and raises an error if it fails, instead of accepting an error page as documentation. |
| Nested `expand(match, entry)` | Replaces a supported Markdown include with text from the same repository snapshot. Rejects paths outside the allowed articles area; unresolved includes are omitted. |

**`src/erp_copilot/ingestion/chunking.py`** converts those files into LangChain `Document` objects: text plus metadata.

| Function | What happens |
|---|---|
| `load_chunks()` | Verifies source hashes, extracts front matter, removes selected publishing markup, splits text, attaches provenance, and removes duplicate chunk IDs. Returns the chunk list. |

A SHA-256 hash is a content fingerprint: changing a file changes its hash. Stable identifiers let us recognize the same chunk on later runs.

### Embeddings

**`src/erp_copilot/embeddings/qwen.py`** implements LangChain's embedding interface.

| Function | What happens |
|---|---|
| `QwenEmbeddings.__init__()` | Stores settings; leaves the model unloaded. |
| `QwenEmbeddings.model()` | Lazily loads Qwen3-Embedding-0.6B, chooses the device and output dimensions, and limits input to 512 tokens. |
| `embed_documents(texts)` | Encodes document text in small batches and normalizes vectors. Used when building the index. |
| `embed_query(text)` | Adds a retrieval instruction to the question and encodes one normalized vector. Used when searching. |

A class groups related data and functions. `self` refers to the current object; `return` sends a result back to the caller. An API lets another program call the app, while the CLI lets you call it from a terminal.

Document and question vectors must use the same embedding model and dimensions. Changing either requires rebuilding the index.

### Retrieval

**`src/erp_copilot/retrieval/store.py`** owns both search paths.

| Function | What happens |
|---|---|
| `join_passages(texts)` | Joins neighboring chunks and removes an exact shared boundary so overlap is not repeated. |
| `tokenize(text)` | Lowercases text and extracts words, numbers, and technical terms for BM25. |
| `SearchStore.__init__()` | Creates lightweight configuration and embedding wrappers; defers database and index loading. |
| `SearchStore.client()` | Lazily opens persistent local Qdrant or connects to the configured remote server. |
| `SearchStore.chunks_path()` | Returns the dimension-specific JSON file used to rebuild the in-memory keyword index. |
| `SearchStore.load()` | Checks the saved embedding model, collection existence, and point count; loads chunks and constructs BM25. |
| `SearchStore.build(progress)` | Rebuilds the selected collection, embeds chunks, uploads vectors and payloads, atomically saves the chunk file, then loads BM25. |
| `SearchStore.expand_context(ranked, limit)` | Groups candidates by document and heading, adds nearby chunks, and limits sections to 2,400 characters and total context to 6,000 characters. |
| Nested `section_key(doc)` | Creates the document, section, and subsection grouping key. |
| `SearchStore.search(query, mode, topic)` | Runs dense search, keyword search, or both. Applies the topic filter to both paths, fuses duplicate IDs with RRF, and returns up to eight candidates by default. |

Qdrant stores dense vectors; BM25 is separate application code. Local Qdrant permits one owning application process at a time.

### Reranking

**`src/erp_copilot/reranking/qwen.py`** scores shortlisted chunks more carefully.

| Function | What happens |
|---|---|
| `QwenReranker.__init__()` | Stores settings and leaves the model unloaded. |
| `QwenReranker.load()` | Loads the tokenizer and model, prepares the documented chat template, and finds the “yes” and “no” token IDs. |
| `QwenReranker.rerank(query, documents)` | Scores each pair without gradient tracking, uses only the final token's logits, attaches relevance scores, and sorts candidates. |
| `QwenReranker.select_units(query, units, limit)` | Focuses on the strongest parent section, shortlists up to 12 units with BM25, reranks them, and returns the best six with relevance scores. |

A logit is a model score before normalization. `logits_to_keep=1` avoids computing vocabulary predictions for every earlier token when only the last position is needed.

### Answer policy and generation

**`src/erp_copilot/generation/policy.py`** prevents specific unsupported requests from reaching generation.

| Function | What happens |
|---|---|
| `question_limitation(question)` | Returns a clear refusal for recognized unsupported ERP products or requests for actual company financial figures. Allows instructional questions such as how to view a balance. Returns `None` otherwise. |

This conservative, rule-based guard is not a complete understanding of every possible question. When Qwen selects nothing, a unit above `MIN_EXCERPT_SCORE` can be shown as “Closest source excerpt.” That threshold is a retrieval setting, not an accuracy probability.

**`src/erp_copilot/generation/runtime.py`** manages the answer model's local server.

| Function | What happens |
|---|---|
| `download_model()` | Downloads the configured Unsloth GGUF into `models/`. |
| `ready()` | Checks the local server's health endpoint; returns a boolean. |
| `stop()` | Terminates only the server this process started; kills it if graceful shutdown times out. Registered for application exit. |
| `start()` | Checks files, starts llama.cpp on localhost, waits for readiness, and retries on CPU if an accelerated attempt fails. Reuses an already-running healthy server. |

**`src/erp_copilot/generation/answer.py`** turns evidence into selectable units. The model chooses IDs; application code controls the displayed wording.

| Function | What happens |
|---|---|
| `normalize_excerpt(text)` | Normalizes whitespace and harmless Markdown for span checks while preserving words, numbers, and negation. |
| `split_evidence_blocks(text)` | Groups consecutive list items so a three-part definition or list stays intact; ignores standalone headings. |
| `build_evidence_units(docs)` | Assigns IDs and heading labels to original paragraphs and list blocks; splits unusually long blocks. |
| `extract_answer(payload, docs, units, question)` | Validates IDs and source spans, retains the strongest ranked unit as the answer anchor, preserves explicit restrictions in yes/no answers, and renders up to two original excerpts. An empty selection gives abstention, or a clearly labeled strong-evidence fallback. |
| `generate_answer(question, docs, selected_units)` | Uses compressed evidence, a JSON schema limited to valid unit IDs, and selection examples. Qwen selects at most two units; the application displays their original wording. |
| `validate_answer(answer, source_count, finish_reason)` | Rejects token-limit truncation, unfinished reasoning, empty output, and invalid or absent citations. Preserves the exact supported abstention message. |

Validation checks output structure. Qwen selects IDs rather than writing the displayed answer. Short fictional examples teach the selection format; they do not train the model or supply the ERP test answers. Exact source matching prevents changed wording, but relevance and completeness still need review.

### Pipeline orchestration

**`src/erp_copilot/pipeline.py`** connects the modules without putting their implementations in one file.

| Function | What happens |
|---|---|
| `Copilot.__init__()` | Creates the shared store, reranker, and reentrant lock; connects three LangChain runnables. |
| `Copilot.retrieve(state)` | Searches for candidates and records retrieval time. |
| `Copilot.rank(state)` | Reranks chunks, expands up to four parent sections, and compresses their units through a second relevance pass. Stores selected units and total reranking time. |
| `Copilot.answer(state)` | Selects the answer excerpts and records timing. |
| `Copilot.ask(question, mode, rerank, topic)` | Validates question length, serializes requests, checks scope, then invokes the chain. Guarded refusals return no evidence and skip model calls. |

`state` is a dictionary passed between stages: question → candidates → context → answer. The lock prevents concurrent heavy requests from competing for limited memory.

### Gradio interface

**`src/erp_copilot/ui/app.py`** displays the service and connects buttons to callbacks.

| Function | What happens |
|---|---|
| `create_app()` | Creates one shared `Copilot` and builds the Ask, Library, and About tabs. Returns the Gradio application. |
| Nested `library_status()` | Reads the manifest and displays document count, dimensions, and answer-model information. |
| Nested `ask(...)` | Calls the service and formats chat, evidence, rankings, and timing. Labels related sources when no answer was established; logs failures. |
| Nested `download(progress)` | Downloads documents while holding the service lock; updates the setup status. |
| Nested `build(progress)` | Builds the knowledge index under the same lock. |
| Nested `model_download()` | Downloads answer-model weights and returns a status message. |
| Nested `model_start()` | Starts the local generation server under the service lock. |
| Clear-button lambda | Returns empty chat, evidence, timing, rankings, and input values. |
| `launch()` | Enables logging, queues work one request at a time, and serves Gradio on `127.0.0.1:7860`. |

Chat history is displayed, but retrieval treats each question independently. Restate context in follow-up questions.

### CLI and support scripts

| File and function | What happens |
|---|---|
| `src/erp_copilot/cli.py` → `main()` | Parses commands for the app, download, indexing, model download, model server, or a single question; delegates to the same modules. |
| `scripts/install_runtime.py` → `main()` | Downloads the pinned Apple Silicon llama.cpp release, checks its digest when supplied, safely extracts it, sets executable permissions, and records runtime provenance. |
| `scripts/check_pipeline.py` → `classify_answer(answer)` | Labels runtime failures, generation failures, abstentions, or ordinary answers using text patterns. This is a mechanical heuristic, not a factual judge. |
| `scripts/check_pipeline.py` → `check_case(client, case)` | Sends a question through the actual Gradio API and records answer, sources, rankings, timing, and structural checks. |
| `scripts/check_pipeline.py` → `main()` | Selects cases, fingerprints application code, runs requests sequentially, and saves results after every question so progress survives a later failure. |

### Regression tests

**`tests/test_answer_policy.py`** uses Python's built-in `unittest`; it does not download or run models.

| Function | Regression checked |
|---|---|
| `test_wrong_product_is_rejected()` | SAP questions do not get Dynamics instructions. |
| `test_actual_company_numbers_are_not_doc_examples()` | Example amounts cannot become live company balances. |
| `test_balance_instructions_stay_allowed()` | Legitimate “how do I view my balance” questions remain allowed. |
| `test_supported_product_question_stays_allowed()` | A normal Dynamics Finance question passes the scope guard. |
| `test_uncited_abstention_is_valid()` | An intentional refusal need not cite an irrelevant source. |
| `test_unknown_citation_is_rejected()` | Invented source numbers are rejected. |
| `test_truncated_answer_is_not_shown_as_complete()` | Token-limit output is marked incomplete. |
| `test_cited_answer_keeps_its_content()` | Valid cited output is preserved. |

**`tests/test_context.py`** verifies evidence expansion without loading models.

| Function | Regression checked |
|---|---|
| `make_chunk(text, identifier, section)` | Creates a small synthetic chunk for the tests. |
| `test_split_options_are_kept_together()` | Adjacent parts of a settings list appear in one evidence section. |
| `test_unrelated_section_is_not_merged()` | Different headings stay separate. |
| `test_overlapping_boundary_is_not_duplicated()` | Shared boundary text appears only once. |

**`tests/test_excerpts.py`** checks that model selections cannot rewrite evidence.

| Function | Regression checked |
|---|---|
| `setUp()` | Creates a small synthetic source before each test. |
| `test_real_excerpt_is_accepted()` | A valid ID returns the original source text. |
| `test_invented_fact_with_real_citation_is_rejected()` | Extra model-authored text is rejected even with a real ID. |
| `test_negation_change_is_rejected()` | A supplied contradiction cannot replace the original statement. |
| `test_wrong_source_is_rejected()` | An unknown unit ID is refused. |
| `test_abstention_is_allowed()` | An empty selection becomes a clear refusal. |
| `test_numbers_are_not_normalized_away()` | Normalization preserves the difference between `-2` and `2`. |
| `test_headings_are_not_answer_units()` | A document heading cannot be selected as the answer. |
| `test_lists_stay_together()` | The items in a components list remain one selectable unit with its heading. |
| `test_best_ranked_evidence_is_not_dropped()` | A weaker model choice cannot discard the strongest ranked passage. |
| `test_list_markers_are_not_doubled()` | Original lists render without a duplicate bullet prefix. |
| `test_secondary_example_does_not_replace_primary_rule()` | A nearby example from another section does not replace the main retrieved rule. |
| Nested `fake_rank(query, documents)` | Supplies controlled scores for that regression test without loading a model. |
| `test_empty_selection_can_show_labeled_strong_evidence()` | A strong-evidence fallback is explicitly labeled rather than pretending it was a model answer. |
| `test_yes_no_answer_preserves_explicit_limit()` | A yes/no answer includes “only one” rather than just “one is allowed.” |

## 4 Other files and folders

| Path | Purpose |
|---|---|
| `config/sources.json` | Public repository, article directory, topic selection, and source metadata. |
| `.env.example` | Copyable runtime settings; `.env` stores your local overrides and is ignored by Git. |
| `pyproject.toml` | Package metadata, dependency ranges, CLI entry point, and formatter settings. |
| `requirements-lock.txt` | Exact dependency versions recorded on the development machine. |
| `.python-version` | Records Python 3.12 for compatible version managers. |
| `.gitignore` | Excludes credentials, environments, models, downloaded documents, caches, and generated indexes. |
| `README.md` and `LICENSE` | Setup instructions and the application's MIT license; upstream materials retain their own licenses. |
| `evaluation/questions.json` | Ten reviewed question specifications and their expected source or abstention behavior. |
| `evaluation/results.json` | Saved full-pipeline results, timings, mechanical checks, and manual review notes. |
| `docs/VALIDATION.md` | Human-readable account of the ten-question check and its limitations. |
| `data/`, `models/`, `.cache/`, `.tools/`, `logs/` | Generated documents, weights, caches, binaries, and local diagnostics. Not committed. |

## 5 Commands to remember

```bash
# Start the app after setup
.venv/bin/python app.py

# Run the ten questions while the app is already running
.venv/bin/python scripts/check_pipeline.py

# Run fast regression checks without model inference
.venv/bin/python -m unittest discover -s tests -v
```

Use `--ids Q02 Q09` to rerun selected pipeline cases. Expect CPU requests to be slower than regression tests. Review the saved evidence before describing an answer as factually correct.
