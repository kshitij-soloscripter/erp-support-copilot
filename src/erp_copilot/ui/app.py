"""Gradio demo: cited answers, inspectable evidence, and local library controls."""

import html
import json
import logging

import gradio as gr

from erp_copilot.config.settings import get_settings
from erp_copilot.generation.runtime import download_model, start
from erp_copilot.ingestion.download import download_corpus
from erp_copilot.pipeline import Copilot

LOGGER = logging.getLogger(__name__)
CSS = """.gradio-container {max-width: 1180px !important;} #hero {padding: 24px 0 8px;} footer {display:none !important;}"""


def create_app() -> gr.Blocks:
    service = Copilot()
    settings = get_settings()

    def library_status():
        path = settings.data_dir / "source-manifest.json"
        count = len(json.loads(path.read_text())["documents"]) if path.exists() else 0
        return f"**{count} public documents** · **{settings.embedding_dimensions}D embeddings** · Qwen3 1.7B Q4_K_M"

    def ask(message, history, mode, rerank, topic):
        history = list(history or [])
        if not message.strip():
            return history, "", "", [], ""
        history.append({"role": "user", "content": message})
        try:
            result = service.ask(message, mode, rerank, topic)
            history.append({"role": "assistant", "content": result["answer"]})
            sources = []
            for i, doc in enumerate(result["context"], 1):
                sources.append(
                    f"### [{i}] {html.escape(doc.metadata['title'])}\n[Open source]({doc.metadata['source']}) · {html.escape(doc.metadata.get('section', 'Accounts payable'))}\n\n> "
                    + html.escape(doc.page_content).replace("\n", "\n> ")
                )
            rows = [
                [
                    d.metadata["title"],
                    round(d.metadata.get("fusion_score", 0), 5),
                    round(d.metadata["rerank_score"], 4) if "rerank_score" in d.metadata else None,
                ]
                for d in result["ranked"]
            ]
            timing = " · ".join(
                f"{label}: {result[key]:.1f}s"
                for label, key in [
                    ("Search", "retrieval_seconds"),
                    ("Rerank", "rerank_seconds"),
                    ("Answer", "generation_seconds"),
                ]
            )
            return history, "\n\n---\n\n".join(sources), timing, rows, ""
        except Exception as exc:
            LOGGER.exception("Question failed")
            history.append(
                {"role": "assistant", "content": f"Unable to answer: {html.escape(str(exc))}"}
            )
            return history, "", "See the Library tab for setup controls.", [], message

    def download(progress=gr.Progress()):  # noqa: B008 - Gradio injects progress via this default
        with service.lock:
            count = len(download_corpus(progress)["documents"])
        return f"Downloaded {count} documents. Build the index next.", library_status()

    def build(progress=gr.Progress()):  # noqa: B008 - Gradio injects progress via this default
        with service.lock:
            count = service.store.build(progress)
        return f"Indexed {count} chunks. Your library is ready.", library_status()

    def model_download():
        with service.lock:
            path = download_model()
        return f"Answer model downloaded: {path.rsplit('/', 1)[-1]}"

    def model_start():
        with service.lock:
            return start()

    with gr.Blocks(
        title="ERP Support Copilot", css=CSS, theme=gr.themes.Soft(primary_hue="teal")
    ) as app:
        gr.Markdown(
            "# ERP Support Copilot\nFind practical answers in public Dynamics 365 Finance documentation.",
            elem_id="hero",
        )
        status = gr.Markdown(library_status())
        with gr.Tab("Ask"):
            with gr.Row():
                with gr.Column(scale=3):
                    chat = gr.Chatbot(
                        type="messages",
                        height=420,
                        label="Support conversation",
                        sanitize_html=True,
                    )
                    question = gr.Textbox(
                        label="Your question",
                        placeholder="How does three-way invoice matching work?",
                        lines=2,
                        max_lines=5,
                    )
                    with gr.Row():
                        send = gr.Button("Find answer", variant="primary")
                        clear = gr.Button("Clear")
                    gr.Examples(
                        examples=[
                            [
                                "What is the difference between two-way and three-way invoice matching?"
                            ],
                            ["How can I configure vendor invoice matching?"],
                            ["What does vendor invoice automation do?"],
                        ],
                        inputs=question,
                    )
                with gr.Column(scale=2):
                    evidence = gr.Markdown(
                        "Sources will appear here after your first question.", label="Evidence"
                    )
            with gr.Accordion("Search details", open=False):
                with gr.Row():
                    mode = gr.Radio(
                        ["Hybrid", "Dense", "Keyword"], value="Hybrid", label="Retrieval"
                    )
                    rerank = gr.Checkbox(value=True, label="Rerank evidence with Qwen")
                    topic = gr.Dropdown(
                        ["All topics", "Matching", "Payments", "Automation", "Vendor invoices"],
                        value="All topics",
                        label="Topic filter",
                    )
                timing = gr.Markdown()
                ranking = gr.Dataframe(
                    headers=["Document", "Fusion score", "Rerank score"], interactive=False
                )
            gr.Markdown(
                "Answers use the indexed library. Check source excerpts before changing an ERP configuration. Each question searches independently."
            )
            inputs = [question, chat, mode, rerank, topic]
            outputs = [chat, evidence, timing, ranking, question]
            send.click(ask, inputs, outputs, concurrency_id="local-models", concurrency_limit=1)
            question.submit(
                ask, inputs, outputs, concurrency_id="local-models", concurrency_limit=1
            )
            clear.click(lambda: ([], "", "", [], ""), outputs=outputs, queue=False)
        with gr.Tab("Library"):
            gr.Markdown(
                "## Prepare your library\nRun these steps once. Downloads and indexes persist locally. Model downloads require several GB of disk space."
            )
            setup_status = gr.Markdown()
            with gr.Row():
                fetch = gr.Button("1 · Download public documents")
                index = gr.Button("2 · Build knowledge index")
            with gr.Row():
                fetch_model = gr.Button("3 · Download answer model")
                launch_model = gr.Button("4 · Start answer model")
            fetch.click(download, outputs=[setup_status, status], concurrency_id="local-models")
            index.click(build, outputs=[setup_status, status], concurrency_id="local-models")
            fetch_model.click(model_download, outputs=setup_status, concurrency_id="local-models")
            launch_model.click(model_start, outputs=setup_status, concurrency_id="local-models")
            gr.Markdown(
                "**Source:** MicrosoftDocs Dynamics 365 public documentation, Accounts payable.\n\nDocumentation: CC BY 4.0. Microsoft is the original author; preprocessing and chunking are applied locally. Repository snapshots and original license notices are saved during download."
            )
        with gr.Tab("About"):
            gr.Markdown(
                "## From question to evidence\nLangChain orchestrates dense Qwen retrieval and BM25 keyword retrieval, Reciprocal Rank Fusion, Qwen reranking, and a local Unsloth quantized Qwen answer model.\n\nThe library covers vendor invoices and matching in Dynamics 365 Finance. It does not connect to a live ERP installation. Citation checks validate source identifiers; they do not prove that every generated claim is supported.\n\nCode modules separate ingestion, embedding, retrieval, reranking, generation, configuration, and interface. Formal quality evaluations are planned after the application is stable."
            )
    return app


def launch():
    logging.basicConfig(level=logging.INFO)
    create_app().queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1", server_port=get_settings().app_port, share=False
    )
