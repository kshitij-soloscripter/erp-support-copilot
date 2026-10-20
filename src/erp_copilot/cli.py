"""Small CLI over the same services used by Gradio."""

import argparse

from erp_copilot.config.settings import get_settings


def main():
    parser = argparse.ArgumentParser(description="ERP Support Copilot")
    parser.add_argument(
        "command", choices=["app", "download", "index", "download-model", "serve-model", "ask"]
    )
    parser.add_argument("question", nargs="?")
    args = parser.parse_args()
    get_settings()
    if args.command == "app":
        from erp_copilot.ui.app import launch

        launch()
    elif args.command == "download":
        from erp_copilot.ingestion.download import download_corpus

        print(f"Downloaded {len(download_corpus()['documents'])} public documents.")
    elif args.command == "index":
        from erp_copilot.retrieval.store import SearchStore

        print(f"Indexed {SearchStore().build()} chunks.")
    elif args.command == "download-model":
        from erp_copilot.generation.runtime import download_model

        print(download_model())
    elif args.command == "serve-model":
        from erp_copilot.generation.runtime import start

        print(start())
        try:
            input("Press Enter to stop the server.\n")
        except EOFError:
            import threading

            threading.Event().wait()
    elif args.command == "ask":
        if not args.question:
            parser.error("ask requires a quoted question")
        from erp_copilot.pipeline import Copilot

        result = Copilot().ask(args.question)
        print(result["answer"])
        for i, doc in enumerate(result["context"], 1):
            print(f"[{i}] {doc.metadata['source']}")


if __name__ == "__main__":
    main()
