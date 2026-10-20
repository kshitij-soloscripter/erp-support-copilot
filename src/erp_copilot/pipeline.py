"""LangChain composition and the shared, serialized application service."""

from threading import RLock
from time import perf_counter

from langchain_core.runnables import RunnableLambda

from erp_copilot.config.settings import get_settings
from erp_copilot.generation.answer import generate_answer
from erp_copilot.reranking.qwen import QwenReranker
from erp_copilot.retrieval.store import SearchStore


class Copilot:
    def __init__(self):
        self.lock = RLock()
        self.store = SearchStore()
        self.reranker = QwenReranker()
        self.chain = (
            RunnableLambda(self.retrieve) | RunnableLambda(self.rank) | RunnableLambda(self.answer)
        )

    def retrieve(self, state):
        started = perf_counter()
        state["documents"] = self.store.search(state["question"], state["mode"], state["topic"])
        state["retrieval_seconds"] = perf_counter() - started
        return state

    def rank(self, state):
        started = perf_counter()
        docs = (
            self.reranker.rerank(state["question"], state["documents"])
            if state["rerank"] and state["documents"]
            else state["documents"]
        )
        state["ranked"] = docs
        state["context"] = docs[: get_settings().context_chunks]
        state["rerank_seconds"] = perf_counter() - started
        return state

    def answer(self, state):
        started = perf_counter()
        state["answer"] = generate_answer(state["question"], state["context"])
        state["generation_seconds"] = perf_counter() - started
        return state

    def ask(self, question, mode="Hybrid", rerank=True, topic="All topics"):
        question = question.strip()
        if not question or len(question) > 1500:
            raise ValueError("Enter a question between 1 and 1,500 characters.")
        with self.lock:
            return self.chain.invoke(
                {"question": question, "mode": mode, "rerank": rerank, "topic": topic}
            )
