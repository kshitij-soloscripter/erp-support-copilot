"""LangChain composition and the shared, serialized application service."""

from threading import RLock
from time import perf_counter

from langchain_core.runnables import RunnableLambda

from erp_copilot.config.settings import get_settings
from erp_copilot.generation.answer import build_evidence_units, generate_answer
from erp_copilot.generation.policy import question_limitation
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
        state["context"] = self.store.expand_context(docs, get_settings().context_chunks)
        units = build_evidence_units(state["context"])
        state["answer_units"] = (
            self.reranker.select_units(state["question"], units) if state["rerank"] else units
        )
        state["rerank_seconds"] = perf_counter() - started
        return state

    def answer(self, state):
        started = perf_counter()
        state["answer"] = generate_answer(
            state["question"], state["context"], state["answer_units"]
        )
        state["generation_seconds"] = perf_counter() - started
        return state

    def ask(self, question, mode="Hybrid", rerank=True, topic="All topics"):
        question = question.strip()
        if not question or len(question) > 1500:
            raise ValueError("Enter a question between 1 and 1,500 characters.")
        with self.lock:
            limitation = question_limitation(question)
            if limitation:
                return {
                    "question": question,
                    "answer": limitation,
                    "documents": [],
                    "ranked": [],
                    "context": [],
                    "retrieval_seconds": 0.0,
                    "rerank_seconds": 0.0,
                    "generation_seconds": 0.0,
                }
            return self.chain.invoke(
                {"question": question, "mode": mode, "rerank": rerank, "topic": topic}
            )
