"""Separate retrieval quality, reference facts, quote checks, and scope behavior."""

import hashlib
import json
import math
import re
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from erp_copilot.config.settings import ROOT, get_settings
from erp_copilot.generation.answer import normalize_excerpt
from erp_copilot.generation.policy import ABSTENTION

GOLD = ROOT / "evaluation/golden.json"


def load_golden(path: Path = GOLD) -> dict:
    dataset = json.loads(path.read_text())
    cases = dataset["cases"]
    if len(cases) != 20 or len({c["id"] for c in cases}) != 20:
        raise ValueError("The golden dataset must contain 20 uniquely identified cases.")
    for case in cases:
        if case["type"] not in ("answer", "abstain"):
            raise ValueError("Unknown expected behavior.")
        if case["type"] == "answer" and not (
            case["facts"] and case["relevant_chunk_ids"] and case["source"]
        ):
            raise ValueError(f"Missing expected evidence or facts: {case['id']}")
    return dataset


def score_case(case: dict, result: dict) -> dict:
    answer = result["answer"]
    text = normalize_excerpt(answer).casefold()
    abstained = answer.startswith(
        (ABSTENTION, "I cannot access live company", "I couldn't find supporting")
    )
    contexts = result.get("context", [])
    parts = re.split(r"\[(\d+)\]", answer.removeprefix("Closest source excerpt:\n\n"))
    citations = [int(parts[i]) for i in range(1, len(parts), 2)]
    valid = bool(citations) and all(1 <= n <= len(contexts) for n in citations)
    grounded = valid and not parts[-1].strip()
    for i, citation in enumerate(citations):
        quote = normalize_excerpt(parts[2 * i].strip().removeprefix("- "))
        grounded = (
            grounded
            and bool(quote)
            and (
                1 <= citation <= len(contexts)
                and quote in normalize_excerpt(contexts[citation - 1].page_content)
            )
        )
    groups = case["facts"]
    matches = [
        any(
            re.search(r"(?<!\w)" + re.escape(normalize_excerpt(term).casefold()) + r"(?!\w)", text)
            for term in group
        )
        for group in groups
    ]
    relevant = set(case["relevant_chunk_ids"])
    ranked = [d.metadata["chunk_id"] for d in result.get("ranked", [])]
    hits = [int(key in relevant) for key in ranked]
    dcg = sum(hit / math.log2(i + 2) for i, hit in enumerate(hits))
    ideal = sum(1 / math.log2(i + 2) for i in range(min(len(relevant), len(ranked))))
    expects_answer = case["type"] == "answer"
    passed = (
        not abstained and grounded and all(matches) and any(hits)
        if expects_answer
        else abstained and not citations
    )
    return {
        "pass": bool(passed),
        "abstained": abstained,
        "fact_coverage": sum(matches) / len(matches) if groups else None,
        "missing_facts": [group for group, matched in zip(groups, matches) if not matched],
        "valid_citations": bool(valid) if expects_answer else None,
        "grounded_quotes": bool(grounded) if expects_answer else None,
        "retrieval_recall_at_k": len(set(ranked) & relevant) / len(relevant) if relevant else None,
        "retrieval_hit_at_k": bool(any(hits)) if relevant else None,
        "mrr_at_k": next((1 / (i + 1) for i, hit in enumerate(hits) if hit), 0)
        if relevant
        else None,
        "ndcg_at_k": dcg / ideal if ideal else None,
        "k": len(ranked),
    }


def summarize(results: list[dict]) -> dict:
    def average(key):
        values = [
            r.get("metrics", {}).get(key, 0)
            for r in results
            if r["type"] == "answer" and r.get("metrics", {}).get(key, 0) is not None
        ]
        return round(sum(values) / len(values), 4) if values else None

    return {
        "completed": len(results),
        "passed": sum(r.get("metrics", {}).get("pass", False) for r in results),
        "errors": sum("error" in r for r in results),
        **{
            key: average(key)
            for key in (
                "fact_coverage",
                "valid_citations",
                "grounded_quotes",
                "retrieval_recall_at_k",
                "retrieval_hit_at_k",
                "mrr_at_k",
                "ndcg_at_k",
            )
        },
        "answer_cases": sum(r["type"] == "answer" for r in results),
        "abstention_cases": sum(r["type"] == "abstain" for r in results),
        "abstention_pass_rate": (
            round(
                sum(
                    r.get("metrics", {}).get("pass", False)
                    for r in results
                    if r["type"] == "abstain"
                )
                / sum(r["type"] == "abstain" for r in results),
                4,
            )
            if any(r["type"] == "abstain" for r in results)
            else None
        ),
    }


def run_suite(service, mode="Hybrid", rerank=True, progress=None):
    dataset = load_golden()
    settings = get_settings()
    with service.lock:
        service.store.load()
        indexed = {d.metadata["file"]: d.metadata["sha256"] for d in service.store.documents}
        changed = [
            name for name, digest in dataset["source_hashes"].items() if indexed.get(name) != digest
        ]
        if changed:
            raise ValueError(
                "Golden source versions differ from the index. Review and rebaseline the dataset first: "
                + ", ".join(changed)
            )
        manifest = json.loads((settings.data_dir / "source-manifest.json").read_text())
        if any(
            hashlib.sha256((settings.data_dir / "raw" / d["file"]).read_bytes()).hexdigest()
            != d["sha256"]
            for d in manifest["documents"]
        ):
            raise ValueError(
                "Raw sources differ from the manifest; restore or ingest the revision first."
            )
        if {d["file"]: d["sha256"] for d in manifest["documents"]} != indexed:
            raise ValueError("Sources and index differ; synchronize the library before evaluation.")
        report = {
            "run_id": uuid4().hex,
            "run_at": datetime.now(UTC).isoformat(),
            "dataset_sha256": hashlib.sha256(GOLD.read_bytes()).hexdigest(),
            "dataset_usage": dataset["usage"],
            "snapshot_commit": dataset["snapshot_commit"],
            "source_hashes": indexed,
            "mode": mode,
            "reranking": rerank,
            "embedding_model": settings.embedding_model,
            "dimensions": settings.embedding_dimensions,
            "reranker_model": settings.reranker_model,
            "answer_model": settings.llm_filename,
            "application_sha256": hashlib.sha256(
                b"".join(
                    p.relative_to(ROOT).as_posix().encode() + p.read_bytes()
                    for p in sorted((ROOT / "src").rglob("*.py"))
                )
            ).hexdigest(),
            "results": [],
        }
        directory = settings.data_dir / "evaluations"
        directory.mkdir(parents=True, exist_ok=True)
        output = directory / f"{report['run_id']}.json"
        for i, case in enumerate(dataset["cases"]):
            if progress:
                progress(i / 20, desc=f"Evaluating {case['id']} ({i + 1}/20)…")
            started = perf_counter()
            row = {
                "id": case["id"],
                "question": case["question"],
                "expected": case["expected"],
                "type": case["type"],
            }
            try:
                result = service.ask(case["question"], mode, rerank, case["topic"])
                row.update(
                    answer=result["answer"],
                    metrics=score_case(case, result),
                    retrieved=[d.metadata for d in result.get("ranked", [])],
                    evidence=[
                        {"text": d.page_content, "metadata": d.metadata} for d in result["context"]
                    ],
                )
            except Exception as exc:  # noqa: BLE001 - retain failed cases in the report
                row.update(error=str(exc), metrics={"pass": False})
            row["seconds"] = round(perf_counter() - started, 2)
            report["results"].append(row)
            report["summary"] = summarize(report["results"])
            temporary = output.with_suffix(".tmp")
            temporary.write_text(json.dumps(report, indent=2))
            temporary.replace(output)
        if progress:
            progress(1, desc="Evaluation complete")
        return report, str(output)
