"""Exercise the live Gradio pipeline; save answers, evidence, and mechanical checks.

Mechanical checks do not establish factual correctness. Review the saved answer
against its expected summary and evidence before marking a case accepted.
"""

import argparse
import hashlib
import json
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from gradio_client import Client

ROOT = Path(__file__).resolve().parents[1]


def classify_answer(answer: str) -> str:
    if answer.startswith("Unable to answer:"):
        return "runtime_error"
    if any(
        text in answer
        for text in (
            "couldn't produce a cited answer",
            "invalid source reference",
            "model did not finish",
            "empty answer",
            "answer was truncated",
        )
    ):
        return "generation_error"
    if re.search(
        r"not (?:covered|available|provided)|cannot (?:determine|answer|access)|couldn't find supporting|no (?:live|company|supporting)|does not (?:contain|cover)",
        answer,
        re.IGNORECASE,
    ):
        return "abstained"
    return "answered"


def check_case(client: Client, case: dict) -> dict:
    started = perf_counter()
    outputs = client.predict(case["question"], [], "Hybrid", True, case["topic"], api_name="/ask")
    answer, evidence = outputs[0][-1]["content"], outputs[1]
    status = classify_answer(answer)
    checks = {
        "no_runtime_error": status != "runtime_error",
        "expected_behavior": status == ("answered" if case["type"] == "answer" else "abstained"),
    }
    if case["type"] == "answer":
        available = {int(n) for n in re.findall(r"### \[(\d+)\]", evidence)}
        cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
        checks.update(
            {
                "valid_citation_ids": bool(cited) and cited <= available,
                "expected_source_retrieved": case["source"] in evidence,
            }
        )
    return {
        **case,
        "answer": answer,
        "evidence": evidence,
        "status": status,
        "timing": outputs[2],
        "elapsed_seconds": round(perf_counter() - started, 2),
        "rankings": outputs[3],
        "checks": checks,
        "mechanical_pass": all(checks.values()),
        "manual_review": "pending",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:7860")
    parser.add_argument("--ids", nargs="*", help="Run only specified case IDs")
    parser.add_argument("--output", type=Path, default=ROOT / "evaluation/results.json")
    args = parser.parse_args()
    cases = json.loads((ROOT / "evaluation/questions.json").read_text())
    if args.ids:
        cases = [c for c in cases if c["id"] in args.ids]
    if not cases:
        parser.error("No matching questions")
    client = Client(args.url, verbose=False)
    source_files = sorted((ROOT / "src").rglob("*.py"))
    code_hash = hashlib.sha256(
        b"".join(str(p.relative_to(ROOT)).encode() + p.read_bytes() for p in source_files)
    ).hexdigest()
    corpus = json.loads((ROOT / "data/source-manifest.json").read_text())
    report = {
        "git_revision": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip(),
        "application_sha256": code_hash,
        "dataset_commit": corpus["commit"],
        "document_count": len(corpus["documents"]),
        "source_content_license": "CC-BY-4.0",
        "source_attribution": "Microsoft and the contributors to MicrosoftDocs/dynamics-365-unified-operations-public; evidence is excerpted and reformatted. Source URLs identify the original documents.",
        "run_at": datetime.now(UTC).isoformat(),
        "pipeline": {"mode": "Hybrid", "reranking": True},
        "results": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for case in cases:
        print(f"START {case['id']}: {case['question']}", flush=True)
        try:
            result = check_case(client, case)
        except Exception as exc:  # noqa: BLE001 - preserve failures and continue the requested batch
            result = {
                **case,
                "status": "runtime_error",
                "error": str(exc),
                "mechanical_pass": False,
                "manual_review": "pending",
            }
        report["results"].append(result)
        args.output.write_text(json.dumps(report, indent=2))
        print(
            f"DONE {case['id']} {result['status']} checks={result['mechanical_pass']} {result.get('timing', '')}",
            flush=True,
        )
        print(result.get("answer", result.get("error")), flush=True)
    print(f"Saved {len(report['results'])} cases to {args.output}", flush=True)


if __name__ == "__main__":
    main()
