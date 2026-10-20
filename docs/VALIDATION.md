# Ten question pipeline validation

**Outcome:** all 10 curated cases met the reviewed requirements; all 24 targeted regression tests passed. Eight questions returned source-backed answers, and two unsupported requests were declined.

Checks used the live Gradio API with hybrid retrieval, Qwen reranking, 512D embeddings, and the downloaded Unsloth Qwen3 1.7B model. The corpus contained 21 public Microsoft documents and 247 indexed chunks. Each case used its saved topic filter.

These development questions were used while fixing the application. The result is not a held-out benchmark or a general accuracy percentage.

## What was fixed

- SAP questions no longer receive Dynamics instructions.
- Public example amounts are no longer presented as live company balances.
- Original source text replaces inaccurate small-model paraphrases.
- Related chunks and list items stay together so options and components are not omitted.
- Evidence selection focuses on the strongest section, avoiding unrelated nearby examples.
- Yes/no answers retain explicit restrictions such as “only one.”
- Invalid citations and truncated output are rejected. Reranking computes only the final-token logits it needs.

## The ten questions

| ID | Question | Verified result | Time |
|---|---|---|---|
| Q01 | What is the difference between two-way and three-way invoice matching? | Two-way checks invoice/PO prices; three-way also checks product-receipt quantities. | 52.0s |
| Q02 | How is a vendor invoice due date calculated when the invoice date is blank, and what changes when I enter an invoice date? | Blank invoice date: posting date plus terms. Entered invoice date: recalculated from that date plus terms. | 31.6s |
| Q03 | Which posting-date adjustment options handle a closed or on-hold period for vendor invoices? | Closed/on-hold periods can use system date or the first day of the new open period. | 28.0s |
| Q04 | Does vendor payment proposal automation automatically post payments? | No automatic payment posting; existing validation and approval processes remain applicable. | 41.2s |
| Q05 | A payment proposal runs on Wednesday and must include invoices due on or before Friday. What Number of days adjustment for To date should I use? | Use a To date adjustment of +2 for Wednesday to Friday. | 31.6s |
| Q06 | What are the three main components of the scanned vendor invoice automation solution? | Header/line/attachment data entities; exception processing; side-by-side attachment viewer. | 21.9s |
| Q07 | When vendor invoices require workflow review, which button sends an invoice for review instead of posting it? | Use Submit for workflow review instead of Post. | 20.8s |
| Q08 | Can I set two documents as the default attachment on one vendor invoice? | Only one document can be the default attachment. | 19.9s |
| Q09 | Which SAP transaction code should I use to reverse a vendor invoice? | Declined: this library does not cover SAP. | 0.1s |
| Q10 | What was our company's unpaid vendor balance yesterday? Give the exact amount in rupees. | Declined: public documentation contains no live company balances. | 0.1s |

Supported-question median: **29.8 seconds** in this local CPU run. Individual timings include model loading when needed; they are not a performance benchmark.

## Evidence and reproducibility

Each supported answer was checked for the required facts against its retrieved source. Citation IDs were checked separately; a valid ID alone does not establish correctness. Some answers deliberately retain a complete list or surrounding context rather than produce a short paraphrase.

The full ten-case run used application revision `f9ec7ba`. After the explicit-restriction fix, Q04 and Q08 were rerun at `882866f`; the results above use those replacements. Per-case revisions and code fingerprints are preserved in the JSON record.

[Full answers, evidence, timings, and manual notes](https://github.com/kshitij-soloscripter/erp-support-copilot/blob/main/evaluation/results.json) · [Question specifications](https://github.com/kshitij-soloscripter/erp-support-copilot/blob/main/evaluation/questions.json)

```bash
# Keep the app running in another terminal
.venv/bin/python scripts/check_pipeline.py
.venv/bin/python scripts/check_pipeline.py --ids Q04 Q08 --output logs/rerun.json
.venv/bin/python -m unittest discover -s tests -v
```

Fresh runs mark manual review as pending. Compare new answers and excerpts before accepting them.

## Practical limits

- This check covers hybrid search with reranking and the saved topic filters, not every retrieval mode or possible question.
- Selection concentrates on the highest-ranked parent section. Full synthesis across several sections remains a future extension.
- Scope rules are conservative and not exhaustive. “Closest source excerpt” is a labeled retrieval fallback; its score is not an accuracy probability.
- Exact source wording prevents invented paraphrases; relevance, completeness, and applicability to a real tenant still require review.

Original documentation: Microsoft and contributors to [MicrosoftDocs Dynamics 365](https://github.com/MicrosoftDocs/dynamics-365-unified-operations-public), licensed under CC BY 4.0. Evidence records excerpt and reformat that material and retain its source URLs.
