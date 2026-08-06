# Local ML Relevance and Evidence Retrieval

## Purpose

The learned system handles one narrow, reviewable task:

```text
one JD requirement + four resume statements
    → strongest evidence
    → Direct / Partial / No Support
```

It helps determine whether an existing resume contains defensible evidence for a job
requirement. It does not predict interviews, offers, candidate quality, or eligibility.

For final metrics and limitations, see the [Model Card](MODEL_CARD.md). For the iteration
history, see the [ML System Case Study](ML_SYSTEM_CASE_STUDY.md).

## Product boundary

The product keeps these signals separate:

- **Role Fit** measures requirement coverage.
- **Eligibility** handles hard constraints independently.
- **Confidence** reflects JD and evidence completeness.
- **JD Quality** measures whether the source description is usable.
- **Evidence retrieval** selects and labels resume statements.

The transparent retriever remains authoritative. MiniLM v21 is available only through an
opt-in, read-only background Web Shadow and cannot change scores, cover letters,
recommendations, or application records. Missing or incompatible artifacts fail closed.

## Current learned candidate

MiniLM v21 uses a shared pair encoder with three heads:

1. support probability;
2. Direct probability conditional on support;
3. strongest-evidence rank score.

Task acceptance uses the maximum support probability across four candidates. Acceptance
and ranking remain separate so semantic similarity alone cannot turn an unsupported
sentence into evidence.

On frozen E15, v21 reached:

| Metric | Result |
|---|---:|
| Task agreement | 90.63% |
| Macro-F1 | 89.62% |
| Direct recall | 100.00% |
| Partial recall | 79.17% |
| No-Support recall | 90.00% |
| Supported Top-1 | 100.00% |

E15 contains 96 tasks and 384 pairs with zero exact training overlap and maximum token
Jaccard 0.333. Its labels are synthetic teacher-proxy contracts, so the results measure
teacher agreement—not human-gold or real-world accuracy.

## Data lifecycle

Every dataset has one declared role:

- **training** may fit weights;
- **development** may guide architecture or thresholds;
- **frozen evaluation** is opened once after configuration is fixed;
- **shadow** measures deployment distribution and never returns to training;
- **sealed holdout** remains unopened until separately authorized.

Isolation checks cover resume groups, source hashes, exact text, and near-duplicates.
Consumed evaluations are retired into diagnostics. Personal data, row-level teacher text,
models, and reports remain under Git-ignored `data/ml/` paths.

## What earlier failures changed

| Failure | Resulting design decision |
|---|---|
| Calibration shifted every task below threshold | Removed unsupported prior correction |
| Strength head collapsed toward one class | Added per-class gates and boundary-focused data |
| Broad fine-tuning damaged existing behavior | Limited adaptation and preserved parent logits |
| Same-domain negatives looked deceptively relevant | Added capability, agency, and ownership counterfactuals |
| Direct recovery hurt Partial or ranking | Added possessive-accountability contrasts and explicit ranking margin |
| A used evaluation influenced the next iteration | Retired it and generated a new frozen evaluation |

Detailed results for rejected candidates are intentionally omitted here; the case study
preserves the useful engineering conclusions without turning this page into an experiment
ledger.

## Runtime and Web shadow

The self-contained 91.6 MB local bundle records file hashes and decision thresholds. On Apple
MPS it measured 137.9 pairs/s, 41.5 ms p95 task latency, and 876.6 MB peak RSS, with exact
offline/runtime parity.

In a completed 57-job local snapshot, v21 accepted 27 requirements versus
18 for the previous shadow candidate. This is an unlabeled coverage observation, not an
accuracy claim.

Live Shadow is disabled by default. Explicit opt-in schedules inference outside the Web
analysis request. Persisted observations contain input hashes and aggregate counts only;
raw requirement and resume-evidence text is not stored in Shadow reports.

## Public verification

```bash
python -m pip install -r requirements-dev.txt -r requirements-ml.txt
python -m ruff check main.py run_dashboard.py scripts src tests
PYTHONPATH=src python -m mypy
PYTHONPATH=src python -m pytest -q
python scripts/privacy_audit.py
python -m pip check
```

Local annotation, evaluation, and artifact commands are documented in
[Requirement–Evidence Annotation](ML_ANNOTATION.md). None of those commands authorizes
product integration.
