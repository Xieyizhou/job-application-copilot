# Semantic Evidence Retrieval Model Card

## Summary

The project's primary ML task is requirement-level evidence retrieval:

```text
one JD requirement + factual statements from an existing resume
    → strongest candidate statement
    → Direct / Partial / No Support
```

The objective is to find defensible resume evidence for human review and cover-letter
preparation. It is not a hiring model, interview predictor, resume generator, or automatic
application decision.

## Current status

| Component | Status | Product effect |
|---|---|---|
| Transparent lexical and concept retriever | Active | Proposes reviewable resume evidence |
| Sentence-embedding two-stage candidate | Rejected after operational shadow review | None |
| Task-level rejection successor | Blocked on development | None |
| Legacy whole-resume/job relevance signal | Experimental auxiliary | Never changes product decisions |

No learned semantic candidate currently controls Role Fit, Eligibility, Confidence,
JD Quality, ranking, recommendation, or Cover Letter evidence.

## Intended use

- Rank factual resume statements for one explicit JD requirement.
- Reject the complete candidate set when no statement provides usable support.
- Preserve the exact requirement, evidence sentence, source section, and support label.
- Support human review of evidence used in application materials.

## Out-of-scope use

- Predicting interviews, offers, hiring success, or candidate quality.
- Inferring protected or sensitive candidate attributes.
- Overriding eligibility or work-authorization review.
- Rewriting or generating a resume.
- Automatically submitting an application.
- Treating retrieval similarity as probability of support.

## Product baseline

The active product path uses a transparent local retriever combining:

- word-level TF-IDF similarity
- exact requirement-term coverage
- explicit concept aliases
- a small concreteness bonus for action-led or quantified statements

Evidence below the configured 42% retrieval threshold is rejected. When the portable
TF-IDF artifact is unavailable, the product falls back to the auditable lexical and
concept layer.

Sensitive eligibility requirements are excluded from employer-facing prose even when a
resume contains matching text.

## Learned candidate

The evaluated candidate separates ranking from rejection:

- a pinned sentence encoder supplies semantic candidate representations
- a hybrid ranker orders the candidate statements
- a support classifier decides whether any candidate should be accepted
- task-level rejection experiments inspect the full distribution of four candidate
  probabilities

This design keeps retrieval rank separate from evidence acceptance.

## Labels

Every candidate statement receives one of three labels:

- **Direct** — explicit evidence satisfies the requirement.
- **Partial** — relevant evidence exists but does not fully satisfy the requirement.
- **No Support** — the statement does not provide defensible evidence.

Reviewed tasks preserve all candidate labels rather than converting unselected candidates
into automatic negatives.

## Data and isolation

| Dataset role | Scale | Allowed use |
|---|---:|---|
| Reviewed training corpus | 207 tasks / 828 candidate judgments | Training and grouped development |
| Real development v3 | 96 tasks | Candidate selection before reserve construction |
| Frozen reserve v4 | 96 tasks | One-time evaluation; consumed |
| Operational shadow v1 review | 24 disagreement-enriched tasks | Deployment diagnostic; consumed |
| Operational development v2 | 64 tasks / 256 candidate judgments | Successor development only |

Resume groups, job groups, semantic groups, exact text, and near-text are checked for
overlap according to each dataset contract. Frozen reserve and shadow labels cannot be
used for training, threshold selection, or architecture selection.

Most row-level data, annotations, fitted artifacts, and reports remain in ignored local
paths because they contain restricted or personally sourced material. The repository
tracks de-identified manifests, evaluation code, contracts, and regression fixtures.

## Evaluation results

Metrics below belong to different dataset roles and should not be compared as if they
came from one shared test population.

| Stage | Metric | Reference | Candidate | Decision |
|---|---|---|---|---|
| Real development v3 | Task-balanced accuracy | LSA 0.530 | Two-stage 0.675 | Passed development gate |
| Frozen reserve v4 | Task-balanced accuracy | LSA 0.403 | Two-stage 0.586 | Authorized isolated shadow only |
| Operational shadow v1 | Reviewed decision correctness on disagreement sample | Transparent 0.625 | Shadow 0.250 | Rejected |
| Operational development v2 | Task-balanced accuracy | Candidate baseline 0.547 | Task rejector 0.551 | Blocked |

On reserve v4, the frozen candidate also improved Recall@1 from 0.400 to 0.533,
Recall@3 from 0.833 to 0.933, and No Support rejection from 0.439 to 0.773. The paired
90% interval for task-outcome improvement was 0.086 to 0.288.

The reserve result did not transfer safely to operational shadow use. On the
disagreement-enriched shadow review, the candidate accepted 23 cases, including 17
judged No Support. This false-accept pattern blocked product integration.

The corrected nested operational-development evaluation raised No Support rejection
from 0.548 to 0.738 but reduced supported-task success from 0.545 to 0.364. Its
task-balanced result changed from 0.547 to 0.551, with a paired 90% interval of
-0.077 to 0.085. No successor artifact was created.

## Known limitations

- Operational development v2 contains only two Direct tasks.
- Generic resume bullets can appear semantically related without satisfying a requirement.
- Repeatedly matching one resume against many jobs differs from balanced candidate pools.
- Compound, numeric, degree, and eligibility constraints require explicit handling.
- Current operational results do not establish population-level performance.
- CPU latency and memory targets must be measured before any future product trial.

## Promotion requirements

A successor may proceed only if:

1. It is selected on new source-isolated development data with stronger Direct and Partial
   coverage.
2. Nested resume-grouped evaluation excludes every outer test group from candidate fitting,
   rejection fitting, and threshold selection.
3. It improves task-balanced outcomes with a positive lower paired interval while
   preserving supported-task success and No Support rejection.
4. The artifact, source revision, dependencies, thresholds, baseline, and success criteria
   are frozen before a new holdout is constructed.
5. A fresh source-isolated holdout passes once without tuning.
6. A later non-decision shadow trial confirms acceptable false-accept behavior.

Passing these gates would authorize another isolated shadow trial, not an automatic product
decision.

## Reproducibility

Public checks:

```bash
PYTHONPATH=src python -m pytest tests/test_ml_*.py -q
python scripts/ml/evaluate_real_validation.py --semantic-only
```

The full operational evaluation requires ignored local reviewed data:

```bash
PYTHONPATH=src python scripts/ml/evaluate_task_rejector_operational_v2.py
```

The 24-case semantic regression result is agreement with a small curated set. It is not
model accuracy or evidence of generalization to all resumes and jobs.

## Version

- Model card: 1.0
- Updated: 2026-07-29
- Learned product integration: not approved
