# MiniLM Resume-Evidence Model Card

## Summary

The model answers one deliberately narrow question:

```text
job requirement + four resume statements
    → Direct / Partial / No Support
    → strongest defensible evidence
```

It supports evidence review for job applications. It does not predict hiring outcomes,
score candidate quality, generate resume claims, or make eligibility decisions.

## Current release candidate

| Item | Value |
|---|---|
| Candidate | MiniLM v21 (`minilm-two-stage-v3`) |
| Architecture | Shared MiniLM encoder with support, strength, and ranking heads |
| Decision rule | Maximum support ≥ 0.30; Direct conditional probability ≥ 0.30 |
| Evaluation | Frozen E15, 96 tasks / 384 candidate pairs |
| Deployment state | Opt-in, read-only background Web shadow |
| Product authority | None; `product_integration_allowed=false` |
| Bundle size | 91.6 MB |

The production-facing transparent retriever remains authoritative. When explicitly enabled,
the learned model runs in a background diagnostic path without changing visible scores or
application text. It is disabled by default in a clean clone.

## Architecture

The model encodes each requirement/evidence pair once and produces three signals:

- `support_probability`: whether the sentence provides usable support;
- `direct_probability_given_support`: whether supported evidence is Direct rather than
  Partial;
- `rank_score`: which supported sentence is strongest.

Task rejection uses the maximum support score across the four candidates. Ranking is kept
separate from acceptance so a fluent but unsupported sentence cannot win solely through
semantic similarity. Invalid or incomplete input fails closed.

## Training and evaluation data

| Dataset role | Scale | Use |
|---|---:|---|
| v21 training corpus | 3,024 tasks / 12,096 pairs | Training; includes 1,296 targeted boundary tasks |
| Frozen E15 | 96 tasks / 384 pairs | One-time candidate evaluation only |
| Web shadow snapshot | 57 jobs / 183 requirements | Unlabeled deployment-distribution diagnostic |
| Sealed operational holdouts | Not opened | Reserved; never used for this result |

Training emphasizes capability-vs-tool distinctions, agency and ownership, compound
requirements, Partial/Direct boundaries, possessive accountability, and strongest-evidence
ranking. Resume groups, source hashes, exact text, and near-text similarity are audited
across splits. E15 had zero exact overlap with training and a maximum token Jaccard of
0.333, below the predeclared 0.85 exclusion threshold.

The E15 labels are privacy-safe synthetic teacher-proxy contracts with automated rule
audits. Agreement with them is useful for model selection, but it is not human-gold
accuracy and does not establish real-world hiring validity.

## Frozen E15 results

| Metric | Result |
|---|---:|
| Task agreement | **90.63%** |
| Task macro-F1 | **89.62%** |
| Direct recall | **100.00%** |
| Partial recall | **79.17%** |
| No-Support recall | **90.00%** |
| Supported Top-1 | **100.00%** |
| Wilson 95% interval for agreement | **83.14%–94.99%** |

Role-family task agreement was 87.50% for Software, 91.67% for Data, 95.83% for ML,
and 87.50% for Business. All predeclared E15 gates passed.

## Runtime validation

Measured locally on Apple MPS over the frozen evaluation packet:

| Gate | Result | Requirement |
|---|---:|---:|
| Throughput | 137.9 pairs/s | ≥20 pairs/s |
| p95 task latency | 41.5 ms | Diagnostic |
| Peak RSS | 876.6 MB | ≤1.5 GB |
| Offline/runtime parity | Exact | Required |
| Invalid-input behavior | Fail closed | Required |

The bundle is self-contained, loads offline, records SHA-256 hashes for runtime files,
and uses local-only inference.

## Web shadow observation

Across a completed local 57-job snapshot, v21 accepted 27 of 183 extracted requirements (14.75%):
12 Direct and 15 Partial. The earlier shadow candidate accepted 18 (9.84%). This shows a
change in coverage, not an accuracy improvement, because these Web observations are not
labeled. They remain excluded from training.

## Failure-driven development

Earlier candidates exposed distinct problems rather than one generic “model failure”:

- fixed calibration corrections caused an all-reject collapse;
- native tri-class objectives failed to reject No Support reliably;
- broader fine-tuning moved established logits and hurt prior behavior;
- v18 improved the hierarchy but failed fresh E12 generalization;
- v19 improved ranking but confused subject, agency, and ownership;
- v20 fixed many support boundaries but over-corrected against Direct evidence;
- v21 added possessive-accountability counterfactuals and an explicit ranking margin,
  then passed a newly generated, frozen E15.

Failed evaluation sets became development diagnostics and were not reused as final tests.
See the [ML System Case Study](ML_SYSTEM_CASE_STUDY.md) for the engineering narrative.

## Intended use

- Rank factual statements from an existing resume for one explicit requirement.
- Reject the candidate set when no statement provides defensible support.
- Preserve the requirement, evidence text, source section, label, and scores for review.
- Support local, human-reviewed application preparation.

## Out-of-scope use

- Predicting interviews, offers, hiring success, or candidate quality.
- Inferring protected or sensitive attributes.
- Overriding eligibility or work-authorization review.
- Inventing or rewriting resume facts.
- Automatically submitting applications.
- Treating teacher agreement as real-world accuracy.

## Known limitations

- Partial evidence remains the weakest E15 class.
- Four-candidate packets do not cover every resume layout or requirement formulation.
- Requirement extraction errors can occur before the model is called.
- Synthetic teacher contracts may encode systematic teacher preferences.
- Web-shadow coverage is unlabeled and cannot establish correctness.
- CPU deployment needs its own latency and memory benchmark.

## Promotion boundary

The candidate may remain in non-decision shadow mode. Product integration requires a
separate authorization, labeled operational review, acceptable false-accept behavior,
runtime monitoring, and a rollback path. No E15 result alone authorizes user-visible use.

## Reproducibility and release hygiene

Public code includes the exact content-free v21 release contract, reusable objective and
evaluation components, leakage checks, runtime adapters, and regression tests. Row-level
teacher data, personal data, final experimental launchers, model weights, and reports stay
in Git-ignored local storage. A clean clone can validate declared hashes and safety flags,
but cannot reproduce training without those intentionally unpublished inputs.

## Version

- Model card: 2.0
- Updated: 2026-08-07
- Web shadow: available, opt-in, disabled by default
- Learned product integration: not approved
