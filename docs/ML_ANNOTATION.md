# Requirement–Evidence Annotation

## Purpose

This local workflow records complete judgments for one requirement and four candidate
resume statements. It is isolated from the application dashboard and writes all queues,
events, labels, and reports to Git-ignored paths.

The repository supports both human-reviewed datasets and synthetic teacher-proxy
contracts. The current MiniLM v21 E15 result uses teacher-proxy labels; it must not be
described as human-gold accuracy. See the [Model Card](MODEL_CARD.md).

## Label contract

Label every candidate before selecting the strongest evidence:

- **Direct** — the sentence independently demonstrates the requirement.
- **Partial** — relevant evidence exists, but an important part remains unproven.
- **No Support** — the sentence does not defensibly support the requirement.
- **Uncertain** — the task needs more context or adjudication and cannot enter gold.

For supported tasks, select exactly one strongest Direct or Partial candidate. Select
`None` only when all four candidates are No Support. Do not infer that an unselected
candidate is negative.

## Minimal local workflow

Install dependencies, generate a queue, and open the reviewer UI:

```bash
python -m pip install -r requirements.txt -r requirements-ml.txt
python scripts/ml/build_annotation_queue.py
python run_annotation.py
```

Audit completed events before export:

```bash
python scripts/ml/audit_annotations.py
python scripts/ml/export_annotation_dataset.py
```

Queues and event logs are append-only local artifacts under `data/ml/annotations/`.
Changing a decision does not erase its earlier state.

## Review requirements

A dataset is usable only when:

- all four candidate labels are present;
- strongest evidence agrees with the candidate labels;
- uncertain or privacy-risk tasks are excluded from gold;
- hidden-repeat agreement is reported without rewriting raw decisions;
- Direct, Partial, and No Support coverage is reported by role family;
- one resume group never crosses train, development, evaluation, or holdout roles;
- exact and near-text overlap checks pass;
- authoring intent, model scores, retrieval rank, and source metadata are hidden from
  reviewers.

Natural label distributions are retained. Do not delete examples or force equal class
counts after labels are known. Coverage gaps require a new, isolated supplement.

## Human review and adjudication

When two independent reviews are required:

1. create separately shuffled reviewer packets;
2. freeze raw agreement before adjudication;
3. send disagreements to a content-only adjudication queue;
4. preserve raw decisions and agreement reports;
5. finalize only complete, internally consistent tasks.

An optional Chinese translation sidecar may assist reading, but the English source remains
authoritative. Translation must never alter task checksums, labels, or model inputs.

## Synthetic teacher-proxy data

Teacher-generated data reduces manual drafting cost but does not become truth merely
because the teacher is confident or self-consistent. Accepted tasks must still pass:

- schema and candidate-completeness checks;
- privacy and text-quality checks;
- source, group, exact-text, and near-duplicate isolation;
- label/strongest-evidence consistency;
- ambiguity exclusion;
- frozen dataset-role and non-integration flags.

Teacher agreement is reported as teacher agreement. It is never presented as hiring
validity, human accuracy, or evidence that the model obtained the teacher's general
reasoning ability.

## Evaluation discipline

- Fit weights only on training data.
- Select architecture and thresholds only on development data.
- Freeze model, thresholds, hashes, and success gates before opening an evaluation.
- Use a frozen evaluation once; after failure, retire it into diagnostics.
- Never train on Web-shadow observations or sealed holdouts.
- Passing offline gates authorizes at most a non-decision shadow trial.

The current v21 training corpus contains 3,024 tasks / 12,096 pairs. Frozen E15 contains
96 new tasks / 384 pairs. Sealed operational holdouts were not opened for that result.

## Privacy and repository policy

Never commit:

- resumes, job descriptions, contact details, or personal annotations;
- reviewer queues, event logs, translations, or adjudication files;
- teacher prompts or row-level teacher outputs;
- model weights, local reports, credentials, or generated documents.

Public Git content is limited to schemas, lifecycle code, de-identified fixtures,
aggregate documentation, and tests. Run before publishing:

```bash
python scripts/privacy_audit.py
git check-ignore data/ml/annotations data/ml/distillation data/ml/shadow
git diff --check
```

## Related documentation

- [Model Card](MODEL_CARD.md) — current candidate, metrics, and limitations
- [ML System Case Study](ML_SYSTEM_CASE_STUDY.md) — failure-driven learning narrative
- [ML Relevance](ML_RELEVANCE.md) — product boundary and runtime integration
