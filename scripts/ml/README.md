# ML Command Map

This directory contains the public, auditable entry points used to demonstrate the current
ML lifecycle. Commands that require private or licensed source data are reference workflows,
not clean-clone reproductions. Historical one-off experiments are summarized in
[`docs/ML_SYSTEM_CASE_STUDY.md`](../../docs/ML_SYSTEM_CASE_STUDY.md) instead of being kept
as dozens of executable scripts.

## Annotation

| Command | Purpose |
|---|---|
| `build_annotation_queue.py` | Build a privacy-screened reviewer queue |
| `audit_annotations.py` | Audit label completeness, balance, and repeats |
| `export_annotation_dataset.py` | Export candidate-complete reviewed data |
| `build_evidence_training_corpus.py` | Materialize the local training contract |

## Training and diagnostics

| Command | Purpose |
|---|---|
| `validate_minilm_v21_release.py` | Validate the public v21/E15 contract and an optional ignored local bundle |
| `build_teacher_error_audit.py` | Build an auditable teacher-error packet without product access |
| `evaluate_real_validation.py` | Run the de-identified semantic regression gate used by CI |

The exact content-free v21 hyperparameters, metrics, thresholds, and hashes live in
`config/ml_minilm_v21_release.json`. Reusable MiniLM objective and evaluation components
live in `src/ml/evidence_multitask_*`. Model weights, teacher text, final experimental
launchers, and row-level reports remain under ignored `data/ml/` paths.

## Source-isolated lifecycle

These commands require locally obtained source corpora and manifests under `data/ml/`.
The repository does not redistribute Djinni, ATS, resume, or job-description source text.
Before running them, the operator must independently obtain data under its original terms,
create the paths requested by `--help`, and keep every source file Git ignored. The commands
remain public to expose validation, grouping, leakage, and sealed-holdout contracts.

| Command | Purpose |
|---|---|
| `build_operational_development_v3.py` | Build a grouped development packet |
| `prepare_operational_development_v3_adjudication.py` | Prepare content-only disagreement review |
| `finalize_operational_development_v3.py` | Freeze complete reviewed tasks |
| `evaluate_operational_development_v3.py` | Evaluate a fixed candidate without opening holdouts |
| `build_successor_source_partition.py` | Assign development and sealed-holdout sources together |
| `materialize_successor_development.py` | Materialize development only |
| `prepare_successor_reviewer_queue.py` | Create blinded reviewer packets |
| `prepare_successor_adjudication.py` | Prepare adjudication while preserving raw decisions |

## Web shadow

`run_web_candidate_shadow_backfill.py` replays local jobs through the candidate adapter and
writes hash-only and aggregate ignored diagnostics. Live Web Shadow is disabled by default;
set `JOB_COPILOT_WEB_SHADOW_ENABLED=true` to opt in. It runs outside the analysis request,
never changes user-visible product state, and never runs in Demo mode.

`sanitize_web_shadow_reports.py` removes raw requirement/evidence fields from observations
created by older versions while preserving hashes and aggregate comparison counts.

All commands fail closed when required local data, permissions, or manifests are missing.
