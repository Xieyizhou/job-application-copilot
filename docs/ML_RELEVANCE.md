# Local ML Relevance and Evidence Retrieval

## Primary ML objective

The project's primary ML task is requirement-level evidence retrieval:

```text
one JD requirement + factual statements from an existing resume
    → strongest candidate statement
    → Direct / Partial / No Support
```

The product uses a transparent retriever. Learned semantic candidates remain offline
unless they pass source-isolated development, a frozen holdout, and operational shadow
review. The current learned candidate is not approved for product use.

- deterministic Role Fit remains the ranking score
- eligibility remains a separate hard-constraint review
- confidence remains tied to JD and candidate-evidence completeness
- JD Quality remains a document-quality signal
- learned evidence predictions do not change Cover Letter evidence
- missing dependencies or incompatible artifacts fail closed

See [Semantic Evidence Retrieval Model Card](MODEL_CARD.md) for the concise task
definition, evaluation summary, limitations, and current integration decision.

## Legacy resume/job relevance experiment

An earlier optional model supplies a second view of whole-resume/job relevance. It is not
an interview-probability model or an automatic application decision. Its probability is
displayed only as an experimental auxiliary signal, and collapsed or out-of-distribution
batches are hidden instead of being rounded into misleading `0%` values.

## Dataset roles

The synthetic candidate-matching dataset contains fictional resumes, jobs, and known
relevant-resume lists. It is used for training because it supports explicit binary
relevance labels and job-grouped evaluation. Synthetic performance does not establish
real-world hiring or application outcomes.

The resume ATS-score dataset is retained only as local research material. Its labels are
algorithmic weak labels and its published partitions share many normalized resumes and
jobs, so it is not part of the product scoring path or the relevance-model training path.

## Model design

`PairRelevanceModel` uses element-wise products of resume and job TF-IDF vectors plus
compact lexical overlap, recall, precision, bigram, and length-ratio features. It does
not receive standalone resume or job vectors, limiting its ability to label a document
positive without pair evidence.

## Requirement evidence retrieval

The same portable TF-IDF vocabulary can support a narrower, human-reviewable retrieval
task. Each required or preferred JD statement is compared with factual lines extracted
from the uploaded resume. The retriever combines:

- portable TF-IDF cosine similarity
- exact requirement-term coverage
- explicit concept aliases such as `data pipeline` ↔ `ETL workflow`
- a small concreteness bonus for action-led or quantified resume statements

The output always retains the original requirement and exact resume sentence. Similarity
is a retrieval score, not a hiring probability. Evidence below the configured 42% threshold
is rejected and cannot enter the employer-facing cover letter. If the portable model is
absent, the feature degrades to the auditable concept-and-lexical layer.

Eligibility and sensitive-status requirements—including work authorization, visa,
citizenship, sponsorship, security clearance, and required degrees—are always excluded
from cover-letter prose even when the resume contains matching evidence.

The Fit page displays the mapping, similarity, match type, and resume section. Generated
internal cover-letter notes record accepted and rejected requirements. This retrieval layer
does not change Role Fit, eligibility, confidence, ranking, or recommendation.

## JD quality classifier

Before presenting experimental comparisons, the toolkit classifies saved job text as
`Scoring-ready`, `Partial JD`, `Likely snippet`, `Requirements missing`,
`Boilerplate-heavy`, or `Empty or unreadable`. The local classifier uses auditable
document features: word count, responsibility and requirement statements, section
coverage, truncation, boilerplate share, and saved-source provenance. Its 0–100 value
describes document quality, never candidate fit.

The classification controls confidence and display boundaries, not candidate ranking. A
discovery snippet may still receive a provisional deterministic review, but it cannot make
an experimental relevance number trustworthy. The next action points back to the original
full posting whenever source evidence is incomplete.

The synthetic pair classifier has not been calibrated on representative real application
traffic. If a real-world batch collapses into a tiny probability range, the UI hides those
raw diagnostic values instead of presenting false precision. The guard does not rewrite or
rescale the probabilities.

## Evaluation protocol

Jobs are assigned deterministically by a stable hash of `job_id`: 70% train, 15%
validation, and 15% test. All pairs for one job remain in one subset. The threshold is
selected on validation data and applied once to the untouched test data.

The local report includes ROC AUC, average precision, accuracy, balanced accuracy,
precision, recall, F1, a confusion matrix, and mean per-job average precision over the
sampled candidate pools. Any published metric must also name the synthetic dataset,
unseen-job protocol, negative-sampling ratio, and sampled-pool limitation.

## Anonymous real-derived validation

Two committed manifests add a separate domain-shift and evidence-quality check without
publishing resumes or job descriptions:

- `semantic_evidence_real_v1.json` contains 24 short, manually reviewed sentence pairs
  derived from the public ATS corpus, with names, employers, contacts, locations, and
  whole-document context removed.
- `relevance_real_holdout_v1.json` contains only SHA-256 resume/job references and the
  source dataset's weak labels. It resolves against the ignored local
  `canonical_pairs.parquet` file and never commits source text or row-level predictions.

The semantic set tests whether a requirement is supported by one factual resume statement.
The relevance set is diagnostic only: its ATS labels are algorithmic weak labels, not human
ground truth or recruiting outcomes. The current synthetic relevance artifact collapses on
this real-data holdout, so the dashboard continues to hide that experimental signal for
collapsed batches. The holdout must not be used for threshold selection or retraining.

The 24/24 semantic result is curated validation-set agreement, not model accuracy or proof
that the retriever generalizes to all resumes and job descriptions.

The standalone local labeling workspace is the path from this small curated check to a
larger independent gold set. Its queue generation, label definitions, privacy boundaries,
and evaluation gates are documented in [Requirement–Evidence Annotation](ML_ANNOTATION.md).

Synthetic expansion never treats construction metadata as a gold label. Local tooling
creates strict blind packets, requires candidate-level review, and routes disagreement to
human adjudication. Anonymous real tasks are isolated by connected resume, job, and
semantic groups before fitting, validation, or fixed-holdout evaluation.

The reviewed-evidence v3 experiment combines the completed human seed with only unanimously
accepted blind-consensus tasks. It compares TF-IDF, local LSA embeddings, binary support
classification, and hybrid/pairwise rerankers under semantic/template-grouped evaluation.
The trainer fits whichever artifact-capable reranker wins the predefined grouped selection;
it does not force a pairwise model. Schema-v2 artifacts include an exact training-corpus
fingerprint, and diagnostics reject legacy or stale files. The selected artifact is still
an offline experiment: it is not loaded by Dashboard, Role Fit, eligibility,
recommendation, or cover-letter generation.

The real-text development experiment now also separates two decisions that must not be
treated as one score:

- word TF-IDF ranks the four resume-evidence candidates
- an explicit-feature support gate decides whether the top-ranked candidate is strong
  enough to use

The support gate exposes its logistic coefficients for lexical coverage, concept overlap,
numeric constraints, and compound-requirement signals. Its threshold is selected only on
the frozen 24-task validation split. The previously evaluated holdout and 16-task reserve
are not read during calibration.

This split architecture improved No Support rejection but did not beat the validation
rank-only reference, so no replacement artifact was frozen and application behavior
remains unchanged. The next development step is a larger, source-isolated validation
extension. A new final reserve should be created only after a candidate wins that
development comparison.

The source-isolated extension now contains 48 frozen development tasks from local Djinni
and SkillSpan sources. Two independently ordered reviews produced 30 exact agreements;
18 disagreements were resolved by human adjudication. The frozen set contains 15 Direct,
10 Partial, and 23 No Support decisions.

On this extension, separately calibrated TF-IDF remained the strongest tested ranker
(0.717 task-balanced accuracy and 0.640 Recall@1). More importantly, transferring the
threshold selected on the older validation split produced 0.671 task-balanced accuracy,
0.640 Recall@1, 0.880 Recall@3, and 0.783 No Support rejection. The learned support gate
reached 0.633 task-balanced accuracy under the same transfer protocol and therefore did
not qualify for product integration. These are development results, not final
generalization claims.

A subsequent four-fold cross-fit experiment added only the other folds' reviewed real
pairs to each training fold. LSA retrieval shortlisted three candidates, then a
three-class local logistic reranker estimated Direct, Partial, and No Support. Unselected
candidates from supported tasks were never treated as negatives. Compared with the
cross-fit TF-IDF reference, the two-stage method improved Recall@1 from 0.560 to 0.640 and
No Support rejection from 0.565 to 0.609, but task-balanced accuracy fell from 0.543 to
0.484. The acceptance gate therefore blocked artifact creation. This result supports LSA
as a candidate generator, not as a product decision model.

The next training pass added 80 source-isolated, human-reviewed acceptance tasks: 33
Direct, 20 Partial, and 27 No Support, balanced equally across Data, ML, Software, and
Business roles. The combined corpus contains 207 tasks and 444 conservatively labeled
pairs. Grouped training selected the hybrid LSA reranker, but the 24-case diagnostic
showed high recall with excessive false acceptance (F1 0.667, balanced accuracy 0.643,
10 of 14 unsupported cases accepted). On the frozen development extension, calibrated
TF-IDF remained strongest at 0.737 task-balanced accuracy, 0.640 Recall@1, and 0.913
No Support rejection. It did not preserve the transparent baseline's 1.000 rejection
rate, so promotion remains blocked.

Repeating the four-fold two-stage comparison with the enlarged corpus improved the
cross-fit TF-IDF reference to 0.626 task-balanced accuracy and the two-stage method to
0.590. The two-stage method increased No Support rejection to 0.739 but reduced Recall@1
to 0.600 versus TF-IDF's 0.640. The learned artifact therefore remains an offline
experiment and the application inference path is unchanged. A new acceptance architecture
must win on development before constructing a fresh source-isolated reserve.

A constraint-aware acceptance experiment now separates TF-IDF candidate ranking from a
standardized local logistic gate. Its auditable features cover named tools and adjacent
skill families, numeric and degree constraints, eligibility statements, compound-clause
coverage, weak assertions, and explicit negation. On the 48-task development extension it
reached 0.740 task-balanced accuracy, 0.640 Recall@1, and 1.000 No Support rejection versus
TF-IDF's 0.737, 0.640, and 0.913. The point improvement is only 0.003, with a paired
stratified-bootstrap 90% interval of -0.078 to 0.087. Because the improvement is not
statistically stable, artifact creation is blocked and no fresh reserve should be opened
yet. More source-isolated development labels are required to narrow this interval.

Candidate-level completion subsequently produced 828 explicit Direct, Partial, or No
Support judgments across 207 four-candidate tasks. Five-fold task-grouped evaluation
compared pair TF-IDF, LSA plus transparent features, a pinned frozen
`all-MiniLM-L6-v2` encoder, and the encoder combined with transparent features. The pure
frozen encoder improved pair macro F1 from the LSA baseline's 0.423 to 0.506 and balanced
accuracy from 0.521 to 0.551.

The operational retrieval trade-off was different. The encoder-plus-transparent hybrid
matched LSA Recall@1 at 0.675, improved Recall@3 from 0.937 to 0.952, and improved No
Support rejection from 0.160 to 0.395. The pure encoder reached 0.635 Recall@1 and 0.481
No Support rejection. A fixed two-stage candidate therefore uses the pure encoder for
support classification and acceptance, while the hybrid score only orders candidates. It
combines 0.506 macro F1, 0.675 Recall@1, 0.952 Recall@3, and 0.481 No Support rejection
without threshold tuning. These results freeze the transformer baseline for the next
source-isolated evaluation; they do not modify application inference or authorize model
promotion.

The locked two-stage candidate was then fitted on all 828 reviewed training pairs and
evaluated without threshold tuning on the 96-task source-isolated real development v3
split. Source job hashes, resume hashes, task ids, and normalized requirement/evidence
pairs had zero overlap with training. Relative to LSA, task-balanced accuracy improved
from 0.530 to 0.675, No Support rejection from 0.423 to 0.827, and Recall@3 from 0.955 to
0.977 while Recall@1 remained 0.659. The paired stratified-bootstrap 90% interval for the
task-balanced improvement was 0.047 to 0.241. This passes the development gate for
constructing a new source-isolated reserve, but the development split is not an untouched
final holdout and cannot authorize product integration.

The candidate and its success criteria were then frozen before constructing real reserve
v4. The reserve contains 96 new source-isolated tasks, balanced across Data, ML, Software,
and Business. Two blind reviews reached 75 exact task decisions; a human adjudicator
resolved all 21 exact disagreements and audited eight randomly selected exact agreements.
The final labels are 8 Direct, 22 Partial, and 66 No Support.

On this one-time reserve, the frozen two-stage candidate improved task-balanced accuracy
from the fixed LSA baseline's 0.403 to 0.586, Recall@1 from 0.400 to 0.533, Recall@3 from
0.833 to 0.933, and No Support rejection from 0.439 to 0.773. The paired
stratified-bootstrap 90% interval for the task-outcome improvement was 0.086 to 0.288.
All precommitted criteria passed. Diagnostic review found 31 baseline failures corrected,
8 regressions, and 33 remaining candidate failures: 15 false accepts, 9 false rejects,
and 9 ranking errors.

This result authorizes failure review and an isolated shadow-mode trial only. It does not
authorize changing Role Fit, Cover Letter evidence, or any user-visible product decision.
Reserve v4 labels and failures must not be used to retune this candidate.

## Local shadow operation

The reserve-qualified heads are packaged separately from the pinned sentence encoder.
The artifact stores the pure acceptance classifier, hybrid ranking classifier, fitted
transparent word scorer, candidate-spec checksum, training checksums, and reserve report
checksum. Packaging is blocked unless the stored heads exactly reproduce the frozen
reserve metrics.

```bash
PYTHONPATH=.:src .venv/bin/python \
  scripts/ml/build_sentence_embedding_shadow_artifact.py
```

Shadow comparison accepts one saved Markdown or text JD and one locally extracted resume
text. It runs the current evidence method and the frozen candidate side by side, then
writes acceptance and evidence-ranking disagreements under the ignored
`data/ml/shadow/` directory.

```bash
PYTHONPATH=.:src .venv/bin/python \
  scripts/ml/run_evidence_shadow.py \
  --job-text data/job_descriptions/private/example.md \
  --resume-text data/resume/resume_source.md \
  --output data/ml/shadow/example.json
```

The runner never writes application state and its output cannot affect Role Fit, Cover
Letter evidence, or recommendations. Input paths are excluded from the report. Before
any user-visible experiment, collect 30–50 real local comparisons and review every
`shadow_only_accept`, `baseline_only_accept`, and `both_accept_different_evidence` case.

The first operational batch used 40 source-isolated public jobs, balanced equally across
Data, ML, Software, and Business, against one real local resume. Across 183 extracted
requirements, the candidate produced 115 shadow-only accepts. A blind, role-balanced
review sampled 24 method disagreements and hid method origin. The baseline accepted 10
reviewed cases, with 6 supported and 4 unsupported evidence selections. Shadow v1
accepted 23, with 2 Direct, 4 Partial, and 17 No Support judgments. Diagnostic accuracy
on this disagreement-enriched sample was 0.625 for the baseline and 0.250 for shadow v1.

Shadow v1 is therefore rejected for product integration. The reviewed 24-task sample is
now a fixed operational diagnostic holdout and must not be added to training. The reserve
v4 result remains a valid four-candidate benchmark result, but it did not cover the
deployment distribution of repeatedly matching one real resume against many jobs.
Development of a successor requires new training/development data with realistic
single-resume candidate pools, generic-bullet hard negatives, and an explicit rejection
objective. Role Fit and Cover Letter behavior remain unchanged.

## Successor operational development v2

The successor development packet mirrors the product operation more closely: each of 16
candidate profiles is matched against four distinct public job requirements. It contains
64 tasks and 256 candidate statements, balanced across Data, ML, Software, and Business.
Within every profile family, three jobs are aligned and one is a cross-role hard negative.
All tasks are isolated from prior annotation, training, reserve, and shadow sources by
source identity and normalized near-text checks.

Candidate profiles come from the English TalentCLEF 2026 Task A corpus, Zenodo record
`10.5281/zenodo.19652670` (CC BY 4.0). TalentCLEF describes these profiles as
privacy-protecting synthetic documents derived from structured real-world data and
manually reviewed for coherence. Job requirements come from unused public Djinni job
descriptions. The packet is therefore valid for successor-model development after blind
human review, but it is not a real-candidate holdout and cannot support product-readiness
claims.

The original local candidate-matching synthetic corpus was considered and rejected for
this packet: its remaining Data profiles had no candidate evidence that passed the
existing-corpus near-text isolation gate. The gate was not relaxed.

```bash
PYTHONPATH=.:src .venv/bin/python \
  scripts/ml/build_operational_development_v2.py
```

The ignored output is stored under
`data/ml/annotations/operational_development_v2/`. Reviewer packets contain no model
predictions or construction labels. Resume groups and job sources must remain intact in
all later train/development splits.

Reviewer A completed all 64 tasks and all 256 candidate judgments. The human labels
contain 2 Direct, 20 Partial, and 42 No Support tasks, or 42 supported and 214 unsupported
candidate pairs. A frozen rejected shadow model was then run independently as Reviewer B
without loading Reviewer A decisions. Candidate labels agreed on 208/256 judgments;
27/64 complete task decisions differed. Per the predeclared human-authority policy, every
gold label comes from Reviewer A. Reviewer B cannot define or override gold.

The first fixed-candidate evaluation did not pass the operational development gate.
The frozen hybrid reached 0.598 task-balanced accuracy but only 0.455 Recall@1. The prior
two-stage candidate reached 0.587 balanced accuracy versus 0.581 for LSA, with a paired
bootstrap 90% interval of -0.109 to 0.121.

A second comparison used four-fold stratified resume-group cross-validation. This keeps
all four tasks from one profile in the same fold. The LSA two-stage method improved the
point estimate to 0.539 balanced accuracy and 0.545 Recall@1, but its paired 90% interval
remained unstable at -0.070 to 0.161. The rank-preserving gate raised No Support rejection
to 0.881 but retained only 0.364 Recall@1. No successor artifact was saved.

An explicit task-level logistic rejector was then evaluated over auditable distribution
features from all four candidate probabilities. Candidate probabilities are generated by
an inner resume-grouped cross-fit inside every outer fold, all tasks from one profile stay
together, and each outer test fold is excluded from candidate fitting, rejector fitting,
and threshold selection. The rejector raised No Support rejection from 0.548 to 0.738,
but reduced supported-task success from 0.545 to 0.364. Task-balanced accuracy changed
from 0.547 to 0.551; the paired stratified-bootstrap 90% interval was -0.077 to 0.085.
The selection gate blocked the model and no artifact was written.

Anonymous failure slices show that cross-role hard negatives are not the primary
bottleneck. Most failures occur on aligned requirements: the rejector reduced false
accepts from 19 to 13 overall, but support rejections increased from 3 to 9. The current
64-task packet contains only 22 supported tasks, including just two Direct tasks. This is
not enough evidence to keep tuning a rejection boundary on the same development packet.
The next data increment should emphasize varied aligned Direct and Partial cases, plus
same-role adjacent No Support cases. It should not add another large block of easy
cross-role negatives.

## Future source-isolated operational holdout

No current successor qualifies for this holdout. The reserve-qualified sentence-embedding
candidate failed operational shadow v1, and every successor evaluated on operational
development v2 remains blocked. Do not construct or label a new operational holdout until
one candidate and its decision rule pass the nested development gate and are frozen.

When that gate is eventually met, use this protocol:

- freeze the candidate artifact, source revision, dependency lock, feature manifest,
  thresholds, baseline, metrics, and success criteria before sampling or review
- sample at least 24 new anonymous real-resume groups, with four independently sourced
  jobs per resume and balanced Data, ML, Software, and Business coverage
- use a source frame not used by training, development, reserve v4, or shadow v1; check
  only source hashes and normalized-text isolation artifacts from consumed datasets, never
  their labels, predictions, failures, or row-level metrics
- select jobs and candidate statements without candidate-model scores or
  model-disagreement enrichment, so the result estimates the intended one-resume-to-many-
  jobs operation rather than a hand-picked error slice
- keep every resume group intact and record zero source, exact-text, and near-text overlap
  before either reviewer receives a packet
- use two independent human reviewers for every candidate judgment, route all
  disagreements to a content-only adjudicator, and freeze the gold checksum once
- open the labels once for the precommitted comparison; require a positive lower 90%
  paired interval for task-balanced improvement, no more than a five-point supported-task
  success regression, and no No Support rejection regression
- treat a pass as authorization for a new isolated, non-decision shadow trial only; it
  still cannot change Role Fit, Eligibility, Confidence, JD Quality, Cover Letter
  evidence, ranking, or recommendation

After labels are opened, the holdout becomes a consumed regression set. Its tasks,
labels, errors, and diagnostics cannot enter training, threshold selection, architecture
selection, or a later promotion claim.

```bash
PYTHONPATH=.:src .venv/bin/python \
  scripts/ml/finalize_operational_development_v2.py

PYTHONPATH=.:src .venv/bin/python \
  scripts/ml/evaluate_two_stage_development.py \
  --annotation-dir data/ml/processed/operational_development_v2 \
  --gold-filename annotated_tasks.jsonl \
  --manifest-filename manifest.json \
  --dataset-name operational_development_v2

PYTHONPATH=.:src .venv/bin/python \
  scripts/ml/evaluate_task_rejector_operational_v2.py
```

## Commands

```bash
python -m pip install -r requirements.txt -r requirements-ml.txt

python scripts/ml/train_relevance_baseline.py

python scripts/ml/evaluate_real_validation.py

python scripts/ml/build_evidence_training_corpus.py

python scripts/ml/train_evidence_reranker.py

python scripts/ml/evaluate_evidence_reranker.py

python scripts/ml/finalize_real_holdout.py

python scripts/ml/evaluate_real_holdout.py

python scripts/ml/calibrate_dual_signal_validation.py

python scripts/ml/build_real_development_extension.py

python scripts/ml/calibrate_real_validation.py --dataset development

python scripts/ml/calibrate_dual_signal_validation.py --dataset development

python scripts/ml/evaluate_two_stage_development.py

python scripts/ml/build_acceptance_supplement.py

python scripts/ml/evaluate_constraint_gate_development.py

PYTHONPATH=.:src .venv/bin/python \
  scripts/ml/evaluate_candidate_complete_multiclass.py \
  --dataset-dir data/ml/processed/reviewed_evidence_training_v4_batch4 \
  --report-path reports/ml/generated/candidate_complete_sentence_embedding_v4.json \
  --local-files-only

PYTHONPATH=.:src .venv/bin/python \
  scripts/ml/evaluate_sentence_embedding_real_development.py \
  --local-files-only

PYTHONPATH=.:src .venv/bin/python \
  scripts/ml/evaluate_sentence_embedding_real_reserve.py

PYTHONPATH=.:src .venv/bin/python \
  scripts/ml/build_sentence_embedding_shadow_artifact.py
```

For a quick implementation check, add `--max-jobs 30 --max-features 1000`. A capped
run is not the evaluated model.

## Local artifacts and privacy

Raw data, caches, processed pairs, fitted models, and evaluation reports are ignored
under `data/ml/` and `reports/ml/`; only the de-identified validation manifests are tracked.
The release audit and pre-push hook also block local
private configuration artifacts and personal candidate data. Do not commit raw resumes,
raw job text, fitted models, or row-level predictions.
