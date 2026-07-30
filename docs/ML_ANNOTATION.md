# Requirement–Evidence Annotation

This local workflow builds a human-reviewed dataset for matching one job requirement to
the strongest factual resume evidence. It is separate from the application dashboard so
labeling controls and research metadata do not add noise to the normal job-search flow.

## Start a pilot

Install the standard and ML dependencies, build a balanced queue, and open the labeling
workspace:

```bash
python -m pip install -r requirements.txt -r requirements-ml.txt
python scripts/ml/build_annotation_queue.py
python run_annotation.py
```

After both reserve passes finish:

```bash
python scripts/ml/prepare_real_adjudication.py --split reserve

JOB_COPILOT_ANNOTATION_QUEUE=data/ml/annotations/real_holdout_v1/real_reserve_adjudication_queue_v1.jsonl \
JOB_COPILOT_ANNOTATION_EVENTS=data/ml/annotations/real_holdout_v1/real_reserve_adjudication_decisions_v1.jsonl \
JOB_COPILOT_ANNOTATION_TRANSLATIONS=data/ml/annotations/real_holdout_v1/real_reserve_translations_zh_v1.jsonl \
python run_annotation.py
```

For the 48-task fresh reserve v2, Reviewer A is the designated human authority.
Reviewer B remains an independent consistency check, and disagreements preserve A's
decision without requiring a second pass:

```bash
python scripts/ml/prepare_real_adjudication.py \
  --split reserve-v2 \
  --annotation-dir data/ml/annotations/real_reserve_v2 \
  --prefer-reviewer-a

python scripts/ml/finalize_real_reserve.py \
  --dataset-version 2 \
  --annotation-dir data/ml/annotations/real_reserve_v2

python scripts/ml/evaluate_fresh_real_reserve.py
```

The final command is a one-time evaluation. Do not tune the saved candidate, its
thresholds, or its selection rule against reserve v2 after opening the labels. A failed
candidate remains rejected; use the failure report to design a new candidate and evaluate
that candidate on a newly constructed reserve.

The conservative semantic-blend candidate uses development v3 for calibration and the
source-disjoint development v2 packet for a non-regression check. It changes evidence
ranking while preserving the lexical accept/reject decision:

```bash
python scripts/ml/evaluate_semantic_blend_development.py
python scripts/ml/build_semantic_blend_candidate.py

python scripts/ml/build_real_development_extension.py \
  --dataset-version 3 \
  --dataset-role reserve \
  --output-dir data/ml/annotations/real_reserve_v3 \
  --tasks-per-family 12 \
  --random-state 20260728 \
  --candidate-path data/ml/models/evidence_semantic_blend_candidate_v2.joblib
```

The reserve builder records only the candidate artifact's byte hash. It does not load the
model or use predictions to select tasks. Existing datasets are used only as exclusion
indexes, and all requirement, evidence, source-job, and source-resume overlaps must remain
zero. The candidate remains unavailable to product inference until reserve v3 is frozen
and evaluated.

The default v3 queue contains 160 unique fictional tasks—40 each across Data, ML,
Software, and Business—plus 16 hidden repeats used to measure labeling consistency.
Completed v1 and v2 queues and event logs remain separate and are never overwritten.

V3 separates semantic relation from surface wording:

- the design target is 56 Direct, 56 Partial, and 48 No Support tasks
- 20 single-skill and six compound requirement constructions are mixed
- 16 candidate surface styles cover concise, context-first, result-first, project-note,
  two-clause, and responsibility-led descriptions
- every semantic candidate type rotates through all surface styles; no style accounts for
  more than 10% of one semantic type
- candidate openings are capped below 8% of all descriptions by regression tests
- the intended best-evidence position is balanced exactly across A, B, C, and D
- blind repeats receive a different candidate order
- retrieval rank, retrieval similarity, semantic stratum, and source labels are not stored

V2 remains the completed 48-task pilot used by the current experimental baseline report.

Queue and event files stay under ignored `data/ml/annotations/`. Labels are written as an
append-only event log, so clearing or changing a decision does not destroy prior work.

## Complete candidate labels

The training workflow records one judgment for every evidence candidate before asking for
the strongest evidence. This preserves same-task hard negatives and keeps `Direct`,
`Partial`, and `No Support` as separate candidate labels. Legacy task-level events remain
readable, but unselected candidates are never silently converted to negatives.

The existing training corpus contains 128 supported tasks whose unselected candidates still
need review. Build the first balanced 32-task batch and open it locally:

```bash
python scripts/ml/build_candidate_completion_batch.py \
  --batch-index 1 \
  --batch-size 32

JOB_COPILOT_ANNOTATION_QUEUE=data/ml/annotations/candidate_completion_v1/candidate_completion_queue_v1_batch1.jsonl \
JOB_COPILOT_ANNOTATION_EVENTS=data/ml/annotations/candidate_completion_v1/candidate_completion_events_v1_batch1.jsonl \
PYTHONPATH=.:src .venv/bin/python -m streamlit run src/ml/annotation_dashboard.py \
  --server.port 8507 \
  --server.headless true
```

After every task reports candidate-complete, merge the reviewed judgments into a new
training corpus. This command rejects incomplete events and does not overwrite v3:

```bash
python scripts/ml/finalize_candidate_completion_batch.py --batch-index 1
```

## Optional Chinese reading aid

Real-text reviewer packets can use a separate Chinese translation sidecar without changing
the English queue, candidate order, task checksum, or event log:

```bash
JOB_COPILOT_ANNOTATION_QUEUE=data/ml/annotations/real_holdout_v1/real_holdout_reviewer_a_queue_v1.jsonl \
JOB_COPILOT_ANNOTATION_EVENTS=data/ml/annotations/real_holdout_v1/reviewer_a_decisions.jsonl \
JOB_COPILOT_ANNOTATION_TRANSLATIONS=data/ml/annotations/real_holdout_v1/real_holdout_translations_zh_v1.jsonl \
python run_annotation.py
```

The sidebar controls whether Chinese text is shown. Translation records are keyed by
`task_id` and `candidate_id`, so independently shuffled reviewer packets can share one
sidecar. The interface verifies every stored English source string before displaying its
translation; stale or mismatched translations are hidden. Chinese text is a reading aid
only, and the English source remains the adjudication authority. Translation text is never
written into label events or model inputs.

## One complete decision

For each requirement:

1. label every candidate `Direct`, `Partial`, or `No Support`
2. choose the strongest candidate among those labeled Direct or Partial, or choose `None`
   when every candidate is No Support
3. if the task cannot be resolved, mark the whole task `Uncertain`
4. for supported evidence, decide whether it is safe to use in a cover
   letter

Use `Direct` only when the sentence independently demonstrates the requested skill or
experience. Use `Partial` when it is relevant but leaves an important part unproven. Use
candidate-level `No Support` when one sentence does not prove the requirement. Choose
task-level `None` only when every candidate is No Support. Use `Uncertain` when more context
or a second reviewer is needed.

Retrieval diagnostics are omitted from the queue rather than merely hidden. Skipped and
uncertain items remain available as dedicated review queues.

After labeling, run the aggregate audit:

```bash
python scripts/ml/audit_annotations.py
```

It reports selected-position balance, label balance, repeated requirement openings,
blind-repeat agreement, and forbidden retrieval metadata without printing resume or job
text. A position above 45%, a label above 70%, or a repeated opening above 35% produces a
warning.

## Pilot data boundaries

The queue generator extracts short requirement and evidence snippets; it does not retain
complete resumes or job descriptions. Contact strings are removed, and only stable source
hashes are kept for document-grouped evaluation. Local pilot queues can mix:

- real-derived ATS pairs, used as research material rather than hiring ground truth
- fictional candidate-matching records, used to fill sparse role families

Each task records its source dataset outside the visible labeling surface. Before any
human-reviewed examples are committed as a public validation set, remove names,
organizations, schools, locations, URLs, and contacts, then review every line manually.
Never commit the local queue, event log, raw resumes, or raw job descriptions.

The ATS research corpus is excluded by default because its flattened source text can join
several resume sections and expose identifying entities. It is available only for local
parser research:

```bash
python scripts/ml/build_annotation_queue.py --include-ats
```

Do not use that option to create a public dataset without a separate entity-redaction and
manual privacy-review stage.

## Quality gates for a useful dataset

A first pilot validates the interface and label definitions; it is not enough to fit a
credible model. The next dataset should contain at least 150–300 reviewed unique pairs and:

- include Direct, Partial, and No Support examples rather than only positive matches
- include hard negatives from adjacent skills and responsibilities
- keep every resume and every job in only one of train, validation, or test
- reserve an untouched real-derived test set
- review uncertain items separately
- report repeat agreement and per-role/per-label counts

The first model comparison should include TF-IDF retrieval, sentence-embedding retrieval,
and a lexical reranker. Report Recall@1, Recall@3, mean reciprocal rank, macro F1, a
confusion matrix, and representative errors. Model scores remain retrieval diagnostics;
they are not hiring probabilities or application decisions.

## Synthetic data expansion

The optional offline process drafts fictional requirement/evidence cases across different
difficulty barriers. It reduces manual drafting time but does not replace human judgment.

Public source code contains only provider-neutral schemas, validation, blinding, consensus,
and audit logic. Private configuration, proposals, reviewer packets, reviewer decisions,
and row-level outputs remain in ignored local paths.

The barrier taxonomy covers:

- explicit and semantic direct support
- adjacent or incomplete support
- compound requirements with partial coverage
- scale, ownership, duration, and deployment mismatches
- same-domain hard negatives
- non-practical mentions such as planning or documentation

Each producer proposal stores its intended label only as diagnostic metadata. An allow-list
reconstruction removes that metadata before review, and it never votes in consensus.
Reviewers see only a requirement and independently shuffled evidence candidates. They label
every candidate, select the strongest candidate, and add quality flags for ambiguity,
multiple plausible answers, wording leakage, unsupported inference, or privacy risk.

Prepare a local proposal batch and three independent blind packets:

```bash
python scripts/ml/prepare_barrier_review.py
```

The preparation audit checks schema and privacy boundaries, producer/barrier/role coverage,
answer-position balance, exact and near duplicates, requirement-style association,
candidate length/opening/number/punctuation association, and hidden metadata leakage. The
initial style-association warning threshold is Cramér's V `0.15`; a supported answer
position above `35%` also produces a warning. These are dataset diagnostics, not model
metrics.

After reviewers write candidate-level decisions into the ignored `reviews/` directory,
unblind and aggregate them:

```bash
python scripts/ml/adjudicate_barrier_reviews.py
```

Automatic gold promotion requires at least three reviewers and unanimous agreement on the
overall label, strongest evidence, and every candidate label, with no critical quality
flag. Anything else goes to human adjudication or rejection. Gold records can be created
only from blind consensus, human annotation, or human adjudication; producer intent is
never an allowed decision source.

### Human adjudication queue

Cases without unanimous blind agreement are written to the local, ignored queue at
`data/ml/annotations/pipeline_v1/adjudication/queue.jsonl`. Open that queue with a separate
append-only event log:

```bash
JOB_COPILOT_ANNOTATION_QUEUE=data/ml/annotations/pipeline_v1/adjudication/queue.jsonl \
JOB_COPILOT_ANNOTATION_EVENTS=data/ml/annotations/pipeline_v1/adjudication/human_decisions.jsonl \
python run_annotation.py
```

For `Direct` or `Partial`, select the single strongest evidence. For `No Support`, select
`None`. Do not use `Uncertain` as a final adjudication: resolve it or leave it incomplete.
When all queue items are resolved, turn the event log into conservative human-adjudication
gold:

```bash
python scripts/ml/import_human_adjudications.py
```

The import verifies that every queue item still matches its source case, rejects skipped,
missing, and uncertain decisions by default, and writes ignored local gold records. It uses
only the selected `Direct`/`Partial` candidate as a positive pair; other candidates remain
unlabeled. A `No Support` decision labels every candidate as negative. This prevents an
unselected but plausible candidate from becoming a synthetic false negative.

The consensus report includes overall-label and candidate-level Fleiss' kappa. A reviewer
pilot should reach at least `0.70` before its labels are used for model comparison; kappa
does not override the stricter per-case unanimity rule for automatic gold promotion.

All runtime data remains under the Git-ignored path:

```text
data/ml/annotations/pipeline_v1/
```

## Real-data calibration and fixed holdout

Synthetic cases broaden coverage but do not establish real-world performance. Anonymous
human-reviewed real tasks must be split by source identity before fitting or threshold
selection. Each real task needs anonymous resume and job hashes, plus a semantic or
near-duplicate group when applicable.

```bash
python scripts/ml/build_real_validation_split.py
```

Tasks connected by the same resume, job, or semantic duplicate group stay in one split.
The output includes train, validation, and fixed holdout files, zero-overlap checks,
per-label counts, and a SHA-256 checksum over holdout task identities. A holdout with fewer
than 40 unique reviewed tasks remains explicitly provisional.

Use real train data only for domain adaptation, real validation only for thresholds and
model selection, and the frozen holdout only for final reporting. Do not fit on the
holdout, then report the same rows as independent evidence.

## Reviewed v3 reranker experiment

After exporting the completed portion of v3 and adjudicating the synthetic expansion,
combine only human annotations and accepted consensus gold:

```bash
python scripts/ml/build_evidence_training_corpus.py
```

Candidate-level consensus labels are preserved. Human-supported tasks continue to use the
conservative policy: selected strongest evidence is positive, while unselected candidates
remain unlabeled rather than being converted into false negatives. Producer intent and
human-review-queue cases are not imported.

Train and compare the local experiment:

```bash
python scripts/ml/train_evidence_reranker.py
```

The grouped comparison includes:

- the transparent concept/lexical rule
- word TF-IDF cosine
- local word/character LSA embeddings
- the existing shared-term pair classifier
- a learned hybrid LSA reranker
- a lexical-guarded reranker
- a pairwise hybrid reranker trained from the human-selected strongest evidence

Every fold keeps one semantic or requirement-template group out for testing and uses a
different mixed-label group for threshold selection. Reports include pair metrics,
Recall@1, Recall@3, MRR, No-Support rejection, task-decision accuracy, and ID-only error
analysis.

Model selection is predefined and lexicographic: task-decision accuracy, average precision,
Recall@1, then No-Support rejection. The trainer fits the selected artifact-capable
reranker rather than requiring pairwise ranking to win. The grouped comparison report is
written before fitting, so an unsupported winner still leaves an auditable result.

The selected schema-v2 joblib artifact remains at the ignored local path
`data/ml/models/evidence_reranker_v3.joblib` and is marked
`experimental_not_used_by_application`. Its metadata includes a fingerprint of the exact
annotated tasks and training pairs. External evaluation rejects legacy artifacts and any
artifact whose fingerprint does not match the current training corpus.

Run the small external diagnostic:

```bash
python scripts/ml/evaluate_evidence_reranker.py
```

This command rejects exact train/evaluation overlap and evaluates the artifact on the
tracked de-identified 24-case semantic set. The set is too small and too familiar to the
project to qualify as the fixed real holdout. Its result is diagnostic only; promotion
still requires at least 40 untouched real tasks with zero resume, job, and semantic-group
overlap.

## Freeze and evaluate the real-text holdout

Reviewer A and reviewer B use independently shuffled queues. Reviewer B may be a clearly
identified model-assisted independent review when a second human is unavailable, but the
result is not represented as two-human agreement. The human adjudicator reviews only
disagreements without seeing either prior answer.

After all disagreement tasks are resolved, freeze the local gold file once:

```bash
python scripts/ml/finalize_real_holdout.py
python scripts/ml/evaluate_real_holdout.py
```

Finalization validates complete coverage, queue-content identity, candidate ids, and final
labels. It then writes deterministic JSONL plus a SHA-256 manifest and refuses to overwrite
an existing frozen holdout. Evaluation verifies the checksum and blocks exact text,
source-job, or source-resume overlap with training. Every threshold comes from the
training-only grouped experiment; the holdout is never used to select a threshold.

The real-text gate compares the transparent concept/lexical rule, local LSA embedding, and
the selected schema-v2 reranker. Promotion requires the learned artifact to improve task
decision accuracy without reducing Recall@1 or No-Support rejection. A failure leaves the
application inference path unchanged.

After validation adjudication, compare and freeze the selected development candidate before
opening any reserve labels:

```bash
python scripts/ml/finalize_real_validation.py
python scripts/ml/calibrate_real_validation.py
python scripts/ml/build_validation_candidate_artifact.py
python scripts/ml/prepare_real_review_packets.py --split reserve
```

The candidate artifact records the training-corpus fingerprint, validation checksum,
selected method, and threshold. It refuses overwrite so reserve labels cannot influence
candidate construction.

Start reserve Reviewer A with its separately approved local translation sidecar:

```bash
JOB_COPILOT_ANNOTATION_QUEUE=data/ml/annotations/real_holdout_v1/real_reserve_reviewer_a_queue_v1.jsonl \
JOB_COPILOT_ANNOTATION_EVENTS=data/ml/annotations/real_holdout_v1/real_reserve_reviewer_a_decisions_v1.jsonl \
JOB_COPILOT_ANNOTATION_TRANSLATIONS=data/ml/annotations/real_holdout_v1/real_reserve_translations_zh_v1.jsonl \
python run_annotation.py
```

After reserve adjudication, freeze both the gold checksum and the already committed
candidate checksum before the one-time evaluation:

```bash
python scripts/ml/finalize_real_reserve.py
python scripts/ml/evaluate_real_reserve.py
```

The 16-task reserve is a direction check, not a promotion set. Failure keeps the candidate
disabled. Passing would justify constructing a new source-isolated 40+ task final reserve,
not immediate product integration.

The first validation-selected threshold did not transfer to the 16-task direction check.
That reserve is therefore consumed and must not be used to tune its replacement. The
follow-up dual-signal experiment can be reproduced with:

```bash
python scripts/ml/calibrate_dual_signal_validation.py
```

It keeps TF-IDF candidate ranking separate from top-candidate acceptance and calibrates
only against the 24-task validation gold. Both the fixed transparent gate and the fitted
explicit-feature gate improved rejection but rejected too many supported tasks, so the
script records `blocked_validation_not_improved` and does not create an artifact.

Do not prepare another final reserve yet. First add a new source-isolated development
extension with enough supported and No Support cases to estimate an acceptance boundary.
After one architecture is selected and frozen without reading prior holdout/reserve labels,
prepare a fresh 40+ task reserve for a single final evaluation.

## Development extension v2

Build the new 48-task development extension from the ignored local Djinni and SkillSpan
sources:

```bash
python scripts/ml/build_real_development_extension.py
```

The builder refuses overwrite and creates 12 tasks each for Data, ML, Software, and
Business. It balances supportive, adjacent, and low-overlap construction strata without
assigning labels or loading model predictions. Every task uses a distinct job and resume,
and source hashes plus exact/near text are checked against training, validation, holdout,
and the consumed reserve.

Reviewer packets are independently ordered. Among tasks with a unique highest lexical
anchor, its A/B/C/D position differs by at most one occurrence. Tied tasks remain
independently shuffled. The anchor and construction stratum are not visible in the
annotation queue.

Create a Chinese reading-aid sidecar before beginning review when the source text is
authorized for Google Translate. The queue itself, candidate ordering, checksum, and event
log remain local and unchanged:

```bash
python scripts/ml/translate_annotation_queue.py \
  --input data/ml/annotations/real_development_v2/real_development_reviewer_a_queue_v2.jsonl \
  --output data/ml/annotations/real_development_v2/real_development_translations_zh_v2.jsonl
```

The script retries bounded requests, refuses overwrite by default, and may resume a
source-verified partial sidecar using `--resume`. The dashboard verifies the English source
for every requirement and candidate before displaying any Chinese text.

Start Reviewer A:

```bash
JOB_COPILOT_ANNOTATION_QUEUE=data/ml/annotations/real_development_v2/real_development_reviewer_a_queue_v2.jsonl \
JOB_COPILOT_ANNOTATION_EVENTS=data/ml/annotations/real_development_v2/real_development_reviewer_a_decisions_v2.jsonl \
JOB_COPILOT_ANNOTATION_TRANSLATIONS=data/ml/annotations/real_development_v2/real_development_translations_zh_v2.jsonl \
python run_annotation.py
```

Reviewer B must use the separate queue and event log:

```bash
JOB_COPILOT_ANNOTATION_QUEUE=data/ml/annotations/real_development_v2/real_development_reviewer_b_queue_v2.jsonl \
JOB_COPILOT_ANNOTATION_EVENTS=data/ml/annotations/real_development_v2/real_development_reviewer_b_decisions_v2.jsonl \
JOB_COPILOT_ANNOTATION_TRANSLATIONS=data/ml/annotations/real_development_v2/real_development_translations_zh_v2.jsonl \
python run_annotation.py
```

After both passes, build the blinded disagreement queue:

```bash
python scripts/ml/prepare_real_adjudication.py \
  --split development \
  --annotation-dir data/ml/annotations/real_development_v2
```

Run the adjudication dashboard with its own event log, then freeze development gold:

```bash
JOB_COPILOT_ANNOTATION_QUEUE=data/ml/annotations/real_development_v2/real_development_adjudication_queue_v2.jsonl \
JOB_COPILOT_ANNOTATION_EVENTS=data/ml/annotations/real_development_v2/real_development_adjudication_decisions_v2.jsonl \
python run_annotation.py

python scripts/ml/finalize_real_development.py
```

This gold is development-only. It may be used to compare acceptance gates and thresholds,
but it must never be reported as an untouched final generalization result.

## Development extension v3

The 48-task v2 development result was too small to establish that the
constraint-aware acceptance gate improves over TF-IDF. Build a larger 96-task
source-isolated extension with 24 tasks per role family and balanced supportive,
adjacent, and low-overlap construction strata:

```bash
python scripts/ml/build_real_development_extension.py \
  --dataset-version 3 \
  --tasks-per-family 24 \
  --random-state 20260727
```

The builder excludes the reviewed training corpus, validation, consumed holdout
and reserve, and development v2. It also rejects exact and near-text overlap
within the new batch. Construction uses deterministic lexical and reviewable
skill-family features, not model predictions or labels.

Create the approved local Chinese sidecar and start Reviewer A:

```bash
python scripts/ml/translate_annotation_queue.py \
  --input data/ml/annotations/real_development_v3/real_development_reviewer_a_queue_v3.jsonl \
  --output data/ml/annotations/real_development_v3/real_development_translations_zh_v3.jsonl

JOB_COPILOT_ANNOTATION_QUEUE=data/ml/annotations/real_development_v3/real_development_reviewer_a_queue_v3.jsonl \
JOB_COPILOT_ANNOTATION_EVENTS=data/ml/annotations/real_development_v3/real_development_reviewer_a_decisions_v3.jsonl \
JOB_COPILOT_ANNOTATION_TRANSLATIONS=data/ml/annotations/real_development_v3/real_development_translations_zh_v3.jsonl \
python run_annotation.py
```

After independent Reviewer B and human adjudication:

```bash
python scripts/ml/prepare_real_adjudication.py \
  --split development-v3 \
  --annotation-dir data/ml/annotations/real_development_v3

python scripts/ml/finalize_real_development.py \
  --dataset-version 3
```

Development v3 remains model-selection data. It cannot be used as a final
generalization result or added to the training corpus before architecture and
threshold selection are frozen.

Once holdout results or errors have been inspected, this version becomes a frozen
regression set for that experiment. Do not tune a replacement model on its labels and then
claim a fresh generalization result on the same tasks. Use only source-isolated development
data for the replacement, then construct a new reserve before the next one-time final test.

## Acceptance training supplement

The acceptance supplement is an active-learning training set, not an evaluation set. Its
80 tasks are balanced across Data, ML, Software, and Business roles. Hidden construction
metadata balances four model-disagreement strata; reviewers never see those scores or any
suggested label. Every task uses a distinct source job and source resume, and all text is
isolated from the consumed training, validation, holdout, reserve, and development sets.

Build the local packets:

```bash
python scripts/ml/build_acceptance_supplement.py
```

Create the approved local Chinese sidecar:

```bash
python scripts/ml/translate_annotation_queue.py \
  --input data/ml/annotations/acceptance_supplement_v1/acceptance_supplement_reviewer_a_queue_v1.jsonl \
  --output data/ml/annotations/acceptance_supplement_v1/acceptance_supplement_translations_zh_v1.jsonl
```

Start Reviewer A:

```bash
JOB_COPILOT_ANNOTATION_QUEUE=data/ml/annotations/acceptance_supplement_v1/acceptance_supplement_reviewer_a_queue_v1.jsonl \
JOB_COPILOT_ANNOTATION_EVENTS=data/ml/annotations/acceptance_supplement_v1/acceptance_supplement_reviewer_a_decisions_v1.jsonl \
JOB_COPILOT_ANNOTATION_TRANSLATIONS=data/ml/annotations/acceptance_supplement_v1/acceptance_supplement_translations_zh_v1.jsonl \
python run_annotation.py
```

Do not use these tasks to report model quality. After independent review and adjudication,
their accepted gold pairs may be added to a future training corpus; a new source-isolated
reserve remains mandatory for evaluation.

After Reviewer A and an independent Reviewer B finish, prepare the content-only
disagreement queue:

```bash
python scripts/ml/prepare_real_adjudication.py \
  --split acceptance \
  --annotation-dir data/ml/annotations/acceptance_supplement_v1
```

Start the local adjudication page with the approved Chinese sidecar:

```bash
streamlit run src/ml/adjudication_dashboard.py -- \
  --queue data/ml/annotations/acceptance_supplement_v1/acceptance_supplement_adjudication_queue_v1.jsonl \
  --events data/ml/annotations/acceptance_supplement_v1/acceptance_supplement_adjudication_decisions_v1.jsonl \
  --translations data/ml/annotations/acceptance_supplement_v1/acceptance_supplement_translations_zh_v1.jsonl
```

When every disagreement has a human decision, freeze the reviewed supplement,
rebuild the combined training corpus, and train the selected evidence model:

```bash
python scripts/ml/finalize_acceptance_supplement.py
python scripts/ml/build_evidence_training_corpus.py
python scripts/ml/train_evidence_reranker.py
```

Finalization refuses to create gold while adjudications are missing. The resulting
supplement remains prohibited from validation, model selection, holdout, and reserve
evaluation.

Prepare independently ordered packets for the real validation split:

```bash
python scripts/ml/prepare_real_review_packets.py --split validation
```

The command verifies that both packets retain identical task and candidate content, writes
an ordered-packet checksum for each reviewer, and refuses to overwrite existing packets.
Reviewer decisions remain in separate ignored event files until both passes are complete.

When an approved local Chinese sidecar is available, start validation Reviewer A with:

```bash
JOB_COPILOT_ANNOTATION_QUEUE=data/ml/annotations/real_holdout_v1/real_validation_reviewer_a_queue_v1.jsonl \
JOB_COPILOT_ANNOTATION_EVENTS=data/ml/annotations/real_holdout_v1/real_validation_reviewer_a_decisions_v1.jsonl \
JOB_COPILOT_ANNOTATION_TRANSLATIONS=data/ml/annotations/real_holdout_v1/real_validation_translations_zh_v1.jsonl \
python run_annotation.py
```

After both reviewers finish, create the content-only disagreement queue:

```bash
python scripts/ml/prepare_real_adjudication.py --split validation
```

The adjudicator does not see either prior answer. Start the resulting queue with:

```bash
JOB_COPILOT_ANNOTATION_QUEUE=data/ml/annotations/real_holdout_v1/real_validation_adjudication_queue_v1.jsonl \
JOB_COPILOT_ANNOTATION_EVENTS=data/ml/annotations/real_holdout_v1/real_validation_adjudication_decisions_v1.jsonl \
JOB_COPILOT_ANNOTATION_TRANSLATIONS=data/ml/annotations/real_holdout_v1/real_validation_translations_zh_v1.jsonl \
python run_annotation.py
```

## Export and baseline evaluation

After all repeat conflicts are resolved, export one row per unique task and a
high-confidence binary pair table. The current export defaults remain pinned to the
completed v2 pilot until v3 labeling is finished:

```bash
python scripts/ml/export_annotation_dataset.py
```

The pair table follows a backward-compatible labeling policy:

- complete reviews preserve every candidate as Direct, Partial, or No Support
- legacy selected evidence from a supported task remains a positive pair
- legacy candidates in a No Support task remain negative pairs
- legacy unselected candidates from supported tasks stay unlabeled until completion
- blind repeats are excluded

To freeze a reviewed v3 seed before the full queue is complete, export only completed,
resolved tasks. This does not modify the append-only event log:

```bash
python scripts/ml/export_annotation_dataset.py \
  --queue-path data/ml/annotations/pilot_queue_v3.jsonl \
  --events-path data/ml/annotations/pilot_annotations_v3.jsonl \
  --output-dir data/ml/processed/reviewed_evidence_v3_seed \
  --allow-partial \
  --dataset-name reviewed_evidence_v3_human_seed
```

Run the experimental comparison:

```bash
python scripts/ml/train_annotation_baseline.py
```

Evaluation leaves one requirement wording group out at a time. Thresholds are selected
using a different mixed-label wording group. The report compares the transparent
concept/lexical rule, TF-IDF cosine, and a trained local pair classifier. It includes pair
F1, Recall@1, Recall@3, mean reciprocal rank, and No Support rejection.

The resulting model is explicitly experimental and is not used by the application.
Promotion requires an independent, anonymous real-data holdout and a retrieval improvement
over the strongest baseline.

## Candidate-complete three-class evaluation

After a candidate-completion batch has been reviewed and finalized, compare the local
three-class methods with task-grouped cross-validation:

```bash
PYTHONPATH=.:src .venv/bin/python scripts/ml/evaluate_candidate_complete_multiclass.py
```

The report compares a majority No Support baseline, pair-text TF-IDF, local LSA plus
transparent features, a pinned frozen sentence encoder, and the frozen encoder combined
with transparent features. All candidates from one requirement remain in the same fold.
This prevents candidates from the same task leaking between training and evaluation.

The sentence model is downloaded once and can then be evaluated without network access:

```bash
PYTHONPATH=.:src .venv/bin/python \
  scripts/ml/evaluate_candidate_complete_multiclass.py \
  --dataset-dir data/ml/processed/reviewed_evidence_training_v4_batch4 \
  --report-path reports/ml/generated/candidate_complete_sentence_embedding_v4.json \
  --local-files-only
```

This is a development comparison only. It does not use the frozen holdout or reserve data,
does not change the application model, and cannot authorize promotion. Transformer weights
remain frozen; only a balanced local logistic head is fitted inside each training fold.

After locking a development candidate, test it without threshold tuning on the
source-isolated real development split:

```bash
PYTHONPATH=.:src .venv/bin/python \
  scripts/ml/evaluate_sentence_embedding_real_development.py \
  --local-files-only
```

This split may support architecture decisions and the decision to construct a new
reserve. It must not be described as an untouched final holdout or used to claim product
readiness.

## Frozen real reserve v4

Reserve v4 contains 96 new source-isolated tasks with 24 tasks in each of Data, ML,
Software, and Business. Its candidate specification, training-corpus hashes, source-code
hashes, development report, and success criteria were frozen before task construction.
The model was not read during sampling.

Two blind reviewers completed all 384 candidate judgments independently. Human review
covered all 21 exact disagreements plus eight deterministically sampled exact agreements.
The adjudicator's decision is final for both groups. The frozen gold is restricted to a
one-time evaluation and may not be added to training or used for threshold selection.

```bash
PYTHONPATH=.:src .venv/bin/python \
  scripts/ml/evaluate_sentence_embedding_real_reserve.py
```

The reserve gate passed. Further work is limited to diagnostic failure review and
non-decision shadow operation; application inference remains unchanged.

Shadow reports are local operational diagnostics, not annotation or training data. Keep
them under `data/ml/shadow/`; do not import them into an annotation corpus, use them to
retune the reserve-tested candidate, or expose their predictions in the application UI.
Only independently reviewed future development data may support a new model version.

The completed shadow v1 blind review is frozen as an operational diagnostic holdout.
Its disagreement-enriched metrics may reject an unsafe deployment but are not population
accuracy estimates. The reviewed rows, their method origins, and their errors are
prohibited from successor-model training and threshold selection.

## Operational successor packet

`operational_development_v2` contains 64 source-isolated tasks built from 16
TalentCLEF 2026 privacy-protecting synthetic profiles and unused public Djinni
requirements. Each profile appears in four tasks. Both role families and requirement
families contain 16 tasks each across Data, ML, Software, and Business.

The two reviewer queues are content-identical but independently shuffled. Reviewers must
label all four candidates as Direct, Partial, or No Support. Resume/job relevance from an
upstream corpus is not a candidate-evidence label and is never shown or imported as gold.
Any reviewer disagreement must go to content-only human adjudication.

This packet may enter successor training/development only after complete review. It is
prohibited from final evaluation because its candidate profiles are synthetic. A future
promotion decision still requires a separately sourced real, anonymous, frozen
operational holdout whose resume and job groups do not occur in this packet.

Reviewer A completed every task and candidate judgment. A frozen local model supplied an
independent diagnostic Reviewer B without reading Reviewer A events. Exact task decisions
agreed on 37/64 tasks, task labels agreed on 45/64, and candidate labels agreed on
208/256. The 27 exact disagreements were resolved by the declared policy that Reviewer A
is the final human authority. The frozen gold contains 64 tasks and 256 fully labeled
pairs; Reviewer B output is diagnostic metadata only.

The gold is stored locally under
`data/ml/processed/operational_development_v2/`. Its candidate profiles are synthetic, so
it may support grouped cross-validation and architecture development but may not be
renamed or reused as a real holdout.

The first explicit task-rejection experiment is also development-only. Its grouped error
report is stored under the ignored
`reports/ml/generated/operational_task_rejector_v2.json`. The report contains aggregate
role, profile, pairing, and source slices but never requirement or evidence text. The
experiment was rejected because improved No Support rejection came with a larger loss on
supported tasks. Do not relabel these tasks or tune individual decisions to improve that
report.

A future supplement should add new profile and job groups rather than alternate versions
of these 64 tasks. Prioritize aligned Direct and Partial support, ambiguous same-role No
Support candidates, and underrepresented Business/Data language. Freeze the construction
rules before review and keep all tasks from one profile in one split.
