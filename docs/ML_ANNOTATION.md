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

## One decision

For each requirement:

1. choose the single strongest resume sentence, or choose `None`
2. assign `Direct`, `Partial`, `No Support`, or `Uncertain`
3. for Direct or Partial support, decide whether the evidence is safe to use in a cover
   letter

Use `Direct` only when the sentence independently demonstrates the requested skill or
experience. Use `Partial` when it is relevant but leaves an important part unproven. Use
`No Support` when none of the choices supports the requirement. Use `Uncertain` when more
context or a second reviewer is needed.

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

## LLM-assisted synthetic expansion

The optional expansion pipeline uses multiple, role-separated producers to draft fictional
requirement/evidence cases at different difficulty barriers. It reduces manual drafting
time; it does not replace human judgment.

Public source code contains only provider-neutral schemas, validation, blinding, consensus,
and audit logic. Generation instructions, provider configuration, proposals, reviewer
packets, reviewer decisions, and row-level outputs remain in ignored local paths. No
generation prompt is required or stored in this repository.

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
Recall@1, then No-Support rejection. The selected joblib artifact remains under ignored
`data/ml/models/` and is marked `experimental_not_used_by_application`.

Run the small external diagnostic:

```bash
python scripts/ml/evaluate_evidence_reranker.py
```

This command rejects exact train/evaluation overlap and evaluates the artifact on the
tracked de-identified 24-case semantic set. The set is too small and too familiar to the
project to qualify as the fixed real holdout. Its result is diagnostic only; promotion
still requires at least 40 untouched real tasks with zero resume, job, and semantic-group
overlap.

## Export and baseline evaluation

After all repeat conflicts are resolved, export one row per unique task and a
high-confidence binary pair table. The current export defaults remain pinned to the
completed v2 pilot until v3 labeling is finished:

```bash
python scripts/ml/export_annotation_dataset.py
```

The pair table follows a conservative labeling policy:

- the selected evidence from a `Direct` or `Partial` task is a positive pair
- every candidate in a `No Support` task is a negative pair
- unselected candidates from supported tasks stay unlabeled
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
