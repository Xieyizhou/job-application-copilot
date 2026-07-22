# Local ML Relevance

## Purpose

The optional model supplies a second view of resume/job relevance. It is not an
interview-probability model and is never an automatic application decision:

- deterministic Role Fit remains the ranking score
- eligibility remains a separate hard-constraint review
- confidence remains tied to JD and candidate-evidence completeness
- the local ML probability is displayed only as an experimental auxiliary signal
- missing dependencies or incompatible artifacts fail closed
- collapsed, out-of-distribution batches are hidden instead of being rounded into misleading `0%` values

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

## Commands

```bash
python -m pip install -r requirements.txt -r requirements-ml.txt

python scripts/ml/train_relevance_baseline.py
```

For a quick implementation check, add `--max-jobs 30 --max-features 1000`. A capped
run is not the evaluated model.

## Local artifacts and privacy

Raw data, caches, processed pairs, fitted models, and evaluation reports are ignored
under `data/ml/` and `reports/ml/`. The release audit and pre-push hook also block local
automation-instruction artifacts and personal candidate data. Do not commit raw resumes,
raw job text, fitted models, or row-level predictions.
