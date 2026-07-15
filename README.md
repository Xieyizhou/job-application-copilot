# Job Application Copilot

**Privacy-first, local-first workflow for job discovery and application preparation**

[![CI](https://github.com/Xieyizhou/job-application-copilot/actions/workflows/ci.yml/badge.svg)](https://github.com/Xieyizhou/job-application-copilot/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Job Application Copilot is a Streamlit application for collecting job listings,
reviewing candidate-role fit, preparing tailored application documents, and
tracking manual applications.

The project uses deterministic requirement matching, eligibility checks,
confidence-aware scoring, and rule-based document generation. It does not submit
applications, scrape restricted job platforms, or make hiring predictions.

## Quick Demo

![Demo walkthrough](docs/assets/demo_walkthrough.gif)

The walkthrough uses only fictional and sanitized data:

**Dashboard → Review Jobs → Fit Analysis → Application Package**

## Highlights

- Searches Adzuna and Jooble through their supported APIs.
- Deduplicates fetched jobs and stores normalized descriptions as local Markdown.
- Accepts manually added jobs from pasted text, documents, PDFs, and screenshots.
- Normalizes company, role, location, source, and job-description fields before use.
- Produces an explainable Role Fit assessment using:
  - required and preferred terms
  - direct and partial evidence
  - missing candidate evidence
  - experience and education requirements
  - eligibility checks
  - scoring confidence
  - a final recommendation
- Treats incomplete job descriptions conservatively by showing
  **Insufficient evidence** instead of a misleading high-confidence score.
- Uses one current scoring result across Dashboard summaries, Review Jobs,
  Fit Analysis, Tracker preparation, and newly generated packages.
- Generates:
  - tailored resume drafts
  - cover-letter drafts
  - fit-analysis reports
  - internal review notes
  - DOCX files
  - a restricted ZIP export
- Separates fictional Demo data from ignored Personal files, API keys,
  generated outputs, and the local SQLite tracker.
- Includes regression, integration, extraction, document-export, privacy,
  and Streamlit runtime tests.

## Product Walkthrough

### Dashboard

![Dashboard](docs/assets/dashboard.png)

### Review Jobs

![Review Jobs](docs/assets/review_jobs.png)

### Fit Analysis

![Fit Analysis](docs/assets/fit_analysis.png)

### Application Package

![Application Package](docs/assets/application_package.png)

## Workflow

1. Fetch jobs through a supported API or add a target role manually.
2. Review normalized company, role, location, and job-description fields.
3. Compare the job requirements with the active candidate profile.
4. Review Role Fit, eligibility, scoring confidence, and missing evidence.
5. Generate a tailored application package.
6. Review and export the generated files.
7. Update the local application tracker manually.

## Quick Start

```bash
git clone https://github.com/Xieyizhou/job-application-copilot.git
cd job-application-copilot

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python run_dashboard.py
```

The application opens in the read-only Demo workspace and does not require API
credentials.

The project has been tested with:

- Python 3.11 on macOS ARM
- Python 3.12 on Linux

Use `python run_dashboard.py` rather than calling Streamlit directly. The
launcher configures the local rendering environment before starting the app.

## Demo and Personal Workspaces

### Demo

The Demo workspace:

- loads fictional jobs from `data/demo/`
- includes a sanitized sample application package
- makes no live API requests
- does not write tracker records
- does not generate Personal application files

Some bundled package files are historical generation-time snapshots and are not
automatically recalculated when the scoring implementation changes.

### Personal

The Personal workspace accepts a candidate source in one of these formats:

- Markdown
- TXT
- DOCX
- text-based PDF

Optional Personal inputs include:

- an experience-bank YAML file
- a cover-letter DOCX template

Personal workspace files are stored under:

```text
data/local_workspace/
```

This directory is ignored by Git.

Generated resumes, cover letters, and reports should always be reviewed before
use.

## Live Job Search

To enable live search, copy the example environment file:

```bash
cp .env.example .env
```

Add credentials for one or both supported providers:

```text
ADZUNA_APP_ID=your_adzuna_app_id
ADZUNA_APP_KEY=your_adzuna_app_key
JOOBLE_API_KEY=your_jooble_api_key
```

The `.env` file is ignored by Git and must never be committed.

Search behavior includes:

- configurable role queries
- preset or custom locations
- configurable source limits
- duplicate detection
- previously seen job tracking
- local Markdown storage
- source-specific error handling

Provider availability, returned descriptions, and result quality may vary.

## Role Fit Assessment

The project uses a deterministic and explainable scoring pipeline rather than a
trained prediction model.

The assessment considers:

- core technical requirements
- domain overlap
- candidate experience evidence
- project relevance
- communication-related requirements
- required versus preferred terms
- direct versus partial matches

Role Fit is kept separate from:

- eligibility
- work-authorization review
- scoring confidence
- final recommendation

A high raw match percentage does not automatically produce an Apply
recommendation.

See [Scoring Method](docs/SCORING_METHOD.md) for the scoring, gating, and public
benchmark methodology.

For example, when only one requirement is recognized, the app may show:

```text
Role Fit: Insufficient evidence
Eligibility: Passed
Scoring Confidence: Low
Recommendation: Manual Review
Recognized requirements: 1
Matched requirements: 1 of 1
```

This prevents a narrow `1 of 1` match from being presented as a confident
perfect fit.

## Application Package

A generated package may include:

- `tailored_resume.md`
- `tailored_resume.docx`
- `cover_letter.md`
- `cover_letter.docx`
- `analysis.md`
- internal tailoring notes
- a restricted ZIP export

The package generator:

- uses the current candidate and job sources
- preserves the canonical company and role
- includes eligibility and scoring-confidence information
- records the generation-time analysis
- adds the application to the local tracker when requested

Existing packages remain historical snapshots and are not automatically
rewritten.

## Local Tracker

The SQLite tracker stores application records such as:

- company
- role
- location
- source URL
- Role Fit score
- recommendation
- application status
- generated file paths
- notes
- application date

The application does not submit forms or change employer-side application
systems.

## Technical Stack

- Python
- Streamlit
- SQLite
- python-docx
- PyMuPDF
- pdfplumber
- pytesseract
- Pillow
- Adzuna API
- Jooble API
- local Markdown storage
- ZIP export

## Privacy and Safety

- Candidate files stay on the local machine.
- Generated application materials stay on the local machine.
- API credentials stay in the ignored `.env` file.
- Personal workspace data is excluded from Git.
- Demo content is fictional and sanitized.
- The application does not scrape LinkedIn, Indeed, or other restricted
  platforms.
- The application does not submit job applications automatically.
- Generated documents require manual review before use.
- Fit results are decision-support signals, not hiring predictions.

Run the release privacy check with:

```bash
python scripts/privacy_audit.py
```

An optional ignored file can add private terms to the scan:

```text
privacy_terms.local.txt
```

See `privacy_terms.local.example.txt` for the expected format.

## Validation

Run the complete automated test suite:

```bash
python -m py_compile main.py scripts/privacy_audit.py scripts/evaluate_scoring.py src/*.py
python -m unittest discover -s tests -v
python scripts/evaluate_scoring.py
python -m pip check
python scripts/privacy_audit.py
```

The release checks cover:

- deterministic scoring regression
- dashboard scoring integration
- title normalization
- tracker behavior
- package parsing
- document extraction
- export behavior
- privacy scanning
- dependency consistency
- Streamlit runtime behavior

## Project Structure

```text
job-application-copilot/
├── data/
│   └── demo/
├── docs/
│   ├── assets/
│   └── USAGE.md
├── scripts/
│   └── privacy_audit.py
├── src/
│   ├── analyze_job.py
│   ├── apply_package.py
│   ├── dashboard.py
│   ├── fetch_jobs.py
│   ├── manual_jobs.py
│   ├── tracker.py
│   └── workspace.py
├── tests/
├── .env.example
├── main.py
├── requirements.txt
├── run_dashboard.py
└── README.md
```

Local Personal data and generated outputs are excluded from the public
repository.

## Limitations

- Live search depends on provider credentials, availability, and rate limits.
- Provider job descriptions may be incomplete or truncated.
- Source listings should be verified on the original employer or provider page.
- Requirement extraction is deterministic and may not recognize every unusual
  phrase, synonym, or job-description format.
- Scoring depends on the quality and completeness of both the job description
  and candidate source.
- Existing Tracker rows and generated packages are historical snapshots.
- OCR quality depends on image clarity and local Tesseract installation.
- Scanned PDFs may require OCR before useful text can be extracted.
- Generated DOCX formatting is intentionally simple.
- Generated application materials require manual review.
- The project is a local single-user application, not a hosted service.

## Documentation

Detailed setup instructions, workspace behavior, and command examples are
available in:

[`docs/USAGE.md`](docs/USAGE.md)

## License

This project is licensed under the [MIT License](LICENSE).
