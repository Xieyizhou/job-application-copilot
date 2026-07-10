# Usage Guide

## Install and Run

```bash
git clone https://github.com/your-account/job-application-copilot.git
cd job-application-copilot
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run src/dashboard.py
```

The initial screen uses the sanitized, read-only Demo workspace.

macOS users can also launch the app after setup by opening
`open_dashboard.command`.

## Workspaces

### Demo

Demo loads tracked fictional jobs and a pre-generated sample package from
`data/demo/`. It requires no credentials and does not create tracker records or
Personal outputs.

### Personal

Select **Personal** in the sidebar and provide a candidate source in Markdown,
UTF-8 text, DOCX, or text-based PDF format. The app extracts a canonical local
Markdown source. Scanned candidate PDFs and legacy `.doc` files are not
supported.

Optional Personal inputs:

- experience-bank YAML
- cover-letter DOCX template

Personal data stays under the ignored directory below:

```text
data/local_workspace/
  workspace.json
  candidate/
  templates/
  jobs/
  generated/
  applications.db
```

Use **Replace candidate files** to update candidate inputs. To reset Personal
data completely, stop the app and remove `data/local_workspace/`.

## Live Job Search

Copy `.env.example` to `.env` and add credentials for the providers you plan to
use:

```text
ADZUNA_APP_ID=your_adzuna_app_id
ADZUNA_APP_KEY=your_adzuna_app_key
JOOBLE_API_KEY=your_jooble_api_key
```

The dashboard supports preset regions and custom locations. It ranks results
after duplicate removal and local filtering. Provider descriptions may be
partial, so confirm requirements on the original job page.

Adzuna command-line example:

```bash
python src/fetch_jobs.py \
  --source adzuna \
  --country us \
  --query "data analyst" \
  --location "Remote" \
  --max-results 10
```

Jooble command-line example:

```bash
python src/fetch_jobs.py \
  --source jooble \
  --query "data analyst" \
  --location "Remote" \
  --max-results 10
```

Jobs saved through Personal workspace are stored below
`data/local_workspace/jobs/`.

## Analyze a Demo Job

Run a read-only analysis without saving Personal output:

```bash
python src/analyze_job.py data/demo/jobs/machine_learning_intern.md --demo
```

The report includes matched evidence, missing keywords, risk notes, a score,
and a recommendation based on deterministic rules.

## Generate an Application Package

After Personal workspace setup, run:

```bash
python src/apply_package.py data/demo/jobs/machine_learning_intern.md \
  --company "Example Robotics Company" \
  --role "Machine Learning Engineer Intern" \
  --location "Remote" \
  --job-url "https://example.com/job"
```

Each run creates a timestamped package:

```text
data/local_workspace/generated/<company_role>/<timestamp>/
  analysis.md
  tailored_resume.md
  tailored_resume.docx
  cover_letter.md
  cover_letter.docx
  tailoring_notes.md
  cover_letter_notes.md
```

To re-export an existing package:

```bash
python src/export_documents.py \
  data/local_workspace/generated/example_company_role/<timestamp>/
```

Review every generated file before sharing it with an employer.

## Local Data Boundaries

Do not commit:

- `.env` or API credentials
- candidate files and experience banks
- fetched job records
- generated application packages
- SQLite databases, logs, or caches

The `.gitignore` covers the standard local paths. Before publishing, run:

```bash
python scripts/privacy_audit.py
git status --short
```

The app never submits applications or automates third-party job platforms.
