# Project Status

Last updated: 2026-07-10

## Current State

Job Application Copilot is a working local-first, human-in-the-loop Streamlit
application. It supports job discovery, manual job capture, explainable fit
review, tailored application package generation, DOCX/ZIP export, and manual
application tracking. It never submits applications automatically.

The public repository is based on one fresh sanitized commit (`0bc8634`). The
archived history must not be imported because earlier commits may contain
personal data.

## Public Product Boundaries

- Fresh sessions open in the sanitized, read-only Demo workspace.
- Personal workflows require candidate files supplied by the local user.
- Personal files, fetched jobs, generated packages, and the SQLite tracker stay
  under ignored local paths.
- API credentials stay in `.env`; `.env.example` contains placeholders only.
- Fit scoring and document generation use local deterministic rules and
  templates.
- Job applications remain manual and human-reviewed.
- Agent-facing instructions, project status, development notes, and release
  checklists stay in the adjacent local-notes directory, outside Git.

## Completed Workflow

1. Find jobs through Adzuna/Jooble or add a target job manually.
2. Normalize and review role metadata.
3. Compare the role with the active candidate profile.
4. Generate a tailored resume, cover letter, fit report, and internal notes.
5. Export DOCX files or a whitelisted ZIP package.
6. Update the local application tracker manually.

## Main Components

- `src/dashboard.py` - Streamlit routing and workflow UI.
- `src/workspace.py` - Demo/Personal isolation and local candidate setup.
- `src/candidate_document.py` - candidate Markdown, text, DOCX, and PDF input.
- `src/manual_jobs.py` - manual job extraction, parsing, and storage.
- `src/fetch_jobs.py` / `src/fetch_history.py` - API intake and deduplication.
- `src/analyze_job.py` - explainable heuristic fit analysis.
- `src/apply_package.py` - application package orchestration.
- `src/generate_tailored_resume.py` / `src/generate_cover_letter.py` - local
  document generation.
- `src/export_documents.py` - DOCX and package export helpers.
- `src/tracker.py` - local SQLite application tracker.

## Public Demo Assets

- Three fictional sample jobs under `data/demo/jobs/`.
- Sanitized read-only Markdown and DOCX package under
  `data/demo/sample_package/`.
- Four reviewed product screenshots under `docs/assets/`.
- Fictional candidate, experience-bank, search-preference, and cover-letter
  examples under `data/`.

## Verification Baseline

Run before publication and after meaningful changes:

```bash
.venv/bin/python -m py_compile main.py scripts/privacy_audit.py src/*.py
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m pip check
python3 scripts/privacy_audit.py
git diff --check
git status --short
```

Latest clean-repository verification (Python 3.12.13 on Linux):

- Direct dependencies installed successfully from `requirements.txt`.
- Python compilation passed for `main.py`, `scripts/privacy_audit.py`, and
  `src/*.py`.
- All 17 unit/integration/runtime tests passed.
- Dependency consistency check passed.
- Privacy audit passed across 43 public-release text candidates while the local
  private-term list was active.
- Streamlit started successfully and its health endpoint returned success.
- Git whitespace/status checks passed; ignored local data remained untracked.

## Known Limitations

- Live job fetching requires user-provided Adzuna and/or Jooble credentials.
- Provider descriptions may be incomplete and require review on the source page.
- OCR requires a local Tesseract installation in addition to Python packages.
- Fit scores and generated text are heuristic decision support and require
  human review.
- DOCX formatting is intentionally simple.
- `src/dashboard.py` and `src/manual_jobs.py` remain large; split them only with
  focused regression coverage.
- The app is local software, not a hosted multi-user service.

## Next Recommended Work

1. Replace the placeholder clone URL after the new repository URL is provided.
2. Push the single sanitized initial commit to the new GitHub repository.
3. Run a final local browser smoke test on macOS after cloning the new repo.
4. Consider module extraction only as a separate, test-backed task.