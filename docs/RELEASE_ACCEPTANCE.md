# v1.0.0 release acceptance

This report separates automated evidence from manual acceptance. A row may be marked
Passed only with a dated result from the final candidate commit. Credentials, Personal
data, raw browser captures, and sensitive logs stay local.

## Candidate identity

- Candidate commit: pending
- Final `main` commit: pending
- Pull request and required CI: pending
- Release tag and URL: pending

## Automated gates

| Gate | Environment | Status | Evidence |
| --- | --- | --- | --- |
| Core suite and coverage >= 68% | Linux, Python 3.11/3.12 | Pending final CI | CI run URL pending |
| ML suite and coverage >= 80% | Linux, Python 3.11/3.12 | Pending final CI | CI run URL pending |
| Ruff, mypy, compileall, pip check | Linux, Python 3.11 | Pending final CI | CI run URL pending |
| Scoring benchmark and privacy audit | Linux, Python 3.11 | Pending final CI | CI run URL pending |
| Public v21 contract | Linux, Python 3.11/3.12 | Pending final CI | CI run URL pending |
| Base install and supported launcher | macOS, Python 3.11/3.12 | Pending final CI | CI run URL pending |

## Manual product acceptance

| Area | Required evidence | Status |
| --- | --- | --- |
| Clean install, first/repeat start, occupied port, stop/restart | Dated macOS notes for Python 3.11 and 3.12 | Pending |
| Demo isolation and read-only behavior | Fixed fictional sample; no Personal writes | Pending |
| Sanitized Personal upgrade | Backup used; original data unchanged | Pending |
| Import, deduplication, JD quality, scoring, evidence | Fixed fictional inputs and explained diffs | Pending |
| Cover letter, confirmation, DOCX/evidence export, tracker, archive | Fixed fictional inputs and artifact checklist | Pending |
| Empty/partial/duplicate/parse failure/timeout/fallback cases | Expected error and recovery recorded | Pending |
| Desktop and narrow layout; critical reruns | Screenshot checklist with fictional data | Pending |

## Browser and public ATS acceptance

| Path | Required environment | Status | Last verified |
| --- | --- | --- | --- |
| Chrome unpacked companion | Clean macOS browser profile | Pending | — |
| Edge unpacked companion | Clean macOS browser profile | Pending | — |
| Greenhouse public posting | Live representative posting; no application submitted | Pending | — |
| Lever public posting | Live representative posting; no application submitted | Pending | — |
| Ordinary-page fallback | Live representative posting; no application submitted | Pending | — |
| Invalid token and offline recovery | Chrome and Edge | Pending | — |

Credential-backed Adzuna, Jooble, and JSearch checks are experimental. Record only whether
a minimal live search was performed and its date; never copy keys or sensitive logs here.

## Release blockers

- Any failed supported macOS, Python, Chrome, Edge, Greenhouse, or Lever acceptance row.
- Unexplained structured-output or exported-document differences.
- Data-format incompatibility, Personal/Demo isolation failure, or resume mutation.
- Credential, Personal data, private label, unpublished weight, or sensitive report found
  in the candidate tree, history under review, screenshots, or release attachments.
- Missing required review or failing CI on the final `main` commit.

## Sign-off

Do not replace Pending with Passed until the evidence is tied to the final candidate. Before
tagging, fetch remote tags again and stop if `v1.0.0` exists. After tagging, verify the source
archive and documented startup from the immutable tag.
