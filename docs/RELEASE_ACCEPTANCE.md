# v1.0.0 release acceptance

This report separates automated evidence from manual acceptance. A row may be marked
Passed only with a dated result from the final candidate commit. Credentials, Personal
data, raw browser captures, and sensitive logs stay local.

## Candidate identity

- Candidate code commit: `45db0d6`
- Final `main` commit: pending
- Pull request: [#4](https://github.com/Xieyizhou/job-application-copilot/pull/4)
- Candidate CI: [run 34088123757](https://github.com/Xieyizhou/job-application-copilot/actions/runs/34088123757)
- Release tag and URL: pending

## Automated gates

| Gate | Environment | Status | Evidence |
| --- | --- | --- | --- |
| Core suite and coverage >= 68% | Linux, Python 3.11/3.12 | Passed on candidate | CI passed; local: 324 passed + 198 subtests, 76.1% |
| ML suite and coverage >= 80% | Linux, Python 3.11/3.12 | Passed on candidate | CI: 319 passed, 1 documented private-data skip, 80.7% |
| Ruff, mypy, compileall, pip check | Linux, Python 3.11 | Passed on candidate | CI run 34088123757 and local validation |
| Scoring benchmark and privacy audit | Linux, Python 3.11 | Passed on candidate | Local benchmark 48/48; privacy audit passed |
| Public v21 contract | Linux, Python 3.11/3.12 | Passed on candidate | CI run 34088123757; product integration intentionally false |
| Base install and supported launcher | macOS, Python 3.11/3.12 | Passed on candidate CI | Both macOS matrix jobs passed in run 34088123757 |

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
| Chrome unpacked companion | Clean macOS browser profile | Partial: load, live import, invalid token passed | 2026-09-07; offline/restart recovery pending |
| Edge unpacked companion | Clean macOS browser profile | Blocked: browser not installed | 2026-09-07; confirmed by user |
| Greenhouse public posting | Live representative posting; no application submitted | Passed on candidate | 2026-09-07, Capco Data analyst, public API extractor, 611 words |
| Lever public posting | Live representative posting; no application submitted | Passed on candidate | 2026-09-07, Portcast Data Analyst, public API extractor, 1,049 words |
| Ordinary-page fallback | Live representative posting; no application submitted | Passed on candidate | 2026-09-07, Unilever Data Analyst; JobPosting JSON-LD, 526 words; responsibilities, qualifications, and Power BI retained |
| Invalid token and offline recovery | Chrome and Edge | Partial: Chrome invalid token rejected | 2026-09-07; remaining recovery and Edge checks pending |

Credential-backed Adzuna, Jooble, and JSearch checks are experimental. Record only whether
a minimal live search was performed and its date; never copy keys or sensitive logs here.

## Execution notes — 2026-09-07

- Public fetch samples: [Greenhouse](https://job-boards.greenhouse.io/capco/jobs/8143432),
  [Lever](https://jobs.lever.co/portcast/e66df487-0622-480f-8506-532b3db5db28), and
  [ordinary HTML](https://careers.unilever.com/en/job/athens/data-analyst/34155/99818333104).
  These checks called the real `fetch_job_page` entry point; they did not submit applications.
  Browser import of these pages remains separate and pending.
- The sampled AssistRx Workable URL redirected to a company listing with `not_found=true`.
  Two Jobs By Workable samples returned HTTP 410; another timed out. These unavailable
  postings were not counted as successful extraction. Linked-Markdown recovery has a
  reproduced failing regression and a passing fix, but no successful live sample yet.
- A second launcher previously deleted an active companion's connection metadata before
  failing to bind its occupied port. Regression commit `19564d9` reproduces the failure;
  fix `e96de46` preserves the active connection. Launcher/companion tests: 13 passed.
- Source archives with fresh Python 3.11 and 3.12 virtual environments and isolated home
  directories are being validated without developer credentials, models, or caches.
  Python 3.11 pip initially reported no pandas distribution; an independent resolver
  subsequently found a compatible wheel. Python 3.12 pip timed out downloading from
  `files.pythonhosted.org`. Installation retries remain pending; neither local clean
  environment has passed the install gate. Do not treat the import-only CI smoke as proof of
  a rendered local application or the complete first/repeat-start lifecycle.
- On the isolated source copy, using preinstalled dependencies (not the new-install gate),
  fictional Personal initialization, manifest-copy reload, cover-letter/DOCX generation,
  `ready -> applied` Tracker update, byte-for-byte original resume preservation, and Demo
  write rejection passed. DOCX opened successfully with expected fictional identity and
  company text and no export warnings. DOCX rendered to one page and passed visual review for clipping, overlap, and text
  layout; UI confirmation remains pending; same-format reload is not proof of a historical-version upgrade.
- The browser tool's security policy blocks extension-management pages. The user manually
  loaded the Chrome extension in an independent profile, then imported the live Capco
  posting successfully. The saved record was independently verified as `browser_companion`
  / `complete` (660 words including metadata); the UI showed Scoring-ready and Role Fit
  4/100 against the fictional resume. A deliberately invalid token returned
  `Invalid companion token.`. Shutdown closed both the dashboard and companion ports
  and removed connection metadata. Browser offline/restart verification is pending.
- Candidate privacy audit passed (332 files). Historical pattern scans checked 905 blobs
  for common credential signatures/sensitive paths and 875 UTF-8 blobs with the repository's
  generic privacy rules, with no findings. These scans exclude binary visual inspection and
  are not a guarantee that all sensitive content has been ruled out. Final screenshots,
  attachments, and any newly introduced history still need review.
- ML CI skips only `test_local_lifecycle_integrity_includes_v9_assignments` because the
  private `source_partitions/successor_v4/partition_manifest.json` is intentionally absent.
  The public lifecycle contract and all other 319 ML tests pass; no core tests are skipped.
- Minimal live searches on Adzuna, Jooble, and JSearch each returned one result for
  `data analyst` in New York on 2026-09-07. Existing local credentials were used only for
  this separate live-service check; no keys, raw responses, or query URLs were recorded.
  Settings, README, and the usage guide retain experimental status because broader
  provider behavior and end-to-end search acceptance remain unverified. Public ML CI evidence is authoritative for the
  clean public contract; local development model availability is not reproducibility evidence.
- Repository rulesets were empty and `main` had no branch protection at inspection time.
  Merge commits are allowed. Re-read rules, reviews, `main`, and tags before final release.

- Real HTTP startup and SIGINT shutdown from an isolated tracked archive now pass in
  macOS CI on Python 3.11 and 3.12 via `scripts/smoke_dashboard.py`. This runs after base
  installation and before development dependencies. Companion lifecycle and rendered
  product flows are separate manual gates.
- Regression `9b45ede` reproduced a prose mention of “job description” being interpreted
  as a responsibility heading. Fix `45db0d6` removes that false Demo requirement while
  preserving flattened explicit headings. All three fixed Demo scores remain unchanged
  (92, 92, 69); the Data Analyst requirement count changes from 9 to 8 as expected.

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
