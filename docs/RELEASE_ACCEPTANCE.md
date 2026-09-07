# v1.0.0 release acceptance

This report separates automated evidence from manual acceptance. Dated manual evidence identifies the code under test; final CI must cover the release commit. Credentials, Personal
data, raw browser captures, and sensitive logs stay local.

## Candidate identity

- Candidate code commit: `73aea72`; documentation/media candidate: `096d56d`
- Final `main` commit and post-merge CI: recorded in the [immutable release notes](https://github.com/Xieyizhou/job-application-copilot/releases/tag/v1.0.0) after the merge gates pass.
- Pull request: [#4](https://github.com/Xieyizhou/job-application-copilot/pull/4)
- Candidate CI: [run 34095864293](https://github.com/Xieyizhou/job-application-copilot/actions/runs/34095864293), all seven jobs passed on `096d56d`
- Release tag: `v1.0.0`; the release notes are the final commit/CI ledger.

## Automated gates

| Gate | Environment | Status | Evidence |
| --- | --- | --- | --- |
| Core suite and coverage >= 68% | Linux, Python 3.11/3.12 | Passed on `096d56d` | CI passed; local: 329 passed + 198 subtests; CI 75.8%; clean macOS 76.3% (3.11), 76.1% (3.12) coverage |
| ML suite and coverage >= 80% | Linux, Python 3.11/3.12 | Passed on `096d56d` | CI: 319 passed, 1 documented private-data skip, 80.7% |
| Ruff, mypy, compileall, pip check | Linux, Python 3.11 | Passed on `096d56d` | CI run 34095864293 and local validation |
| Scoring benchmark and privacy audit | Linux, Python 3.11 | Passed on `096d56d` | Local benchmark 48/48; privacy audit passed |
| Public v21 contract | Linux, Python 3.11/3.12 | Passed on `096d56d` | CI run 34095864293; product integration intentionally false |
| Base install and supported launcher | macOS, Python 3.11/3.12 | Passed on `096d56d` | Both macOS matrix jobs passed in run 34095864293 |

## Manual product acceptance

| Area | Required evidence | Status |
| --- | --- | --- |
| Clean base install, first/repeat start, occupied port, stop/restart | macOS 26.6.2 arm64, Python 3.11.9 / 3.12.13 | Passed, 2026-09-07; fresh source archives, virtual environments and HOME; real HTTP/SIGINT lifecycle |
| Demo isolation and read-only behavior | Fixed fictional sample; no Personal writes | Passed; disabled imports/search writes, workspace switch, unchanged resume |
| Sanitized Personal upgrade | Historical `62f181a` fictional workspace copied to candidate | Passed; 10 file hashes unchanged, score/Tracker/DOCX load correctly |
| Import, deduplication, JD quality, scoring, evidence | Fictional text/Markdown and live public ATS | Passed; 8 live Lever records, repeat search 0 new / 8 already seen; partial JD remains unscored |
| Cover letter, confirmation, DOCX/evidence export, tracker, archive | Fictional inputs, UI downloads and local artifact inspection | Passed; manual-claim checkbox required, DOCX render checked, ZIP 5 entries; Tracker and archive lifecycle verified |
| Empty/partial/duplicate/parse failure/timeout/fallback cases | UI errors, local parser tests and live fetch results | Passed; empty title/query, duplicate and unsafe URL rejected; invalid PDF/image/EXE safe failure; unavailable postings not counted as success |
| Desktop layout and critical reruns | 1280 × 720 screenshots with fictional data | Passed, 2026-09-07; narrow screens deferred by release owner |

## Browser and public ATS acceptance

| Path | Required environment | Status | Last verified |
| --- | --- | --- | --- |
| Chrome unpacked companion | Clean macOS browser profile | Passed: load, live import, invalid token, offline recovery | 2026-09-07; user-operated independent Chrome, saved result verified locally |
| Edge unpacked companion | Outside stable support scope | Not verified; experimental compatibility | 2026-09-07; scope narrowed by release owner |
| Greenhouse public posting | Live representative posting; no application submitted | Passed; extractor unchanged since live verification | 2026-09-07, Capco Data analyst, public API extractor, 611 words |
| Lever public posting | Live representative posting; no application submitted | Passed; extractor unchanged since live verification | 2026-09-07, Portcast Data Analyst, public API extractor, 1,049 words |
| Ordinary-page fallback | Live representative posting; no application submitted | Passed; extractor unchanged since live verification | 2026-09-07, Unilever Data Analyst; JobPosting JSON-LD, 526 words; responsibilities, qualifications, and Power BI retained |
| Invalid token and offline recovery | Chrome on macOS | Passed | 2026-09-07; invalid-token rejection, offline error, same-token restart recovery |

Credential-backed Adzuna, Jooble, and JSearch checks are experimental. Record only whether
a minimal live search was performed and its date; never copy keys or sensitive logs here.

## Browser scope decision

On 2026-09-07 the release owner approved omitting Edge from formal acceptance because
it is not installed in the available environment. Stable browser-import support therefore
covers Chrome on macOS only. Edge code is retained as experimental, unverified compatibility;
Safari extension import is outside scope. This is a support-scope change, not an Edge pass.

## Execution notes — 2026-09-07

- Public fetch samples: [Greenhouse](https://job-boards.greenhouse.io/capco/jobs/8143432),
  [Lever](https://jobs.lever.co/portcast/e66df487-0622-480f-8506-532b3db5db28), and
  [ordinary HTML](https://careers.unilever.com/en/job/athens/data-analyst/34155/99818333104).
  These checks called the real `fetch_job_page` entry point; they did not submit applications.
  Native Chrome import of the saved Greenhouse posting was separately verified.
- The sampled AssistRx Workable URL redirected to a company listing with `not_found=true`.
  Two Jobs By Workable samples returned HTTP 410; another timed out. These unavailable
  postings were not counted as successful extraction. Linked-Markdown recovery has a
  reproduced failing regression and a passing fix, but no successful live sample yet.
- A second launcher previously deleted an active companion's connection metadata before
  failing to bind its occupied port. Regression commit `19564d9` reproduces the failure;
  fix `e96de46` preserves the active connection. Launcher/companion tests: 13 passed.
- Fresh base dependencies installed successfully on both Python versions using the Tsinghua
  PyPI mirror after official-index timeouts. The same pinned requirements were used with
  no developer credentials, models, caches, or historical Personal data. Both supported
  launchers passed first start, occupied-port rejection without corrupting the active
  connection, SIGINT shutdown of both services, and restart with the original token.
  Both Python versions passed development/ML installation and all clean-environment
  checks: core 329 + 198 subtests (3.11: 76.3%; 3.12: 76.1%), ML 319 with one documented
  skip (80.7%), benchmark 48/48, semantic fixture validation and the v21 public contract.
  The semantic fixture result remains 23/24 (95.8%), F1 94.7%, with known case
  `real-sem-005` unmatched; this is a measured model limitation, not hiring-outcome accuracy.
  Python 3.12 mypy uses `--python-version 3.12`: its installed NumPy stubs use Python 3.12
  syntax. The default Python 3.11 compatibility target remains checked by CI.
- Rendered Personal setup, resume upload, manual text and Markdown import, full/partial JD
  quality, scoring and evidence passed. Public Company ATS search returned eight Lever
  records; repeating it returned zero new records and eight duplicates. Empty queries and
  missing provider credentials produced clear errors. Demo remained read-only.
- Historical upgrade used a newly generated fictional workspace from `62f181a`, copied
  to the candidate. The old and upgraded copies retained all ten file hashes, Tracker
  state, score and readable DOCX. No original Personal data was used or changed.
- Cover-letter generation, manual factual-claim confirmation, DOCX and bundle downloads
  passed in the desktop UI. The DOCX rendered to one readable page. Local artifact checks
  verified five ZIP entries, unchanged resume, Tracker applied/interview/archived states,
  job archive content preservation and cache invalidation after JD/workspace changes.
- A saved full-JD record intentionally uses structured evidence scoring (87 for the fixed
  Northstar sample), while the legacy raw Demo fixture uses keyword fallback (92). These
  existing provenance rules explain the difference. A separate bug made new exported
  reports retain the fallback score; regression `f01e47e` and fix `73aea72` align new report,
  Tracker and dashboard scores. The rendered generation summary and report now both show
  87, and the new Tracker stored 87 before its UI transition from Ready to Applied. Historical exports are not rewritten.
- Base-only Personal startup exposed an eager optional sklearn import, fixed in `65ac9fd`.
  Empty Visa Note metadata no longer creates a false work-authorization warning (`b392e41`).
  Demo report evidence now uses actual fictional resume snippets (`8d92001`). Each change
  has an independently committed reproducing test.
- Desktop screenshots and the demonstration GIF were refreshed from the fictional workspace.
  Narrow-screen acceptance was explicitly deferred by the release owner on 2026-09-07.
- The browser tool's security policy blocks extension-management pages. The user manually
  loaded the Chrome extension in an independent profile, then imported the live Capco
  posting successfully. The saved record was independently verified as `browser_companion`
  / `complete` (660 words including metadata); the UI showed Scoring-ready and Role Fit
  4/100 against the fictional resume. A deliberately invalid token returned
  `Invalid companion token.`. Shutdown closed both the dashboard and companion ports
  and removed connection metadata. The browser displayed the expected offline error, then succeeded with the unchanged
  token after restart on runtime source `5c630c2`; both health endpoints returned 200.
  An unsaved Portcast posting was rejected with “Save this job in JobCopilot first”;
  independent inspection confirmed Capco remained complete and contained no Portcast text.
- Candidate privacy audit passed (337 files). Historical scans checked 948 blobs,
  including 909 UTF-8 text blobs. One phone-pattern match was a documented Chrome version
  in an earlier README commit; the current README uses a version prefix. No credential or
  local-identifier finding remained unexplained. Historical binary review covered seven
  DOCX files (including XML metadata), first-frame/static contact sheets, and OCR of 272
  unique GIF samples/static images. GIF sampling used approximately one-second intervals,
  not every animation frame. The only DOCX contact matches were explicit fictional
  example identities. All seven new screenshot/GIF assets were visually reviewed as well.
  Pattern scans and sampled review are bounded checks, not a guarantee of exhaustive detection.
- GitHub private vulnerability reporting is enabled; the documented security channel is live.
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

## Independent rollback points

Release fixes retain separate regression and implementation commits. Revert a selected
implementation on a new branch, keep its regression test, and validate the resulting
behavior before publishing a patch. Some reverts intentionally restore the reproduced bug;
they are diagnostic rollback points, not automatically safe release candidates.

| Regression | Implementation | Scope |
| --- | --- | --- |
| `d659034` | `00750f4` | Advertised Markdown JD recovery |
| `19564d9` | `e96de46` | Preserve active companion on occupied port |
| `9b45ede` | `45db0d6` | Distinguish prose from a JD heading |
| `6071de9` | `abb9659` | Manually entered title submission |
| `1bdba93` | `65ac9fd` | Base Personal startup without optional ML |
| `3ef9a71` | `8d92001` | Demo evidence freshness |
| `71f3978` | `b392e41` | Empty Visa Note warning |
| `f01e47e` | `73aea72` | Dashboard/report/Tracker score agreement |

The earlier internal-refactor rollback sequence remains in
[the simplification review](CODE_SIMPLIFICATION_REVIEW.md#回滚与复现).

## Release blockers

- Any failed supported macOS, Python, Chrome, Greenhouse, or Lever acceptance row.
- Unexplained structured-output or exported-document differences.
- Data-format incompatibility, Personal/Demo isolation failure, or resume mutation.
- Credential, Personal data, private label, unpublished weight, or sensitive report found
  in the candidate tree, history under review, screenshots, or release attachments.
- Missing required review or failing CI on the final `main` commit.

## Sign-off

Local acceptance and candidate CI have passed. The final documentation commit must pass
CI before merge; the exact merged commit must then pass main CI, fresh installation and
Demo startup before tagging. Those post-merge results and commit identities are recorded
in the release notes, avoiding a self-referential source commit. Re-read remote tags and
stop if `v1.0.0` exists. After tagging, verify the source archive and documented startup
from the immutable tag.
