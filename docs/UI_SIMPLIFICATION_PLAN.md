# UI Simplification Plan

## Goal

Make the toolkit answer three questions in order:

1. What should I do next?
2. Is this job worth more time?
3. What evidence supports that decision?

The interface should keep audit detail available without placing every diagnostic above
the fold.

## Current-state audit

The audit used the running Personal workspace at desktop width. No personal resume content
was copied into this document.

### Dashboard

- Five counters, three next-action cards, and three opportunity summaries compete for the
  same first-screen attention.
- “Need full JD” is repeated as a counter, an action card, and inside every opportunity.
- The strongest element is the next-action group; it should become the page's primary view.

### Find Jobs

- The core search form is understandable and reasonably compact.
- Source limitations and API setup are shown inline even when the user only wants to search.
- Provider configuration belongs in a short source-status row with details in Settings.

### Add Target Job

- Three instruction banners, three tabs, two cleanup actions, upload controls, and a long
  metadata form appear before the main save action.
- Destructive or maintenance actions are visually equal to the primary task.
- Most optional metadata can be inferred or placed behind “More details.”

### Review Jobs

- This is the densest screen: six inbox views, three controls, filters, four summary metrics,
  repeated job-card actions, five detail metrics, four detail tabs, an alert, and multiple
  analysis expanders.
- Role Fit, observed coverage, evidence count, confidence, recommendation, JD completeness,
  and experimental ML are presented as peers even though they have different decision roles.
- Cards repeat `Select`, `Fit`, and `Cover Letter`; selecting the card plus one contextual
  primary action is enough.
- The same low-JD warning is repeated in cards, metrics, the detail alert, and explanatory text.

### Cover Letter

- The page shows source selection, four counters, a five-row materials table, a warning,
  four download buttons, three previews, status actions, and local file details.
- Users primarily need the draft preview, evidence/gap warning, and one download action.
- Internal notes, match reports, bundle composition, and local paths are audit details.

### Tracker

- The pipeline is understandable, but five counters precede the actual application list.
- The selected-record detail is useful; archive/delete controls are correctly secondary.

### Settings

- The structure is sound, but scoring terminology should be a reference, not required reading.
- Experimental model status and developer notes should remain collapsed by default.

## Target information hierarchy

### Level 1 — always visible

- Recommendation or current stage
- Role Fit only when confidence permits it
- Confidence and JD quality as compact status labels
- One next best action
- One primary button

### Level 2 — visible after one click

- Top supporting evidence
- Main gap or hard constraint
- Up to three requirement-to-resume matches
- Cover-letter preview and employer-facing download

### Level 3 — advanced / audit

- Observed coverage and parser counts
- Full requirement map and source excerpts
- Experimental ML diagnostics
- Internal notes, raw reports, local file paths, and maintenance actions

## Proposed screen changes

### 1. Dashboard

- Keep three counters: active opportunities, ready to apply, and follow-ups due.
- Make “Next actions” the first content block and make each card clickable.
- Show at most three opportunities with title, recommendation, confidence, and one-line next
  action. Remove long scoring explanations from this page.

### 2. Find Jobs

- Keep query, region, sources, and one `Find Jobs` button.
- Replace the long API paragraph with a compact source-health label such as
  `2 discovery sources · full-JD source not configured`.
- Move per-source limits into an expanded `Search options` section.

### 3. Add Target Job

- Use one linear form: paste/upload JD first, then verify inferred company/title/location.
- Collapse optional salary, visa note, status, and notes under `More details`.
- Move cleanup actions to Settings or a bottom `Maintenance` expander.
- Show JD quality immediately after extraction and disable trustworthy-fit language until the
  posting is complete.

### 4. Review Jobs

- Reduce inbox views to `Recommended`, `Needs attention`, `Ready`, and `All`; move ignored and
  untracked states into filters.
- Keep search and sort visible; keep all other filters collapsed.
- Remove the four aggregate metrics from the default view unless one is actionable.
- Make the whole job card select the role. Card content: company/title, recommendation or Role
  Fit, confidence/JD-quality badges, and one next action.
- Replace the five-metric detail strip with four fields: recommendation, Role Fit, confidence,
  and JD quality. Never show experimental ML beside the canonical score.
- Default detail content should be `Why this result`: strongest evidence, main gap, hard
  constraint, then one primary action.
- Show only the top three semantic evidence matches. Keep coverage, parser terms, source
  snippets, and experimental diagnostics in `Advanced analysis`.

### 5. Cover Letter

- Put the editable/previewable letter first.
- Show one compact readiness statement: factual evidence used, unresolved gaps, and employer
  details to verify.
- Primary action: `Download Cover Letter DOCX`. Secondary menu: bundle, match report, internal
  notes.
- Collapse the materials table, stored-score snapshot, tracker notes, and file paths.

### 6. Tracker

- Keep three counters: active, follow-ups due, interviews.
- Place the application table before selected-record details.
- Keep stage update as the primary detail action and move document metadata into a collapsed
  section.

### 7. Settings

- Separate `Workspace`, `Job sources`, `Scoring`, `Privacy`, and `Advanced`.
- Show a one-line health state per section; reveal definitions and technical details on demand.

## Delivery sequence

Implementation status (2026-07-22): Phases 1–3 are complete. Review Jobs now uses four
inbox views, one action per result card, four decision fields, and a decision-first overview.
Coverage/parser detail, source excerpts, and the experimental local model are available under
Advanced analysis. Cover Letter now leads with an editable draft and DOCX download; supporting
artifacts are collapsed. Add Target Job now captures the JD first, reports quality immediately,
then asks users to verify inferred fields. Optional metadata and maintenance actions are
collapsed. Component and function-size limits are enforced by tests.
Dashboard now leads with at most three direct next actions and keeps only Active, Ready to
apply, and Follow-ups due counters. Tracker puts the application table and stage update ahead
of stored scores and document metadata. Settings reports health for Workspace, Job sources,
Scoring, Privacy, and Advanced areas before revealing detailed reference text.

### Phase 1 — Review Jobs and shared components

- Introduce reusable status badges and one next-action callout.
- Simplify job cards and the selected-job summary.
- Move ML, parser coverage, source excerpts, and file paths into advanced sections.
- Add UI tests for the new information hierarchy.

### Phase 2 — Cover Letter and Add Target Job

- Make preview/download and JD capture the dominant workflows.
- Relegate maintenance and audit artifacts to secondary menus.

### Phase 3 — Dashboard, Tracker, and Settings

- Reduce counters, remove repeated explanations, and align terminology across pages.

### Phase 4 — usability validation

- Test the flows `add full JD → judge fit`, `review fit → generate letter`, and
  `ready → applied → follow-up` with both complete and snippet JDs.
- Record desktop and narrow-width screenshots and run keyboard/accessibility checks.

## Acceptance criteria

- No screen shows more than four top-level metrics.
- Every screen has one visually dominant primary action.
- Review Jobs answers recommendation, confidence, JD quality, why, and next action without
  scrolling the detail panel.
- Default job cards have no more than two status lines and one action.
- Experimental ML values are absent when inputs are incomplete or outputs are out of
  distribution.
- Internal paths and diagnostic artifacts are hidden by default.
- The default Review and Cover Letter views contain at least 35% fewer visible information
  blocks than the current desktop layout, without removing audit access.
