"""Compact health and reference sections for Settings."""

from __future__ import annotations

import os
from typing import Any

from fetch_jobs import jsearch_configured


def job_source_health() -> dict[str, bool]:
    """Return provider configuration without exposing any credential values."""
    return {
        "JSearch · full JD": jsearch_configured(),
        "Adzuna · discovery": bool(os.getenv("ADZUNA_APP_ID", "").strip() and os.getenv("ADZUNA_APP_KEY", "").strip()),
        "Jooble · discovery": bool(os.getenv("JOOBLE_API_KEY", "").strip()),
    }


def render_settings_sections(
    ui: Any,
    *,
    workspace: Any,
    jobs_count: int,
    tracker_count: int,
    demo_mode: bool,
) -> None:
    """Render health first and explanations only inside their relevant tab."""
    sources = job_source_health()
    ui.table(
        [
            {"Section": "Workspace", "Status": "Ready" if workspace.ready else "Needs setup"},
            {"Section": "Job sources", "Status": f"{sum(sources.values())}/3 configured"},
            {"Section": "Scoring", "Status": "Available" if workspace.ready else "Waiting for resume"},
            {"Section": "Privacy", "Status": "Local workspace"},
        ]
    )

    workspace_tab, sources_tab, scoring_tab, privacy_tab = ui.tabs(
        ["Workspace", "Job sources", "Scoring", "Privacy"]
    )
    with workspace_tab:
        _render_workspace(ui, workspace, jobs_count, tracker_count, demo_mode)
    with sources_tab:
        _render_sources(ui, sources)
    with scoring_tab:
        _render_scoring(ui)
    with privacy_tab:
        _render_privacy(ui)


def _render_workspace(ui: Any, workspace: Any, jobs_count: int, tracker_count: int, demo_mode: bool) -> None:
    ui.markdown(f"**{workspace.mode.title()} workspace**")
    ui.write("Candidate source: " + ("Ready" if workspace.resume_source_path else "Missing"))
    ui.write(f"Saved jobs: {jobs_count}")
    ui.write(f"Tracker records: {tracker_count if not demo_mode else 'Disabled in Demo'}")
    ui.caption("Use Manage workspace files in the sidebar to update your resume or optional template.")


def _render_sources(ui: Any, sources: dict[str, bool]) -> None:
    for label, configured in sources.items():
        ui.write(f"{'Ready' if configured else 'Not configured'} · {label}")
    ui.caption("JSearch supports complete postings. Adzuna and Jooble may return discovery snippets.")
    if not sources["JSearch · full JD"]:
        ui.info("Configure JSEARCH_API_KEY before relying on automatic full-JD retrieval.")


def _render_scoring(ui: Any) -> None:
    ui.markdown("**Interpret the decision signals separately**")
    ui.write("Role Fit ranks resume evidence against recognized requirements.")
    ui.write("Confidence says whether the available JD and resume evidence can support that score.")
    ui.write("Eligibility checks hard constraints; JD Quality evaluates the posting text itself.")
    with ui.expander("Scoring glossary", expanded=False):
        ui.write("Observed coverage considers only requirements recognized by the parser.")
        ui.write("A high provisional score is not a reliable recommendation when confidence is low.")
        ui.write("Role Fit is not an interview probability and does not override eligibility.")


def _render_privacy(ui: Any) -> None:
    ui.markdown("**Local by default**")
    ui.write("Saved jobs, tracker records, candidate files, and cover-letter bundles remain on this machine.")
    ui.write("The toolkit does not submit applications or complete external forms.")
    ui.write("Review every resume claim, cover letter, and application answer before use.")
