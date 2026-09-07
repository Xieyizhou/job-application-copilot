"""Compact health and reference sections for Settings."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dashboard_ats_settings import render_ats_boards
from fetch_jobs import jsearch_configured


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPANION_CONNECTION_PATH = PROJECT_ROOT / "data/local_workspace/browser_companion/connection.json"
COMPANION_EXTENSION_PATH = PROJECT_ROOT / "browser_companion"


def browser_companion_connection() -> dict[str, str]:
    """Return public loopback connection details without reading browser state."""
    try:
        payload = json.loads(COMPANION_CONNECTION_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    endpoint = str(payload.get("endpoint", "") or "")
    token = str(payload.get("token", "") or "")
    if endpoint != "http://127.0.0.1:8765" or len(token) < 24:
        return {}
    return {"endpoint": endpoint, "token": token}


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
        _render_sources(ui, sources, workspace=workspace, demo_mode=demo_mode)
    with scoring_tab:
        _render_scoring(ui)
    with privacy_tab:
        _render_privacy(ui)


def _render_workspace(ui: Any, workspace: Any, jobs_count: int, tracker_count: int, demo_mode: bool) -> None:
    ui.markdown(f"**{workspace.mode.title()} workspace**")
    ui.write("Candidate source: " + ("Ready" if workspace.resume_source_path else "Missing"))
    ui.write(f"Saved jobs: {jobs_count}")
    ui.write(f"Tracker records: {tracker_count if not demo_mode else 'Disabled in Demo'}")
    ui.caption("Use Resume in the sidebar to replace your resume or optional template.")


def _render_sources(ui: Any, sources: dict[str, bool], *, workspace: Any, demo_mode: bool) -> None:
    for label, configured in sources.items():
        ui.write(f"{'Ready' if configured else 'Not configured'} · {label}")
    ui.caption("JSearch supports complete postings. Adzuna and Jooble may return discovery snippets.")
    ui.caption(
        "Experimental services: JSearch, Adzuna, and Jooble have limited live validation. "
        "Provider quotas, availability, and response formats may vary."
    )
    if not sources["JSearch · full JD"]:
        ui.info("Configure JSEARCH_API_KEY before relying on automatic full-JD retrieval.")
    render_ats_boards(ui, workspace=workspace, demo_mode=demo_mode)
    ui.divider()
    ui.markdown("**Browser companion · current-page import**")
    connection = browser_companion_connection()
    if not connection:
        ui.warning("Start JobCopilot with `python run_dashboard.py` to enable local browser import.")
        return
    ui.write("Ready · listens only on 127.0.0.1 and writes only to the local workspace.")
    ui.caption(
        "Install once in Chrome (Edge compatibility is experimental and unverified). "
        "On a saved job's original page, choose "
        "Import and verify this posting, then Open JobCopilot."
    )
    ui.code(str(COMPANION_EXTENSION_PATH), language=None)
    ui.markdown(
        "1. Open `chrome://extensions` (or `edge://extensions`).\n"
        "2. Enable **Developer mode** and choose **Load unpacked**.\n"
        "3. Select the folder above, open the extension once, and paste this local token:"
    )
    ui.code(connection["token"], language=None)
    ui.caption("The token remains local, is stored in a Git-ignored directory with file mode 600, and may be replaced by deleting the local browser_companion folder.")


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
