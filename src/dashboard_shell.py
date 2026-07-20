"""Streamlit application shell shared by all dashboard pages."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Mapping
from typing import Any


PAGE_NAMES = (
    "Dashboard",
    "Find Jobs",
    "Add Target Job",
    "Review Jobs",
    "Cover Letter",
    "Tracker",
    "Settings",
)

PAGE_DESCRIPTIONS = {
    "Dashboard": "Start with the next-action cards.",
    "Find Jobs": "Search broadly, then verify full JDs.",
    "Add Target Job": "Capture one complete source of truth.",
    "Review Jobs": "Compare evidence, gaps, and risks.",
    "Cover Letter": "Review the draft, evidence trace, and gaps.",
    "Tracker": "Keep stages and follow-ups current.",
    "Settings": "Review workspace and scoring boundaries.",
}

GLOBAL_STYLES = """
<style>
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
}
.app-title-safe-area {
    padding-top: 2.25rem;
    padding-bottom: 0.75rem;
    overflow: visible !important;
}
.app-title-text {
    font-size: 2.35rem;
    font-weight: 750;
    line-height: 1.25;
    letter-spacing: -0.02em;
    margin: 0;
    padding: 0;
    overflow: visible !important;
}
.page-header {
    margin-top: 0.35rem;
    margin-bottom: 0.65rem;
}
.page-title {
    font-size: 1.55rem;
    font-weight: 700;
    line-height: 1.25;
    margin: 0 0 0.35rem 0;
}
.page-subtitle {
    color: rgba(49, 51, 63, 0.72);
    font-size: 0.92rem;
    line-height: 1.35;
    margin: 0;
}
h2 {
    margin-top: 0.55rem !important;
    margin-bottom: 0.4rem !important;
}
h3 {
    margin-top: 0.45rem !important;
    margin-bottom: 0.35rem !important;
}
div[data-testid="stRadio"] {
    margin-top: 0rem !important;
    margin-bottom: 0.2rem !important;
}
</style>
"""


def render_global_styles(st: Any) -> None:
    """Apply compact spacing shared by every page."""
    st.markdown(GLOBAL_STYLES, unsafe_allow_html=True)


def switch_workspace_mode(session_state: Any, mode: str) -> None:
    """Switch isolated workspace modes and clear cross-workspace selections."""
    if mode not in {"Demo", "Personal"}:
        raise ValueError(f"Unsupported workspace mode: {mode}")
    session_state["workspace_mode"] = mode
    session_state["workspace_setup_open"] = False
    for key in [
        "latest_generated_package_dir",
        "latest_generated_package_summary",
        "package_viewer_tracker_id",
        "selected_review_job_path",
        "selected_review_job_label",
        "review_workspace_mode",
    ]:
        session_state.pop(key, None)


def render_sidebar(
    st: Any,
    *,
    current_workspace: Callable[[], Any],
    list_job_description_files: Callable[..., list[Any]],
    count_generated_packages: Callable[[], int],
    load_tracker_rows: Callable[..., list[dict[str, Any]]],
    demo_mode_enabled: Callable[[], bool],
) -> None:
    """Render a Personal-first workflow with Demo as a separate experience."""
    active_mode = str(st.session_state.get("workspace_mode", "Personal"))
    if active_mode == "Demo":
        if st.sidebar.button("Back to Personal Workspace", type="primary", width="stretch"):
            switch_workspace_mode(st.session_state, "Personal")
            st.rerun()
    elif st.sidebar.button("Explore Read-only Demo", width="stretch"):
        switch_workspace_mode(st.session_state, "Demo")
        st.rerun()

    st.sidebar.title("Your Job Search Flow")
    workspace = current_workspace()
    if workspace.mode == "personal":
        st.sidebar.caption("Personal workspace · local and private")
        st.sidebar.caption("Configured" if workspace.ready else "Setup required")
        st.sidebar.write(f"Candidate source: {'Ready' if workspace.resume_source_path else 'Missing'}")
        st.sidebar.write(
            f"Experience bank: {'Provided' if workspace.experience_bank_path else 'Optional · resume-grounded'}"
        )
        st.sidebar.write(
            f"Cover-letter template: {'Provided' if workspace.cover_letter_template_path else 'Generic template'}"
        )
        if workspace.ready and st.sidebar.button("Replace candidate files"):
            st.session_state["workspace_setup_open"] = True
            st.rerun()
    else:
        st.sidebar.info("Demo mode · sanitized and read-only")

    if workspace.ready:
        try:
            sidebar_jobs = len(list_job_description_files())
            sidebar_packages = count_generated_packages()
            sidebar_cover_label = "cover letter" if sidebar_packages == 1 else "cover letters"
            sidebar_tracker = 0 if demo_mode_enabled() else len(load_tracker_rows(sort_by="created_at", descending=True))
            st.sidebar.caption(
                f"{sidebar_jobs} jobs · {sidebar_packages} {sidebar_cover_label} · {sidebar_tracker} tracker records"
            )
        except (OSError, sqlite3.Error):
            pass

    active_page = str(st.session_state.get("active_page", "Dashboard"))
    st.sidebar.markdown(f"**Current:** {active_page}")
    st.sidebar.caption(PAGE_DESCRIPTIONS.get(active_page, "Follow the workflow one decision at a time."))
    st.sidebar.caption("Local-first. Human-reviewed. No automatic submissions.")


def render_app_title(st: Any) -> None:
    """Render the global product title below the navigation safe area."""
    st.markdown(
        """
        <div class="app-title-safe-area">
          <div class="app-title-text">Job Application Toolkit</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def run_app(
    st: Any,
    *,
    current_workspace: Callable[[], Any],
    list_job_description_files: Callable[..., list[Any]],
    count_generated_packages: Callable[[], int],
    load_tracker_rows: Callable[..., list[dict[str, Any]]],
    demo_mode_enabled: Callable[[], bool],
    render_candidate_workspace_setup: Callable[[Any], None],
    manual_jobs_module: Any,
    page_renderers: Mapping[str, Callable[[], None]],
) -> None:
    """Configure Streamlit, resolve navigation, and dispatch one page renderer."""
    st.set_page_config(page_title="Job Application Toolkit", layout="wide")
    render_global_styles(st)
    render_sidebar(
        st,
        current_workspace=current_workspace,
        list_job_description_files=list_job_description_files,
        count_generated_packages=count_generated_packages,
        load_tracker_rows=load_tracker_rows,
        demo_mode_enabled=demo_mode_enabled,
    )
    render_app_title(st)

    workspace = current_workspace()
    if workspace.mode == "personal" and (
        not workspace.ready or st.session_state.get("workspace_setup_open", False)
    ):
        render_candidate_workspace_setup(workspace)
        return
    if workspace.mode == "personal":
        manual_jobs_module.MANUAL_SAVED_JOBS_DIR = workspace.jobs_dir

    active_page = str(st.session_state.get("active_page", "Dashboard"))
    if active_page not in PAGE_NAMES:
        active_page = "Dashboard"
        st.session_state["active_page"] = active_page

    selected_page = st.radio(
        "Navigation",
        PAGE_NAMES,
        index=PAGE_NAMES.index(active_page),
        horizontal=True,
        label_visibility="collapsed",
    )
    if selected_page != active_page:
        st.session_state["active_page"] = selected_page
        st.rerun()

    page_renderers[selected_page]()
