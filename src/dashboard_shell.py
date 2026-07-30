"""Streamlit application shell shared by all dashboard pages."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from scoring_types import TrackerRow


PAGE_NAMES = (
    "Dashboard",
    "Find Jobs",
    "Add Target Job",
    "Review Jobs",
    "Cover Letter",
    "Tracker",
    "Settings",
)

GLOBAL_STYLES = """
<style>
header[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
#MainMenu,
footer {
    display: none !important;
}
.block-container {
    max-width: 1320px !important;
    padding-top: .75rem !important;
    padding-bottom: 2rem !important;
    margin-left: auto !important;
    margin-right: auto !important;
}
div[data-testid="stMain"]:has(.desktop-workspace-marker) {
    overflow: hidden !important;
}
div[data-testid="stMainBlockContainer"]:has(.desktop-workspace-marker) {
    height: 100vh !important;
    overflow: hidden !important;
    padding-bottom: .5rem !important;
}
.desktop-workspace-marker {
    display: none;
}
.page-header {
    margin-top: 0;
    margin-bottom: 0.55rem;
}
.page-title {
    font-size: 1.55rem;
    font-weight: 700;
    line-height: 1.25;
    margin: 0 0 0.35rem 0;
}
.page-subtitle {
    color: var(--text-color);
    opacity: 0.68;
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
@media (max-width: 1099px) {
    div[data-testid="stMain"]:has(.desktop-workspace-marker),
    div[data-testid="stMainBlockContainer"]:has(.desktop-workspace-marker) {
        height: auto !important;
        overflow: auto !important;
    }
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
        "selected_review_tab",
        "review_workspace_mode",
    ]:
        session_state.pop(key, None)


def render_sidebar(
    st: Any,
    *,
    current_workspace: Callable[[], Any],
    list_job_description_files: Callable[..., list[Any]],
    count_generated_packages: Callable[[], int],
    load_tracker_rows: Callable[..., list[TrackerRow]],
    demo_mode_enabled: Callable[[], bool],
) -> None:
    """Render a Personal-first workflow with Demo as a separate experience."""
    active_page = str(st.session_state.get("active_page", "Dashboard"))
    if active_page not in PAGE_NAMES:
        active_page = "Dashboard"
        st.session_state["active_page"] = active_page

    st.sidebar.title("Job Application Toolkit")
    st.sidebar.caption("Your local job application workflow")
    selected_page = st.sidebar.radio(
        "Navigation",
        PAGE_NAMES,
        index=PAGE_NAMES.index(active_page),
        label_visibility="collapsed",
    )
    if selected_page != active_page:
        st.session_state["active_page"] = selected_page
        st.rerun()
    st.sidebar.divider()

    active_mode = str(st.session_state.get("workspace_mode", "Personal"))
    if active_mode == "Demo":
        if st.sidebar.button("Back to Personal Workspace", type="primary", width="stretch"):
            switch_workspace_mode(st.session_state, "Personal")
            st.rerun()
    elif st.sidebar.button("Explore Read-only Demo", width="stretch"):
        switch_workspace_mode(st.session_state, "Demo")
        st.rerun()

    workspace = current_workspace()
    if workspace.mode == "personal":
        workspace_status = "Ready" if workspace.ready else "Setup required"
        st.sidebar.caption(f"Personal · {workspace_status} · Local")
        if workspace.ready and st.sidebar.button("Manage workspace files", width="stretch"):
            st.session_state["workspace_setup_open"] = True
            st.rerun()
    else:
        st.sidebar.caption("Demo · Read-only")


def run_app(
    st: Any,
    *,
    current_workspace: Callable[[], Any],
    list_job_description_files: Callable[..., list[Any]],
    count_generated_packages: Callable[[], int],
    load_tracker_rows: Callable[..., list[TrackerRow]],
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

    workspace = current_workspace()
    if workspace.mode == "personal" and (
        not workspace.ready or st.session_state.get("workspace_setup_open", False)
    ):
        render_candidate_workspace_setup(workspace)
        return
    if workspace.mode == "personal":
        manual_jobs_module.MANUAL_SAVED_JOBS_DIR = workspace.jobs_dir

    page_renderers[str(st.session_state.get("active_page", "Dashboard"))]()
