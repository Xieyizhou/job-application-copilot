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

NAVIGATION_ITEMS = (
    ("Dashboard", "Dashboard", ":material/home:"),
    ("Resume", "Resume", ":material/description:"),
    ("Find Jobs", "Find Jobs", ":material/search:"),
    ("Review Jobs", "Review Jobs", ":material/assignment:"),
    ("Cover Letters", "Cover Letter", ":material/edit_note:"),
    ("Settings", "Settings", ":material/settings:"),
)

GLOBAL_STYLES = """
<style>
:root {
    --app-canvas: #f6f7f9;
    --app-surface: #ffffff;
    --app-sidebar: #f1f3f6;
    --app-text: #20242d;
    --app-muted: #68707d;
    --app-border: #dfe3e8;
    --app-border-strong: #cdd3da;
    --app-accent: #d94f55;
    --app-accent-dark: #b93840;
    --app-accent-soft: #fff0f1;
    --app-radius-sm: 8px;
    --app-radius-md: 12px;
    --app-shadow: 0 1px 2px rgba(20, 24, 32, .04), 0 10px 28px rgba(20, 24, 32, .035);
}
html, body, [class*="css"] {
    font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont,
        "Segoe UI", sans-serif;
}
[data-testid="stAppViewContainer"] {
    background: var(--app-canvas);
    color: var(--app-text);
}
header[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
#MainMenu,
footer {
    display: none !important;
}
[data-testid="stSidebar"] {
    background: var(--app-sidebar);
    border-right: 1px solid var(--app-border);
}
[data-testid="stSidebar"][aria-expanded="true"],
[data-testid="stSidebar"][aria-expanded="true"] [data-testid="stSidebarContent"] {
    width: 242px !important;
    min-width: 242px !important;
    max-width: 242px !important;
}
[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
    padding: 1.55rem 1.1rem 1.2rem;
}
@media (min-width: 1100px) {
    [data-testid="stSidebarHeader"] {height:0 !important;min-height:0 !important;padding:0 !important}
    [data-testid="stSidebarCollapseButton"] {display:none !important}
    div[data-testid="stMainBlockContainer"]:has(.desktop-workspace-marker) {
        margin-top:-2.7rem !important;
    }
}
[data-testid="stSidebar"] h1 {
    color: var(--app-text);
    font-size: 1.22rem !important;
    letter-spacing: -.025em;
    line-height: 1.2 !important;
    margin-bottom: .25rem !important;
}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    color: var(--app-muted);
}
[data-testid="stSidebar"] hr {
    border-color: var(--app-border);
    margin: 1.15rem 0 1rem;
}
.sidebar-brand {display:flex;align-items:center;gap:.62rem;margin:.2rem .1rem .1rem}
.sidebar-brand-mark {
    color:var(--app-accent);font-family:"Material Symbols Rounded";
    font-size:1.65rem;font-weight:400;line-height:1;
}
.sidebar-brand-name {color:var(--app-accent);font-size:1.24rem;font-weight:760;letter-spacing:-.025em}
.sidebar-brand-caption {color:var(--app-muted);font-size:.78rem;margin:.05rem 0 1.45rem 2.27rem}
[data-testid="stSidebar"] [class*="st-key-sidebar_nav_"] button {
    justify-content:flex-start !important;gap:.66rem;border:0 !important;border-radius:6px !important;
    background:transparent;box-shadow:none !important;color:var(--app-text);
    min-height:2.7rem;padding:.48rem .62rem;font-weight:520;
}
[data-testid="stSidebar"] [class*="st-key-sidebar_nav_"] button > div {
    justify-content:flex-start !important;width:100%;
}
[data-testid="stSidebar"] [class*="st-key-sidebar_nav_"] button > div > span {
    display:grid !important;grid-template-columns:1.45rem minmax(0,1fr);
    align-items:center;gap:.66rem !important;width:100%;
}
[data-testid="stSidebar"] [class*="st-key-sidebar_nav_"] button:hover {
    background:rgba(32,36,45,.05);transform:none;
}
[data-testid="stSidebar"] [class*="st-key-sidebar_nav_"] button[kind="secondary"] {
    background:var(--app-accent-soft) !important;color:#d93b35 !important;
    box-shadow:inset 3px 0 0 #f04438 !important;font-weight:680;
}
[data-testid="stSidebar"] [class*="st-key-sidebar_nav_"] button span[data-testid="stIconMaterial"] {
    display:inline-flex;align-items:center;justify-content:center;
    flex:0 0 1.45rem;width:1.45rem;font-size:1.28rem;
}
[data-testid="stSidebar"] [class*="st-key-sidebar_nav_"] button p {
    margin:0;text-align:left;
}
[data-testid="stMain"] {
    background: var(--app-surface);
}
.block-container {
    max-width: none !important;
    padding-top: 1.15rem !important;
    padding-left: 1.65rem !important;
    padding-right: 1.65rem !important;
    padding-bottom: 2rem !important;
    margin-left: auto !important;
    margin-right: auto !important;
}
button, input, textarea, [data-baseweb="select"] > div {
    font-family: inherit !important;
}
[data-testid="stButton"] button,
[data-testid="stLinkButton"] a,
[data-testid="stDownloadButton"] button {
    border-radius: var(--app-radius-sm) !important;
    min-height: 2.35rem;
    font-weight: 620;
    transition: transform 120ms ease, border-color 120ms ease,
        background-color 120ms ease, box-shadow 120ms ease;
}
[data-testid="stButton"] button:hover,
[data-testid="stLinkButton"] a:hover,
[data-testid="stDownloadButton"] button:hover {
    border-color: var(--app-border-strong) !important;
    box-shadow: 0 2px 8px rgba(20, 24, 32, .07);
    transform: translateY(-1px);
}
[data-testid="stButton"] button[kind="primary"],
[data-testid="stDownloadButton"] button[kind="primary"] {
    background: var(--app-accent) !important;
    border-color: var(--app-accent) !important;
    color: #fff !important;
}
[data-testid="stButton"] button[kind="primary"]:hover,
[data-testid="stDownloadButton"] button[kind="primary"]:hover {
    background: var(--app-accent-dark) !important;
    border-color: var(--app-accent-dark) !important;
}
[data-testid="stExpander"] {
    border-color: var(--app-border) !important;
    border-radius: var(--app-radius-sm) !important;
    overflow: hidden;
}
[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: var(--app-border) !important;
    border-radius: var(--app-radius-md) !important;
    box-shadow: var(--app-shadow);
}
[data-baseweb="select"] > div,
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {
    background: #f7f8fa !important;
    border-color: transparent !important;
    border-radius: var(--app-radius-sm) !important;
}
[data-baseweb="select"] > div:focus-within,
[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {
    background: var(--app-surface) !important;
    border-color: var(--app-accent) !important;
    box-shadow: 0 0 0 2px rgba(217, 79, 85, .12) !important;
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
    margin-bottom: 0.8rem;
}
.page-title {
    color: var(--app-text);
    font-size: 1.62rem;
    font-weight: 760;
    letter-spacing: -.03em;
    line-height: 1.15;
    margin: 0 0 0.38rem 0;
}
.page-subtitle {
    color: var(--app-muted);
    font-size: 0.92rem;
    line-height: 1.45;
    margin: 0;
}
h1, h2, h3 {
    color: var(--app-text);
    letter-spacing: -.025em;
}
p, li {
    line-height: 1.5;
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
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        scroll-behavior: auto !important;
        transition-duration: .01ms !important;
        animation-duration: .01ms !important;
    }
}
[data-testid="stAppViewContainer"]:has(
    [data-testid="stSidebar"][aria-expanded="false"]
) header[data-testid="stHeader"] {
    display: flex !important;height: 2.6rem !important;
    background: rgba(255,255,255,.94) !important;
    border-bottom: 1px solid var(--app-border) !important;
    backdrop-filter: blur(10px);
}
[data-testid="stAppViewContainer"]:has(
    [data-testid="stSidebar"][aria-expanded="false"]
) header[data-testid="stHeader"] [data-testid="stToolbar"] {
    display: flex !important;
}
[data-testid="stAppViewContainer"]:has(
    [data-testid="stSidebar"][aria-expanded="false"]
) header[data-testid="stHeader"] button {
    display: none !important;
}
[data-testid="stAppViewContainer"]:has(
    [data-testid="stSidebar"][aria-expanded="false"]
) header[data-testid="stHeader"] [data-testid="stExpandSidebarButton"] {
    display: inline-flex !important;
}
[data-testid="stAppViewContainer"]:has(
    [data-testid="stSidebar"][aria-expanded="false"]
) .block-container {
    padding-top: 3.25rem !important;
}
@media (max-width: 1099px) {
    header[data-testid="stHeader"] {
        display: flex !important;
        height: 2.6rem !important;
        background: rgba(255,255,255,.94) !important;
        border-bottom: 1px solid var(--app-border) !important;
        backdrop-filter: blur(10px);
    }
    header[data-testid="stHeader"] [data-testid="stToolbar"] {
        display: flex !important;
    }
    header[data-testid="stHeader"] [data-testid="stAppDeployButton"],
    header[data-testid="stHeader"] [data-testid="stToolbarActions"] {
        display: none !important;
    }
    header[data-testid="stHeader"] button {
        display: none !important;
    }
    header[data-testid="stHeader"] [data-testid="stExpandSidebarButton"] {
        display: inline-flex !important;
    }
    .block-container {
        padding: 3.25rem 1.25rem 2rem !important;
    }
    div[data-testid="stMain"]:has(.desktop-workspace-marker),
    div[data-testid="stMainBlockContainer"]:has(.desktop-workspace-marker) {
        height: auto !important;
        overflow: auto !important;
    }
}
@media (max-width: 640px) {
    .block-container {
        padding: 3.1rem .9rem 1.75rem !important;
    }
    .page-title {font-size: 1.42rem}
    .page-subtitle {font-size: .86rem}
    [data-testid="stVerticalBlockBorderWrapper"] {box-shadow: none}
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
        "review_detail_tabs",
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

    st.sidebar.markdown(
        '<div class="sidebar-brand"><span class="sidebar-brand-mark">task_alt</span>'
        '<span class="sidebar-brand-name">JobCopilot</span></div>'
        '<div class="sidebar-brand-caption">Privacy-first</div>',
        unsafe_allow_html=True,
    )
    for label, page, icon in NAVIGATION_ITEMS:
        resume_open = bool(st.session_state.get("workspace_setup_open", False))
        selected = resume_open if page == "Resume" else active_page == page and not resume_open
        if st.sidebar.button(
            label,
            key=f"sidebar_nav_{page.lower().replace(' ', '_')}",
            icon=icon,
            type="secondary" if selected else "tertiary",
            width="stretch",
        ):
            if page == "Resume":
                st.session_state["workspace_setup_open"] = True
            else:
                st.session_state["workspace_setup_open"] = False
                st.session_state["active_page"] = page
            st.rerun()
    st.sidebar.divider()

    active_mode = str(st.session_state.get("workspace_mode", "Personal"))
    if active_mode == "Demo":
        if st.sidebar.button("Back to Personal Workspace", type="primary", width="stretch"):
            switch_workspace_mode(st.session_state, "Personal")
            st.rerun()
    elif st.sidebar.button("Explore Read-only Demo", width="stretch"):
        switch_workspace_mode(st.session_state, "Demo")
        st.session_state["active_page"] = "Dashboard"
        st.rerun()

    workspace = current_workspace()
    if workspace.mode == "personal":
        workspace_status = "Resume ready" if workspace.ready else "Resume required"
        st.sidebar.caption(f"Personal · {workspace_status} · Local")
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
    if st.session_state.get("workspace_setup_open", False) or (
        workspace.mode == "personal" and not workspace.ready
    ):
        render_candidate_workspace_setup(workspace)
        return
    if workspace.mode == "personal":
        manual_jobs_module.MANUAL_SAVED_JOBS_DIR = workspace.jobs_dir

    page_renderers[str(st.session_state.get("active_page", "Dashboard"))]()
