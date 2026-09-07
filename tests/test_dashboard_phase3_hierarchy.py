"""Guard the simplified Dashboard, Tracker, and Settings hierarchy."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

from dashboard_home import home_summary, primary_home_action
from dashboard_shell import NAVIGATION_ITEMS
from dashboard_settings_sections import job_source_health
from dashboard_tracker_components import tracker_summary


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_dashboard_keeps_one_primary_action_and_collapsed_counts() -> None:
    jobs = [
        {
            "eligibility": {"status": "passed"},
            "score": 88,
            "confidence": {"level": "high"},
            "jd_quality": {"reliable_scoring_ready": True},
        },
        {
            "eligibility": {"status": "failed"},
            "score": 40,
            "confidence": {"level": "low"},
            "jd_quality": {"reliable_scoring_ready": False},
        },
    ]
    assert home_summary(jobs, 3) == {
        "saved": 2,
        "strong": 1,
        "needs_full_jd": 1,
        "cover_letters": 3,
    }

    source = "\n".join(
        (PROJECT_ROOT / "src" / filename).read_text(encoding="utf-8")
        for filename in ("dashboard_home.py", "dashboard_home_components.py")
    )
    assert "Saved opportunities" not in source
    assert "Applications sent" not in source
    assert "Next best action" in source
    assert "Priority opportunities" not in source
    assert 'st.markdown("**Workspace summary**")' in source
    assert "Recent opportunities" in source
    assert "Search fresh roles" in source
    assert primary_home_action(jobs, 0)[2] == "Review Jobs"


def test_shell_keeps_navigation_in_sidebar_and_bounds_main_width() -> None:
    source = (PROJECT_ROOT / "src" / "dashboard_shell.py").read_text(encoding="utf-8")
    assert "NAVIGATION_ITEMS" in source
    assert "st.sidebar.button(" in source
    assert "max-width: none" in source
    assert 'header[data-testid="stHeader"]' in source
    assert "desktop-workspace-marker" in source
    assert "@media (max-width: 1099px)" in source
    assert "render_app_title" not in source
    assert "st.sidebar.radio(" not in source
    assert "Personal · {workspace_status} · Local" in source
    assert "Manage workspace files" not in source
    assert "Resume ready" in source
    assert 'resume_open if page == "Resume"' in source
    assert 'else:\n                st.session_state["workspace_setup_open"] = False' in source
    assert "Candidate source:" not in source
    assert "sidebar_jobs" not in source
    assert "No automatic submissions." not in source
    assert [label for label, _page, _icon in NAVIGATION_ITEMS] == [
        "Dashboard",
        "Resume",
        "Find Jobs",
        "Review Jobs",
        "Cover Letters",
        "Settings",
    ]
    assert all(page != "Tracker" for _label, page, _icon in NAVIGATION_ITEMS)
    assert "justify-content:flex-start !important" in source
    assert "grid-template-columns:1.45rem minmax(0,1fr)" in source
    assert "flex:0 0 1.45rem" in source


def test_tracker_defaults_to_stage_and_next_action() -> None:
    rows = [
        {"status": "saved"},
        {"status": "interview"},
        {"status": "archived"},
    ]
    with patch("dashboard_tracker_components.tracker_follow_up_due", return_value=False):
        assert tracker_summary(rows) == {"active": 2, "follow_ups": 0, "interviews": 1}

    source = (PROJECT_ROOT / "src" / "dashboard_tracker_components.py").read_text(encoding="utf-8")
    assert '"Stage":' in source
    assert '"Next action":' in source
    assert source.index('ui.button("Update Stage"') < source.index(
        'with ui.expander("Application details"'
    )


def test_settings_reports_source_health_without_key_values() -> None:
    clean_env = {key: value for key, value in os.environ.items() if not key.endswith("API_KEY")}
    clean_env.pop("ADZUNA_APP_ID", None)
    clean_env.pop("ADZUNA_APP_KEY", None)
    with (
        patch.dict(os.environ, clean_env, clear=True),
        patch("dashboard_settings_sections.jsearch_configured", return_value=False),
    ):
        health = job_source_health()
    assert health == {
        "JSearch · full JD": False,
        "Adzuna · discovery": False,
        "Jooble · discovery": False,
    }

    source = (PROJECT_ROOT / "src" / "dashboard_settings_sections.py").read_text(encoding="utf-8")
    assert '["Workspace", "Job sources", "Scoring", "Privacy"]' in source
    assert "local relevance model" not in source.lower()
    assert "os.getenv" in source
    assert "API key:" not in source


def test_cover_letter_uses_aligned_context_and_document_panels() -> None:
    source = (PROJECT_ROOT / "src" / "dashboard_cover_letter.py").read_text(encoding="utf-8")
    styles = (PROJECT_ROOT / "src" / "dashboard_desktop_styles.py").read_text(encoding="utf-8")
    assert "st.columns(" in source
    assert 'vertical_alignment="top"' in source
    assert 'key="cover_letter_context_panel"' in source
    assert 'key="cover_letter_document_panel"' in source
    assert "render_cover_letter_context(" in source
    assert "render_cover_letter_document(" in source
    assert ".st-key-cover_letter_context_panel" in styles
    assert ".st-key-cover_letter_document_panel" in styles
    assert ":has(.cover-letter-editor-detail)" in styles
    assert "flex:0 0 calc(76% - 2.1rem) !important" in styles
    assert "max-height:calc(100vh - 4.7rem) !important" in styles
    assert "flex:0 0 calc(100vh - 4.7rem) !important" in styles
    assert "overflow-y:auto !important" in styles
    assert "min-height:34rem !important" in styles
    assert "height:clamp(260px,34vh,390px) !important" in styles
