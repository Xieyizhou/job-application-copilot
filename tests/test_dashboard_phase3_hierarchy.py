"""Guard the simplified Dashboard, Tracker, and Settings hierarchy."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

from dashboard_home import home_summary, primary_home_action
from dashboard_settings_sections import job_source_health
from dashboard_tracker_components import tracker_summary


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_dashboard_keeps_one_primary_action_and_collapsed_counts() -> None:
    jobs = [
        {"eligibility": {"status": "passed"}},
        {"eligibility": {"status": "failed"}},
    ]
    rows = [{"status": "ready"}, {"status": "interview"}]
    with patch("dashboard_home.tracker_follow_up_due", return_value=False):
        assert home_summary(jobs, rows) == {"active": 1, "ready": 1, "follow_ups": 0}

    source = (PROJECT_ROOT / "src" / "dashboard_home.py").read_text(encoding="utf-8")
    assert "Saved opportunities" not in source
    assert "Applications sent" not in source
    assert "**Your next step**" in source
    assert "Priority opportunities" not in source
    assert 'with st.expander("Workspace summary"' in source
    assert primary_home_action(jobs, rows, 0)[2] == "Review Jobs"


def test_shell_keeps_navigation_in_sidebar_and_bounds_main_width() -> None:
    source = (PROJECT_ROOT / "src" / "dashboard_shell.py").read_text(encoding="utf-8")
    assert "st.sidebar.radio(" in source
    assert 'max-width: 1320px' in source
    assert 'header[data-testid="stHeader"]' in source
    assert "desktop-workspace-marker" in source
    assert "@media (max-width: 1099px)" in source
    assert "render_app_title" not in source
    assert "st.radio(" not in source.replace("st.sidebar.radio(", "")


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
    with patch.dict(os.environ, clean_env, clear=True), patch(
        "dashboard_settings_sections.jsearch_configured", return_value=False
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
    source = (PROJECT_ROOT / "src" / "dashboard_cover_letter.py").read_text(
        encoding="utf-8"
    )
    styles = (PROJECT_ROOT / "src" / "dashboard_desktop_styles.py").read_text(
        encoding="utf-8"
    )
    assert 'st.columns(' in source
    assert 'vertical_alignment="top"' in source
    assert 'key="cover_letter_context_panel"' in source
    assert 'key="cover_letter_document_panel"' in source
    assert "render_cover_letter_context(" in source
    assert "render_cover_letter_document(" in source
    assert ".st-key-cover_letter_context_panel" in styles
    assert ".st-key-cover_letter_document_panel" in styles
