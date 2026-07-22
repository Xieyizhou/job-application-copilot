"""Guard the simplified Dashboard, Tracker, and Settings hierarchy."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

from dashboard_home import home_summary
from dashboard_settings_sections import job_source_health
from dashboard_tracker_components import tracker_summary


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_dashboard_keeps_three_decision_counts() -> None:
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
    assert source.index("**Next actions**") < source.index("**Priority opportunities**")


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
    assert '["Workspace", "Job sources", "Scoring", "Privacy", "Advanced"]' in source
    assert "os.getenv" in source
    assert "API key:" not in source
