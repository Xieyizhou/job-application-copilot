"""Focused regression tests for canonical job titles and Custom Location UI."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import dashboard


class DashboardTitleTests(unittest.TestCase):
    def test_exact_placeholder_titles_are_invalid(self) -> None:
        for value in [
            "Sample Job",
            " test   job ",
            "DEMO JOB",
            "Unknown Role",
            "Untitled Job",
            "Not Provided",
            "N/A",
        ]:
            with self.subTest(value=value):
                self.assertTrue(dashboard.is_placeholder_job_title(value))

    def test_real_titles_containing_placeholder_words_remain_valid(self) -> None:
        for value in [
            "Sample Management Analyst",
            "Test Automation Engineer",
            "Demo Platform Manager",
            "Senior Data Manager",
        ]:
            with self.subTest(value=value):
                self.assertFalse(dashboard.is_placeholder_job_title(value))
                self.assertEqual(dashboard.resolve_canonical_job_title({"role": value}), value)

    def test_markdown_role_replaces_placeholder_object_role(self) -> None:
        job = {
            "company": "Axle",
            "role": "Sample Job",
            "preview": """
# Data Science Fellow - AI/NLP

Company: Axle
Role: Data Science Fellow - AI/NLP
Location: Remote
""",
        }
        self.assertEqual(
            dashboard.resolve_canonical_job_title(job),
            "Data Science Fellow - AI/NLP",
        )

    def test_title_priority_and_heading_fallback(self) -> None:
        job = {
            "company": "Fictional Axle",
            "display_role": "Research Analyst",
            "title": "Data Analyst",
            "role": "Analytics Associate",
            "preview": "Role: ML Analyst",
        }
        self.assertEqual(dashboard.resolve_canonical_job_title(job), "Research Analyst")

        heading_only = {
            "company": "Fictional Axle",
            "role": "Unknown Role",
            "preview": "# Applied Scientist\n\nCompany: Fictional Axle",
        }
        self.assertEqual(dashboard.resolve_canonical_job_title(heading_only), "Applied Scientist")

    def test_company_or_metadata_heading_is_not_used_as_role(self) -> None:
        for preview in [
            "# Fictional Axle\n\nCompany: Fictional Axle",
            "# Job Description\n\nCompany: Fictional Axle",
            "# Company: Fictional Axle\n\nLocation: Remote",
        ]:
            with self.subTest(preview=preview):
                self.assertEqual(
                    dashboard.resolve_canonical_job_title(
                        {"company": "Fictional Axle", "role": "N/A", "preview": preview}
                    ),
                    "Missing job title",
                )


def run_custom_location_child() -> None:
    import pyarrow

    pyarrow.set_memory_pool(pyarrow.system_memory_pool())
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(PROJECT_ROOT / "src" / "dashboard.py")
    app.session_state["workspace_mode"] = "Demo"
    app.run(timeout=30)
    app.radio[0].set_value("Find Jobs").run(timeout=30)
    assert not list(app.exception)
    region = next(widget for widget in app.selectbox if widget.label == "Region")
    region.set_value("Custom").run(timeout=30)
    assert not list(app.exception)
    assert "Custom Location" in [widget.label for widget in app.text_input]


class CustomLocationRuntimeTests(unittest.TestCase):
    def test_custom_location_appears_on_region_selection(self) -> None:
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--child"],
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    if "--child" in sys.argv:
        run_custom_location_child()
    else:
        unittest.main()
