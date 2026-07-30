"""Tests for shared company verification controls."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import dashboard_company_verification as company_ui


class CompanyVerificationTests(unittest.TestCase):
    def test_generation_requires_verified_company_without_review_flag(self) -> None:
        self.assertTrue(
            company_ui.company_generation_allowed(
                {
                    "company_normalized": "Example Labs",
                    "company_confidence": "high",
                    "company_needs_review": False,
                }
            )
        )
        self.assertTrue(
            company_ui.company_generation_allowed(
                {
                    "company_normalized": "Example Labs",
                    "company_confidence": "low",
                    "company_confirmed_by_user": True,
                    "company_needs_review": False,
                }
            )
        )
        self.assertFalse(
            company_ui.company_generation_allowed(
                {
                    "company_normalized": "Example Labs",
                    "company_confidence": "high",
                    "company_needs_review": True,
                }
            )
        )

    def test_candidate_names_are_normalized_and_deduplicated(self) -> None:
        fields = {
            "company_normalized": "Example Labs",
            "company_candidates": [
                {"normalized_company": "Example Labs"},
                {"company": "Second Company"},
                "Second Company",
            ],
        }

        self.assertEqual(
            company_ui.company_candidate_names(fields),
            ["Example Labs", "Second Company"],
        )
        self.assertEqual(
            company_ui.compact_company_evidence(
                {"company_evidence": "domain: example.test | title: Example"}
            ),
            "domain: example.test | title: Example",
        )

    def test_markdown_confirmation_updates_valid_selected_company(self) -> None:
        ui = MagicMock()
        ui.selectbox.return_value = "Example Labs"
        ui.text_input.return_value = "Example Labs"
        ui.button.return_value = True
        fields = {
            "company_normalized": "Example Labs",
            "company_confidence": "medium",
            "company_needs_review": True,
            "company_candidates": ["Example Labs"],
        }
        path = Path("sample-job.md")

        with (
            patch.object(company_ui, "st", ui),
            patch.object(company_ui, "verification_from_markdown", return_value=fields),
            patch.object(
                company_ui,
                "verification_status_label",
                return_value="Needs review",
            ),
            patch.object(
                company_ui,
                "normalize_company_name",
                return_value="Example Labs",
            ),
            patch.object(company_ui, "confirm_markdown_company") as confirm,
        ):
            result = company_ui.render_markdown_company_confirmation(path, "sample")

        self.assertIs(result, fields)
        confirm.assert_called_once_with(path, "Example Labs")
        ui.success.assert_called_once_with("Company name confirmed.")
        ui.rerun.assert_called_once()

    def test_manual_confirmation_reports_failed_record_update(self) -> None:
        ui = MagicMock()
        ui.selectbox.return_value = "Example Labs"
        ui.text_input.return_value = "Example Labs"
        ui.button.return_value = True
        fields = {
            "company_normalized": "Example Labs",
            "company_candidates": ["Example Labs"],
            "company_needs_review": True,
        }
        record = {
            "id": "manual-7",
            "company": "Example Labs",
            "title": "Data Analyst",
        }

        with (
            patch.object(company_ui, "st", ui),
            patch.object(company_ui, "company_verification_fields", return_value=fields),
            patch.object(
                company_ui,
                "verification_status_label",
                return_value="Needs review",
            ),
            patch.object(
                company_ui,
                "normalize_company_name",
                return_value="Example Labs",
            ),
            patch.object(
                company_ui,
                "confirm_manual_job_company",
                return_value=None,
            ) as confirm,
        ):
            result = company_ui.render_manual_company_confirmation(record, "manual")

        self.assertIs(result, fields)
        confirm.assert_called_once_with("manual-7", "Example Labs")
        ui.error.assert_called_once_with("Could not update that target job record.")
        ui.rerun.assert_not_called()


if __name__ == "__main__":
    unittest.main()
