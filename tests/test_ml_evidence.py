"""Tests for requirement-to-resume semantic evidence retrieval."""

from __future__ import annotations

from pathlib import Path
import unittest

from ml.evidence import (
    build_semantic_evidence_index,
    extract_requirement_records,
    extract_resume_evidence_records,
    score_evidence_pair,
)
from scoring_engine import score_job_texts
from scoring_report import analyze_job_structured


NO_MODEL = Path("/missing/portable-relevance-model.json")


class SemanticEvidenceTests(unittest.TestCase):
    def test_etl_workflow_semantically_supports_data_pipeline_requirement(self) -> None:
        score = score_evidence_pair(
            "Build production data pipelines",
            "Developed automated ETL workflows for multi-source research data.",
            model_path=NO_MODEL,
        )
        self.assertTrue(score["accepted"])
        self.assertEqual(score["match_type"], "Semantic support")
        self.assertIn("data_pipeline", score["shared_concepts"])

    def test_direct_technical_overlap_is_identified(self) -> None:
        score = score_evidence_pair(
            "Use Python and SQL for analysis",
            "Built a Python and SQL reporting dashboard for operational datasets.",
            model_path=NO_MODEL,
        )
        self.assertTrue(score["accepted"])
        self.assertEqual(score["match_type"], "Direct support")
        self.assertEqual(set(score["shared_terms"]), {"python", "sql"})

    def test_unrelated_resume_line_is_rejected(self) -> None:
        score = score_evidence_pair(
            "Deploy machine learning services to production",
            "Presented historical research findings to a student seminar.",
            model_path=NO_MODEL,
        )
        self.assertFalse(score["accepted"])
        self.assertEqual(score["match_type"], "Insufficient evidence")

    def test_index_preserves_exact_resume_wording_and_rejects_gaps(self) -> None:
        job = """# Data Engineer

## Requirements
- Build production data pipelines
- Manage Kubernetes infrastructure
"""
        resume = """# Fictional Candidate

## Research Engineering
- Developed automated ETL workflows for multi-source research data.
- Presented findings to project stakeholders.
"""
        index = build_semantic_evidence_index(job, resume, model_path=NO_MODEL)
        self.assertEqual(index["requirement_count"], 2)
        self.assertEqual(index["accepted_count"], 1)
        accepted = index["accepted_matches"][0]
        self.assertEqual(accepted["requirement"], "Build production data pipelines")
        self.assertEqual(
            accepted["evidence"],
            "Developed automated ETL workflows for multi-source research data.",
        )
        self.assertEqual(accepted["section_evidence"], "Research Engineering")
        self.assertEqual(index["unmatched_requirements"], ["Manage Kubernetes infrastructure"])

    def test_required_requirements_precede_preferred(self) -> None:
        records = extract_requirement_records(
            """## Preferred Qualifications
- Tableau is preferred

## Requirements
- Python and SQL
"""
        )
        self.assertEqual([record["demand"] for record in records], ["required", "preferred"])

    def test_plain_section_labels_are_recognized_and_negated_requirements_are_ignored(self) -> None:
        records = extract_requirement_records(
            """Requirements:
- Python
- SQL

No specific degree level is required for this role.
"""
        )
        self.assertEqual([record["text"] for record in records], ["Python", "SQL"])

    def test_resume_extractor_omits_contact_and_keeps_section(self) -> None:
        records = extract_resume_evidence_records(
            """# Fictional Candidate

## Experience
- Built a reporting dashboard for operational datasets.
"""
        )
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["section"], "Experience")

    def test_sensitive_eligibility_evidence_is_not_cover_letter_eligible(self) -> None:
        index = build_semantic_evidence_index(
            """## Requirements
- Current work authorization is required
""",
            """# Fictional Candidate

## Additional Information
- Current work authorization is explicitly documented in the candidate source.
""",
            model_path=NO_MODEL,
        )
        self.assertEqual(index["accepted_count"], 1)
        self.assertFalse(index["accepted_matches"][0]["cover_letter_eligible"])
        self.assertEqual(index["cover_letter_eligible_count"], 0)

    def test_structured_analysis_adds_evidence_without_changing_canonical_score(self) -> None:
        job = """## Requirements
- Build production data pipelines
"""
        resume = """# Fictional Candidate

## Engineering Project
- Developed automated ETL workflows for multi-source research data.
"""
        baseline = score_job_texts(job, resume)
        analysis = analyze_job_structured(job, resume)
        self.assertEqual(analysis["score"], baseline["score"])
        self.assertEqual(analysis["eligibility"], baseline["eligibility"])
        self.assertEqual(analysis["semantic_evidence"]["accepted_count"], 1)
        self.assertEqual(
            analysis["profile_evidence"],
            ["Developed automated ETL workflows for multi-source research data."],
        )


if __name__ == "__main__":
    unittest.main()
