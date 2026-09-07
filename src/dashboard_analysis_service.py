"""Canonical dashboard analysis orchestration without UI dependencies."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, MutableMapping
from typing import Any

from scoring_types import DashboardJob


StructuredAnalyzer = Callable[..., Mapping[str, Any]]
AnalysisCache = MutableMapping[str, dict[str, Any]]


def unavailable_dashboard_analysis(reason: str) -> dict[str, Any]:
    """Return a safe analysis result without promoting a legacy score."""
    return {
        "score": None,
        "recommendation": "Manual Review",
        "score_breakdown": [],
        "eligibility": {"status": "manual_review", "reasons": []},
        "confidence": {
            "level": "low",
            "active_requirement_count": 0,
            "candidate_evidence_count": 0,
            "reasons": [reason],
        },
        "candidate_profile": {
            "career_level": "unknown",
            "years_experience": None,
            "highest_degree": "unknown",
            "evidence": [],
        },
        "parsed_job": {
            "required_skills": [],
            "preferred_skills": [],
            "experience_level": [],
        },
        "matched_skills": [],
        "partial_matches": [],
        "missing_skills": [],
        "main_reason": reason,
        "main_risk": "Review the full job description manually.",
        "analysis_available": False,
    }


def dashboard_analysis_cache_key(
    job: DashboardJob,
    job_text: str,
    candidate_text: str,
    *,
    scoring_version: str,
    workspace_mode: str = "provided",
    workspace_root: str = "provided",
) -> str:
    """Build a content-addressed key scoped to scoring and workspace inputs."""
    cache_material = "\0".join(
        [
            scoring_version,
            workspace_mode,
            workspace_root,
            str(job.get("canonical_job_key", "") or job.get("path", "")),
            hashlib.sha256(job_text.encode("utf-8")).hexdigest(),
            hashlib.sha256(candidate_text.encode("utf-8")).hexdigest(),
        ]
    )
    return hashlib.sha256(cache_material.encode("utf-8")).hexdigest()


def analyze_dashboard_job(
    job: DashboardJob,
    job_text: str,
    candidate_text: str,
    *,
    analyzer: StructuredAnalyzer,
    scoring_version: str,
    cache: AnalysisCache | None = None,
    workspace_mode: str = "provided",
    workspace_root: str = "provided",
) -> dict[str, Any]:
    """Run canonical analysis with safe fallbacks and optional content caching."""
    if not job_text.strip():
        return unavailable_dashboard_analysis("Job description is empty or unreadable.")
    if not candidate_text.strip():
        return unavailable_dashboard_analysis("Candidate source is missing or empty.")

    cache_key = dashboard_analysis_cache_key(
        job,
        job_text,
        candidate_text,
        scoring_version=scoring_version,
        workspace_mode=workspace_mode,
        workspace_root=workspace_root,
    )
    if cache is not None and cache_key in cache:
        return dict(cache[cache_key])

    try:
        analysis = dict(analyzer(job_text, candidate_text))
    except (OSError, ValueError, TypeError) as error:
        return unavailable_dashboard_analysis(f"Current analysis could not run: {error}")

    confidence_value = analysis.get("confidence", {})
    confidence = dict(confidence_value) if isinstance(confidence_value, dict) else {}
    if int(confidence.get("active_requirement_count", 0) or 0) == 0:
        pipeline = analysis.get("jd_pipeline", {})
        if isinstance(pipeline, dict):
            from structured_jd import pipeline_diagnostic_message

            reason = pipeline_diagnostic_message(pipeline)
        else:
            reason = "Structured requirements could not be extracted reliably."
        return unavailable_dashboard_analysis(
            reason
        )

    analysis["analysis_available"] = True
    if cache is not None:
        cache[cache_key] = dict(analysis)
    return analysis
