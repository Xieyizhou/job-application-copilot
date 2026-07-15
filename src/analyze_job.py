"""Public facade and CLI for deterministic job-fit analysis.

The implementation is split by responsibility across extraction, matching,
eligibility, scoring, and reporting modules. Existing imports from
``analyze_job`` remain supported for backward compatibility.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from output_paths import application_package_dir
from scoring_config import (
    CAREER_LEVELS,
    DEGREE_RANK,
    DIRECT_MATCH_STRENGTH,
    EXPERIENCE_LEVEL_KEYWORDS,
    EXPERIENCE_THEMES,
    KEYWORD_CATALOG,
    NO_MATCH_STRENGTH,
    PARTIAL_MATCH_STRENGTH,
    PARTIAL_RESUME_MATCHES,
    PENALTY_RULES,
    PREFERRED_LANGUAGE,
    PREFERRED_REQUIREMENT_WEIGHT,
    RED_FLAG_RULES,
    REQUIRED_REQUIREMENT_WEIGHT,
    ROLE_FOCUS_RULES,
    SCORE_CATEGORIES,
    UK_ALREADY_AUTHORIZED_WARNING,
    UK_HPI_MANUAL_REVIEW_WARNING,
    UK_HPI_NOTE,
)
from scoring_eligibility import (
    _eligibility_reason,
    _hard_seniority_requirement,
    _required_degree,
    _required_experience_years,
    evaluate_eligibility,
)
from scoring_engine import (
    apply_uk_work_authorization_score_cap,
    apply_role_focus_adjustment,
    calculate_match_score,
    calculate_score_breakdown,
    calculate_scoring_confidence,
    calibrate_score_for_evidence,
    assess_job_description_quality,
    evaluate_role_focus_alignment,
    extract_job_description_body,
    extract_saved_job_title,
    explain_final_decision,
    final_recommendation,
    find_penalties,
    recommendation_for_score,
    score_job_texts,
    score_note,
)
from scoring_extraction import (
    add_unique,
    all_scored_keywords,
    asks_for_uk_work_authorization_review,
    contains_alias,
    explicit_job_experience_levels,
    find_keywords,
    find_red_flags,
    infer_candidate_experience_profile,
    is_preferred_line,
    is_uk_job,
    must_already_have_uk_work_authorization,
    normalize_text,
    parse_job_description,
    split_job_description_lines,
)
from scoring_matching import (
    choose_relevant_themes,
    collect_report_matches,
    demand_type_for_keyword,
    experience_match_strength,
    match_strength_for_keyword,
    resume_suggestions_for_keywords,
    short_evidence_snippets,
)
from scoring_report import (
    analyze_job,
    analyze_job_structured,
    build_markdown_report,
    explain_overall_score,
    find_resume_evidence,
    format_bullets,
    format_inline_list,
    format_penalties,
    format_reason_messages,
    format_score_breakdown,
    save_report,
)
from workspace import Workspace, WorkspaceError, demo_workspace, personal_workspace


PROJECT_ROOT = Path(__file__).resolve().parents[1]


# Explicitly document the compatibility surface. Direct imports also continue to
# work for the private helpers historically exposed by this module.
__all__ = [
    "CAREER_LEVELS",
    "DEGREE_RANK",
    "DIRECT_MATCH_STRENGTH",
    "EXPERIENCE_LEVEL_KEYWORDS",
    "EXPERIENCE_THEMES",
    "KEYWORD_CATALOG",
    "NO_MATCH_STRENGTH",
    "PARTIAL_MATCH_STRENGTH",
    "PARTIAL_RESUME_MATCHES",
    "PENALTY_RULES",
    "PREFERRED_LANGUAGE",
    "PREFERRED_REQUIREMENT_WEIGHT",
    "RED_FLAG_RULES",
    "REQUIRED_REQUIREMENT_WEIGHT",
    "ROLE_FOCUS_RULES",
    "SCORE_CATEGORIES",
    "UK_ALREADY_AUTHORIZED_WARNING",
    "UK_HPI_MANUAL_REVIEW_WARNING",
    "UK_HPI_NOTE",
    "Workspace",
    "WorkspaceError",
    "add_unique",
    "all_scored_keywords",
    "analyze_job",
    "analyze_job_structured",
    "apply_uk_work_authorization_score_cap",
    "apply_role_focus_adjustment",
    "asks_for_uk_work_authorization_review",
    "build_markdown_report",
    "calculate_match_score",
    "calculate_score_breakdown",
    "calculate_scoring_confidence",
    "calibrate_score_for_evidence",
    "assess_job_description_quality",
    "evaluate_role_focus_alignment",
    "extract_job_description_body",
    "extract_saved_job_title",
    "choose_relevant_themes",
    "collect_report_matches",
    "contains_alias",
    "demand_type_for_keyword",
    "evaluate_eligibility",
    "experience_match_strength",
    "explicit_job_experience_levels",
    "explain_final_decision",
    "explain_overall_score",
    "final_recommendation",
    "find_keywords",
    "find_penalties",
    "find_red_flags",
    "find_resume_evidence",
    "format_bullets",
    "format_inline_list",
    "format_penalties",
    "format_reason_messages",
    "format_score_breakdown",
    "infer_candidate_experience_profile",
    "is_preferred_line",
    "is_uk_job",
    "match_strength_for_keyword",
    "must_already_have_uk_work_authorization",
    "normalize_text",
    "parse_job_description",
    "recommendation_for_score",
    "resume_suggestions_for_keywords",
    "save_report",
    "score_job_texts",
    "score_note",
    "short_evidence_snippets",
    "split_job_description_lines",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze a job description against the selected candidate workspace."
    )
    parser.add_argument(
        "job_description",
        help="Path to a Markdown or text job description file.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Analyze with sanitized Demo candidate data without writing files.",
    )
    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""
    args = parse_args()
    job_description_path = Path(args.job_description).expanduser()

    if not job_description_path.is_absolute():
        job_description_path = PROJECT_ROOT / job_description_path

    if not job_description_path.exists():
        raise FileNotFoundError(f"Job description file was not found: {job_description_path}")

    if args.demo:
        workspace = demo_workspace()
        workspace.require_ready()
        assert workspace.resume_source_path is not None
        analysis = analyze_job_structured(
            job_description_path.read_text(encoding="utf-8"),
            workspace.resume_source_path.read_text(encoding="utf-8"),
        )
        print(json.dumps(analysis, indent=2))
        return

    workspace = personal_workspace()
    try:
        workspace.require_ready()
    except WorkspaceError as error:
        raise SystemExit(str(error)) from None
    report, report_path = analyze_job(job_description_path, workspace)
    print(report)
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
