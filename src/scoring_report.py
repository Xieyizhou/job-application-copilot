"""Structured and Markdown reporting for deterministic job-fit analysis."""

from __future__ import annotations

from document_text import format_bullets

from datetime import datetime
from pathlib import Path
from typing import Any, cast

from ml.evidence import build_semantic_evidence_index
from output_paths import application_package_dir
from scoring_engine import (
    apply_role_focus_adjustment,
    calibrate_score_for_evidence,
    explain_final_decision,
    final_recommendation,
    score_job_texts,
)
from scoring_matching import (
    choose_relevant_themes,
    collect_report_matches,
    resume_suggestions_for_keywords,
    short_evidence_snippets,
)
from scoring_types import (
    EligibilityResult,
    ParsedJob,
    Penalty,
    RoleAlignment,
    ScoreBreakdownItem,
    ScoreCalibration,
    ScoringConfidence,
    ScoringResult,
    StructuredAnalysis,
)
from workspace import Workspace
from structured_jd import (
    StructuredJob,
    pipeline_trace,
    requirement_records,
    scoring_requirements,
    structure_job_description,
)


def _structured_requirement_score(
    structured: StructuredJob,
    semantic_evidence: dict[str, Any],
    legacy: ScoringResult,
) -> dict[str, Any]:
    """Score extracted requirements while retaining the deterministic result as audit data."""
    rows = list(scoring_requirements(structured))
    matches = list(semantic_evidence.get("matches", []) or [])
    by_text = {str(row.get("requirement", "")): row for row in matches}
    matched: list[str] = []
    partial: list[str] = []
    missing: list[str] = []
    earned = 0.0
    possible = 0.0
    for value in rows:
        row = dict(value)
        text = str(row.get("text", ""))
        weight = (
            0.5
            if row.get("type") == "preferred"
            else 0.75
            if row.get("type") == "responsibility"
            else 1.0
        )
        possible += weight
        evidence = dict(by_text.get(text, {}) or {})
        if not evidence.get("accepted"):
            missing.append(text)
            continue
        if evidence.get("match_type") == "Direct support":
            matched.append(text)
            earned += weight
        else:
            partial.append(text)
            earned += weight * 0.6
    coverage_score = round(100 * earned / possible) if possible else 0
    role_alignment = legacy["role_alignment"]
    observed_score, role_adjustment = apply_role_focus_adjustment(coverage_score, role_alignment)
    confidence = cast(ScoringConfidence, dict(legacy["confidence"]))
    confidence["active_requirement_count"] = len(rows)
    confidence["reasons"] = []
    quality = dict(confidence.get("job_description_quality", {}) or {})
    if len(rows) < 4:
        confidence["level"] = "low"
        confidence["reasons"].append("Fewer than four structured requirements were extracted.")
    elif len(rows) >= 8 and quality.get("explicit_full_source"):
        confidence["level"] = "high"
        confidence["reasons"].append(
            "At least eight source-backed requirements were recovered from the full posting."
        )
        quality.update(
            {
                "label": "scoring_ready",
                "display_label": "Scoring-ready",
                "appears_incomplete": False,
                "provisional_scoring_ready": True,
                "reliable_scoring_ready": True,
                "requirement_statement_count": max(
                    len(rows), int(quality.get("requirement_statement_count", 0) or 0)
                ),
                "next_action": "Review the extracted requirements before trusting the fit result.",
            }
        )
    elif quality.get("appears_incomplete"):
        confidence["level"] = "low"
        confidence["reasons"].append("The saved job description still appears incomplete.")
    elif len(rows) >= 8:
        confidence["level"] = "high"
        confidence["reasons"].append("At least eight source-backed requirements were extracted.")
    else:
        confidence["level"] = "medium"
        confidence["reasons"].append("Four to seven source-backed requirements were extracted.")
    score, calibration = calibrate_score_for_evidence(observed_score, confidence)
    confidence["coverage_score"] = coverage_score
    confidence["observed_score"] = observed_score
    confidence["score_calibration"] = calibration
    confidence["job_description_quality"] = quality
    recommendation = final_recommendation(score, legacy["eligibility"], confidence)
    breakdown = [
        {
            "category": "Structured requirement coverage",
            "earned": float(score),
            "possible": 100,
            "active_terms": [str(row.get("text", "")) for row in rows],
            "matched": matched,
            "partial": partial,
            "missing": missing,
            "note": "Source-backed requirements matched to the strongest resume evidence.",
        }
    ]
    return {
        "score": score,
        "coverage_score": coverage_score,
        "observed_score": observed_score,
        "score_calibration": calibration,
        "role_focus_adjustment": role_adjustment,
        "confidence": confidence,
        "recommendation": recommendation,
        "score_breakdown": breakdown,
        "matched": matched,
        "partial": partial,
        "missing": missing,
    }


def _enforce_evidence_consistency(
    result: ScoringResult,
    semantic_evidence: dict[str, Any],
) -> bool:
    """Prevent keyword coverage from contradicting an all-missing evidence table."""
    matches = list(semantic_evidence.get("matches", []) or [])
    quality = dict(result["confidence"].get("job_description_quality", {}) or {})
    if (
        not quality.get("explicit_full_source")
        or not matches
        or int(semantic_evidence.get("accepted_count", 0) or 0) > 0
    ):
        return False

    requirements = [str(match.get("requirement", "")).strip() for match in matches]
    requirements = [value for value in requirements if value]
    confidence = cast(ScoringConfidence, dict(result["confidence"]))
    confidence["active_requirement_count"] = len(requirements)
    confidence["coverage_score"] = 0
    confidence["observed_score"] = 0
    confidence["reasons"] = [
        "No resume statement passed the evidence threshold for the extracted requirements."
    ]
    if len(requirements) < 4:
        confidence["level"] = "low"
        if quality.get("explicit_full_source"):
            quality.update(
                {
                    "label": "requirements_missing",
                    "display_label": "Requirements unclear",
                    "reliable_scoring_ready": False,
                    "provisional_scoring_ready": False,
                    "next_action": "Review the extracted requirements before using the fit score.",
                }
            )
        confidence["job_description_quality"] = quality

    calibration: ScoreCalibration = {
        "applied": int(result["score"]) != 0,
        "observed_score": 0,
        "calibrated_score": 0,
        "active_requirement_count": len(requirements),
        "role_signal_count": 0,
        "effective_evidence_count": len(requirements),
        "evidence_target": 4,
        "evidence_factor": min(1.0, len(requirements) / 4.0),
        "reason": "The score was reset because every extracted requirement lacked accepted resume evidence.",
    }
    confidence["score_calibration"] = calibration
    result["score"] = 0
    result["coverage_score"] = 0
    result["observed_score"] = 0
    result["score_calibration"] = calibration
    result["confidence"] = confidence
    result["recommendation"] = final_recommendation(
        0, result["eligibility"], confidence
    )
    result["score_breakdown"] = [
        {
            "category": "Requirement evidence coverage",
            "earned": 0.0,
            "possible": 100,
            "active_terms": requirements,
            "matched": [],
            "partial": [],
            "missing": requirements,
            "note": "Every extracted requirement lacked accepted resume evidence.",
        }
    ]
    return True


def format_reason_messages(reasons: object) -> str:
    """Format structured eligibility reasons without exposing implementation detail."""
    if not isinstance(reasons, list) or not reasons:
        return "None"
    return "; ".join(
        str(reason.get("message", "")) for reason in reasons if isinstance(reason, dict)
    )


def format_inline_list(items: object) -> str:
    """Format a list for use inside a sentence."""
    if not isinstance(items, list) or not items:
        return "None"
    return ", ".join(str(item) for item in items)


def format_score_breakdown(score_breakdown: list[ScoreBreakdownItem]) -> str:
    """Format weighted category scores for the report."""
    lines = []
    for item in score_breakdown:
        if item["earned"] is None:
            lines.append(f"- {item['category']}: **N/A**")
            lines.append(f"  - {item['note']}")
            continue
        lines.append(f"- {item['category']}: **{item['earned']}/{item['possible']}**")
        lines.append(f"  - JD terms scored: {format_inline_list(item['active_terms'])}")
        lines.append(f"  - Matched: {format_inline_list(item['matched'])}")
        lines.append(f"  - Partial / adjacent: {format_inline_list(item['partial'])}")
        lines.append(
            f"  - Missing required or preferred terms: {format_inline_list(item['missing'])}"
        )
        lines.append(f"  - Note: {item['note']}")
    return "\n".join(lines)


def format_penalties(penalties: list[Penalty]) -> str:
    """Format score penalties for the report."""
    if not penalties:
        return "- None found"
    return "\n".join(f"- -{item['points']}: {item['name']}" for item in penalties)


def build_markdown_report(
    job_description_path: Path,
    resume_source_path: Path,
    parsed_job: ParsedJob,
    matched_skills: list[str],
    partial_matches: list[str],
    missing_skills: list[str],
    themes: list[str],
    score_breakdown: list[ScoreBreakdownItem],
    penalties: list[Penalty],
    red_flags: list[str],
    resume_evidence: list[str],
    score: int,
    coverage_score: int,
    role_alignment: RoleAlignment,
    recommendation: str,
    eligibility: EligibilityResult,
    confidence: ScoringConfidence,
) -> str:
    """Build the Markdown report saved for human review."""
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    return "\n".join(
        [
            "# Job Match Analysis",
            "",
            f"- Job description file: `{job_description_path}`",
            f"- Candidate source: `{resume_source_path}`",
            f"- Generated at: {generated_at}",
            "",
            "## Summary",
            "",
            f"- Role Fit Score: **{score}/100**",
            f"- Observed Requirement Coverage: **{coverage_score}%**",
            (
                f"- Role Focus Alignment: **{role_alignment.get('focus', 'Not detected')} — "
                f"{'Supported' if role_alignment.get('score') == 100 else 'Candidate evidence not found'}**"
                if role_alignment.get("detected")
                else "- Role Focus Alignment: **Not detected from saved title**"
            ),
            f"- Eligibility: **{str(eligibility['status']).replace('_', ' ').title()}**",
            f"- Scoring Confidence: **{str(confidence['level']).title()}**",
            f"- Recommendation: **{recommendation}**",
            f"- Why: {explain_final_decision(score, recommendation, eligibility, confidence)}",
            f"- Eligibility reasons: {format_reason_messages(eligibility.get('reasons', []))}",
            f"- Confidence reasons: {format_inline_list(confidence.get('reasons', []))}",
            "",
            "## Parsed Job Requirements",
            "",
            f"- Required skills: {format_inline_list(parsed_job['required_skills'])}",
            f"- Preferred / plus skills: {format_inline_list(parsed_job['preferred_skills'])}",
            f"- Experience level: {format_inline_list(parsed_job['experience_level'])}",
            f"- Degree requirements: {format_inline_list(parsed_job['degree_requirements'])}",
            f"- Domain keywords: {format_inline_list(parsed_job['domain_keywords'])}",
            "",
            "## Score Breakdown",
            "",
            format_score_breakdown(score_breakdown),
            "",
            "## Penalties",
            "",
            format_penalties(penalties),
            "",
            "## Matched Skills",
            "",
            format_bullets(matched_skills),
            "",
            "## Partial / Adjacent Matches",
            "",
            format_bullets(partial_matches),
            "",
            "## Missing Skills",
            "",
            format_bullets(missing_skills),
            "",
            "## Relevant Experience Themes",
            "",
            format_bullets(themes),
            "",
            "## Red Flags",
            "",
            format_bullets(red_flags),
            "",
            "## Relevant Resume Evidence",
            "",
            format_bullets(resume_evidence),
            "",
            "## Human Review Notes",
            "",
            "- This report uses weighted keyword matching and simple penalty rules.",
            "- Required and preferred JD terms use symmetric requirement weights; preferred gaps have less impact.",
            "- Categories the JD does not mention are marked N/A and are not counted against the final score.",
            "- High coverage from sparse or truncated evidence is calibrated toward a neutral score before recommendation.",
            "- Eligibility and scoring confidence are separate from the role-fit score and may override the recommendation.",
            "- It should be reviewed by a person before preparing application materials.",
            "- It does not invent experience, skills, degree level, metrics, visa status, or work authorization.",
            "- Confirm the resume source's degree level before relying on education-related statements.",
            "- It does not submit applications or interact with job platforms.",
            "",
        ]
    )


def analyze_job_structured(
    job_text: str, resume_text: str, raw_analysis: str = ""
) -> StructuredAnalysis:
    """Return structured, dependency-light fit analysis for UI display."""
    result = score_job_texts(job_text, resume_text)
    legacy_score = result["score"]
    structured_job = structure_job_description(job_text)
    structured_records = requirement_records(structured_job)
    semantic_limit = max(1, len(structured_records))
    semantic_evidence = build_semantic_evidence_index(
        job_text,
        resume_text,
        max_requirements=semantic_limit,
        requirement_records=structured_records or None,
    )
    scoring_method = "deterministic_keyword_fallback"
    legacy_requirement_count = int(result["confidence"].get("active_requirement_count", 0) or 0)
    # Controlled migration: fill the legacy parser's blind spots without changing
    # already-recognized product results. The audit fields keep both paths visible.
    legacy_quality = dict(result["confidence"].get("job_description_quality", {}) or {})
    structured_takeover = len(structured_records) >= 4 and (
        legacy_requirement_count <= 2 or legacy_quality.get("explicit_full_source")
    )
    if structured_takeover:
        promoted = _structured_requirement_score(structured_job, semantic_evidence, result)
        result["score"] = promoted["score"]
        result["coverage_score"] = promoted["coverage_score"]
        result["observed_score"] = promoted["observed_score"]
        result["score_calibration"] = promoted["score_calibration"]
        result["role_focus_adjustment"] = promoted["role_focus_adjustment"]
        result["confidence"] = promoted["confidence"]
        result["recommendation"] = promoted["recommendation"]
        result["score_breakdown"] = promoted["score_breakdown"]
        scoring_method = "structured_requirement_evidence_v1"
    elif _enforce_evidence_consistency(result, semantic_evidence):
        scoring_method = "semantic_evidence_consistency_guard_v1"
    job_keywords = result["job_keywords"]
    parsed_job = result["parsed_job"]
    score_breakdown = result["score_breakdown"]
    matched_keywords, partial_matches, missing_keywords = collect_report_matches(score_breakdown)
    red_flags = list(parsed_job["red_flags"])
    score = result["score"]
    recommendation = result["recommendation"]
    main_reason = explain_final_decision(
        score, recommendation, result["eligibility"], result["confidence"]
    )

    matched_strengths = [
        (
            f"{match['requirement']} — {match['evidence']} "
            f"({float(match['similarity']):.0%}, {match['match_type']})."
        )
        for match in semantic_evidence["accepted_matches"][:3]
    ]
    matched_strengths.extend(
        f"Resume source supports requested keyword: {keyword}." for keyword in matched_keywords[:6]
    )
    if partial_matches and len(matched_strengths) < 6:
        matched_strengths.extend(
            f"Adjacent evidence may support: {keyword}."
            for keyword in partial_matches[: 6 - len(matched_strengths)]
        )
    if not matched_strengths:
        matched_strengths.append("No strong keyword overlap was detected; review manually.")

    weak_areas = [
        f"Missing or unclear evidence for: {keyword}." for keyword in missing_keywords[:6]
    ]
    role_alignment = result["role_alignment"]
    if role_alignment.get("detected"):
        focus = str(role_alignment.get("focus", "the title's core domain"))
        if role_alignment.get("score") == 100:
            matched_strengths.insert(
                0, f"Candidate source supports the title's core role focus: {focus}."
            )
        else:
            weak_areas.insert(
                0,
                f"Candidate source does not clearly support the title's core role focus: {focus}.",
            )
    if red_flags and len(weak_areas) < 6:
        weak_areas.extend(red_flags[: 6 - len(weak_areas)])
    if not weak_areas:
        weak_areas.append("No major weak areas were detected by the lightweight analyzer.")

    resume_evidence = list(
        dict.fromkeys(str(match["evidence"]) for match in semantic_evidence["accepted_matches"])
    )
    return {
        "score": score,
        "coverage_score": result["coverage_score"],
        "observed_score": result["observed_score"],
        "score_calibration": result["score_calibration"],
        "role_alignment": result["role_alignment"],
        "role_focus_adjustment": result["role_focus_adjustment"],
        "recommendation": recommendation,
        "score_breakdown": score_breakdown,
        "eligibility": result["eligibility"],
        "confidence": result["confidence"],
        "candidate_profile": result["candidate_profile"],
        "parsed_job": parsed_job,
        "job_keywords": result["job_keywords"],
        "resume_keywords": result["resume_keywords"],
        "penalties": result["penalties"],
        "matched_skills": matched_keywords,
        "partial_matches": partial_matches,
        "missing_skills": missing_keywords,
        "main_reason": main_reason,
        "main_risk": red_flags[0]
        if red_flags
        else (weak_areas[0] if weak_areas else "No major risk detected."),
        "matched_strengths": matched_strengths[:6],
        "weak_areas": weak_areas[:6],
        "matched_keywords": matched_keywords,
        "missing_keywords": missing_keywords,
        "optional_keywords": list(parsed_job["preferred_skills"]),
        "resume_suggestions": resume_suggestions_for_keywords(
            matched_keywords, missing_keywords, red_flags
        ),
        "jd_evidence": short_evidence_snippets(job_text, matched_keywords or job_keywords),
        "profile_evidence": resume_evidence[:3],
        "semantic_evidence": semantic_evidence,
        "jd_quality": dict(result["confidence"].get("job_description_quality", {})),
        "raw_analysis": raw_analysis,
        "scoring_method": scoring_method,
        "legacy_score": legacy_score,
        "structured_job": structured_job,
        "jd_pipeline": pipeline_trace(job_text, structured_job, semantic_evidence),
    }


def save_report(
    job_description_path: Path,
    report: str,
    workspace: Workspace,
    package_dir: Path | None = None,
) -> Path:
    """Save a Markdown report in a structured generated application folder."""
    if package_dir is None:
        package_dir = application_package_dir(workspace.generated_dir, job_description_path.stem)
    package_dir.mkdir(parents=True, exist_ok=True)
    report_path = package_dir / "analysis.md"
    report_path.write_text(report, encoding="utf-8")
    return report_path


def analyze_job(
    job_description_path: Path,
    workspace: Workspace,
    package_dir: Path | None = None,
) -> tuple[str, Path]:
    """Run the analysis and return the report text plus saved report path."""
    workspace.require_writable()
    assert workspace.resume_source_path is not None
    resume_text = workspace.resume_source_path.read_text(encoding="utf-8")
    job_text = job_description_path.read_text(encoding="utf-8")

    result = score_job_texts(job_text, resume_text)
    themes = choose_relevant_themes(result["job_keywords"], result["resume_keywords"])
    score_breakdown = result["score_breakdown"]
    matched_skills, partial_matches, missing_skills = collect_report_matches(score_breakdown)
    parsed_job = result["parsed_job"]
    red_flags = parsed_job["red_flags"]
    semantic_evidence = build_semantic_evidence_index(job_text, resume_text)
    resume_evidence = [
        (
            f"{match['requirement']} => {match['evidence']} "
            f"({float(match['similarity']):.0%}, {match['match_type']})"
        )
        for match in semantic_evidence["accepted_matches"]
    ]
    resume_evidence.extend(
        f"No accepted resume evidence for: {requirement}"
        for requirement in semantic_evidence["unmatched_requirements"]
    )

    report = build_markdown_report(
        job_description_path=job_description_path,
        resume_source_path=workspace.resume_source_path,
        parsed_job=parsed_job,
        matched_skills=matched_skills,
        partial_matches=partial_matches,
        missing_skills=missing_skills,
        themes=themes,
        score_breakdown=score_breakdown,
        penalties=result["penalties"],
        red_flags=red_flags,
        resume_evidence=resume_evidence,
        score=result["score"],
        coverage_score=result["coverage_score"],
        role_alignment=result["role_alignment"],
        recommendation=result["recommendation"],
        eligibility=result["eligibility"],
        confidence=result["confidence"],
    )
    report_path = save_report(job_description_path, report, workspace, package_dir)
    return report, report_path
