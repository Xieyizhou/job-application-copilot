"""Structured and Markdown reporting for deterministic job-fit analysis."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from ml.evidence import build_semantic_evidence_index
from output_paths import application_package_dir
from scoring_engine import explain_final_decision, score_job_texts
from scoring_matching import (
    choose_relevant_themes,
    collect_report_matches,
    resume_suggestions_for_keywords,
    short_evidence_snippets,
)
from workspace import Workspace


def find_resume_evidence(themes: list[str]) -> list[str]:
    """Return resume-backed evidence bullets for the selected themes."""
    return [f"Candidate source contains keywords related to {theme}." for theme in themes]


def format_bullets(items: list[str]) -> str:
    """Format a list as Markdown bullets, or show a placeholder if empty."""
    if not items:
        return "- None found"
    return "\n".join(f"- {item}" for item in items)


def format_reason_messages(reasons: object) -> str:
    """Format structured eligibility reasons without exposing implementation detail."""
    if not isinstance(reasons, list) or not reasons:
        return "None"
    return "; ".join(str(reason.get("message", "")) for reason in reasons if isinstance(reason, dict))


def format_inline_list(items: object) -> str:
    """Format a list for use inside a sentence."""
    if not isinstance(items, list) or not items:
        return "None"
    return ", ".join(str(item) for item in items)


def format_score_breakdown(score_breakdown: list[dict[str, object]]) -> str:
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
        lines.append(f"  - Missing required or preferred terms: {format_inline_list(item['missing'])}")
        lines.append(f"  - Note: {item['note']}")
    return "\n".join(lines)


def format_penalties(penalties: list[dict[str, object]]) -> str:
    """Format score penalties for the report."""
    if not penalties:
        return "- None found"
    return "\n".join(f"- -{item['points']}: {item['name']}" for item in penalties)


def build_markdown_report(
    job_description_path: Path,
    resume_source_path: Path,
    parsed_job: dict[str, object],
    matched_skills: list[str],
    partial_matches: list[str],
    missing_skills: list[str],
    themes: list[str],
    score_breakdown: list[dict[str, object]],
    penalties: list[dict[str, object]],
    red_flags: list[str],
    resume_evidence: list[str],
    score: int,
    coverage_score: int,
    role_alignment: dict[str, object],
    recommendation: str,
    eligibility: dict[str, object],
    confidence: dict[str, object],
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


def explain_overall_score(
    score: int,
    recommendation: str,
    penalties: list[dict[str, object]],
    red_flags: list[str],
) -> str:
    """Write a short plain-English explanation for the final score."""
    explanation = (
        f"{recommendation} because the score is {score}/100 after normalizing only "
        "the categories the job description actually mentions."
    )
    if penalties:
        explanation += " Penalties were applied for seniority, experience, or degree requirements."
    if red_flags:
        explanation += " Red flags need human review before applying."
    return explanation


def analyze_job_structured(job_text: str, resume_text: str, raw_analysis: str = "") -> dict[str, object]:
    """Return structured, dependency-light fit analysis for UI display."""
    result = score_job_texts(job_text, resume_text)
    job_keywords = result["job_keywords"]
    resume_keywords = result["resume_keywords"]
    parsed_job = result["parsed_job"]
    themes = choose_relevant_themes(job_keywords, resume_keywords)
    score_breakdown = result["score_breakdown"]
    matched_keywords, partial_matches, missing_keywords = collect_report_matches(score_breakdown)
    red_flags = list(parsed_job["red_flags"])
    score = result["score"]
    recommendation = result["recommendation"]
    main_reason = explain_final_decision(score, recommendation, result["eligibility"], result["confidence"])
    semantic_evidence = build_semantic_evidence_index(job_text, resume_text)

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

    weak_areas = [f"Missing or unclear evidence for: {keyword}." for keyword in missing_keywords[:6]]
    role_alignment = dict(result.get("role_alignment", {}) or {})
    if role_alignment.get("detected"):
        focus = str(role_alignment.get("focus", "the title's core domain"))
        if role_alignment.get("score") == 100:
            matched_strengths.insert(0, f"Candidate source supports the title's core role focus: {focus}.")
        else:
            weak_areas.insert(0, f"Candidate source does not clearly support the title's core role focus: {focus}.")
    if red_flags and len(weak_areas) < 6:
        weak_areas.extend(red_flags[: 6 - len(weak_areas)])
    if not weak_areas:
        weak_areas.append("No major weak areas were detected by the lightweight analyzer.")

    resume_evidence = list(
        dict.fromkeys(
            str(match["evidence"])
            for match in semantic_evidence["accepted_matches"]
        )
    )
    return {
        "score": score,
        "coverage_score": result["coverage_score"],
        "score_calibration": result["score_calibration"],
        "role_alignment": result["role_alignment"],
        "recommendation": recommendation,
        "score_breakdown": score_breakdown,
        "eligibility": result["eligibility"],
        "confidence": result["confidence"],
        "candidate_profile": result["candidate_profile"],
        "parsed_job": parsed_job,
        "matched_skills": matched_keywords,
        "partial_matches": partial_matches,
        "missing_skills": missing_keywords,
        "main_reason": main_reason,
        "main_risk": red_flags[0] if red_flags else (weak_areas[0] if weak_areas else "No major risk detected."),
        "matched_strengths": matched_strengths[:6],
        "weak_areas": weak_areas[:6],
        "matched_keywords": matched_keywords,
        "missing_keywords": missing_keywords,
        "optional_keywords": list(parsed_job["preferred_skills"]),
        "resume_suggestions": resume_suggestions_for_keywords(matched_keywords, missing_keywords, red_flags),
        "jd_evidence": short_evidence_snippets(job_text, matched_keywords or job_keywords),
        "profile_evidence": resume_evidence[:3],
        "semantic_evidence": semantic_evidence,
        "jd_quality": dict(result["confidence"].get("job_description_quality", {})),
        "raw_analysis": raw_analysis,
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
