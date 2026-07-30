"""Progressive-disclosure Fit evidence renderer for Review Jobs."""

from __future__ import annotations

import html
from typing import Any, Callable

import streamlit as st

from dashboard_fit import apply_canonical_analysis, build_fit_presentation
from scoring_types import DashboardJob, FitPresentation


Sanitizer = Callable[[Any], str]


def as_int(value: object) -> int:
    """Convert stored numeric diagnostics without trusting their runtime type."""
    try:
        return int(str(value or 0))
    except ValueError:
        return 0


def accepted_semantic_matches(analysis: dict[str, Any], limit: int = 3) -> list[dict[str, Any]]:
    """Return only accepted requirement matches for the concise Fit view."""
    semantic = dict(analysis.get("semantic_evidence", {}) or {})
    matches = semantic.get("matches", []) or []
    return [match for match in matches if isinstance(match, dict) and match.get("accepted")][:limit]


def render_top_evidence(analysis: dict[str, Any], sanitize: Sanitizer) -> None:
    """Render at most three evidence matches above the fold."""
    matches = accepted_semantic_matches(analysis)
    st.markdown("**Top supporting evidence**")
    if matches:
        for match in matches:
            requirement = sanitize(match.get("requirement", "Requirement"))
            evidence = sanitize(match.get("evidence", ""))
            support = str(match.get("match_type", "Evidence support"))
            st.markdown(
                '<div class="review-evidence-row">'
                f'<div><strong>{html.escape(requirement)}</strong>: '
                f"“{html.escape(evidence)}”</div>"
                f"<span>{html.escape(support)}</span>"
                "</div>",
                unsafe_allow_html=True,
            )
        return
    strengths = [str(item).strip() for item in analysis.get("matched_strengths", []) if str(item).strip()]
    for item in strengths[:3]:
        st.write(f"- {sanitize(item)}")
    if not strengths:
        st.info("No resume statement passed the evidence threshold yet.")


def render_main_gap(analysis: dict[str, Any], sanitize: Sanitizer) -> None:
    """Render one gap or risk after the top evidence."""
    risk = str(analysis.get("main_risk", "") or "").strip()
    weak_areas = [str(item).strip() for item in analysis.get("weak_areas", []) if str(item).strip()]
    st.markdown("**Main gap**")
    st.write(sanitize(risk or (weak_areas[0] if weak_areas else "No major gap detected.")))


def render_requirement_details(terms: dict[str, Any], sanitize: Sanitizer) -> None:
    """Render parser-level term buckets inside Advanced analysis."""
    if not terms.get("active_requirement_count"):
        st.write("Requirements could not be extracted reliably.")
        return
    rows = [
        ("Matched required", terms.get("matched_required", [])),
        ("Matched preferred", terms.get("matched_preferred", [])),
        ("Partial required", terms.get("partial_required", [])),
        ("Partial preferred", terms.get("partial_preferred", [])),
        ("Missing required", terms.get("missing_required", [])),
        ("Missing preferred", terms.get("missing_preferred", [])),
    ]
    for label, values in rows:
        cleaned = [sanitize(value) for value in values if str(value).strip()]
        st.write(f"{label}: {', '.join(cleaned) if cleaned else 'None'}")


def render_advanced_analysis(
    job: DashboardJob | dict[str, Any],
    analysis: dict[str, Any],
    presentation: FitPresentation,
    sanitize: Sanitizer,
    *,
    demo_mode: bool,
    show_debug: bool,
) -> None:
    """Keep requirement and source-level detail collapsed."""
    terms = dict(presentation.get("terms", {}) or {})
    with st.expander("Evidence details", expanded=False):
        coverage = presentation.get("coverage_score")
        st.markdown("**Recognized requirement details**")
        st.caption(
            f"{as_int(terms.get('active_requirement_count', 0))} recognized · "
            + (
                f"{int(coverage)}% observed coverage"
                if coverage is not None
                else "coverage unavailable"
            )
        )
        render_requirement_details(terms, sanitize)

        suggestions = [str(item).strip() for item in analysis.get("resume_suggestions", []) if str(item).strip()]
        if suggestions:
            st.markdown("**Tailoring suggestions**")
            for item in suggestions[:5]:
                st.write(f"- {sanitize(item)}")

        jd_evidence = list(analysis.get("jd_evidence", []) or [])
        profile_evidence = list(analysis.get("profile_evidence", []) or [])
        if jd_evidence or profile_evidence:
            st.markdown("**Source evidence**")
            for item in jd_evidence[:3]:
                st.write(f"- {sanitize(item)}")
            profile_label = "Demo-profile" if demo_mode else "Candidate-profile"
            for item in profile_evidence[:3]:
                st.write(f"- {profile_label}: {sanitize(item)}")

        if show_debug and analysis.get("raw_analysis"):
            with st.expander("Full analysis report", expanded=False):
                st.markdown(sanitize(analysis["raw_analysis"]))


def render_fit_analysis_sections(
    job: DashboardJob,
    job_text: str,
    *,
    analyze: Callable[[DashboardJob, str], dict[str, Any]],
    sanitize: Sanitizer,
    demo_mode: bool,
    show_debug: bool = False,
) -> None:
    """Render concise evidence first and diagnostics after one click."""
    analysis = analyze(job, job_text)
    presentation = build_fit_presentation(apply_canonical_analysis(job, analysis))
    render_top_evidence(analysis, sanitize)
    render_main_gap(analysis, sanitize)
    render_advanced_analysis(
        job,
        analysis,
        presentation,
        sanitize,
        demo_mode=demo_mode,
        show_debug=show_debug,
    )
