"""Progressive-disclosure Fit evidence renderer for Review Jobs."""

from __future__ import annotations

import html
from typing import Any, Callable

import streamlit as st

from dashboard_analysis_details import render_analysis_action, render_analysis_inspector
from dashboard_evidence_map import render_evidence_map
from dashboard_evidence_styles import render_evidence_map_styles
from dashboard_fit import apply_canonical_analysis, build_fit_presentation
from scoring_types import DashboardJob


Sanitizer = Callable[[Any], str]


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
    render_evidence_map_styles()
    risk = str(analysis.get("main_risk", "") or "").strip()
    weak_areas = [str(item).strip() for item in analysis.get("weak_areas", []) if str(item).strip()]
    gap = html.escape(sanitize(risk or (weak_areas[0] if weak_areas else "No major gap detected.")))
    st.markdown(
        '<div class="evidence-gap-row">'
        '<i aria-hidden="true">warning</i>'
        '<strong>Other key gap</strong>'
        f"<span>{gap}</span>"
        "</div>",
        unsafe_allow_html=True,
    )


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
    def analysis_action() -> None:
        render_analysis_action(
            analysis,
            presentation,
            sanitize,
            demo_mode=demo_mode,
            show_debug=show_debug,
        )

    with st.container(border=False, key="evidence_workspace"):
        evidence_column, inspector_column = st.columns(
            [0.7, 0.3], vertical_alignment="top", gap="medium"
        )
        with evidence_column, st.container(border=False, key="evidence_main_column"):
            with st.container(border=False, key="evidence_main_scroll"):
                if not render_evidence_map(analysis, sanitize, analysis_action):
                    render_top_evidence(analysis, sanitize)
                render_main_gap(analysis, sanitize)
        with inspector_column, st.container(border=False, key="evidence_inspector_column"):
            render_analysis_inspector(
                analysis,
                presentation,
                sanitize,
                demo_mode=demo_mode,
                show_debug=show_debug,
            )
