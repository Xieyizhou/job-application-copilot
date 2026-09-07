"""Non-blocking analysis inspector for the Evidence workspace."""

from __future__ import annotations

import html
from typing import Any, Callable

import streamlit as st

from scoring_types import FitPresentation

from dashboard_evidence_map import build_evidence_cards, evidence_counts


Sanitizer = Callable[[Any], str]


def _render_requirement_details(terms: dict[str, Any], sanitize: Sanitizer) -> None:
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
    rendered = 0
    for label, values in rows:
        cleaned = [sanitize(value) for value in values if str(value).strip()]
        if not cleaned:
            continue
        rendered += 1
        st.markdown(
            '<div class="analysis-detail-group">'
            f"<strong>{html.escape(label)}</strong>"
            f"<span>{html.escape(', '.join(cleaned))}</span>"
            "</div>",
            unsafe_allow_html=True,
        )
    if not rendered:
        st.caption("No additional parser-level requirement groups were available.")


def _render_items(items: list[str], sanitize: Sanitizer, prefix: str = "") -> None:
    for item in items:
        st.markdown(
            '<div class="analysis-source-item">• '
            f"{html.escape(prefix)}{html.escape(sanitize(item))}</div>",
            unsafe_allow_html=True,
        )


def _render_coverage(analysis: dict[str, Any]) -> None:
    cards = build_evidence_cards(analysis)
    counts = evidence_counts(cards)
    rows = (
        ("Total requirements", len(cards), ""),
        ("Directly evidenced", counts["Direct"], "direct"),
        ("Partially evidenced", counts["Partial"], "partial"),
        ("Missing", counts["No Support"], "missing"),
    )
    st.markdown('<div class="analysis-drawer-section">Coverage</div>', unsafe_allow_html=True)
    for label, value, tone in rows:
        share = f" ({round(value / len(cards) * 100)}%)" if cards and tone else ""
        st.markdown(
            '<div class="analysis-drawer-metric">'
            f"<span>{html.escape(label)}</span>"
            f'<strong class="analysis-drawer-{tone or "total"}">{value}{share}</strong>'
            "</div>",
            unsafe_allow_html=True,
        )


def _render_requirement_signals(analysis: dict[str, Any], sanitize: Sanitizer) -> None:
    cards = build_evidence_cards(analysis)
    if not cards:
        return
    st.markdown('<div class="analysis-drawer-section">Requirement signals</div>', unsafe_allow_html=True)
    for card in cards[:8]:
        tone = card["status"].lower().replace(" ", "-")
        st.markdown(
            '<div class="analysis-drawer-signal">'
            '<div><strong>'
            f'{html.escape(sanitize(card["requirement"]))}</strong>'
            f'<span>{html.escape(sanitize(card["context"]))}</span></div>'
            f'<em class="analysis-drawer-{tone}">{html.escape(card["status"])}</em>'
            "</div>",
            unsafe_allow_html=True,
        )


def _render_analysis_content(
    analysis: dict[str, Any],
    presentation: FitPresentation,
    sanitize: Sanitizer,
    *,
    demo_mode: bool,
    show_debug: bool,
) -> None:
    """Render compact, safely escaped diagnostics inside the inspector."""
    terms = dict(presentation.get("terms", {}) or {})
    _render_coverage(analysis)
    _render_requirement_signals(analysis, sanitize)

    st.markdown('<div class="analysis-drawer-section">Requirement groups</div>', unsafe_allow_html=True)
    _render_requirement_details(terms, sanitize)

    suggestions = [str(item).strip() for item in analysis.get("resume_suggestions", []) if str(item).strip()]
    if suggestions:
        st.markdown('<div class="analysis-drawer-section">Tailoring suggestions</div>', unsafe_allow_html=True)
        _render_items(suggestions[:5], sanitize)

    jd_evidence = [str(item) for item in list(analysis.get("jd_evidence", []) or [])]
    profile_evidence = [str(item) for item in list(analysis.get("profile_evidence", []) or [])]
    if jd_evidence or profile_evidence:
        st.markdown('<div class="analysis-drawer-section">Source evidence</div>', unsafe_allow_html=True)
        _render_items(jd_evidence[:3], sanitize)
        profile_label = "Demo profile: " if demo_mode else "Candidate profile: "
        _render_items(profile_evidence[:3], sanitize, profile_label)

    if show_debug and analysis.get("raw_analysis"):
        with st.expander("Full analysis report", expanded=False):
            st.markdown(sanitize(analysis["raw_analysis"]))


@st.dialog("Analysis details", width="large")
def _show_analysis_dialog(
    analysis: dict[str, Any],
    presentation: FitPresentation,
    sanitize: Sanitizer,
    *,
    demo_mode: bool,
    show_debug: bool,
) -> None:
    _render_analysis_content(
        analysis,
        presentation,
        sanitize,
        demo_mode=demo_mode,
        show_debug=show_debug,
    )


def render_analysis_action(
    analysis: dict[str, Any],
    presentation: FitPresentation,
    sanitize: Sanitizer,
    *,
    demo_mode: bool,
    show_debug: bool,
) -> None:
    """Open diagnostics as an overlay instead of extending the page."""
    if st.button(
        "View analysis",
        key="evidence_view_analysis",
        icon=":material/info:",
        type="tertiary",
    ):
        _show_analysis_dialog(
            analysis,
            presentation,
            sanitize,
            demo_mode=demo_mode,
            show_debug=show_debug,
        )


def render_analysis_inspector(
    analysis: dict[str, Any],
    presentation: FitPresentation,
    sanitize: Sanitizer,
    *,
    demo_mode: bool,
    show_debug: bool,
) -> None:
    """Render the same diagnostics as a compact, persistent desktop inspector."""
    with st.container(height=430, border=False, key="evidence_analysis_inspector"):
        st.markdown('<div class="analysis-inspector-title">Analysis</div>', unsafe_allow_html=True)
        _render_analysis_content(
            analysis,
            presentation,
            sanitize,
            demo_mode=demo_mode,
            show_debug=show_debug,
        )
