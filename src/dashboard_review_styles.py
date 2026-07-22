"""Escaped HTML and scoped styles for compact Review Jobs components."""

from __future__ import annotations

import html

import streamlit as st


def badge_html(label: str, value: str, tone: str = "neutral") -> str:
    """Return one escaped, compact status badge."""
    safe_tone = tone if tone in {"positive", "warning", "neutral"} else "neutral"
    return (
        f'<span class="review-badge review-badge-{safe_tone}">'
        f"{html.escape(label)}: {html.escape(value)}</span>"
    )


def decision_field_html(label: str, value: object) -> str:
    """Return one escaped field for the compact decision grid."""
    return (
        '<div class="review-decision-field">'
        f'<div class="review-decision-label">{html.escape(label)}</div>'
        f'<div class="review-decision-value">{html.escape(str(value))}</div>'
        "</div>"
    )


def render_review_component_styles() -> None:
    """Install the small Review Jobs visual vocabulary once per render."""
    st.markdown(
        """
        <style>
        .review-card-company {font-size:.82rem;font-weight:700;line-height:1.2}
        .review-card-role {font-size:.96rem;font-weight:600;line-height:1.3;margin-bottom:.35rem}
        .review-card-result {font-size:.86rem;line-height:1.3;margin-bottom:.4rem}
        .review-badge-row {display:flex;gap:.4rem;flex-wrap:wrap;margin:.2rem 0 .45rem}
        .review-badge {border-radius:999px;padding:.12rem .48rem;font-size:.72rem;font-weight:600}
        .review-badge-positive {background:#e8f5ec;color:#27633b}
        .review-badge-warning {background:#fff3d6;color:#7a5700}
        .review-badge-neutral {background:#eef0f4;color:#4b5260}
        .review-decision-grid {display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:.55rem;margin:.6rem 0}
        .review-decision-field {border:1px solid #e4e7ec;border-radius:.55rem;padding:.5rem .65rem;min-width:0}
        .review-decision-label {color:#69707d;font-size:.72rem;line-height:1.2}
        .review-decision-value {font-size:.94rem;font-weight:650;line-height:1.3;overflow-wrap:anywhere}
        </style>
        """,
        unsafe_allow_html=True,
    )
