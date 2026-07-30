"""Escaped HTML and scoped styles for compact Review Jobs components."""

from __future__ import annotations

import html

import streamlit as st

def decision_field_html(label: str, value: object) -> str:
    """Return one escaped field for the compact decision grid."""
    return (
        '<div class="review-decision-field">'
        f'<div class="review-decision-label">{html.escape(label)}</div>'
        f'<div class="review-decision-value">{html.escape(str(value))}</div>'
        "</div>"
    )

def action_note_html(message: str, *, caution: bool) -> str:
    """Return one escaped next-action line without a full alert box."""
    class_name = " review-action-note-caution" if caution else ""
    return (
        f'<div class="review-action-note{class_name}">'
        f"<strong>Next:</strong> {html.escape(message)}"
        "</div>"
    )


def render_review_component_styles() -> None:
    """Install the small Review Jobs visual vocabulary once per render."""
    st.markdown(
        """
        <style>
        .review-decision-grid {
            display:grid;
            grid-template-columns:repeat(4,minmax(0,1fr));
            border-top:1px solid color-mix(in srgb,var(--text-color) 16%,transparent);
            border-bottom:1px solid color-mix(in srgb,var(--text-color) 16%,transparent);
            margin:.45rem 0 .5rem;
        }
        .review-decision-field {
            padding:.5rem .7rem;
            min-width:0;
        }
        .review-decision-field + .review-decision-field {
            border-left:1px solid color-mix(in srgb,var(--text-color) 12%,transparent);
        }
        .review-decision-label {color:var(--text-color);opacity:.62;font-size:.72rem;line-height:1.2}
        .review-decision-value {font-size:.94rem;font-weight:650;line-height:1.3;overflow-wrap:anywhere}
        .review-action-note {
            color:var(--text-color);
            font-size:.88rem;
            line-height:1.35;
            margin:.35rem 0 .55rem;
            opacity:.74;
        }
        .review-action-note-caution {
            border-left:3px solid #d97706;
            padding:.3rem .55rem;
            opacity:.92;
        }
        .review-evidence-row {
            display:grid;grid-template-columns:minmax(0,1fr) auto;
            gap:.75rem;align-items:start;padding:.45rem 0;
            border-bottom:1px solid color-mix(in srgb,var(--text-color) 10%,transparent);
            font-size:.9rem;line-height:1.35;
        }
        .review-evidence-row span {
            color:var(--text-color);opacity:.58;font-size:.74rem;white-space:nowrap;
        }
        @media (max-width: 760px) {
            .review-decision-grid {grid-template-columns:repeat(2,minmax(0,1fr))}
            .review-decision-field:nth-child(3) {border-left:0}
            .review-decision-field:nth-child(n+3) {
                border-top:1px solid color-mix(in srgb,var(--text-color) 12%,transparent);
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
