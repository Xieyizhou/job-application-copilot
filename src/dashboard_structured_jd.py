"""Standardized, source-backed presentation for saved job descriptions."""

from __future__ import annotations

import html

import streamlit as st

from structured_jd import StructuredJob


def render_structured_job(structured: StructuredJob, original_body: str) -> None:
    """Render available canonical fields and keep original employer text accessible."""
    facts = [
        ("Location", structured["location"]),
        ("Work mode", structured["work_mode"]),
        ("Employment", structured["employment_type"]),
        ("Salary", structured["salary"]),
    ]
    available = [(label, value) for label, value in facts if value and value.lower() != "not provided"]
    if available:
        st.markdown(
            '<div class="jd-fact-strip">'
            + "".join(
                f'<div><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong></div>'
                for label, value in available
            )
            + "</div>",
            unsafe_allow_html=True,
        )
    if structured["overview"]:
        st.markdown(
            f'<div class="jd-overview">{html.escape(structured["overview"])}</div>',
            unsafe_allow_html=True,
        )
    sections = [
        ("Responsibilities", structured["responsibilities"]),
        ("Required qualifications", structured["required_qualifications"]),
        ("Preferred qualifications", structured["preferred_qualifications"]),
        ("Experience", structured["experience"]),
        ("Education", structured["education"]),
        ("Eligibility", structured["eligibility"]),
        ("Benefits", structured["benefits"]),
    ]
    rendered = False
    for label, items in sections:
        unique = list(dict.fromkeys(items))
        if not unique:
            continue
        rendered = True
        st.markdown(f"### {label}")
        st.markdown("\n".join(f"- {item}" for item in unique[:12]))
    with st.container(border=False, key="jd_document_body"):
        if not rendered:
            st.markdown(original_body)
        else:
            with st.expander("View original posting text", expanded=False):
                st.markdown(original_body)
