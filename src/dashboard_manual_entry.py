"""Linear job-description capture and verification components."""

from __future__ import annotations

from typing import Any, Callable

import streamlit as st

from manual_jobs import (
    SOURCE_OPTIONS,
    STATUS_OPTIONS,
    job_description_quality_warnings,
    normalize_job_title,
)
from ml.jd_quality import classify_jd_quality


def render_jd_capture(upload_renderer: Callable[[], list[Any]]) -> list[Any]:
    """Render the JD source first and keep uploaded files for persistence."""
    st.markdown("**1. Add the full job description**")
    st.caption("Paste the posting directly, or extract it from screenshots, PDF, TXT, or Markdown files.")
    uploaded_files = upload_renderer()
    st.text_area(
        "Full job description",
        height=340,
        key="manual_job_description",
        placeholder="Paste the complete responsibilities and requirements here.",
    )
    return uploaded_files


def render_jd_quality(job_description: str) -> dict[str, Any]:
    """Show whether the current JD supports a trustworthy fit result."""
    quality = classify_jd_quality(job_description)
    if not job_description.strip():
        st.caption("JD quality will appear after text is added.")
        return quality

    message = (
        f"JD quality: {quality['display_label']} · {quality['word_count']} words · "
        f"{quality['requirement_statement_count']} requirement statements"
    )
    if quality["reliable_scoring_ready"]:
        st.success(message)
        st.caption("The posting has enough role-specific evidence for a reliable fit review.")
    else:
        st.warning(message)
        st.caption(f"Fit results remain provisional. Next: {quality['next_action']}")
    return quality


def render_verification_form(
    *,
    suggestions: dict[str, Any],
    job_description: str,
) -> dict[str, Any]:
    """Render required identity fields before optional application metadata."""
    st.markdown("**2. Verify the job details**")
    st.caption("Confirm inferred values against the employer posting before saving.")
    with st.form("manual_job_form"):
        company_col, title_col, location_col = st.columns(3)
        with company_col:
            company = st.text_input("Company name", key="manual_company")
        with title_col:
            title = st.text_input("Job title", key="manual_title")
        with location_col:
            location = st.text_input("Location", key="manual_location")

        url_col, source_col = st.columns([0.65, 0.35])
        with url_col:
            url = st.text_input("Official job URL", key="manual_url")
        with source_col:
            source = st.selectbox("Job source", SOURCE_OPTIONS, key="manual_source")

        with st.expander("More details", expanded=False):
            optional_left, optional_right = st.columns(2)
            with optional_left:
                salary_range = st.text_input("Salary range", key="manual_salary_range")
                visa_note = st.text_input("Work authorization / visa note", key="manual_visa_note")
            with optional_right:
                status = st.selectbox("Status", STATUS_OPTIONS, key="manual_status")
                notes = st.text_area("Notes", height=100, key="manual_notes")

        normalized_title = normalize_job_title(title)
        warnings = job_description_quality_warnings(
            company=company,
            title=normalized_title,
            location=location,
            url=url,
            job_description=job_description,
        )
        if warnings:
            with st.expander("Fields to verify", expanded=False):
                for warning in warnings:
                    st.warning(warning)
        submitted = st.form_submit_button("Save Target Job", type="primary", width="stretch")

    return {
        "submitted": submitted,
        "company": company,
        "title": normalized_title,
        "location": location,
        "source": source,
        "url": url,
        "salary_range": salary_range,
        "visa_note": visa_note,
        "status": status,
        "notes": notes,
        "job_description": job_description,
        "suggestions": suggestions,
    }
