"""Company verification controls shared by dashboard workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from company_verification import (
    company_verification_fields,
    confirm_markdown_company,
    dedupe_strings,
    normalize_company_name,
    verification_from_markdown,
    verification_status_label,
)
from manual_jobs import confirm_manual_job_company


def company_generation_allowed(fields: dict[str, Any]) -> bool:
    """Return True when a verified company name may be used for generation."""
    company = str(
        fields.get("company_normalized", "")
        or fields.get("normalized_company", "")
    ).strip()
    confidence = str(
        fields.get("company_confidence", "") or fields.get("confidence", "")
    ).lower()
    confirmed = bool(
        fields.get("company_confirmed_by_user") or fields.get("confirmed_by_user")
    )
    needs_review = bool(
        fields.get("company_needs_review", fields.get("needs_review", True))
    )
    return bool(company) and (confidence == "high" or confirmed) and not needs_review


def compact_company_evidence(fields: dict[str, Any]) -> str:
    """Format short company verification evidence for captions."""
    evidence = fields.get("company_evidence", fields.get("evidence", [])) or []
    if isinstance(evidence, str):
        evidence = [item.strip() for item in evidence.split("|") if item.strip()]
    return " | ".join(str(item) for item in evidence[:2]) or "-"


def company_candidate_names(fields: dict[str, Any]) -> list[str]:
    """Return normalized, deduplicated candidate names for selectors."""
    candidates = fields.get("company_candidates", fields.get("candidates", [])) or []
    names = []
    for item in candidates:
        if isinstance(item, dict):
            name = str(
                item.get("normalized_company", "") or item.get("company", "")
            ).strip()
        else:
            name = str(item).strip()
        if name:
            names.append(name)
    current = str(
        fields.get("company_normalized", "")
        or fields.get("normalized_company", "")
        or fields.get("company_raw", "")
    ).strip()
    if current:
        names.insert(0, current)
    return dedupe_strings(names)


def render_company_verification_summary(fields: dict[str, Any]) -> None:
    """Show compact verification status and supporting evidence."""
    status = verification_status_label(fields)
    company = str(
        fields.get("company_normalized", "")
        or fields.get("normalized_company", "")
        or fields.get("company_raw", "")
        or "-"
    )
    st.write(f"Company: {company}")
    st.write(f"Verification: {status}")
    if status in {"Needs review", "Missing"}:
        st.warning("Company name needs confirmation before generating a cover letter.")
    evidence_text = compact_company_evidence(fields)
    if evidence_text != "-":
        st.caption(f"Evidence: {evidence_text}")


def render_markdown_company_confirmation(
    path: Path,
    key_prefix: str,
) -> dict[str, Any]:
    """Render editable company confirmation controls for a Markdown job file."""
    fields = verification_from_markdown(path)
    render_company_verification_summary(fields)
    candidates = company_candidate_names(fields)
    if candidates:
        selected = st.selectbox(
            "Suggested company candidates",
            candidates,
            key=f"{key_prefix}_company_candidate",
        )
    else:
        selected = str(fields.get("company_raw", "") or "")
    edited_company = st.text_input(
        "Editable company name",
        value=selected,
        key=f"{key_prefix}_company_confirm_input",
    )
    if st.button("Confirm company name", key=f"{key_prefix}_company_confirm_button"):
        normalized = normalize_company_name(edited_company)
        if not normalized:
            st.error("Enter a valid company name before confirming.")
        else:
            confirm_markdown_company(path, normalized)
            st.success("Company name confirmed.")
            st.rerun()
    return fields


def render_manual_company_confirmation(
    record: dict[str, Any],
    key_prefix: str,
) -> dict[str, Any]:
    """Render editable company confirmation controls for a saved manual record."""
    fields = company_verification_fields(
        str(record.get("company_raw") or record.get("company", "")),
        {
            "job_text": str(record.get("job_description", "") or ""),
            "role": str(record.get("title", "") or ""),
            "location": str(record.get("location", "") or ""),
            "job_url": str(record.get("url", "") or ""),
            "company_confirmed_by_user": bool(
                record.get("company_confirmed_by_user")
            ),
            "company_source_confidence": str(
                record.get("company_confidence", "") or ""
            ),
            "company_source_evidence": compact_company_evidence(record),
        },
        confirmed_by_user=bool(record.get("company_confirmed_by_user")),
        confirmed_at=str(record.get("company_confirmed_at", "") or ""),
    )
    fields.update(
        {key: record.get(key) for key in record if key.startswith("company_")}
    )
    render_company_verification_summary(fields)
    candidates = company_candidate_names(fields)
    selected = st.selectbox(
        "Suggested company candidates",
        candidates or [str(record.get("company", "") or "")],
        key=f"{key_prefix}_company_candidate",
    )
    edited_company = st.text_input(
        "Editable company name",
        value=selected,
        key=f"{key_prefix}_company_confirm_input",
    )
    if st.button("Confirm company name", key=f"{key_prefix}_company_confirm_button"):
        normalized = normalize_company_name(edited_company)
        if not normalized:
            st.error("Enter a valid company name before confirming.")
        else:
            updated = confirm_manual_job_company(str(record["id"]), normalized)
            if updated:
                st.success("Company name confirmed.")
                st.rerun()
            else:
                st.error("Could not update that target job record.")
    return fields
