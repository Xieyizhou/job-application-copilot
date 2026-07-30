"""Evidence and gap presentation for generated cover letters."""

from __future__ import annotations

from typing import Any

import streamlit as st


def markdown_section_items(
    text: str,
    headings: tuple[str, ...],
    *,
    limit: int = 3,
) -> list[str]:
    """Extract concise bullet items from the first matching Markdown section."""
    targets = {heading.casefold() for heading in headings}
    collecting = False
    items: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("## "):
            heading = line[3:].strip().casefold()
            if collecting:
                break
            collecting = heading in targets
            continue
        if collecting and line.startswith(("- ", "* ")):
            item = line[2:].strip()
            if item and item.casefold() not in {"none", "none found"}:
                items.append(item)
                if len(items) == limit:
                    break
    return items


def render_evidence_and_gaps(
    artifacts: Any,
    services: Any,
) -> None:
    """Show resume evidence and unresolved gaps before the editable draft."""
    notes_text = "\n\n".join(
        services.read_text_file(path) for path in artifacts.internal_notes
    )
    analysis_text = (
        services.read_text_file(artifacts.analysis)
        if artifacts.analysis.exists()
        else ""
    )
    evidence = markdown_section_items(
        notes_text,
        (
            "Claim Trace — Exact Resume Evidence",
            "Requirement-to-Resume Evidence Map",
        ),
    ) or markdown_section_items(
        analysis_text,
        ("Relevant Resume Evidence",),
    )
    gaps = markdown_section_items(
        notes_text,
        ("Weak Or Missing Areas", "Rejected Requirement Evidence"),
    ) or markdown_section_items(
        analysis_text,
        ("Missing Skills",),
    )

    st.markdown("**Evidence and gaps**")
    evidence_column, gaps_column = st.columns(2)
    with evidence_column:
        st.markdown("**Resume evidence**")
        if evidence:
            for item in evidence:
                st.write(f"- {item}")
        else:
            st.caption("No structured evidence trace was found. Review the match report.")
    with gaps_column:
        st.markdown("**Gaps to review**")
        if gaps:
            for item in gaps:
                st.write(f"- {item}")
        else:
            st.caption("No structured gap list was found. Verify every claim before use.")
