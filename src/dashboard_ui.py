"""Small shared presentation functions, independent of page orchestration."""

import html
from typing import Any

import streamlit as st


def render_page_header(title: str, subtitle: str | None = None) -> None:
    """Render a compact page heading below the top navigation."""
    subtitle_html = ""
    if subtitle:
        subtitle_html = f'<div class="page-subtitle">{html.escape(subtitle)}</div>'
    st.markdown(
        f"""
        <div class="page-header">
          <div class="page-title">{html.escape(title)}</div>
          {subtitle_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_action_callout(action: str, *, caution: bool = False) -> None:
    """Render a consistent, compact next-action callout."""
    message = f"Next best action: {action}"
    if caution:
        st.warning(message)
    else:
        st.info(message)


def sanitize_fit_text(value: Any) -> str:
    """Hide local implementation filenames from user-facing fit text."""
    text = str(value)
    replacements = {
        "`resume_source.md`": "the candidate profile",
        "resume_source.md": "the candidate profile",
        "resume_source.example.md": "the demo candidate profile",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text
