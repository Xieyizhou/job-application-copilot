"""Read-only resume preview confined to the Demo workspace."""

import streamlit as st

from workspace import (
    Workspace, SUPPORTED_RESUME_EXTENSIONS, SUPPORTED_EXPERIENCE_BANK_EXTENSIONS,
    SUPPORTED_COVER_LETTER_TEMPLATE_EXTENSIONS,
)


def render_demo_resume(workspace: Workspace) -> None:
    """Mirror the Personal form with isolated fictional values and disabled controls."""
    st.success("Demo resume ready. Uploads and edits are disabled in this read-only workspace.")
    st.markdown("**Cover letter contact details**")
    st.caption("Fictional Demo information only. Your Personal contact details are not shown or changed.")
    for row in (
        (("Full name", "Alex Morgan", "name"), ("Email (optional)", "alex.morgan@example.com", "email")),
        (("Location (optional)", "Example City", "location"), ("LinkedIn URL (optional)", "https://example.com/demo-linkedin-profile", "linkedin")),
    ):
        for column, (label, value, field) in zip(st.columns(2), row):
            with column:
                st.text_input(label, value=value, key=f"demo_resume_profile_{field}", disabled=True)
    st.button("Save contact details", key="demo_resume_save_profile", disabled=True)
    st.divider()
    st.file_uploader("Resume file", type=sorted(ext.lstrip(".") for ext in SUPPORTED_RESUME_EXTENSIONS), key="demo_resume_upload", disabled=True)
    with st.expander("Optional supporting files", expanded=False):
        st.file_uploader("Experience bank", type=sorted(ext.lstrip(".") for ext in SUPPORTED_EXPERIENCE_BANK_EXTENSIONS), key="demo_experience_upload", disabled=True)
        st.file_uploader("Cover-letter template", type=sorted(ext.lstrip(".") for ext in SUPPORTED_COVER_LETTER_TEMPLATE_EXTENSIONS), key="demo_template_upload", disabled=True)
    st.caption("Read-only Demo. Choose Back to Personal Workspace to upload or save your own files.")
    st.button("Save resume", type="primary", key="demo_resume_save", disabled=True)
