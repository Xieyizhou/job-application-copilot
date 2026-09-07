"""Local candidate profile editor used by Cover Letter generation."""

from __future__ import annotations

from typing import Any

from workspace import CandidateProfile, Workspace, WorkspaceError, update_candidate_profile


def render_candidate_profile_editor(ui: Any, workspace: Workspace) -> None:
    """Confirm employer-facing contact details without leaving the local workspace."""
    ui.markdown("**Cover letter contact details**")
    ui.caption(
        "Stored only in the local Git-ignored workspace. Name is required; other fields are optional."
    )
    profile = workspace.candidate_profile
    name_column, email_column = ui.columns(2)
    with name_column:
        candidate_name = ui.text_input(
            "Full name", value=profile.name, key="candidate_profile_name"
        )
    with email_column:
        candidate_email = ui.text_input(
            "Email (optional)", value=profile.email, key="candidate_profile_email"
        )
    location_column, linkedin_column = ui.columns(2)
    with location_column:
        candidate_location = ui.text_input(
            "Location (optional)", value=profile.location, key="candidate_profile_location"
        )
    with linkedin_column:
        candidate_linkedin = ui.text_input(
            "LinkedIn URL (optional)", value=profile.linkedin, key="candidate_profile_linkedin"
        )
    if not ui.button("Save contact details", key="save_candidate_profile"):
        return
    try:
        update_candidate_profile(
            CandidateProfile(
                name=candidate_name,
                email=candidate_email,
                location=candidate_location,
                linkedin=candidate_linkedin,
            )
        )
        ui.success("Cover letter contact details saved locally.")
        ui.rerun()
    except WorkspaceError as error:
        ui.error(str(error))
