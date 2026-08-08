"""Compact decision components shared by the Review Jobs page."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Protocol

import streamlit as st

from dashboard_fit import build_fit_presentation, confidence_level, eligibility_status
from dashboard_review import ReviewAction, job_needs_full_jd, primary_review_action
from dashboard_review_styles import (
    action_note_html,
    decision_field_html,
    render_review_component_styles,
)
from output_paths import safe_slug
from scoring_types import DashboardJob, TrackerRow


class ReviewComponentServices(Protocol):
    @property
    def demo_mode_enabled(self) -> Callable[[], bool]: ...

    @property
    def package_status_for_job(self) -> Callable[..., str]: ...

    @property
    def tracker_status_for_job(self) -> Callable[..., str]: ...


def jd_quality_label(job: DashboardJob | dict[str, Any]) -> str:
    """Return the concise JD-quality label used in default UI."""
    quality = dict(job.get("jd_quality", {}) or {})
    if not quality:
        confidence = dict(job.get("confidence", {}) or {})
        quality = dict(confidence.get("job_description_quality", {}) or {})
    if quality.get("display_label"):
        return str(quality["display_label"])
    return "Complete" if not job_needs_full_jd(job) else "Needs full JD"


def visible_role_fit(job: DashboardJob | dict[str, Any]) -> str:
    """Hide numeric fit when the scoring confidence is low."""
    if confidence_level(job.get("confidence")) not in {"medium", "high"}:
        return "Not reliable"
    return str(build_fit_presentation(job)["role_fit"])


def strongest_evidence(job: DashboardJob | dict[str, Any]) -> str:
    """Return one strongest supported statement for the default decision view."""
    analysis = dict(job.get("analysis_result", {}) or {})
    strengths = [str(item).strip() for item in analysis.get("matched_strengths", []) if str(item).strip()]
    if strengths:
        return strengths[0]
    semantic = dict(analysis.get("semantic_evidence", {}) or {})
    for match in semantic.get("matches", []) or []:
        if isinstance(match, dict) and match.get("accepted") and str(match.get("evidence", "")).strip():
            return str(match["evidence"]).strip()
    return "No strong resume evidence was identified yet."


def main_gap(job: DashboardJob | dict[str, Any]) -> str:
    """Return one material gap without exposing parser diagnostics."""
    analysis = dict(job.get("analysis_result", {}) or {})
    risk = str(analysis.get("main_risk", "") or "").strip()
    if risk:
        return risk
    weak_areas = [str(item).strip() for item in analysis.get("weak_areas", []) if str(item).strip()]
    if weak_areas:
        return weak_areas[0]
    return "No major gap was detected in the recognized requirements."


def hard_constraint(job: DashboardJob | dict[str, Any]) -> str:
    """Return the first eligibility constraint, or a compact safe state."""
    eligibility = job.get("eligibility", {})
    if not isinstance(eligibility, dict):
        return "Eligibility needs manual review."
    reasons = eligibility.get("reasons", [])
    if isinstance(reasons, list):
        for reason in reasons:
            if isinstance(reason, dict):
                message = str(reason.get("message") or reason.get("code") or "").strip()
                if message:
                    return message.replace("_", " ")
    if eligibility_status(job) == "passed":
        return "No hard constraint detected."
    return "Eligibility needs manual review."


def render_review_action_buttons(
    job: DashboardJob,
    key_prefix: str,
    action: ReviewAction,
    on_select: Callable[[DashboardJob, str], None],
) -> None:
    """Render the single action derived for the selected job."""
    st.button(
        action.label,
        key=f"{key_prefix}_primary",
        type="primary",
        width="content",
        on_click=on_select,
        args=(job, action.target_section),
    )


def render_selected_review_header(
    job: DashboardJob,
    tracker_rows: list[TrackerRow],
    services: ReviewComponentServices,
) -> dict[str, Any]:
    """Render identity and exactly four decision fields."""
    render_review_component_styles()
    tracker_status = services.tracker_status_for_job(job, tracker_rows)
    package_status = job.get("package_status") or services.package_status_for_job(job, tracker_rows)
    presentation = build_fit_presentation(job)
    decision_fields = [
        ("Role Fit", visible_role_fit(job)),
        (
            "Eligibility",
            eligibility_status(job).replace("_", " ").title(),
        ),
        ("Confidence", confidence_level(job.get("confidence")).title()),
        ("JD Quality", jd_quality_label(job)),
    ]
    st.markdown(
        '<div class="review-decision-grid">'
        + "".join(decision_field_html(label, value) for label, value in decision_fields)
        + "</div>",
        unsafe_allow_html=True,
    )
    action = primary_review_action(
        job,
        tracker_status,
        package_status,
        demo=services.demo_mode_enabled(),
    )
    st.markdown(
        action_note_html(action.message, caution=action.caution),
        unsafe_allow_html=True,
    )
    return {
        "tracker_status": tracker_status,
        "package_status": package_status,
        "presentation": presentation,
        "confidence": dict(job.get("confidence", {}) or {}),
        "jd_quality": dict(job.get("jd_quality", {}) or {}),
        "action": action,
    }


def render_review_overview_section(
    job: DashboardJob,
    selected_path: Path,
    context: dict[str, Any],
    on_select: Callable[[DashboardJob, str], None],
) -> None:
    """Render why, risk, constraint, and one primary next action."""
    st.markdown("**Decision basis**")
    st.write(f"Strongest evidence: {strongest_evidence(job)}")
    st.write(f"Main gap: {main_gap(job)}")
    st.write(f"Hard constraint: {hard_constraint(job)}")
    render_review_action_buttons(
        job,
        key_prefix=f"overview_actions_{safe_slug(str(selected_path))}",
        action=context["action"],
        on_select=on_select,
    )
