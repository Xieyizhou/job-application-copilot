"""Compact decision components shared by the Review Jobs page."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Protocol

import streamlit as st

from dashboard_fit import build_fit_presentation, confidence_level, eligibility_status
from dashboard_review import job_evidence_label, job_needs_full_jd, review_job_next_action
from dashboard_review_styles import badge_html, decision_field_html, render_review_component_styles
from dashboard_titles import get_job_display_title
from output_paths import safe_slug
from scoring_types import DashboardJob


class ReviewComponentServices(Protocol):
    build_job_snippet: Callable[[dict[str, Any]], str]
    card_html: Callable[[Any, str], str]
    demo_mode_enabled: Callable[[], bool]
    mark_job_not_interested: Callable[..., tuple[Any, str]]
    package_status_for_job: Callable[..., str]
    render_action_callout: Callable[..., None]
    save_job_to_tracker: Callable[..., tuple[Any, str]]
    tracker_status_for_job: Callable[..., str]
    show_debug_ui: bool


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


def render_badges(job: DashboardJob | dict[str, Any]) -> None:
    """Render confidence and JD quality as secondary status labels."""
    confidence = confidence_level(job.get("confidence")).title()
    quality = jd_quality_label(job)
    confidence_tone = "positive" if confidence.lower() in {"medium", "high"} else "warning"
    quality_tone = "warning" if job_needs_full_jd(job) else "positive"
    st.markdown(
        '<div class="review-badge-row">'
        + badge_html("Confidence", confidence, confidence_tone)
        + badge_html("JD", quality, quality_tone)
        + "</div>",
        unsafe_allow_html=True,
    )


def render_review_action_buttons(
    job: dict[str, Any],
    tracker_rows: list[dict[str, Any]],
    key_prefix: str,
    services: ReviewComponentServices,
    on_select: Callable[[dict[str, Any], str], None],
) -> None:
    """Render one primary action and keep secondary actions collapsed."""
    package_status = job.get("package_status") or services.package_status_for_job(job, tracker_rows)
    needs_full_jd = job_needs_full_jd(job)
    if needs_full_jd:
        action_label, focus = "Get Full JD", "JD"
    elif package_status in {"Cover letter ready", "Demo cover letter"}:
        action_label, focus = "Review Cover Letter", "Cover Letter"
    else:
        action_label, focus = "Prepare Cover Letter", "Cover Letter"
    if st.button(action_label, key=f"{key_prefix}_primary", type="primary", width="stretch"):
        on_select(job, focus)
        st.rerun()

    with st.expander("Secondary actions", expanded=False):
        left, right = st.columns(2)
        with left:
            if st.button("View Fit Evidence", key=f"{key_prefix}_fit", width="stretch"):
                on_select(job, "Fit")
                st.rerun()
        with right:
            if services.demo_mode_enabled():
                st.caption("Tracker disabled in Demo.")
            elif st.button("Add to Tracker", key=f"{key_prefix}_track", width="stretch"):
                try:
                    tracker_id, _output = services.save_job_to_tracker(job)
                    st.success(f"Saved to tracker #{tracker_id}.")
                except Exception as error:  # noqa: BLE001
                    st.error(str(error))
        if not services.demo_mode_enabled() and st.button("Ignore", key=f"{key_prefix}_ignore"):
            try:
                tracker_id, _output = services.mark_job_not_interested(job, tracker_rows)
                st.success(f"Marked tracker #{tracker_id} as not interested.")
            except Exception as error:  # noqa: BLE001
                st.error(str(error))


def render_job_result_cards(
    jobs: list[dict[str, Any]],
    tracker_rows: list[dict[str, Any]],
    services: ReviewComponentServices,
    on_select: Callable[[dict[str, Any], str], None],
) -> None:
    """Render one decision-focused action per result card."""
    render_review_component_styles()
    for index, job in enumerate(jobs, start=1):
        file_key = safe_slug(str(job["path"]))
        tracker_status = services.tracker_status_for_job(job, tracker_rows)
        package_status = job.get("package_status") or services.package_status_for_job(job, tracker_rows)
        next_action = review_job_next_action(job, tracker_status, package_status)
        result = visible_role_fit(job)
        if result == "Not reliable":
            result = str(job.get("recommendation", "Manual Review"))
        with st.container(border=True):
            st.markdown(
                services.card_html(job["company"], "review-card-company")
                + services.card_html(get_job_display_title(job), "review-card-role")
                + services.card_html(f"Result: {result}", "review-card-result"),
                unsafe_allow_html=True,
            )
            render_badges(job)
            st.caption(f"Next: {next_action}")
            if st.button("Review", key=f"review_{file_key}_{index}", width="stretch"):
                on_select(job, "Overview")
                st.rerun()


def render_selected_review_header(
    job: dict[str, Any],
    tracker_rows: list[dict[str, Any]],
    services: ReviewComponentServices,
) -> dict[str, Any]:
    """Render identity and exactly four decision fields."""
    tracker_status = services.tracker_status_for_job(job, tracker_rows)
    package_status = job.get("package_status") or services.package_status_for_job(job, tracker_rows)
    presentation = build_fit_presentation(job)
    st.markdown(f"**{job['company']} · {get_job_display_title(job)}**")
    st.caption(str(job.get("normalized_location", "")))
    decision_fields = [
        ("Recommendation", job.get("recommendation", "Manual Review")),
        ("Role Fit", visible_role_fit(job)),
        ("Confidence", confidence_level(job.get("confidence")).title()),
        ("JD Quality", jd_quality_label(job)),
    ]
    st.markdown(
        '<div class="review-decision-grid">'
        + "".join(decision_field_html(label, value) for label, value in decision_fields)
        + "</div>",
        unsafe_allow_html=True,
    )
    next_action = review_job_next_action(job, tracker_status, package_status)
    services.render_action_callout(
        next_action,
        caution=confidence_level(job.get("confidence")) == "low" or eligibility_status(job) == "failed",
    )
    return {
        "tracker_status": tracker_status,
        "package_status": package_status,
        "presentation": presentation,
        "confidence": dict(job.get("confidence", {}) or {}),
        "jd_quality": dict(job.get("jd_quality", {}) or {}),
    }


def render_review_overview_section(
    job: dict[str, Any],
    tracker_rows: list[dict[str, Any]],
    selected_path: Path,
    context: dict[str, Any],
    services: ReviewComponentServices,
    on_select: Callable[[dict[str, Any], str], None],
) -> None:
    """Render why, risk, constraint, and one primary next action."""
    st.markdown("**Why this result**")
    st.write(f"Strongest evidence: {strongest_evidence(job)}")
    st.write(f"Main gap: {main_gap(job)}")
    st.write(f"Hard constraint: {hard_constraint(job)}")
    render_review_action_buttons(
        job,
        tracker_rows,
        key_prefix=f"overview_actions_{safe_slug(str(selected_path))}",
        services=services,
        on_select=on_select,
    )

    with st.expander("Advanced analysis", expanded=False):
        confidence = context["confidence"]
        coverage = context["presentation"].get("coverage_score")
        st.write(f"Tracker: {context['tracker_status']}")
        st.write(f"Cover letter: {context['package_status']}")
        st.write(f"Recognized requirements: {int(confidence.get('active_requirement_count', 0) or 0)}")
        st.write(f"Observed coverage: {int(coverage)}%" if coverage is not None else "Observed coverage: unavailable")
        learned = dict(job.get("ml_relevance", {}) or {})
        if learned.get("available") and learned.get("displayable", True) and not job_needs_full_jd(job):
            st.write(f"Experimental local model: {float(learned.get('probability', 0.0)):.0%}")
        snippet = services.build_job_snippet(job)
        if snippet:
            st.caption(snippet)
        st.caption(job_evidence_label(job))
