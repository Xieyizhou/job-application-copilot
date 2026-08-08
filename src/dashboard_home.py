"""Home dashboard page with concise next-action summaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from dashboard_home_components import (
    render_home_metrics,
    render_home_styles,
    render_primary_action,
    render_recent_opportunities,
)
from dashboard_review import job_needs_full_jd, review_fit_band
from scoring_types import DashboardJob, TrackerRow


@dataclass(frozen=True)
class HomePageServices:
    """Shared dashboard operations required by the home page."""

    count_generated_packages: Callable[[], int]
    demo_mode_enabled: Callable[[], bool]
    go_to_page: Callable[[str], None]
    load_screened_jobs: Callable[..., list[DashboardJob]]
    load_tracker_rows: Callable[..., list[TrackerRow]]
    render_page_header: Callable[[str, str | None], None]


def home_summary(
    jobs: list[DashboardJob],
    packages: int,
) -> dict[str, int]:
    """Return the four workflow counts shown on Dashboard."""
    return {
        "saved": len(jobs),
        "strong": sum(1 for job in jobs if review_fit_band(job) == "Strong"),
        "needs_full_jd": sum(1 for job in jobs if job_needs_full_jd(job)),
        "cover_letters": packages,
    }


def dashboard_tab(services: HomePageServices) -> None:
    """Render the current workflow state and the most useful next action."""
    render_home_styles()
    services.render_page_header(
        "Dashboard",
        "Move from fresh opportunities to evidence-backed applications.",
    )
    jobs = services.load_screened_jobs()
    packages = services.count_generated_packages()
    render_primary_action(jobs, packages, services, primary_home_action)
    render_home_metrics(home_summary(jobs, packages))
    render_recent_opportunities(jobs, services)


def primary_home_action(
    jobs: list[DashboardJob],
    packages: int,
) -> tuple[str, str, str]:
    """Select one deterministic next action for the Dashboard."""
    low_evidence = sum(1 for job in jobs if job_needs_full_jd(job))
    if packages:
        return (
            f"Review {packages} cover letter(s)",
            "Verify evidence, gaps, and employer details before applying.",
            "Cover Letter",
        )
    if low_evidence:
        return (
            f"Complete {low_evidence} job description(s)",
            "Add the original posting before trusting fit.",
            "Review Jobs",
        )
    if jobs:
        return (
            "Review your strongest opportunities",
            "Compare fit, eligibility, confidence, and JD quality.",
            "Review Jobs",
        )
    return (
        "Add your first target job",
        "Paste a complete posting to begin a reliable fit review.",
        "Add Target Job",
    )
