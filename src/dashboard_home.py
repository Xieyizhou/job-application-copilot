"""Home dashboard page with concise next-action summaries."""

from __future__ import annotations

from dashboard_ui import render_page_header

from dataclasses import dataclass
from typing import Callable

from dashboard_home_components import (
    render_home_metrics,
    render_home_styles,
    render_primary_action,
    render_recent_opportunities,
)
from dashboard_review import job_needs_full_jd, review_fit_band
from scoring_types import DashboardJob


@dataclass(frozen=True)
class HomePageServices:
    """Shared dashboard operations required by the home page."""

    count_generated_packages: Callable[[], int]
    demo_mode_enabled: Callable[[], bool]
    go_to_page: Callable[[str], None]
    load_screened_jobs: Callable[..., list[DashboardJob]]


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
    render_page_header(
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
    strong_matches = sum(1 for job in jobs if review_fit_band(job) == "Strong")
    if strong_matches:
        return (
            f"Review {strong_matches} strong match{'es' if strong_matches != 1 else ''}",
            "Inspect the evidence and gaps before preparing an application.",
            "Review Jobs",
        )
    if low_evidence:
        return (
            "Complete job descriptions before trusting fit",
            f"{low_evidence} saved role{'s' if low_evidence != 1 else ''} still need the original posting.",
            "Review Jobs",
        )
    if packages:
        return (
            f"Review {packages} cover letter{'s' if packages != 1 else ''}",
            "Verify evidence, gaps, and employer details before applying.",
            "Cover Letter",
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
