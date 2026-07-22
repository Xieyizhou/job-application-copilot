"""Pure Review Jobs filters, sorting, and next-action guidance."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from dashboard_fit import build_fit_presentation, confidence_level, eligibility_status
from scoring_types import DashboardJob, TrackerRow


RECOMMENDATION_RANK = {
    "Apply": 5,
    "Apply / Maybe Apply": 4,
    "Maybe Apply": 3,
    "Manual Review": 2,
    "Skip or Low Priority": 1,
    "Skip / Not Eligible": 0,
}

REVIEW_INBOX_OPTIONS = ["Recommended", "Needs attention", "Ready", "All"]


def is_ignored_tracker_status(status: str) -> bool:
    """Return True for tracker states that intentionally remove a job from consideration."""
    return str(status or "").strip().lower() in {"archived", "ignored", "not interested", "rejected", "skip"}


def review_inbox_view_matches(
    job: DashboardJob,
    inbox_view: str,
    tracker_status: str,
    package_status: str,
) -> bool:
    """Map Review Jobs inbox views to job, tracker, and cover-letter state."""
    recommendation = str(job.get("recommendation", ""))
    score = int(job.get("score") or 0)
    confidence = confidence_level(job.get("confidence"))
    eligibility = eligibility_status(job)
    has_package = package_status in {"Cover letter ready", "Demo cover letter"}
    is_tracked = tracker_status not in {"Not tracked", "Demo only"}
    is_ignored = is_ignored_tracker_status(tracker_status)

    if inbox_view == "Recommended":
        return (
            bool(job.get("analysis_available"))
            and eligibility == "passed"
            and confidence in {"medium", "high"}
            and recommendation in {"Apply", "Apply / Maybe Apply", "Maybe Apply"}
            and score >= 50
            and not is_ignored
        )
    if inbox_view in {"Needs attention", "Needs Review"}:
        canonical_review_needed = (
            not bool(job.get("analysis_available"))
            or eligibility == "manual_review"
            or confidence == "low"
            or recommendation == "Manual Review"
        )
        operational_review_needed = tracker_status != "Demo only" and (not has_package or not is_tracked)
        return not is_ignored and (canonical_review_needed or operational_review_needed)
    if inbox_view in {"Ready", "Cover Letter Ready"}:
        return (has_package or str(tracker_status).lower() == "ready") and not is_ignored
    if inbox_view == "Not Tracked":
        return not is_ignored and not is_tracked
    if inbox_view == "Ignored":
        return is_ignored
    return inbox_view in {"All", "All Jobs"}


def review_job_sort_key(job: DashboardJob, sort_by: str) -> tuple[Any, ...]:
    """Sort Review Jobs rows by the selected user-facing option."""
    score = int(job.get("score", 0) or 0)
    newest = str(job.get("last_seen_at", "") or job.get("first_seen_at", ""))
    recommendation_rank = RECOMMENDATION_RANK.get(str(job.get("recommendation", "")), 0)
    package_rank = 1 if job.get("package_status") in {"Cover letter ready", "Demo cover letter"} else 0
    tracker_rank = 1 if job.get("tracker_status") not in {"Not tracked", "Demo only"} else 0
    if sort_by == "Newest first":
        return (newest, score, recommendation_rank)
    if sort_by == "Recommendation":
        return (recommendation_rank, score, newest)
    if sort_by == "Company A-Z":
        return (str(job.get("company", "")).lower(), -score, newest)
    if sort_by in {"Package status", "Cover letter status"}:
        return (package_rank, score, newest)
    if sort_by == "Tracker status":
        return (tracker_rank, score, newest)
    return (score, newest, recommendation_rank)


def is_strong_match(job: DashboardJob) -> bool:
    """Return True only for a confident, eligible canonical Apply result."""
    return (
        bool(job.get("analysis_available"))
        and eligibility_status(job) == "passed"
        and confidence_level(job.get("confidence")) in {"medium", "high"}
        and str(job.get("recommendation", "")) == "Apply"
        and int(job.get("score") or 0) >= 80
    )


def is_current_recommendation(job: DashboardJob) -> bool:
    """Return True for current eligible recommendations shown on Dashboard."""
    return (
        bool(job.get("analysis_available"))
        and eligibility_status(job) == "passed"
        and confidence_level(job.get("confidence")) in {"medium", "high"}
        and str(job.get("recommendation", "")) in {"Apply", "Apply / Maybe Apply", "Maybe Apply"}
    )


def sorted_review_jobs(jobs: list[DashboardJob], sort_by: str) -> list[DashboardJob]:
    """Return Review Jobs sorted for inbox display."""
    reverse = sort_by != "Company A-Z"
    return sorted(jobs, key=lambda job: review_job_sort_key(job, sort_by), reverse=reverse)


def parse_local_datetime(value: object) -> datetime | None:
    """Parse tracker timestamps without assuming one historical format."""
    text = str(value or "").strip()
    if not text:
        return None
    for pattern in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"]:
        try:
            return datetime.strptime(text[: len(datetime.now().strftime(pattern))], pattern)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def tracker_age_days(row: TrackerRow | dict[str, Any]) -> int | None:
    """Return days since application, or since tracking when not yet applied."""
    reference = parse_local_datetime(row.get("applied_date") or row.get("created_at"))
    if reference is None:
        return None
    return max(0, (datetime.now() - reference).days)


def tracker_next_action(row: TrackerRow | dict[str, Any]) -> str:
    """Translate tracker state into the most useful user action."""
    status = str(row.get("status", "saved") or "saved").lower()
    age_days = tracker_age_days(row)
    if status == "saved":
        return "Review fit and decide whether this role deserves time."
    if status == "ready":
        return "Review the generated materials, then apply manually."
    if status == "applied":
        if age_days is not None and age_days >= 7:
            return f"Follow up or record an outcome — applied {age_days} days ago."
        wait_days = max(1, 7 - (age_days or 0))
        return f"Monitor for a response; consider following up in {wait_days} days."
    if status == "interview":
        return "Prepare role-specific stories, questions, and company research."
    if status == "rejected":
        return "Capture what you learned, then archive this application."
    return "No immediate action; this record is archived."


def tracker_follow_up_due(row: TrackerRow | dict[str, Any]) -> bool:
    """Return True when an applied role has had no stage update for at least a week."""
    age_days = tracker_age_days(row)
    return str(row.get("status", "")).lower() == "applied" and age_days is not None and age_days >= 7


def job_needs_full_jd(job: DashboardJob | dict[str, Any]) -> bool:
    """Return whether low confidence is specifically caused by incomplete job text."""
    direct_quality = dict(job.get("jd_quality", {}) or {})
    if "reliable_scoring_ready" in direct_quality:
        return not bool(direct_quality.get("reliable_scoring_ready"))
    confidence = dict(job.get("confidence", {}) or {})
    quality = dict(confidence.get("job_description_quality", {}) or {})
    if "appears_incomplete" in quality:
        return bool(quality.get("appears_incomplete"))
    return str(job.get("jd_fetch_status", "")).lower() in {"", "missing", "snippet_only"} and int(
        job.get("description_word_count", 0) or 0
    ) < 80


def review_job_next_action(
    job: DashboardJob | dict[str, Any],
    tracker_status: str = "Not tracked",
    package_status: str = "No cover letter",
) -> str:
    """Return one prioritized action for a saved job."""
    if not job.get("analysis_available"):
        return "Add a complete job description before judging fit."
    if eligibility_status(job) == "failed":
        return "Review the hard constraint, then ignore unless the source is wrong."
    if confidence_level(job.get("confidence")) == "low":
        if job_needs_full_jd(job):
            return "Get the full job description before trusting fit."
        return "Review the extracted requirements and candidate evidence."
    if bool(job.get("company_needs_review")):
        return "Confirm the company name before generating documents."
    if package_status in {"No cover letter", "Not generated", "-"}:
        return "Review the gaps, then generate a resume-grounded cover letter."
    if tracker_status == "Not tracked":
        return "Add this opportunity to the tracker."
    if str(tracker_status).lower() == "ready":
        return "Review the cover letter and apply manually."
    return "Open the tracker and record the latest outcome."


def job_evidence_label(job: DashboardJob | dict[str, Any]) -> str:
    """Summarize evidence count, observed coverage, and JD completeness."""
    presentation = build_fit_presentation(job)
    confidence = dict(job.get("confidence", {}) or {})
    requirement_count = int(confidence.get("active_requirement_count", 0) or 0)
    coverage = presentation.get("coverage_score")
    source_quality = "API snippet" if job_needs_full_jd(job) else "Full JD"
    coverage_text = f"{int(coverage)}% coverage" if coverage is not None else "Coverage unavailable"
    return f"{requirement_count} requirements · {coverage_text} · {source_quality}"
