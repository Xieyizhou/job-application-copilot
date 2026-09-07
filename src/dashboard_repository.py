"""Read-only job and tracker repository operations for the dashboard."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Callable, cast

from analyze_job import extract_job_description_body
from company_verification import verification_from_markdown, verification_status_label
from dashboard_fit import apply_canonical_analysis, build_fit_presentation
from dashboard_job_dedup import description_fingerprint
from dashboard_regions import (
    infer_high_level_region,
    infer_location_from_path,
    normalize_location,
)
from dashboard_titles import (
    get_job_display_title,
    read_markdown_field,
    resolve_canonical_job_title,
)
from ml.jd_quality import classify_jd_quality
from scoring_types import DashboardJob, TrackerRow


TextReader = Callable[[Path], str]
JobAnalyzer = Callable[[DashboardJob, str, str | None], dict[str, Any]]
JobTextClassifier = Callable[[str], list[str]]
JobRankKey = Callable[[DashboardJob], tuple[int, int, int, int]]
RelevancePredictor = Callable[[list[tuple[str, str]]], list[dict[str, Any]]]
RelevanceSuppressor = Callable[[list[dict[str, Any]]], list[dict[str, Any]]]


def list_job_description_files(
    *,
    demo_mode: bool,
    demo_job_dir: Path,
    jobs_dir: Path,
    is_job_description: Callable[[Path], bool],
    search_text: str = "",
) -> list[Path]:
    """Return filtered job-description paths for one resolved workspace."""
    query = search_text.strip().lower()
    if demo_mode:
        if not demo_job_dir.exists():
            return []
        candidates = demo_job_dir.glob("*.md")
    else:
        if not jobs_dir.exists():
            return []
        candidates = jobs_dir.rglob("*.md")
    return sorted(
        [
            path
            for path in candidates
            if is_job_description(path)
            and (not query or query in str(path).lower())
        ],
        key=lambda path: str(path).lower(),
    )


def infer_source_from_path(path: Path) -> str:
    """Infer source from an organized job-description path."""
    parts = list(path.parts)
    if "manual_jobs" in parts:
        return "manual"
    if "job_descriptions" in parts:
        index = parts.index("job_descriptions")
        if len(parts) > index + 2:
            return parts[index + 1]
    return "unknown"


def build_dashboard_job_record(
    path: Path,
    *,
    read_text: TextReader,
    detect_red_flags: JobTextClassifier,
    collect_warnings: JobTextClassifier,
) -> DashboardJob:
    """Build one unscored dashboard record from saved Markdown."""
    job_text = read_text(path)
    jd_quality = classify_jd_quality(job_text)
    company_fields = verification_from_markdown(path)
    company = str(
        company_fields.get("company_normalized")
        or read_markdown_field(job_text, "Company", "Not provided")
    )
    stored_role = read_markdown_field(job_text, "Role", path.stem)
    role = resolve_canonical_job_title(
        {"company": company, "role": stored_role, "preview": job_text}
    )
    location = normalize_location(
        read_markdown_field(
            job_text,
            "Location",
            infer_location_from_path(path),
        )
    )
    first_seen_at = read_markdown_field(
        job_text,
        "First Seen At",
        read_markdown_field(job_text, "Created at", "unknown"),
    )
    latest_fetch_run_id = read_markdown_field(
        job_text,
        "Latest Fetch Run ID",
        read_markdown_field(job_text, "Last Seen Fetch Run ID", ""),
    )
    score_text = read_markdown_field(job_text, "Match Score", "")
    red_flags = detect_red_flags(job_text)
    return {
        "company": company,
        "company_raw": company_fields.get("company_raw", ""),
        "company_normalized": company_fields.get("company_normalized", company),
        "company_confidence": company_fields.get("company_confidence", ""),
        "company_needs_review": company_fields.get("company_needs_review", True),
        "company_evidence": company_fields.get("company_evidence", []),
        "company_candidates": company_fields.get("company_candidates", []),
        "company_confirmed_by_user": company_fields.get("company_confirmed_by_user", False),
        "company_confirmed_at": company_fields.get("company_confirmed_at", ""),
        "company_status": verification_status_label(company_fields),
        "role": role,
        "display_role": role,
        "location": location,
        "normalized_location": location,
        "high_level_region": infer_high_level_region(location),
        "source": read_markdown_field(
            job_text,
            "Source",
            infer_source_from_path(path),
        ).lower(),
        "score": None,
        "recommendation": "Manual Review",
        "confidence": {
            "level": "low",
            "active_requirement_count": 0,
            "candidate_evidence_count": 0,
            "reasons": [],
        },
        "eligibility": {"status": "manual_review", "reasons": []},
        "score_breakdown": [],
        "analysis_result": {},
        "analysis_available": False,
        "legacy_score": int(score_text) if score_text.isdigit() else None,
        "legacy_recommendation": read_markdown_field(job_text, "Recommendation", ""),
        "red_flags": red_flags,
        "warnings": collect_warnings(job_text),
        "hard_red_flag": bool(red_flags),
        "job_url": read_markdown_field(job_text, "Job URL", ""),
        "canonical_job_key": read_markdown_field(job_text, "Canonical Job Key", ""),
        "description_source": read_markdown_field(job_text, "Description Source", ""),
        "jd_fetch_status": read_markdown_field(job_text, "JD Fetch Status", ""),
        "description_word_count": len(extract_job_description_body(job_text).split()),
        "description_fingerprint": description_fingerprint(job_text),
        "jd_quality": jd_quality,
        "first_seen_at": first_seen_at,
        "last_seen_at": read_markdown_field(job_text, "Last Seen At", first_seen_at),
        "first_seen_fetch_run_id": read_markdown_field(
            job_text,
            "First Seen Fetch Run ID",
            "",
        ),
        "latest_fetch_run_id": latest_fetch_run_id,
        "is_manual": "manual_jobs" in path.parts,
        "path": path,
        "preview": job_text[:1200],
        "label": f"{company} | {role} | {location}",
    }


def job_duplicate_key(job: DashboardJob) -> tuple[str, str, str, str]:
    """Return URL key plus company, role, and location fallback fields."""
    return (
        str(job["job_url"]).strip().lower(),
        str(job["company"]).strip().lower(),
        str(job["role"]).strip().lower(),
        str(job["location"]).strip().lower(),
    )


def job_description_preference(job: DashboardJob) -> tuple[int, int, int]:
    """Prefer duplicate records with the strongest JD evidence."""
    quality = dict(job.get("jd_quality", {}) or {})
    readiness = (
        2
        if quality.get("reliable_scoring_ready", False)
        else 1
        if quality.get("provisional_scoring_ready", False)
        else 0
    )
    return (
        readiness,
        int(quality.get("quality_score", 0) or 0),
        int(job.get("description_word_count", 0) or 0),
    )


def deduplicate_dashboard_jobs(jobs: list[DashboardJob]) -> list[DashboardJob]:
    """Deduplicate exact records and syndicated copies of the same preview."""
    unique_jobs: list[DashboardJob] = []
    url_indexes: dict[str, int] = {}
    fallback_indexes: dict[tuple[str, str, str], int] = {}
    syndicated_preview_indexes: dict[tuple[str, str, str], int] = {}
    for job in jobs:
        job_url, company, role, location = job_duplicate_key(job)
        fallback = (company, role, location)
        existing_index = url_indexes.get(job_url) if job_url else None
        if existing_index is None and all(fallback):
            existing_index = fallback_indexes.get(fallback)
        fingerprint = str(job.get("description_fingerprint", "") or "")
        syndicated_preview = (
            (company, role, fingerprint)
            if company
            and role
            and fingerprint
            and str(job.get("jd_fetch_status", "")).lower() == "snippet_only"
            else None
        )
        if existing_index is None and syndicated_preview is not None:
            existing_index = syndicated_preview_indexes.get(syndicated_preview)
        if existing_index is None:
            existing_index = len(unique_jobs)
            unique_jobs.append(job)
        elif job_description_preference(job) > job_description_preference(
            unique_jobs[existing_index]
        ):
            unique_jobs[existing_index] = job
        if job_url:
            url_indexes[job_url] = existing_index
        if all(fallback):
            fallback_indexes[fallback] = existing_index
        if syndicated_preview is not None:
            syndicated_preview_indexes[syndicated_preview] = existing_index
    return unique_jobs


def load_screened_jobs(
    paths: list[Path],
    *,
    candidate_path: Path | None,
    read_text: TextReader,
    build_record: Callable[[Path], DashboardJob],
    analyze_job: JobAnalyzer,
    rank_key: JobRankKey,
    predict_relevance: RelevancePredictor,
    suppress_relevance: RelevanceSuppressor,
) -> list[DashboardJob]:
    """Load, analyze, rank, and attach diagnostic relevance signals."""
    unique_records = deduplicate_dashboard_jobs(
        [build_record(path) for path in paths]
    )
    candidate_text = (
        read_text(candidate_path)
        if candidate_path is not None and candidate_path.is_file()
        else ""
    )
    analyzed_records: list[DashboardJob] = []
    relevance_pairs: list[tuple[str, str]] = []
    for job in unique_records:
        job_text = read_text(Path(job["path"]))
        analysis = analyze_job(job, job_text, candidate_text)
        analyzed = apply_canonical_analysis(job, analysis)
        relevance_pairs.append(
            (candidate_text, extract_job_description_body(job_text))
        )
        presentation = build_fit_presentation(analyzed)
        analyzed["label"] = (
            f"{analyzed['company']} | {get_job_display_title(analyzed)} | "
            f"{analyzed['location']} | {presentation['role_fit']}"
        )
        analyzed_records.append(analyzed)
    relevance_signals = suppress_relevance(
        predict_relevance(relevance_pairs)
    )
    for analyzed, relevance_signal in zip(analyzed_records, relevance_signals):
        analyzed["ml_relevance"] = relevance_signal
    analyzed_records.sort(key=rank_key, reverse=True)
    return analyzed_records


def load_tracker_rows(
    database_path: Path | None,
    *,
    statuses: list[str] | None = None,
    minimum_score: int = 0,
    company_search: str = "",
    sort_by: str = "created_at",
    descending: bool = True,
) -> list[TrackerRow]:
    """Query tracker rows without depending on Streamlit session state."""
    if database_path is None or not database_path.exists():
        return []
    order_map = {
        "match_score": "COALESCE(match_score, -1)",
        "created_at": "created_at",
        "status": "status",
    }
    order_column = order_map.get(sort_by, "created_at")
    order_direction = "DESC" if descending else "ASC"
    conditions: list[str] = []
    params: list[Any] = []
    if statuses:
        placeholders = ", ".join("?" for _ in statuses)
        conditions.append(f"status IN ({placeholders})")
        params.extend(statuses)
    if minimum_score > 0:
        conditions.append("COALESCE(match_score, 0) >= ?")
        params.append(minimum_score)
    if company_search.strip():
        conditions.append("LOWER(company) LIKE ?")
        params.append(f"%{company_search.strip().lower()}%")
    where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
    query = f"""
        SELECT
            id,
            company,
            role,
            location,
            job_url,
            match_score,
            recommendation,
            status,
            resume_file,
            cover_letter_file,
            notes,
            created_at,
            applied_date
        FROM applications
        {where_clause}
        ORDER BY {order_column} {order_direction}, id DESC
    """
    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(query, params).fetchall()
    return cast(list[TrackerRow], [dict(row) for row in rows])
