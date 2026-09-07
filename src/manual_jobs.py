"""Manual job record persistence and metadata updates."""

from __future__ import annotations
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from company_verification import (
    company_verification_fields,
    markdown_metadata_from_verification,
    normalize_company_name,
)
from output_paths import safe_slug
from document_text import read_markdown_field
from ml.jd_quality import (
    classify_jd_quality,
    extract_description_body,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


MANUAL_JOBS_DIR = PROJECT_ROOT / "data" / "manual_jobs"


MANUAL_JOBS_JSONL = MANUAL_JOBS_DIR / "manual_jobs.jsonl"


MANUAL_SAVED_JOBS_DIR = MANUAL_JOBS_DIR / "saved_jobs"


MANUAL_UPLOADS_DIR = MANUAL_JOBS_DIR / "uploads"


STATUS_OPTIONS = [
    "Saved",
    "Analyzed",
    "Cover Letter Generated",
    "Applied",
    "Interview",
    "Rejected",
    "Offer",
]


SOURCE_OPTIONS = ["LinkedIn", "Company website", "Indeed", "Handshake", "Other"]


def ensure_manual_job_dirs() -> None:
    """Create manual job directories if they are missing."""
    MANUAL_SAVED_JOBS_DIR.mkdir(parents=True, exist_ok=True)
    MANUAL_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


def utc_timestamp() -> str:
    """Return an ISO-like timestamp for local JSONL records."""
    return datetime.now().replace(microsecond=0).isoformat()


def normalize_record_key(company: str, title: str, url: str) -> tuple[str, str, str]:
    """Return the duplicate key used for manual job records."""
    normalized_company = normalize_company_name(company) or company
    return (normalized_company.strip().lower(), title.strip().lower(), url.strip().lower())


def is_valid_url(url: str) -> bool:
    """Return True for empty or normal HTTP(S) URLs."""
    if not url.strip():
        return True
    parsed = urlsplit(url.strip())
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def normalize_manual_location(location: str) -> str:
    """Store a cleaned location hint while keeping runtime normalization separate."""
    value = " ".join(str(location or "").replace("\n", " ").split())
    value = re.sub(r"\.{2,}$", "", value).strip()
    value = value.strip(" ,;:|·•-–—")
    aliases = {
        "uk": "United Kingdom",
        "u.k.": "United Kingdom",
        "gb": "United Kingdom",
        "usa": "United States",
        "u.s.": "United States",
        "us": "United States",
    }
    return aliases.get(value.lower(), value)


def infer_manual_high_level_region(location: str) -> str:
    """Store a broad region hint for new manual jobs."""
    normalized = f" {normalize_manual_location(location).lower()} "
    if " remote " in normalized:
        return "Remote"
    if any(marker in normalized for marker in [" china ", " beijing ", " shanghai ", " shenzhen ", " hangzhou "]):
        return "China"
    if " singapore " in normalized:
        return "Singapore"
    if any(marker in normalized for marker in [" united kingdom ", " london ", " uk "]):
        return "United Kingdom"
    if any(marker in normalized for marker in [" united states ", " california ", " ca "]):
        return "United States"
    return "Other"


def load_manual_jobs() -> list[dict[str, Any]]:
    """Load manual job records from JSONL, ignoring malformed legacy lines."""
    ensure_manual_job_dirs()
    if not MANUAL_JOBS_JSONL.exists():
        return []

    records = []
    with MANUAL_JOBS_JSONL.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                records.append(record)
    return records


def write_manual_jobs(records: list[dict[str, Any]]) -> None:
    """Rewrite the JSONL index after a create or edit operation."""
    ensure_manual_job_dirs()
    with MANUAL_JOBS_JSONL.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def duplicate_manual_job_exists(company: str, title: str, url: str, exclude_id: str = "") -> bool:
    """Detect duplicates by company + title + URL."""
    key = normalize_record_key(company, title, url)
    for record in load_manual_jobs():
        if exclude_id and record.get("id") == exclude_id:
            continue
        record_key = normalize_record_key(
            str(record.get("company", "")),
            str(record.get("title", "")),
            str(record.get("url", "")),
        )
        if key == record_key:
            return True
    return False


def markdown_escape_value(value: str) -> str:
    """Keep single-line metadata fields readable in Markdown."""
    return " ".join(str(value or "").split()) or "Not provided"


def build_manual_job_markdown(record: dict[str, Any]) -> str:
    """Build a human-readable job Markdown file compatible with existing tools."""
    title = markdown_escape_value(record.get("title", ""))
    company = markdown_escape_value(record.get("company_normalized") or record.get("company", ""))
    verification_metadata = markdown_metadata_from_verification(record)
    metadata_lines = [
            f"# {title} at {company}",
            "",
            f"Company: {company}",
            f"Role: {title}",
            f"Location: {markdown_escape_value(record.get('location', ''))}",
            f"Source: {markdown_escape_value(record.get('source', 'Manual'))}",
            f"Job URL: {markdown_escape_value(record.get('url', ''))}",
            f"Salary Range: {markdown_escape_value(record.get('salary_range', ''))}",
            f"Visa Note: {markdown_escape_value(record.get('visa_note', ''))}",
            f"Status: {markdown_escape_value(record.get('status', 'Saved'))}",
            f"Description Source: {markdown_escape_value(record.get('description_source', 'manual_input'))}",
            f"JD Fetch Status: {markdown_escape_value(record.get('jd_fetch_status', 'user_provided'))}",
            f"Created At: {markdown_escape_value(record.get('created_at', ''))}",
            f"Updated At: {markdown_escape_value(record.get('updated_at', ''))}",
            f"Source Upload Filename: {markdown_escape_value(record.get('source_upload_filename', ''))}",
            f"Source Upload Filenames: {markdown_escape_value(', '.join(record.get('source_upload_filenames', []) or []))}",
    ]
    for field_name, value in verification_metadata.items():
        metadata_lines.append(f"{field_name}: {markdown_escape_value(value)}")
    for field_name, record_key in [
        ("JD Enriched By", "jd_enriched_by"),
        ("JD Enriched At", "jd_enriched_at"),
        ("JD Enrichment Match", "jd_enrichment_match"),
        ("JD Enrichment Source Job ID", "jd_enrichment_source_job_id"),
        ("JD Enrichment Source URL", "jd_enrichment_source_url"),
    ]:
        value = str(record.get(record_key, "") or "").strip()
        if value:
            metadata_lines.append(f"{field_name}: {markdown_escape_value(value)}")
    return "\n".join(
        [
            *metadata_lines,
            "",
            "## Notes",
            str(record.get("notes", "") or "").strip() or "Not provided",
            "",
            "## Job Description",
            str(record.get("job_description", "") or "").strip(),
            "",
        ]
    )


def save_uploaded_file(upload_filename: str, upload_bytes: bytes, job_id: str) -> str:
    """Save the original upload for later manual review."""
    ensure_manual_job_dirs()
    suffix = Path(upload_filename).suffix.lower()
    safe_name = f"{safe_slug(Path(upload_filename).stem)}{suffix}"
    destination = MANUAL_UPLOADS_DIR / f"{job_id}_{safe_name}"
    destination.write_bytes(upload_bytes)
    return str(destination.relative_to(PROJECT_ROOT))


def save_manual_job(
    *,
    company: str,
    title: str,
    location: str,
    source: str,
    url: str,
    salary_range: str,
    visa_note: str,
    status: str,
    notes: str,
    job_description: str,
    extracted_text: str = "",
    raw_extracted_text: str = "",
    cleaned_extracted_text: str = "",
    parser_suggestions: dict[str, Any] | None = None,
    upload_filename: str = "",
    upload_bytes: bytes | None = None,
    upload_files: list[tuple[str, bytes]] | None = None,
) -> dict[str, Any]:
    """Persist a manual job as JSONL plus a human-readable Markdown file."""
    ensure_manual_job_dirs()
    now = utc_timestamp()
    job_id = safe_slug(f"{company}_{title}_{now}")
    md_filename = f"{safe_slug(company)}-{safe_slug(title)}-{datetime.now().strftime('%Y%m%d')}-{job_id[:10]}.md"
    markdown_path = MANUAL_SAVED_JOBS_DIR / md_filename
    saved_upload_filename = ""
    saved_upload_filenames = []

    files_to_save = upload_files or []
    if not files_to_save and upload_filename and upload_bytes:
        files_to_save = [(upload_filename, upload_bytes)]

    for filename, file_bytes in files_to_save:
        if not filename or not file_bytes:
            continue
        saved_upload_filenames.append(save_uploaded_file(filename, file_bytes, job_id))
    if saved_upload_filenames:
        saved_upload_filename = saved_upload_filenames[0]

    suggestion_company = str((parser_suggestions or {}).get("company", "") or "").strip()
    suggestion_confidence = str((parser_suggestions or {}).get("company_confidence", "") or "").lower()
    source_confidence = "high"
    source_evidence = "Found in saved manual company field."
    if suggestion_company and normalize_company_name(suggestion_company).lower() == normalize_company_name(company).lower():
        source_confidence = suggestion_confidence if suggestion_confidence in {"high", "medium", "low"} else "medium"
        source_evidence = str((parser_suggestions or {}).get("company_evidence", "") or "Found in parser suggestion.")
    company_fields = company_verification_fields(
        company,
        {
            "job_text": job_description,
            "role": title,
            "location": location,
            "job_url": url,
            "manual_company_entered": source_confidence == "high",
            "company_source_confidence": source_confidence,
            "company_source_evidence": source_evidence,
            "metadata": {
                "manual_company": company,
                "filename": saved_upload_filename,
                "metadata_titles": [
                    str(report.get("metadata_title", "") or "")
                    for report in (parser_suggestions or {}).get("extraction_reports", []) or []
                    if str(report.get("metadata_title", "") or "")
                ],
            },
        },
    )

    initial_quality = classify_jd_quality(job_description)
    record = {
        "id": job_id,
        "company": company_fields["company_normalized"] or company.strip(),
        "title": title.strip(),
        "location": location.strip(),
        "normalized_location": normalize_manual_location(location),
        "inferred_region": infer_manual_high_level_region(location),
        "source": source.strip(),
        "url": url.strip(),
        "salary_range": salary_range.strip(),
        "visa_note": visa_note.strip(),
        "status": status.strip() or "Saved",
        "notes": notes.strip(),
        "job_description": job_description.strip(),
        "description_source": (
            "full_jd_manual" if initial_quality["reliable_scoring_ready"] else "manual_input"
        ),
        "jd_fetch_status": (
            "complete" if initial_quality["reliable_scoring_ready"] else "partial"
        ),
        "created_at": now,
        "updated_at": now,
        "source_upload_filename": saved_upload_filename,
        "source_upload_filenames": saved_upload_filenames,
        "extracted_text": extracted_text.strip(),
        "raw_extracted_text": raw_extracted_text.strip(),
        "cleaned_extracted_text": cleaned_extracted_text.strip(),
        "parser_suggestions": parser_suggestions or {},
        **company_fields,
        "job_title_confidence": (parser_suggestions or {}).get("job_title_confidence", ""),
        "job_title_evidence": (parser_suggestions or {}).get("job_title_evidence", ""),
        "location_options": (parser_suggestions or {}).get("location_options", []),
        "location_confidence": (parser_suggestions or {}).get("location_confidence", ""),
        "location_evidence": (parser_suggestions or {}).get("location_evidence", ""),
        "visa_confidence": (parser_suggestions or {}).get("visa_confidence", ""),
        "visa_evidence": (parser_suggestions or {}).get("visa_evidence", ""),
        "markdown_path": str(markdown_path.relative_to(PROJECT_ROOT)),
    }

    markdown_path.write_text(build_manual_job_markdown(record), encoding="utf-8")
    records = load_manual_jobs()
    records.append(record)
    write_manual_jobs(records)
    return record


def update_manual_job(record_id: str, *, status: str, notes: str) -> dict[str, Any] | None:
    """Update editable manual job fields and refresh its Markdown file."""
    records = load_manual_jobs()
    updated_record: dict[str, Any] | None = None
    for record in records:
        if record.get("id") != record_id:
            continue
        record["status"] = status
        record["notes"] = notes
        record["updated_at"] = utc_timestamp()
        updated_record = record
        markdown_path = PROJECT_ROOT / str(record.get("markdown_path", ""))
        if markdown_path.exists():
            markdown_path.write_text(build_manual_job_markdown(record), encoding="utf-8")
        break

    if updated_record is not None:
        write_manual_jobs(records)
    return updated_record


def sync_manual_job_from_markdown(record_id: str, markdown_path: Path) -> dict[str, Any] | None:
    """Copy an enriched Markdown JD back into its manual JSONL record."""
    if not markdown_path.exists():
        return None
    markdown_text = markdown_path.read_text(encoding="utf-8")
    records = load_manual_jobs()
    updated_record: dict[str, Any] | None = None
    metadata_fields = {
        "description_source": "Description Source",
        "jd_fetch_status": "JD Fetch Status",
        "jd_enriched_by": "JD Enriched By",
        "jd_enriched_at": "JD Enriched At",
        "jd_enrichment_match": "JD Enrichment Match",
        "jd_enrichment_source_job_id": "JD Enrichment Source Job ID",
        "jd_enrichment_source_url": "JD Enrichment Source URL",
    }
    for record in records:
        if record.get("id") != record_id:
            continue
        record["job_description"] = extract_description_body(markdown_text)
        for record_key, field_name in metadata_fields.items():
            record[record_key] = read_markdown_field(markdown_text, field_name)
        record["updated_at"] = utc_timestamp()
        updated_record = record
        break
    if updated_record is not None:
        write_manual_jobs(records)
    return updated_record


def confirm_manual_job_company(record_id: str, company: str) -> dict[str, Any] | None:
    """Persist a user-confirmed company name on a saved manual job."""
    records = load_manual_jobs()
    updated_record: dict[str, Any] | None = None
    for record in records:
        if record.get("id") != record_id:
            continue
        company_fields = company_verification_fields(
            company,
            {
                "job_text": str(record.get("job_description", "") or ""),
                "role": str(record.get("title", "") or ""),
                "location": str(record.get("location", "") or ""),
                "job_url": str(record.get("url", "") or ""),
            },
            confirmed_by_user=True,
        )
        record.update(company_fields)
        record["company"] = company_fields["company_normalized"] or company.strip()
        record["updated_at"] = utc_timestamp()
        updated_record = record
        markdown_path = PROJECT_ROOT / str(record.get("markdown_path", ""))
        if markdown_path.exists():
            markdown_path.write_text(build_manual_job_markdown(record), encoding="utf-8")
        break

    if updated_record is not None:
        write_manual_jobs(records)
    return updated_record
