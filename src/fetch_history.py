"""Fetch-run history and fetched-job identity helpers.

The app stores job descriptions as Markdown files, so fetch history lives in a
small JSONL/JSON sidecar instead of requiring a database migration.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from company_verification import normalize_company_name, parse_bool
from output_paths import safe_slug


PROJECT_ROOT = Path(__file__).resolve().parents[1]
JOB_DESCRIPTION_DIR = PROJECT_ROOT / "data" / "job_descriptions"
FETCH_RUNS_DIR = PROJECT_ROOT / "data" / "fetch_runs"
FETCH_RUNS_JSONL = FETCH_RUNS_DIR / "fetch_runs.jsonl"
FETCH_JOB_INDEX_JSON = FETCH_RUNS_DIR / "job_index.json"
TRACKING_QUERY_PARAMETERS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "fbclid",
    "gclid",
    "msclkid",
    "app_id",
    "app_key",
    "aztt",
}


def now_timestamp() -> str:
    """Return a local timestamp suitable for user-facing fetch history."""
    return datetime.now().replace(microsecond=0).isoformat(sep=" ")


def ensure_fetch_history_dirs() -> None:
    """Create fetch-history storage if it is missing."""
    FETCH_RUNS_DIR.mkdir(parents=True, exist_ok=True)


def normalize_source(source: str) -> str:
    """Normalize source names for stable comparisons."""
    return " ".join(str(source or "").strip().lower().replace("_", " ").split()).replace(" ", "_")


def normalize_identity_text(value: str) -> str:
    """Normalize company/title/location values for canonical job identity."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w+#.-]+", " ", str(value or "").lower())).strip()


def canonicalize_job_url(job_url: str) -> str:
    """Remove tracking parameters and normalize URL casing where safe."""
    if not str(job_url or "").strip():
        return ""
    split_url = urlsplit(str(job_url).strip())
    safe_pairs = [
        (key, value)
        for key, value in parse_qsl(split_url.query, keep_blank_values=True)
        if key.lower() not in TRACKING_QUERY_PARAMETERS
    ]
    return urlunsplit(
        (
            split_url.scheme.lower(),
            split_url.netloc.lower(),
            re.sub(r"/{2,}", "/", split_url.path.rstrip("/")),
            urlencode(safe_pairs),
            "",
        )
    )


def source_job_id_from_job(job: dict[str, Any]) -> str:
    """Return an explicit source id when the fetch API exposes one."""
    for key in ["source_job_id", "id", "job_id", "ats_job_id"]:
        value = str(job.get(key, "") or "").strip()
        if value:
            return value
    return ""


def make_canonical_job_key(job: dict[str, Any], source: str | None = None) -> str:
    """Build a stable source-scoped identity key for duplicate detection."""
    normalized_source = normalize_source(source or str(job.get("source", "")))
    source_job_id = source_job_id_from_job(job)
    if source_job_id:
        return f"{normalized_source}|source_id:{normalize_identity_text(source_job_id)}"

    url = canonicalize_job_url(str(job.get("job_url", "") or job.get("url", "")))
    if url:
        return f"{normalized_source}|url:{url}"

    company = normalize_identity_text(str(job.get("company_normalized", "") or normalize_company_name(str(job.get("company", ""))) or job.get("company", "")))
    role = normalize_identity_text(str(job.get("role", "") or job.get("title", "")))
    location = normalize_identity_text(str(job.get("location", "")))
    if company and role and location:
        return f"{normalized_source}|company_role_location:{company}|{role}|{location}"

    snippet = normalize_identity_text(
        " ".join(
            [
                company,
                role,
                location,
                str(job.get("description", ""))[:300],
            ]
        )
    )
    digest = hashlib.sha1(snippet.encode("utf-8")).hexdigest()[:16] if snippet else uuid.uuid4().hex[:16]
    return f"{normalized_source}|snippet:{digest}"


def read_markdown_field(markdown_text: str, field_name: str, default: str = "") -> str:
    """Read a simple Markdown metadata field."""
    prefix = f"{field_name}:"
    for line in markdown_text.splitlines():
        if line.lower().startswith(prefix.lower()):
            value = line.split(":", 1)[1].strip()
            if value and value.lower() != "not provided":
                return value
    return default


def relative_path(path: Path) -> str:
    """Return project-relative paths for JSON sidecar storage."""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def path_from_record(record: dict[str, Any]) -> Path:
    """Resolve a stored job path."""
    path = Path(str(record.get("path", "")))
    return path if path.is_absolute() else PROJECT_ROOT / path


def existing_markdown_record(path: Path) -> dict[str, Any]:
    """Create a previously-seen index record for an existing Markdown job."""
    text = path.read_text(encoding="utf-8")
    source = normalize_source(read_markdown_field(text, "Source", "unknown"))
    job = {
        "company": read_markdown_field(text, "Company Normalized") or normalize_company_name(read_markdown_field(text, "Company")) or read_markdown_field(text, "Company"),
        "role": read_markdown_field(text, "Role", path.stem),
        "location": read_markdown_field(text, "Location"),
        "job_url": read_markdown_field(text, "Job URL"),
        "source": source,
        "description": text[:1000],
    }
    canonical_key = read_markdown_field(text, "Canonical Job Key") or make_canonical_job_key(job, source)
    first_seen = read_markdown_field(text, "First Seen At") or read_markdown_field(text, "Created at")
    if not first_seen:
        first_seen = datetime.fromtimestamp(path.stat().st_mtime).replace(microsecond=0).isoformat(sep=" ")
    last_seen = read_markdown_field(text, "Last Seen At") or first_seen
    fetch_run_ids = [
        item.strip()
        for item in read_markdown_field(text, "Fetch Run IDs").split(",")
        if item.strip()
    ]
    return {
        "canonical_job_key": canonical_key,
        "source": source,
        "source_job_id": read_markdown_field(text, "Source Job ID"),
        "job_url": job["job_url"],
        "company": job["company"],
        "role": job["role"],
        "location": job["location"],
        "path": relative_path(path),
        "first_seen_at": first_seen,
        "last_seen_at": last_seen,
        "first_seen_fetch_run_id": read_markdown_field(text, "First Seen Fetch Run ID"),
        "last_seen_fetch_run_id": read_markdown_field(text, "Last Seen Fetch Run ID"),
        "fetch_run_ids": fetch_run_ids,
        "company_raw": read_markdown_field(text, "Company Raw", read_markdown_field(text, "Company")),
        "company_normalized": read_markdown_field(text, "Company Normalized", job["company"]),
        "company_confidence": read_markdown_field(text, "Company Confidence"),
        "company_needs_review": parse_bool(read_markdown_field(text, "Company Needs Review", "true")),
        "company_evidence": [
            item.strip()
            for item in read_markdown_field(text, "Company Evidence").split("|")
            if item.strip()
        ],
        "company_candidates": [
            item.strip()
            for item in read_markdown_field(text, "Company Candidates").split(",")
            if item.strip()
        ],
        "company_confirmed_by_user": parse_bool(read_markdown_field(text, "Company Confirmed By User")),
        "company_confirmed_at": read_markdown_field(text, "Company Confirmed At"),
    }


def load_job_index() -> dict[str, dict[str, Any]]:
    """Load fetched-job index and backfill records for pre-existing files."""
    ensure_fetch_history_dirs()
    try:
        raw_index = json.loads(FETCH_JOB_INDEX_JSON.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        raw_index = {}
    index = {str(key): value for key, value in raw_index.items() if isinstance(value, dict)}

    if JOB_DESCRIPTION_DIR.exists():
        for path in JOB_DESCRIPTION_DIR.rglob("*.md"):
            record = existing_markdown_record(path)
            index.setdefault(record["canonical_job_key"], record)
    return index


def write_job_index(index: dict[str, dict[str, Any]]) -> None:
    """Persist fetched-job index."""
    ensure_fetch_history_dirs()
    FETCH_JOB_INDEX_JSON.write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")


def new_fetch_run_id(source: str) -> str:
    """Create a readable fetch run id."""
    return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe_slug(source)}_{uuid.uuid4().hex[:6]}"


def job_summary_for_run(record: dict[str, Any], is_new: bool) -> dict[str, Any]:
    """Store compact job info inside each fetch-run record for review UI."""
    return {
        "canonical_job_key": record.get("canonical_job_key", ""),
        "company": record.get("company", ""),
        "role": record.get("role", ""),
        "location": record.get("location", ""),
        "job_url": record.get("job_url", ""),
        "path": record.get("path", ""),
        "is_new": is_new,
        "company_confidence": record.get("company_confidence", ""),
        "company_needs_review": record.get("company_needs_review", True),
        "company_confirmed_by_user": record.get("company_confirmed_by_user", False),
    }


def append_fetch_run(run_record: dict[str, Any]) -> None:
    """Append one fetch run to JSONL history."""
    ensure_fetch_history_dirs()
    with FETCH_RUNS_JSONL.open("a", encoding="utf-8") as file:
        file.write(json.dumps(run_record, ensure_ascii=False, sort_keys=True) + "\n")


def load_fetch_runs(limit: int | None = None) -> list[dict[str, Any]]:
    """Load fetch runs newest first."""
    ensure_fetch_history_dirs()
    if not FETCH_RUNS_JSONL.exists():
        return []
    records = []
    with FETCH_RUNS_JSONL.open(encoding="utf-8") as file:
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
    records.sort(key=lambda record: str(record.get("created_at", "")), reverse=True)
    return records[:limit] if limit else records


def latest_successful_fetch_run() -> dict[str, Any] | None:
    """Return the newest run that completed without failing."""
    for run in load_fetch_runs():
        if str(run.get("fetch_status", "")).lower() in {"success", "partial"}:
            return run
    return None


def upsert_markdown_metadata(path: Path, fields: dict[str, str]) -> None:
    """Insert or update metadata fields before the job description body."""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    body_index = next((index for index, line in enumerate(lines) if line.startswith("## ")), len(lines))
    existing_names = {line.split(":", 1)[0].strip().lower(): index for index, line in enumerate(lines[:body_index]) if ":" in line}

    for field_name, value in fields.items():
        line = f"{field_name}: {value or 'Not provided'}"
        existing_index = existing_names.get(field_name.lower())
        if existing_index is not None:
            lines[existing_index] = line
        else:
            lines.insert(body_index, line)
            body_index += 1
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
