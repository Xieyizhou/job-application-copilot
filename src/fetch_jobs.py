"""Fetch job descriptions from official job APIs.

This script supports a small set of official/public APIs and saves fetched jobs
as Markdown files under data/job_descriptions/ so the local package workflow can
use them later.

It does not submit applications or scrape job boards.
"""

from __future__ import annotations

from job_urls import sanitize_job_url

import argparse
import html
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlsplit

from company_verification import (
    company_verification_fields,
    markdown_metadata_from_verification,
)
from company_ats import search_configured_ats_boards
from dotenv import load_dotenv
from fetch_history import append_fetch_run, job_summary_for_run, load_job_index, make_canonical_job_key, new_fetch_run_id, normalize_source, now_timestamp, path_from_record, upsert_markdown_metadata, write_job_index
from output_paths import relative_path
from output_paths import date_slug, safe_slug
from ml.jd_quality import classify_jd_quality


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"
JOB_DESCRIPTION_DIR = PROJECT_ROOT / "data" / "job_descriptions"
ADZUNA_API_URL = "https://api.adzuna.com/v1/api/jobs/{country}/search/1"
JOOBLE_API_URL = "https://jooble.org/api/{api_key}"
JSEARCH_API_URL = "https://api.openwebninja.com/jsearch/search-v2"
LOCAL_WORKSPACE_ROOT = PROJECT_ROOT / "data" / "local_workspace"
API_REQUEST_DELAY_SECONDS = 0
DEFAULT_MAX_RESULTS = 8
MAX_RESULTS_PER_SOURCE = 20
AGGREGATOR_HOST_MARKERS = ("adzuna.", "jooble.")


class JSearchNoResultsError(RuntimeError):
    """Raised when JSearch returns no candidate jobs for a valid request."""


class JSearchNoFullDescriptionsError(RuntimeError):
    """Raised when JSearch candidates omit the full description field."""


def cap_max_results(max_results: int) -> int:
    """Keep API requests conservative and predictable."""
    if max_results < 1:
        raise ValueError("--max-results must be at least 1.")
    if max_results > 10:
        print(
            "Warning: requesting more than 10 results may increase rate-limit risk. "
            "The hard cap is 20."
        )
    if max_results > MAX_RESULTS_PER_SOURCE:
        print(
            f"Warning: --max-results is capped at {MAX_RESULTS_PER_SOURCE} "
            "to keep requests conservative."
        )
        return MAX_RESULTS_PER_SOURCE
    return max_results


def load_adzuna_credentials() -> tuple[str, str]:
    """Load Adzuna credentials from .env and return app id/key."""
    if not ENV_PATH.exists():
        raise FileNotFoundError(
            "Missing .env file. Create one in the project root with "
            "ADZUNA_APP_ID and ADZUNA_APP_KEY."
        )

    load_dotenv(ENV_PATH)
    app_id = os.getenv("ADZUNA_APP_ID", "").strip()
    app_key = os.getenv("ADZUNA_APP_KEY", "").strip()

    if not app_id:
        raise ValueError("Missing ADZUNA_APP_ID in .env.")
    if not app_key:
        raise ValueError("Missing ADZUNA_APP_KEY in .env.")

    return app_id, app_key


def load_jooble_api_key() -> str:
    """Load the Jooble API key from .env without printing it."""
    if not ENV_PATH.exists():
        raise FileNotFoundError(
            "Missing .env file. Create one in the project root with JOOBLE_API_KEY."
        )

    load_dotenv(ENV_PATH)
    api_key = os.getenv("JOOBLE_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "Missing JOOBLE_API_KEY in .env. Add a line like "
            "JOOBLE_API_KEY=your_jooble_api_key."
        )

    return api_key


def load_jsearch_api_key() -> str:
    """Load the OpenWeb Ninja JSearch key without printing it."""
    if not ENV_PATH.exists():
        raise FileNotFoundError(
            "Missing .env file. Create one in the project root with JSEARCH_API_KEY."
        )

    load_dotenv(ENV_PATH)
    api_key = os.getenv("JSEARCH_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "Missing JSEARCH_API_KEY in .env. JSearch is the preferred source for full job descriptions."
        )
    return api_key


def jsearch_configured() -> bool:
    """Return whether the optional full-description search source is configured."""
    if not ENV_PATH.exists():
        return False
    load_dotenv(ENV_PATH)
    return bool(os.getenv("JSEARCH_API_KEY", "").strip())


def fetch_adzuna_jobs(
    country: str,
    query: str,
    location: str,
    max_results: int,
) -> list[dict[str, str]]:
    """Call the official Adzuna search API and return job result dictionaries."""
    if not query.strip():
        raise ValueError("--query is required for Adzuna searches.")

    app_id, app_key = load_adzuna_credentials()

    try:
        import requests
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "The requests package is not installed. Run `pip install -r requirements.txt` first."
        ) from error

    url = ADZUNA_API_URL.format(country=country.lower())
    params = {
        "app_id": app_id,
        "app_key": app_key,
        "results_per_page": max_results,
        "what": query,
        "where": location,
        "content-type": "application/json",
    }

    try:
        # This script currently makes one API request per run. The optional
        # delay is here so future multi-page support can remain rate-limit
        # friendly without changing the request code.
        if API_REQUEST_DELAY_SECONDS > 0:
            time.sleep(API_REQUEST_DELAY_SECONDS)
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
    except requests.RequestException as error:
        raise RuntimeError(
            f"Adzuna API request failed: {sanitize_error_message(str(error))}"
        ) from error

    data = response.json()
    jobs = [
        normalize_adzuna_job(job)
        for job in data.get("results", [])
    ]

    if not jobs:
        raise RuntimeError("No jobs found for this query and location.")

    return jobs


def fetch_jooble_jobs(
    query: str,
    location: str,
    max_results: int,
) -> list[dict[str, str]]:
    """Search Jooble through its REST API and return normalized jobs."""
    if not query.strip():
        raise ValueError("--query is required for Jooble searches.")

    api_key = load_jooble_api_key()

    try:
        import requests
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "The requests package is not installed. Run `pip install -r requirements.txt` first."
        ) from error

    url = JOOBLE_API_URL.format(api_key=api_key)
    payload = {
        "keywords": query,
        "location": location,
        "page": 1,
    }

    try:
        if API_REQUEST_DELAY_SECONDS > 0:
            time.sleep(API_REQUEST_DELAY_SECONDS)
        response = requests.post(url, json=payload, timeout=20)
        response.raise_for_status()
    except requests.RequestException as error:
        raise RuntimeError(
            f"Jooble API request failed: {sanitize_error_message(str(error))}"
        ) from error

    data = response.json()
    jobs = [
        normalize_jooble_job(job, location)
        for job in data.get("jobs", [])
    ][:max_results]

    if not jobs:
        raise RuntimeError("No Jooble jobs found for this query and location.")

    return jobs


def fetch_jsearch_jobs(
    country: str,
    query: str,
    location: str,
    max_results: int,
) -> list[dict[str, Any]]:
    """Search JSearch for jobs whose response includes the full description."""
    if not query.strip():
        raise ValueError("--query is required for JSearch searches.")

    api_key = load_jsearch_api_key()
    try:
        import requests
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "The requests package is not installed. Run `pip install -r requirements.txt` first."
        ) from error

    location_query = f" in {location.strip()}" if location.strip() else ""
    params = {
        "query": f"{query.strip()}{location_query}",
        "country": (country or "us").lower(),
        "language": "en",
    }
    try:
        if API_REQUEST_DELAY_SECONDS > 0:
            time.sleep(API_REQUEST_DELAY_SECONDS)
        response = requests.get(
            JSEARCH_API_URL,
            params=params,
            headers={"x-api-key": api_key},
            timeout=25,
        )
        response.raise_for_status()
    except requests.RequestException as error:
        raise RuntimeError(
            f"JSearch API request failed: {sanitize_error_message(str(error))}"
        ) from error

    data = response.json()
    response_data = data.get("data", []) if isinstance(data, dict) else []
    if isinstance(response_data, dict):
        raw_jobs = response_data.get("jobs", [])
    else:
        raw_jobs = response_data
    if not isinstance(raw_jobs, list):
        raw_jobs = []
    if not raw_jobs:
        raise JSearchNoResultsError("JSearch returned no jobs for this query and location.")
    jobs = [normalize_jsearch_job(job, location) for job in raw_jobs if isinstance(job, dict)]
    full_description_jobs = [job for job in jobs if job.get("description")][:max_results]
    if not full_description_jobs:
        raise JSearchNoFullDescriptionsError(
            "JSearch found candidate jobs, but none included a full description."
        )
    return full_description_jobs


def fetch_company_ats_jobs(
    query: str,
    location: str,
    max_results: int,
) -> list[dict[str, Any]]:
    """Search the enabled employer ATS boards stored in the Personal workspace."""
    if not query.strip():
        raise ValueError("--query is required for Company ATS searches.")
    jobs, errors = search_configured_ats_boards(
        LOCAL_WORKSPACE_ROOT,
        query=query,
        location=location,
        max_results=max_results,
    )
    normalized = [
        enrich_job_with_company_verification(job, structured_company=True)
        for job in jobs
    ]
    normalized = [
        job
        for job in normalized
        if classify_jd_quality(
            "Description Source: company_ats_public_api\n"
            "JD Fetch Status: complete\n\n## Job Description\n\n"
            + str(job.get("description", ""))
        )["reliable_scoring_ready"]
    ]
    if not normalized:
        detail = f" Board issues: {'; '.join(errors)}" if errors else ""
        raise RuntimeError(f"No matching jobs were found on the configured employer boards.{detail}")
    if errors:
        print("Company ATS board issues: " + "; ".join(errors))
    return normalized


def clean_text(value: object) -> str:
    """Convert API text into readable plain text for Markdown."""
    if value is None:
        return ""

    text = html.unescape(str(value))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_adzuna_job(job: dict[str, object]) -> dict[str, Any]:
    """Convert an Adzuna result into the local normalized job shape."""
    company_data = job.get("company")
    location_data = job.get("location")
    company = clean_text(
        company_data.get("display_name") if isinstance(company_data, dict) else ""
    )
    role = clean_text(job.get("title"))
    location = clean_text(
        location_data.get("display_name") if isinstance(location_data, dict) else ""
    )
    listing_url = sanitize_job_url(clean_text(job.get("redirect_url")))
    normalized = {
        "source_job_id": clean_text(job.get("id")),
        "company": company,
        "role": role,
        "location": location,
        "job_url": listing_url,
        "discovery_url": listing_url,
        "description": clean_text(job.get("description")),
        "requirements": extract_requirements(job),
        "salary": format_salary(job),
        "source": "adzuna",
        "description_source": "api_snippet",
        "jd_fetch_status": "snippet_only",
        "ats_company_token": "",
    }
    return enrich_job_with_company_verification(normalized, structured_company=True)


def normalize_jooble_job(job: dict[str, object], fallback_location: str) -> dict[str, Any]:
    """Convert a Jooble result into the local normalized job shape."""
    description = clean_text(job.get("snippet") or job.get("description"))
    listing_url = sanitize_job_url(clean_text(job.get("link")))
    normalized = {
        "source_job_id": clean_text(job.get("id")),
        "company": clean_text(job.get("company")),
        "role": clean_text(job.get("title")),
        "location": clean_text(job.get("location")) or clean_text(fallback_location),
        "job_url": listing_url,
        "discovery_url": listing_url,
        "description": description,
        "requirements": extract_requirements_from_text(description),
        "salary": clean_text(job.get("salary")),
        "source": "jooble",
        "description_source": "api_snippet",
        "jd_fetch_status": "snippet_only",
        "ats_company_token": "",
    }
    return enrich_job_with_company_verification(normalized, structured_company=True)


def normalize_jsearch_job(job: dict[str, object], fallback_location: str) -> dict[str, Any]:
    """Convert a JSearch result into the local normalized job shape."""
    description = clean_text(job.get("job_description"))
    city = clean_text(job.get("job_city"))
    state = clean_text(job.get("job_state"))
    country = clean_text(job.get("job_country"))
    structured_location = ", ".join(part for part in [city, state, country] if part)
    if bool(job.get("job_is_remote")) and not structured_location:
        structured_location = "Remote"

    highlights = job.get("job_highlights")
    requirement_parts: list[str] = []
    if isinstance(highlights, dict):
        for key in ["Qualifications", "Responsibilities"]:
            values = highlights.get(key, [])
            if isinstance(values, list):
                requirement_parts.extend(clean_text(value) for value in values if clean_text(value))
    requirements = " ".join(requirement_parts) or extract_requirements_from_text(description)

    apply_url = sanitize_job_url(
        clean_text(job.get("job_apply_link") or job.get("job_google_link"))
    )
    discovery_url = sanitize_job_url(clean_text(job.get("job_google_link")))
    normalized = {
        "source_job_id": clean_text(job.get("job_id")),
        "company": clean_text(job.get("employer_name")),
        "role": clean_text(job.get("job_title")),
        "location": structured_location or clean_text(fallback_location),
        "job_url": apply_url,
        "discovery_url": discovery_url,
        "description": description,
        "requirements": requirements,
        "salary": format_jsearch_salary(job),
        "source": "jsearch",
        "description_source": "full_jd_api",
        "jd_fetch_status": "complete" if description else "missing",
        "ats_company_token": "",
    }
    return enrich_job_with_company_verification(normalized, structured_company=True)


def format_jsearch_salary(job: dict[str, object]) -> str:
    """Format JSearch salary fields without inventing a currency or period."""
    minimum = job.get("job_min_salary")
    maximum = job.get("job_max_salary")
    currency = clean_text(job.get("job_salary_currency"))
    period = clean_text(job.get("job_salary_period"))
    if minimum is None and maximum is None:
        return ""
    if minimum is not None and maximum is not None:
        value = f"{minimum} - {maximum}"
    elif minimum is not None:
        value = f"From {minimum}"
    else:
        value = f"Up to {maximum}"
    suffix = " ".join(part for part in [currency, f"per {period.lower()}" if period else ""] if part)
    return f"{value} {suffix}".strip()


def enrich_job_with_company_verification(job: dict[str, str], *, structured_company: bool = False) -> dict[str, Any]:
    """Attach company verification metadata to one normalized fetched job."""
    source_confidence = "high" if structured_company else "medium"
    fields = company_verification_fields(
        job.get("company", ""),
        {
            "job_text": job.get("description", ""),
            "role": job.get("role", ""),
            "location": job.get("location", ""),
            "job_url": job.get("job_url", ""),
            "company_source_confidence": source_confidence,
            "company_source_evidence": "Found in structured source field." if structured_company else "",
            "metadata": {
                "structured_company": job.get("company", "") if structured_company else "",
                "job_url": job.get("job_url", ""),
                "source": job.get("source", ""),
            },
        },
    )
    enriched: dict[str, Any] = dict(job)
    enriched.update(fields)
    if fields["company_normalized"]:
        enriched["company"] = str(fields["company_normalized"])
    return enriched


def sanitize_error_message(message: str) -> str:
    """Remove credential query parameters from API error messages."""
    message = re.sub(r"app_id=[^&\s)]+", "app_id=[hidden]", message)
    message = re.sub(r"app_key=[^&\s)]+", "app_key=[hidden]", message)
    message = re.sub(r"jooble\.org/api/[^?\s)]+", "jooble.org/api/[hidden]", message)
    return message


def format_salary(job: dict[str, object]) -> str:
    """Format salary fields if Adzuna provides them."""
    salary_min = job.get("salary_min")
    salary_max = job.get("salary_max")

    if salary_min and salary_max:
        return f"{salary_min} - {salary_max}"
    if salary_min:
        return f"From {salary_min}"
    if salary_max:
        return f"Up to {salary_max}"
    return ""


def extract_requirements(job: dict[str, object]) -> str:
    """Return requirements text only if the source provides a dedicated field."""
    for key in ["requirements", "skills", "qualifications"]:
        value = clean_text(job.get(key))
        if value:
            return value
    return ""


def extract_requirements_from_text(description: str) -> str:
    """Pull a lightweight requirements excerpt from description text when present."""
    if not description:
        return ""

    match = re.search(
        r"(requirements?|qualifications?|what you bring|you have)(:?\s+.+)",
        description,
        flags=re.IGNORECASE,
    )
    if not match:
        return ""
    return clean_text(match.group(0))[:1200]


def build_job_markdown(job: dict[str, Any]) -> str:
    """Build one Markdown job description from an API job result."""
    company = clean_text(job.get("company_normalized") or job.get("company"))
    role = clean_text(job.get("role"))
    location = clean_text(job.get("location"))
    job_url = sanitize_job_url(clean_text(job.get("job_url")))
    discovery_url = sanitize_job_url(clean_text(job.get("discovery_url")))
    source = clean_text(job.get("source"))
    source_job_id = clean_text(job.get("source_job_id"))
    description = clean_text(job.get("description"))
    requirements = clean_text(job.get("requirements"))
    salary = clean_text(job.get("salary"))
    ats_company_token = clean_text(job.get("ats_company_token"))
    ats_provider = clean_text(job.get("ats_provider"))
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    description_source = clean_text(job.get("description_source"))
    jd_fetch_status = clean_text(job.get("jd_fetch_status"))

    lines = [
        f"# {role or 'Untitled Role'}",
        "",
        f"Company: {company or 'Not provided'}",
        f"Role: {role or 'Not provided'}",
        f"Location: {location or 'Not provided'}",
        f"Job URL: {job_url or 'Not provided'}",
        f"Discovery URL: {discovery_url or job_url or 'Not provided'}",
        f"Source: {source.title() if source else 'Not provided'}",
        f"Source Job ID: {source_job_id or 'Not provided'}",
        f"Created at: {created_at}",
        f"Description Source: {description_source or 'Not provided'}",
        f"JD Fetch Status: {jd_fetch_status or 'Not provided'}",
    ]
    if ats_provider:
        lines.append(f"ATS Provider: {ats_provider}")
    if ats_company_token:
        lines.append(f"ATS Board: {ats_company_token}")
    for field_name, value in markdown_metadata_from_verification(job).items():
        lines.append(f"{field_name}: {clean_text(value) or 'Not provided'}")
    lines.extend([
        "",
        "## Job Description",
        "",
        description or "Not provided",
    ])

    if requirements:
        lines.extend(["", "## Requirements", "", requirements])

    if salary:
        lines.extend(["", "## Salary", "", salary])

    lines.append("")
    return "\n".join(lines)


def save_job_markdown(
    job: dict[str, Any],
    source: str,
    source_scope: str,
    query: str,
    index: int,
    location_scope: str = "",
) -> Path:
    """Save one API job result as a Markdown file."""
    output_dir = (
        JOB_DESCRIPTION_DIR
        / safe_slug(source)
        / safe_slug(source_scope)
        / safe_slug(location_scope or "all_locations")
        / safe_slug(query or "all_jobs")
        / date_slug()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    company = clean_text(job.get("company_normalized") or job.get("company"))
    role = clean_text(job.get("role"))
    base_filename = (
        f"{index:02d}_"
        f"{safe_slug(role)}"
    )
    if source in {"adzuna", "jsearch", "company_ats"}:
        base_filename = f"{index:02d}_{safe_slug(company)}_{safe_slug(role)}"

    output_path = unique_output_path(output_dir, base_filename)
    output_path.write_text(build_job_markdown(job), encoding="utf-8")
    return output_path


def jooble_region_scope(location: str) -> str:
    """Group Jooble output by broad region for cleaner UK folders."""
    location_slug = safe_slug(location or "remote")
    if location_slug in {"london", "united_kingdom", "uk", "great_britain"}:
        return "uk"
    return location_slug or "remote"


def unique_output_path(output_dir: Path, base_filename: str) -> Path:
    """Return a non-conflicting Markdown path for a fetched job."""
    output_path = output_dir / f"{base_filename}.md"
    if not output_path.exists():
        return output_path

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_dir / f"{base_filename}_{timestamp}.md"


def has_recoverable_original_url(job_url: object) -> bool:
    """Return whether a saved URL can plausibly recover the employer posting."""
    sanitized = sanitize_job_url(clean_text(job_url))
    split_url = urlsplit(sanitized)
    host = split_url.netloc.lower().split(":", 1)[0]
    if split_url.scheme not in {"http", "https"} or not host:
        return False
    return not any(marker in host for marker in AGGREGATOR_HOST_MARKERS)


def should_save_fetched_job(job: dict[str, Any]) -> tuple[bool, str]:
    """Keep full JDs and previews that retain a recoverable original link."""
    if clean_text(job.get("jd_fetch_status")).lower() == "complete":
        return True, "full_description"
    if has_recoverable_original_url(job.get("job_url")):
        return True, "recoverable_original_url"
    return False, "preview_without_original"


def fetch_and_save_jobs(args: argparse.Namespace) -> dict[str, object]:
    """Fetch jobs, record a fetch run, and save only newly discovered jobs."""
    source = args.source.lower()
    if source not in {"adzuna", "jooble", "jsearch", "company_ats"}:
        raise ValueError(
            f"Invalid source '{args.source}'. Supported sources: company_ats, jsearch, adzuna, jooble."
        )
    max_results = cap_max_results(args.max_results)
    fetch_run_id = new_fetch_run_id(source)
    created_at = now_timestamp()
    source_scope = ""
    location_scope = args.location or ""

    try:
        if source == "adzuna":
            jobs = fetch_adzuna_jobs(
                country=args.country,
                query=args.query,
                location=args.location,
                max_results=max_results,
            )
            source_scope = args.country
            location_scope = args.location or "all_locations"
        elif source == "jooble":
            jobs = fetch_jooble_jobs(
                query=args.query,
                location=args.location,
                max_results=max_results,
            )
            source_scope = jooble_region_scope(args.location)
            location_scope = args.location or "remote"
        elif source == "jsearch":
            jobs = fetch_jsearch_jobs(
                country=args.country,
                query=args.query,
                location=args.location,
                max_results=max_results,
            )
            source_scope = args.country
            location_scope = args.location or "all_locations"
        else:
            jobs = fetch_company_ats_jobs(
                query=args.query,
                location=args.location,
                max_results=max_results,
            )
            source_scope = "configured_boards"
            location_scope = args.location or "all_locations"
    except Exception as error:
        append_fetch_run(
            {
                "fetch_run_id": fetch_run_id,
                "source": normalize_source(source),
                "region": args.location or args.country or "",
                "query": args.query,
                "created_at": created_at,
                "total_jobs_returned": 0,
                "new_jobs_count": 0,
                "duplicate_jobs_count": 0,
                "saved_jobs_count": 0,
                "skipped_jobs_count": 0,
                "fetch_status": "failed",
                "notes": sanitize_error_message(str(error)),
                "new_jobs": [],
                "previously_seen_jobs": [],
            }
        )
        raise

    job_index = load_job_index()
    saved_paths = []
    duplicate_jobs = []
    new_jobs = []
    skipped_jobs_count = 0
    skipped_jobs = []

    # Duplicate detection is source-scoped and based on canonical keys. Repeated
    # jobs update last-seen metadata instead of creating duplicate Markdown files.
    for index, job in enumerate(jobs, start=1):
        should_save, excluded_reason = should_save_fetched_job(job)
        if not should_save:
            skipped_jobs_count += 1
            skipped_jobs.append(
                {
                    "company": clean_text(job.get("company_normalized") or job.get("company")),
                    "role": clean_text(job.get("role")),
                    "location": clean_text(job.get("location")),
                    "source": normalize_source(source),
                    "job_url": sanitize_job_url(clean_text(job.get("job_url"))),
                    "excluded_reason": excluded_reason,
                }
            )
            continue
        canonical_key = make_canonical_job_key(job, source)
        existing_record = job_index.get(canonical_key)
        if existing_record:
            existing_record["last_seen_at"] = created_at
            existing_record["last_seen_fetch_run_id"] = fetch_run_id
            fetch_run_ids = list(existing_record.get("fetch_run_ids", []) or [])
            if fetch_run_id not in fetch_run_ids:
                fetch_run_ids.append(fetch_run_id)
            existing_record["fetch_run_ids"] = fetch_run_ids
            job_index[canonical_key] = existing_record
            duplicate_jobs.append(job_summary_for_run(existing_record, is_new=False))

            markdown_path = path_from_record(existing_record)
            if markdown_path.exists():
                upsert_markdown_metadata(
                    markdown_path,
                    {
                        "Canonical Job Key": canonical_key,
                        "First Seen At": str(existing_record.get("first_seen_at", "")),
                        "Last Seen At": created_at,
                        "First Seen Fetch Run ID": str(existing_record.get("first_seen_fetch_run_id", "")),
                        "Last Seen Fetch Run ID": fetch_run_id,
                        "Latest Fetch Run ID": fetch_run_id,
                        "Fetch Run IDs": ", ".join(fetch_run_ids),
                    },
                )
            continue

        output_path = save_job_markdown(
            job=job,
            source=source,
            source_scope=source_scope,
            query=args.query,
            index=index,
            location_scope=location_scope,
        )
        saved_paths.append(output_path)
        record = {
            "canonical_job_key": canonical_key,
            "source": normalize_source(source),
            "source_job_id": clean_text(job.get("source_job_id")),
            "job_url": sanitize_job_url(clean_text(job.get("job_url"))),
            "company": clean_text(job.get("company_normalized") or job.get("company")),
            "role": clean_text(job.get("role")),
            "location": clean_text(job.get("location")),
            "path": relative_path(output_path),
            "first_seen_at": created_at,
            "last_seen_at": created_at,
            "first_seen_fetch_run_id": fetch_run_id,
            "last_seen_fetch_run_id": fetch_run_id,
            "fetch_run_ids": [fetch_run_id],
            "company_raw": clean_text(job.get("company_raw") or job.get("company")),
            "company_normalized": clean_text(job.get("company_normalized")),
            "company_confidence": clean_text(job.get("company_confidence")),
            "company_needs_review": bool(job.get("company_needs_review")),
            "company_evidence": job.get("company_evidence", []),
            "company_candidates": job.get("company_candidates", []),
            "company_confirmed_by_user": bool(job.get("company_confirmed_by_user")),
            "company_confirmed_at": clean_text(job.get("company_confirmed_at")),
            "description_source": clean_text(job.get("description_source")),
            "jd_fetch_status": clean_text(job.get("jd_fetch_status")),
        }
        job_index[canonical_key] = record
        new_jobs.append(job_summary_for_run(record, is_new=True))
        upsert_markdown_metadata(
            output_path,
            {
                "Canonical Job Key": canonical_key,
                "First Seen At": created_at,
                "Last Seen At": created_at,
                "First Seen Fetch Run ID": fetch_run_id,
                "Last Seen Fetch Run ID": fetch_run_id,
                "Latest Fetch Run ID": fetch_run_id,
                "Fetch Run IDs": fetch_run_id,
            },
        )

    write_job_index(job_index)
    duplicate_jobs_count = len(duplicate_jobs)
    run_record = {
        "fetch_run_id": fetch_run_id,
        "source": normalize_source(source),
        "region": args.location or args.country or "",
        "query": args.query,
        "created_at": created_at,
        "total_jobs_returned": len(jobs),
        "new_jobs_count": len(new_jobs),
        "duplicate_jobs_count": duplicate_jobs_count,
        "saved_jobs_count": len(saved_paths),
        "full_descriptions_count": sum(
            1 for job in jobs if clean_text(job.get("jd_fetch_status")) == "complete"
        ),
        "skipped_jobs_count": skipped_jobs_count,
        "fetch_status": "partial" if skipped_jobs_count else "success",
        "notes": (
            f"Skipped {skipped_jobs_count} preview-only result(s) without a recoverable "
            "original posting."
            if skipped_jobs_count
            else ""
        ),
        "new_jobs": new_jobs,
        "previously_seen_jobs": duplicate_jobs,
        "skipped_jobs": skipped_jobs,
    }
    append_fetch_run(run_record)

    print(f"Fetch run: {fetch_run_id}")
    print(f"Returned: {len(jobs)}")
    print(f"New jobs: {len(new_jobs)}")
    print(f"Previously seen: {duplicate_jobs_count}")
    print(f"Skipped previews: {skipped_jobs_count}")

    return {
        "fetch_run": run_record,
        "saved_paths": saved_paths,
        "skipped_duplicates": duplicate_jobs_count,
    }


def build_parser() -> argparse.ArgumentParser:
    """Build command-line parser."""
    parser = argparse.ArgumentParser(
        description="Fetch jobs from official APIs and save Markdown job descriptions."
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--country", default="sg", help="Adzuna country code, such as sg or us.")
    parser.add_argument("--query", required=True, help="Search query, such as 'machine learning intern'.")
    parser.add_argument("--location", default="", help="Search location.")
    parser.add_argument("--max-results", type=int, default=DEFAULT_MAX_RESULTS)
    return parser


def main() -> None:
    """Command-line entry point."""
    parser = build_parser()
    args = parser.parse_args()

    try:
        result = fetch_and_save_jobs(args)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"Error: {error}")
        print("")
        print("Expected .env format:")
        print("ADZUNA_APP_ID=your_adzuna_app_id")
        print("ADZUNA_APP_KEY=your_adzuna_app_key")
        print("JOOBLE_API_KEY=your_jooble_api_key")
        print("")
        print("Example command:")
        print(
            "python3 src/fetch_jobs.py --source adzuna --country sg "
            '--query "machine learning intern" --location "Singapore" --max-results 8'
        )
        print(
            'python3 src/fetch_jobs.py --source jooble --query "machine learning intern" '
            '--location "Singapore" --max-results 8'
        )
        return

    saved_paths = cast(list[Path], result["saved_paths"])
    fetch_run = cast(dict[str, Any], result["fetch_run"])
    print(f"Created {len(saved_paths)} job description file(s).")
    print(f"Previously seen {fetch_run.get('duplicate_jobs_count', 0)} job(s).")
    print("")
    print("Generated file(s):")
    for path in saved_paths:
        print(f"- {path}")
    print("")
    print("Next step: use a saved .md file with src/apply_package.py.")


if __name__ == "__main__":
    main()
