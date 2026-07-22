"""Fetch job descriptions from official job APIs.

This script supports a small set of official/public APIs and saves fetched jobs
as Markdown files under data/job_descriptions/ so the local package workflow can
use them later.

It does not submit applications or scrape job boards.
"""

from __future__ import annotations

import argparse
import html
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from company_verification import (
    company_verification_fields,
    markdown_metadata_from_verification,
    normalize_company_name,
)
from dotenv import load_dotenv
from fetch_history import (
    append_fetch_run,
    job_summary_for_run,
    load_job_index,
    make_canonical_job_key,
    new_fetch_run_id,
    normalize_source,
    now_timestamp,
    path_from_record,
    relative_path,
    upsert_markdown_metadata,
    write_job_index,
)
from output_paths import date_slug, safe_slug


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"
JOB_DESCRIPTION_DIR = PROJECT_ROOT / "data" / "job_descriptions"
ADZUNA_API_URL = "https://api.adzuna.com/v1/api/jobs/{country}/search/1"
JOOBLE_API_URL = "https://jooble.org/api/{api_key}"
JSEARCH_API_URL = "https://api.openwebninja.com/jsearch/search-v2"
API_REQUEST_DELAY_SECONDS = 0
DEFAULT_MAX_RESULTS = 8
MAX_RESULTS_PER_SOURCE = 20
TRACKING_QUERY_PARAMETERS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "app_id",
    "app_key",
    "aztt",
}


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
    jobs = [normalize_jsearch_job(job, location) for job in raw_jobs if isinstance(job, dict)]
    jobs = [job for job in jobs if job.get("description")][:max_results]
    if not jobs:
        raise RuntimeError("No JSearch jobs with full descriptions were found for this query and location.")
    return jobs


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
    company = clean_text(job.get("company", {}).get("display_name") if isinstance(job.get("company"), dict) else "")
    role = clean_text(job.get("title"))
    location = clean_text(job.get("location", {}).get("display_name") if isinstance(job.get("location"), dict) else "")
    normalized = {
        "source_job_id": clean_text(job.get("id")),
        "company": company,
        "role": role,
        "location": location,
        "job_url": sanitize_job_url(clean_text(job.get("redirect_url"))),
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
    normalized = {
        "source_job_id": clean_text(job.get("id")),
        "company": clean_text(job.get("company")),
        "role": clean_text(job.get("title")),
        "location": clean_text(job.get("location")) or clean_text(fallback_location),
        "job_url": sanitize_job_url(clean_text(job.get("link"))),
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

    normalized = {
        "source_job_id": clean_text(job.get("job_id")),
        "company": clean_text(job.get("employer_name")),
        "role": clean_text(job.get("job_title")),
        "location": structured_location or clean_text(fallback_location),
        "job_url": sanitize_job_url(
            clean_text(job.get("job_apply_link") or job.get("job_google_link"))
        ),
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


def matches_query(job: dict[str, str], query: str) -> bool:
    """Return True when a locally fetched job matches the query text."""
    query = query.strip()
    if not query:
        return True

    searchable_text = normalize_text_for_search(
        " ".join([job.get("role", ""), job.get("description", ""), job.get("location", "")])
    )
    query_words = normalize_text_for_search(query).split()
    return all(word in searchable_text for word in query_words)


def normalize_text_for_search(text: str) -> str:
    """Normalize free text for query filtering."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9+#.-]+", " ", text.lower())).strip()


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
    source = clean_text(job.get("source"))
    source_job_id = clean_text(job.get("source_job_id"))
    description = clean_text(job.get("description"))
    requirements = clean_text(job.get("requirements"))
    salary = clean_text(job.get("salary"))
    ats_company_token = clean_text(job.get("ats_company_token"))
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
        f"Source: {source.title() if source else 'Not provided'}",
        f"Source Job ID: {source_job_id or 'Not provided'}",
        f"Created at: {created_at}",
        f"Description Source: {description_source or 'Not provided'}",
        f"JD Fetch Status: {jd_fetch_status or 'Not provided'}",
    ]
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

    if ats_company_token:
        lines.extend(["", "ATS company token:", "", ats_company_token])

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
    if source in {"adzuna", "jsearch"}:
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


def job_duplicate_key(job: dict[str, str]) -> tuple[str, str, str, str]:
    """Return duplicate keys for URL and company/role/location fallback."""
    company = clean_text(job.get("company_normalized") or normalize_company_name(job.get("company", "")) or job.get("company"))
    role = clean_text(job.get("role"))
    location = clean_text(job.get("location"))
    job_url = sanitize_job_url(clean_text(job.get("job_url")))
    return (
        normalize_key(job_url),
        normalize_key(company),
        normalize_key(role),
        normalize_key(location),
    )


def normalize_key(value: str) -> str:
    """Normalize text for duplicate comparisons."""
    return re.sub(r"\s+", " ", value.strip().lower())


def read_markdown_field(markdown_text: str, field_name: str) -> str:
    """Read a simple 'Field: value' line from Markdown."""
    prefix = f"{field_name}:"
    for line in markdown_text.splitlines():
        if line.lower().startswith(prefix.lower()):
            value = line.split(":", 1)[1].strip()
            if value.lower() != "not provided":
                return value
    return ""


def existing_job_keys() -> tuple[set[str], set[tuple[str, str, str]]]:
    """Collect duplicate keys from existing Markdown job descriptions."""
    existing_urls: set[str] = set()
    existing_company_role_locations: set[tuple[str, str, str]] = set()

    if not JOB_DESCRIPTION_DIR.exists():
        return existing_urls, existing_company_role_locations

    for markdown_path in JOB_DESCRIPTION_DIR.rglob("*.md"):
        markdown_text = markdown_path.read_text(encoding="utf-8")
        job_url = sanitize_job_url(read_markdown_field(markdown_text, "Job URL"))
        company = read_markdown_field(markdown_text, "Company Normalized") or normalize_company_name(read_markdown_field(markdown_text, "Company")) or read_markdown_field(markdown_text, "Company")
        role = read_markdown_field(markdown_text, "Role")
        location = read_markdown_field(markdown_text, "Location")

        normalized_url = normalize_key(job_url)
        if normalized_url:
            existing_urls.add(normalized_url)

        fallback_key = (
            normalize_key(company),
            normalize_key(role),
            normalize_key(location),
        )
        if all(fallback_key):
            existing_company_role_locations.add(fallback_key)

    return existing_urls, existing_company_role_locations


def unique_output_path(output_dir: Path, base_filename: str) -> Path:
    """Return a non-conflicting Markdown path for a fetched job."""
    output_path = output_dir / f"{base_filename}.md"
    if not output_path.exists():
        return output_path

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_dir / f"{base_filename}_{timestamp}.md"


def sanitize_job_url(job_url: str) -> str:
    """Remove tracking parameters and canonicalize common Adzuna redirect URLs."""
    if not job_url:
        return ""

    split_url = urlsplit(job_url)

    # Adzuna API results often contain /land/ad/<id> links with tracking
    # parameters. The cleaner public details URL is easier to save and compare.
    land_ad_match = re.match(r"^/land/ad/(\d+)", split_url.path)
    if "adzuna." in split_url.netloc.lower() and land_ad_match:
        return urlunsplit(
            (
                split_url.scheme,
                split_url.netloc,
                f"/details/{land_ad_match.group(1)}",
                "",
                "",
            )
        )

    safe_query_pairs = [
        (key, value)
        for key, value in parse_qsl(split_url.query, keep_blank_values=True)
        if key.lower() not in TRACKING_QUERY_PARAMETERS
    ]
    safe_query = urlencode(safe_query_pairs)
    return urlunsplit(
        (
            split_url.scheme,
            split_url.netloc,
            split_url.path,
            safe_query,
            split_url.fragment,
        )
    )


def fetch_and_save_jobs(args: argparse.Namespace) -> dict[str, object]:
    """Fetch jobs, record a fetch run, and save only newly discovered jobs."""
    source = args.source.lower()
    if source not in {"adzuna", "jooble", "jsearch"}:
        raise ValueError(
            f"Invalid source '{args.source}'. Supported sources: jsearch, adzuna, jooble."
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
        else:
            jobs = fetch_jsearch_jobs(
                country=args.country,
                query=args.query,
                location=args.location,
                max_results=max_results,
            )
            source_scope = args.country
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

    # Duplicate detection is source-scoped and based on canonical keys. Repeated
    # jobs update last-seen metadata instead of creating duplicate Markdown files.
    for index, job in enumerate(jobs, start=1):
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
        "fetch_status": "success",
        "notes": "",
        "new_jobs": new_jobs,
        "previously_seen_jobs": duplicate_jobs,
    }
    append_fetch_run(run_record)

    print(f"Fetch run: {fetch_run_id}")
    print(f"Returned: {len(jobs)}")
    print(f"New jobs: {len(new_jobs)}")
    print(f"Previously seen: {duplicate_jobs_count}")

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

    saved_paths = list(result["saved_paths"])
    fetch_run = dict(result["fetch_run"])
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
