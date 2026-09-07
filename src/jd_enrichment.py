"""Safely replace discovery snippets with matched full job descriptions."""

from __future__ import annotations

import re
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from fetch_history import canonicalize_job_url, read_markdown_field, sync_job_index_record
from fetch_jobs import (
    JSearchNoFullDescriptionsError,
    JSearchNoResultsError,
    fetch_jsearch_jobs,
    jsearch_configured,
)
from job_page_fetch import FetchedJobPage, JobPageFetchError, fetch_job_page
from ml.jd_quality import classify_jd_quality


SearchFunction = Callable[[str, str, str, int], list[dict[str, Any]]]
PageFetchFunction = Callable[[str], FetchedJobPage]
MIN_MATCH_SCORE = 0.76
AMBIGUITY_MARGIN = 0.03
DESCRIPTION_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in",
    "is", "of", "on", "or", "our", "that", "the", "this", "to", "we", "will",
    "with", "you", "your",
}


def _identity_tokens(value: str, *, company: bool = False) -> set[str]:
    normalized = re.sub(r"[^a-z0-9+#]+", " ", str(value or "").lower())
    tokens = set(normalized.split())
    if company:
        tokens -= {"inc", "incorporated", "llc", "ltd", "limited", "corp", "corporation", "company", "co"}
    return tokens


def _similarity(left: str, right: str, *, company: bool = False) -> float:
    if company:
        suffixes = {
            "co",
            "company",
            "corp",
            "corporation",
            "inc",
            "incorporated",
            "limited",
            "llc",
            "ltd",
        }
        left_parts = [
            part
            for part in re.sub(r"[^a-z0-9]+", " ", str(left or "").lower()).split()
            if part not in suffixes
        ]
        right_parts = [
            part
            for part in re.sub(r"[^a-z0-9]+", " ", str(right or "").lower()).split()
            if part not in suffixes
        ]
        left_acronym = "".join(part[0] for part in left_parts)
        right_acronym = "".join(part[0] for part in right_parts)
        if (
            len(left_parts) == 1
            and len(right_parts) > 1
            and left_parts[0] == right_acronym
        ) or (
            len(right_parts) == 1
            and len(left_parts) > 1
            and right_parts[0] == left_acronym
        ):
            return 0.95
    left_tokens = _identity_tokens(left, company=company)
    right_tokens = _identity_tokens(right, company=company)
    if not left_tokens or not right_tokens:
        return 0.0
    intersection = len(left_tokens & right_tokens)
    union = len(left_tokens | right_tokens)
    jaccard = intersection / union if union else 0.0
    left_text = " ".join(sorted(left_tokens))
    right_text = " ".join(sorted(right_tokens))
    sequence = SequenceMatcher(None, left_text, right_text).ratio()
    containment = intersection / min(len(left_tokens), len(right_tokens))
    return max(jaccard, sequence * 0.9, containment * 0.95)


def _country_code(location: str) -> str:
    lowered = str(location or "").lower()
    country_markers = [
        (("canada", "toronto", "vancouver", "montreal"), "ca"),
        (("united kingdom", " uk", "london", "england", "scotland"), "gb"),
        (("australia", "sydney", "melbourne", "brisbane"), "au"),
        (("singapore",), "sg"),
    ]
    for markers, code in country_markers:
        if any(marker in f" {lowered}" for marker in markers):
            return code
    return "us"


def _candidate_match_score(target: dict[str, str], candidate: dict[str, Any]) -> dict[str, float]:
    company_score = _similarity(target["company"], str(candidate.get("company", "")), company=True)
    role_score = _similarity(target["role"], str(candidate.get("role", "")))
    location_score = _similarity(target["location"], str(candidate.get("location", ""))) if target["location"] else 0.5
    target_url = canonicalize_job_url(target["job_url"])
    candidate_url = canonicalize_job_url(str(candidate.get("job_url", "")))
    url_match = bool(target_url and candidate_url and target_url == candidate_url)
    description_score = _description_similarity(
        target.get("description", ""),
        str(candidate.get("description", "")),
    )
    if url_match:
        total = 1.0
    elif description_score:
        total = (
            company_score * 0.45
            + role_score * 0.30
            + description_score * 0.20
            + location_score * 0.05
        )
    else:
        total = company_score * 0.55 + role_score * 0.4 + location_score * 0.05
    return {
        "total": total,
        "company": company_score,
        "role": role_score,
        "location": location_score,
        "description": description_score,
        "url": 1.0 if url_match else 0.0,
    }


def _description_similarity(left: str, right: str) -> float:
    """Measure whether a saved snippet is contained in a candidate full JD."""
    def meaningful_tokens(value: str) -> set[str]:
        tokens = set(re.findall(r"[a-z0-9+#]+", str(value or "").lower()))
        return {token for token in tokens if token not in DESCRIPTION_STOPWORDS and len(token) > 1}

    left_tokens = meaningful_tokens(left)
    right_tokens = meaningful_tokens(right)
    if len(left_tokens) < 6 or len(right_tokens) < 12:
        return 0.0
    overlap = len(left_tokens & right_tokens)
    containment = overlap / len(left_tokens)
    jaccard = overlap / len(left_tokens | right_tokens)
    return max(containment, jaccard)


def _candidate_markdown(candidate: dict[str, Any]) -> str:
    description = str(candidate.get("description", "") or "").strip()
    requirements = str(candidate.get("requirements", "") or "").strip()
    sections = [
        f"# {candidate.get('role', 'Job')}",
        f"Company: {candidate.get('company', '')}",
        "Source: JSearch",
        "Description Source: full_jd_api",
        "JD Fetch Status: complete",
        "",
        "## Job Description",
        description,
    ]
    if requirements:
        sections.extend(["", "## Requirements", requirements])
    return "\n".join(sections)


def _local_full_jd_candidates(path: Path) -> list[dict[str, Any]]:
    """Reuse verified JSearch descriptions already present in this workspace."""
    jobs_root = next((parent for parent in path.parents if parent.name == "jobs"), None)
    if jobs_root is None or not jobs_root.is_dir():
        return []
    candidates: list[dict[str, Any]] = []
    for candidate_path in jobs_root.rglob("*.md"):
        if candidate_path == path:
            continue
        try:
            text = candidate_path.read_text(encoding="utf-8")
        except OSError:
            continue
        source = read_markdown_field(text, "Source").lower()
        fetch_status = read_markdown_field(text, "JD Fetch Status").lower()
        description_source = read_markdown_field(text, "Description Source").lower()
        if source != "jsearch" or (
            fetch_status != "complete" and description_source != "full_jd_api"
        ):
            continue
        if not classify_jd_quality(text)["reliable_scoring_ready"]:
            continue
        marker = re.search(r"(?im)^##\s+Job Description\s*$", text)
        description = text[marker.end() :].strip() if marker else ""
        if not description:
            continue
        candidates.append(
            {
                "source_job_id": read_markdown_field(text, "Source Job ID"),
                "company": read_markdown_field(text, "Company Normalized")
                or read_markdown_field(text, "Company"),
                "role": read_markdown_field(text, "Role"),
                "location": read_markdown_field(text, "Location"),
                "job_url": read_markdown_field(text, "Job URL"),
                "description": description,
                "requirements": "",
                "salary": read_markdown_field(text, "Salary"),
                "source": "jsearch",
            }
        )
    return candidates


def _search_location(location: str, country: str) -> str:
    """Use a broad location phrase that JSearch can reliably interpret."""
    if "remote" in str(location or "").lower():
        return "Remote"
    country_names = {
        "au": "Australia",
        "ca": "Canada",
        "gb": "United Kingdom",
        "sg": "Singapore",
        "us": "United States",
    }
    if country in country_names:
        return country_names[country]
    parts = [part.strip() for part in str(location or "").split(",") if part.strip()]
    return parts[-1] if parts else str(location or "").strip()


def _search_attempts(target: dict[str, str]) -> list[tuple[str, str, str, int]]:
    """Return progressively broader JSearch requests without repeating a query."""
    country = _country_code(target["location"])
    location = _search_location(target["location"], country)
    attempts = [
        (country, f"{target['role']} {target['company']}", location, 10),
        (country, target["role"], location, 10),
        (country, f"{target['role']} jobs at {target['company']}", "", 10),
        (country, f"jobs at {target['company']}", "", 10),
        (country, f"{target['role']} {target['company']}", "", 10),
    ]
    unique: list[tuple[str, str, str, int]] = []
    for attempt in attempts:
        if attempt not in unique:
            unique.append(attempt)
    return unique


def _deduplicate_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for index, candidate in enumerate(candidates):
        source_id = str(candidate.get("source_job_id", "") or "").strip()
        canonical_url = canonicalize_job_url(str(candidate.get("job_url", "") or ""))
        identity = (
            source_id or f"anonymous:{index}",
            f"{candidate.get('company', '')}|{candidate.get('role', '')}|"
            f"{canonical_url}|{str(candidate.get('description', ''))[:160]}",
        )
        if identity in seen:
            continue
        seen.add(identity)
        unique.append(candidate)
    return unique


def _rank_safe_candidates(
    target: dict[str, str],
    candidates: list[dict[str, Any]],
) -> list[tuple[float, dict[str, Any], dict[str, float], dict[str, Any]]]:
    ranked: list[tuple[float, dict[str, Any], dict[str, float], dict[str, Any]]] = []
    for candidate in _deduplicate_candidates(candidates):
        scores = _candidate_match_score(target, candidate)
        candidate_quality = classify_jd_quality(_candidate_markdown(candidate))
        identity_safe = (
            (
                scores["url"] == 1.0
                and scores["company"] >= 0.65
                and scores["role"] >= 0.35
            )
            or (
                scores["company"] >= 0.78
                and (
                    scores["role"] >= 0.55
                    or scores["description"] >= 0.70
                )
                and scores["total"] >= MIN_MATCH_SCORE
            )
        )
        if identity_safe and candidate_quality["reliable_scoring_ready"]:
            ranked.append((scores["total"], candidate, scores, candidate_quality))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked


def _upsert_metadata_text(markdown_text: str, fields: dict[str, str]) -> str:
    lines = markdown_text.splitlines()
    body_index = next((index for index, line in enumerate(lines) if line.startswith("## ")), len(lines))
    existing = {
        line.split(":", 1)[0].strip().lower(): index
        for index, line in enumerate(lines[:body_index])
        if ":" in line
    }
    for field_name, value in fields.items():
        line = f"{field_name}: {value or 'Not provided'}"
        existing_index = existing.get(field_name.lower())
        if existing_index is not None:
            lines[existing_index] = line
        else:
            lines.insert(body_index, line)
            body_index += 1
    return "\n".join(lines).rstrip() + "\n"


def _replace_description(markdown_text: str, candidate: dict[str, Any]) -> str:
    marker = re.search(r"(?im)^##\s+Job Description\s*$", markdown_text)
    prefix = markdown_text[: marker.end()].rstrip() if marker else markdown_text.rstrip() + "\n\n## Job Description"
    description = str(candidate.get("description", "") or "").strip()
    requirements = str(candidate.get("requirements", "") or "").strip()
    salary = str(candidate.get("salary", "") or "").strip()
    sections = [prefix, "", description]
    if requirements and requirements.lower() not in description.lower():
        sections.extend(["", "## Requirements", "", requirements])
    if salary:
        sections.extend(["", "## Salary", "", salary])
    return "\n".join(sections).rstrip() + "\n"


def _atomic_write(path: Path, text: str) -> None:
    temporary_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary_path.write_text(text, encoding="utf-8")
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def replace_saved_job_description(
    path: Path,
    description: str,
    *,
    description_source: str = "manual_full_jd",
    enriched_by: str = "User verified paste",
    extractor: str = "",
    source_url: str = "",
) -> dict[str, Any]:
    """Validate and atomically save a complete JD supplied through a local source."""
    path = Path(path)
    original_text = path.read_text(encoding="utf-8")
    current_quality = classify_jd_quality(original_text)
    cleaned = str(description or "").strip()
    if not cleaned:
        return {
            "status": "empty_description",
            "updated": False,
            "quality": current_quality,
            "message": "Paste the complete responsibilities and requirements before saving.",
        }
    updated_text = _replace_description(original_text, {"description": cleaned})
    updated_text = _upsert_metadata_text(
        updated_text,
        {
            "Description Source": description_source,
            "JD Fetch Status": "complete",
        },
    )
    updated_quality = classify_jd_quality(updated_text)
    if not updated_quality["reliable_scoring_ready"]:
        return {
            "status": "description_not_ready",
            "updated": False,
            "quality": updated_quality,
            "message": (
                "This text still does not contain enough responsibilities and requirements "
                "for reliable scoring, so the saved JD was not changed."
            ),
        }
    metadata = {
        "Description Source": description_source,
        "JD Fetch Status": "complete",
        "JD Enriched By": enriched_by,
        "JD Enriched At": datetime.now().replace(microsecond=0).isoformat(sep=" "),
    }
    if extractor:
        metadata["JD Enrichment Extractor"] = extractor
    if source_url:
        metadata["JD Enrichment Source URL"] = source_url
    updated_text = _upsert_metadata_text(updated_text, metadata)
    _atomic_write(path, updated_text)
    sync_job_index_record(path)
    return {
        "status": "updated",
        "updated": True,
        "quality": updated_quality,
        "message": "Full JD verified and saved. Fit results can now be recalculated.",
    }


def replace_saved_job_description_from_candidate(
    path: Path,
    candidate: dict[str, Any],
    *,
    enriched_by: str,
    match_score: float,
) -> dict[str, Any]:
    """Persist one already identity-verified full-JD candidate and promote its URL."""
    path = Path(path)
    original_text = path.read_text(encoding="utf-8")
    original_url = read_markdown_field(original_text, "Job URL")
    verified_url = str(candidate.get("job_url", "") or "").strip()
    updated_text = _replace_description(original_text, candidate)
    metadata = {
        "Description Source": str(
            candidate.get("description_source", "company_ats_public_api")
        ),
        "JD Fetch Status": "complete",
        "JD Enriched By": enriched_by,
        "JD Enriched At": datetime.now().replace(microsecond=0).isoformat(sep=" "),
        "JD Enrichment Match": f"{match_score:.0%}",
        "JD Enrichment Source Job ID": str(candidate.get("source_job_id", "")),
        "JD Enrichment Source URL": verified_url,
    }
    if verified_url:
        metadata["Job URL"] = verified_url
        if original_url and canonicalize_job_url(original_url) != canonicalize_job_url(verified_url):
            metadata["Discovery URL"] = original_url
    provider = str(candidate.get("ats_provider", "") or "").strip()
    token = str(candidate.get("ats_company_token", "") or "").strip()
    if provider:
        metadata["ATS Provider"] = provider
    if token:
        metadata["ATS Board"] = token
    updated_text = _upsert_metadata_text(updated_text, metadata)
    updated_quality = classify_jd_quality(updated_text)
    if not updated_quality["reliable_scoring_ready"]:
        return {
            "status": "candidate_not_ready",
            "updated": False,
            "quality": updated_quality,
            "message": "The matched posting was still incomplete, so the saved JD was not changed.",
        }
    _atomic_write(path, updated_text)
    sync_job_index_record(path)
    return {
        "status": "updated",
        "updated": True,
        "quality": updated_quality,
        "match_score": round(match_score, 3),
        "source_url": verified_url,
        "message": "Full JD matched to the employer posting, verified, and saved.",
    }


def enrich_saved_job_description_from_url(
    path: Path,
    *,
    fetch_page: PageFetchFunction | None = None,
) -> dict[str, Any]:
    """Fetch the exact saved job URL and persist only a complete matching JD."""
    path = Path(path)
    original_text = path.read_text(encoding="utf-8")
    current_quality = classify_jd_quality(original_text)
    if current_quality["reliable_scoring_ready"]:
        return {
            "status": "already_ready",
            "updated": False,
            "quality": current_quality,
            "message": "The saved JD is already scoring-ready.",
        }
    target = {
        "company": read_markdown_field(original_text, "Company Normalized")
        or read_markdown_field(original_text, "Company"),
        "role": read_markdown_field(original_text, "Role"),
        "job_url": read_markdown_field(original_text, "Job URL"),
    }
    if not target["job_url"] or target["job_url"].lower() == "not provided":
        return {
            "status": "missing_url",
            "updated": False,
            "quality": current_quality,
            "message": "This saved job has no original URL to fetch.",
        }
    try:
        fetched = (fetch_page or fetch_job_page)(target["job_url"])
    except JobPageFetchError as error:
        return {
            "status": "page_fetch_failed",
            "updated": False,
            "quality": current_quality,
            "message": str(error),
        }
    except Exception as error:  # noqa: BLE001
        return {
            "status": "page_fetch_failed",
            "updated": False,
            "quality": current_quality,
            "message": f"The original job page could not be read: {error}",
        }

    title_score = _similarity(target["role"], fetched.title) if fetched.title else 1.0
    company_score = (
        _similarity(target["company"], fetched.company, company=True) if fetched.company else 1.0
    )
    if title_score < 0.35 or company_score < 0.65:
        return {
            "status": "page_identity_mismatch",
            "updated": False,
            "quality": current_quality,
            "message": (
                "The original URL now appears to show a different company or role, "
                "so the saved JD was not changed."
            ),
        }

    updated_text = _replace_description(original_text, {"description": fetched.description})
    updated_text = _upsert_metadata_text(
        updated_text,
        {
            "Description Source": "company_site",
            "JD Fetch Status": "complete",
            "JD Enriched By": "Original job URL",
            "JD Enriched At": datetime.now().replace(microsecond=0).isoformat(sep=" "),
            "JD Enrichment Extractor": fetched.extractor,
            "JD Enrichment Source URL": fetched.source_url,
        },
    )
    updated_quality = classify_jd_quality(updated_text)
    if not updated_quality["reliable_scoring_ready"]:
        return {
            "status": "page_description_not_ready",
            "updated": False,
            "quality": updated_quality,
            "message": (
                "The original page loaded, but it did not expose a complete set of "
                "responsibilities and requirements. The saved JD was not changed."
            ),
        }
    _atomic_write(path, updated_text)
    sync_job_index_record(path)
    return {
        "status": "updated",
        "updated": True,
        "quality": updated_quality,
        "extractor": fetched.extractor,
        "source_url": fetched.source_url,
        "message": "Full JD fetched from the original posting, verified, and saved.",
    }


def enrich_saved_job_description(
    path: Path,
    *,
    search_jobs: SearchFunction | None = None,
    configured: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Find a strict JSearch match and atomically persist its complete JD."""
    path = Path(path)
    original_text = path.read_text(encoding="utf-8")
    current_quality = classify_jd_quality(original_text)
    if current_quality["reliable_scoring_ready"]:
        return {"status": "already_ready", "updated": False, "quality": current_quality, "message": "The saved JD is already scoring-ready."}

    configured = configured or jsearch_configured
    if not configured():
        return {
            "status": "not_configured",
            "updated": False,
            "quality": current_quality,
            "message": "JSearch is not configured. Add JSEARCH_API_KEY or paste the complete posting.",
        }

    target = {
        "company": read_markdown_field(original_text, "Company Normalized") or read_markdown_field(original_text, "Company"),
        "role": read_markdown_field(original_text, "Role"),
        "location": read_markdown_field(original_text, "Location"),
        "job_url": read_markdown_field(original_text, "Job URL"),
        "description": re.split(
            r"(?im)^##\s+Job Description\s*$",
            original_text,
            maxsplit=1,
        )[-1].strip(),
    }
    if not target["company"] or not target["role"]:
        return {
            "status": "missing_metadata",
            "updated": False,
            "quality": current_quality,
            "message": "Company and role are required before searching for a full JD.",
        }

    search_jobs = search_jobs or fetch_jsearch_jobs
    candidates = _local_full_jd_candidates(path)
    def related(candidate: dict[str, Any]) -> bool:
        scores = _candidate_match_score(target, candidate)
        return bool(
            scores["url"]
            or (
                scores["company"] >= 0.65
                and (scores["role"] >= 0.35 or scores["description"] >= 0.45)
            )
        )

    found_related = any(related(candidate) for candidate in candidates)
    ranked = _rank_safe_candidates(target, candidates)
    saw_candidate_without_description = False
    if not ranked:
        for attempt in _search_attempts(target):
            try:
                new_candidates = search_jobs(*attempt)
                candidates.extend(new_candidates)
                found_related = found_related or any(related(candidate) for candidate in new_candidates)
            except JSearchNoResultsError:
                continue
            except JSearchNoFullDescriptionsError:
                saw_candidate_without_description = True
                continue
            except Exception as error:  # noqa: BLE001
                return {
                    "status": "provider_failed",
                    "updated": False,
                    "quality": current_quality,
                    "message": f"JSearch request failed: {error}",
                }
            ranked = _rank_safe_candidates(target, candidates)
            if ranked:
                break
    if not ranked:
        if found_related:
            status = "no_safe_match"
            message = (
                "JSearch found related jobs, but none matched this company and role "
                "confidently enough to replace the saved JD."
            )
        elif saw_candidate_without_description:
            status = "description_unavailable"
            message = (
                "JSearch found related jobs, but none included a full description. "
                "Open the original posting and paste the complete JD instead."
            )
        else:
            status = "posting_not_found"
            message = (
                "JSearch could not find this exact live posting after broader searches. "
                "It may be expired or not indexed; open the original posting and paste the full JD."
            )
        return {
            "status": status,
            "updated": False,
            "quality": current_quality,
            "message": message,
        }
    if len(ranked) > 1 and ranked[0][0] - ranked[1][0] < AMBIGUITY_MARGIN:
        first_id = str(ranked[0][1].get("source_job_id", ""))
        second_id = str(ranked[1][1].get("source_job_id", ""))
        if not first_id or not second_id or first_id != second_id:
            return {
                "status": "ambiguous_match",
                "updated": False,
                "quality": current_quality,
                "message": "Multiple similar postings were found, so the saved JD was not changed.",
            }

    score, candidate, scores, _candidate_quality = ranked[0]
    updated_text = _replace_description(original_text, candidate)
    verified_source_url = str(candidate.get("job_url", "")).strip()
    original_job_url = str(target.get("job_url", "")).strip()
    promoted_link_metadata: dict[str, str] = {}
    if verified_source_url:
        if original_job_url and original_job_url.lower() != "not provided":
            promoted_link_metadata["Discovery URL"] = original_job_url
        promoted_link_metadata["Job URL"] = verified_source_url
    updated_text = _upsert_metadata_text(
        updated_text,
        {
            "Description Source": "full_jd_api",
            "JD Fetch Status": "complete",
            "JD Enriched By": "JSearch",
            "JD Enriched At": datetime.now().replace(microsecond=0).isoformat(sep=" "),
            "JD Enrichment Match": f"{score:.0%}",
            "JD Enrichment Source Job ID": str(candidate.get("source_job_id", "")),
            "JD Enrichment Source URL": str(candidate.get("job_url", "")),
            **promoted_link_metadata,
        },
    )
    updated_quality = classify_jd_quality(updated_text)
    if not updated_quality["reliable_scoring_ready"]:
        return {
            "status": "candidate_not_ready",
            "updated": False,
            "quality": current_quality,
            "message": "The matched posting was still incomplete, so the saved JD was not changed.",
        }

    _atomic_write(path, updated_text)
    sync_job_index_record(path)
    return {
        "status": "updated",
        "updated": True,
        "quality": updated_quality,
        "match_score": round(score, 3),
        "match_components": scores,
        "source_job_id": candidate.get("source_job_id", ""),
        "source_url": verified_source_url,
        "message": "Full JD found, verified, and saved. Fit results can now be recalculated.",
    }


def ensure_saved_job_description_ready(path: Path) -> dict[str, Any]:
    """Return readiness, attempting one safe full-JD lookup when needed."""
    quality = classify_jd_quality(Path(path).read_text(encoding="utf-8"))
    if quality["reliable_scoring_ready"]:
        return {"status": "already_ready", "updated": False, "quality": quality, "message": "The saved JD is already scoring-ready."}
    return enrich_saved_job_description(path)
