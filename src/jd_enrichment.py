"""Safely replace discovery snippets with matched full job descriptions."""

from __future__ import annotations

import re
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from fetch_history import canonicalize_job_url, read_markdown_field
from fetch_jobs import fetch_jsearch_jobs, jsearch_configured
from ml.jd_quality import classify_jd_quality


SearchFunction = Callable[[str, str, str, int], list[dict[str, Any]]]
MIN_MATCH_SCORE = 0.76
AMBIGUITY_MARGIN = 0.03


def _identity_tokens(value: str, *, company: bool = False) -> set[str]:
    normalized = re.sub(r"[^a-z0-9+#]+", " ", str(value or "").lower())
    tokens = set(normalized.split())
    if company:
        tokens -= {"inc", "incorporated", "llc", "ltd", "limited", "corp", "corporation", "company", "co"}
    return tokens


def _similarity(left: str, right: str, *, company: bool = False) -> float:
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
    total = 1.0 if url_match else (company_score * 0.55 + role_score * 0.4 + location_score * 0.05)
    return {
        "total": total,
        "company": company_score,
        "role": role_score,
        "location": location_score,
        "url": 1.0 if url_match else 0.0,
    }


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
    }
    if not target["company"] or not target["role"]:
        return {
            "status": "missing_metadata",
            "updated": False,
            "quality": current_quality,
            "message": "Company and role are required before searching for a full JD.",
        }

    search_jobs = search_jobs or fetch_jsearch_jobs
    query = f"{target['role']} at {target['company']}"
    try:
        candidates = search_jobs(_country_code(target["location"]), query, target["location"], 10)
    except Exception as error:  # noqa: BLE001
        return {
            "status": "lookup_failed",
            "updated": False,
            "quality": current_quality,
            "message": f"Full-JD lookup failed: {error}",
        }

    ranked: list[tuple[float, dict[str, Any], dict[str, float], dict[str, Any]]] = []
    for candidate in candidates:
        scores = _candidate_match_score(target, candidate)
        candidate_quality = classify_jd_quality(_candidate_markdown(candidate))
        if (
            scores["total"] >= MIN_MATCH_SCORE
            and scores["company"] >= 0.78
            and scores["role"] >= 0.55
            and candidate_quality["reliable_scoring_ready"]
        ):
            ranked.append((scores["total"], candidate, scores, candidate_quality))
    ranked.sort(key=lambda item: item[0], reverse=True)
    if not ranked:
        return {
            "status": "no_safe_match",
            "updated": False,
            "quality": current_quality,
            "message": "No unambiguous scoring-ready JSearch match was found. Paste the complete posting instead.",
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

    temporary_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary_path.write_text(updated_text, encoding="utf-8")
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return {
        "status": "updated",
        "updated": True,
        "quality": updated_quality,
        "match_score": round(score, 3),
        "match_components": scores,
        "source_job_id": candidate.get("source_job_id", ""),
        "message": "Full JD found, verified, and saved. Fit results can now be recalculated.",
    }


def ensure_saved_job_description_ready(path: Path) -> dict[str, Any]:
    """Return readiness, attempting one safe full-JD lookup when needed."""
    quality = classify_jd_quality(Path(path).read_text(encoding="utf-8"))
    if quality["reliable_scoring_ready"]:
        return {"status": "already_ready", "updated": False, "quality": quality, "message": "The saved JD is already scoring-ready."}
    return enrich_saved_job_description(path)
