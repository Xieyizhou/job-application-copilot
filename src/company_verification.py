"""Company-name normalization, evidence, and generation gating helpers.

Company names feed directly into cover letters, so noisy parser output must be
cleaned, scored, and either trusted at high confidence or confirmed by the user.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


CONFIDENCE_ORDER = {"missing": 0, "low": 1, "medium": 2, "high": 3}
COMPANY_CONFIRMATION_PLACEHOLDER = "[COMPANY NAME NEEDS CONFIRMATION]"
NOISE_COMPANY_VALUES = {
    "about",
    "about the job",
    "apply",
    "beta",
    "continue",
    "easy apply",
    "jobs",
    "linkedin",
    "login",
    "log in",
    "more",
    "options",
    "saved",
    "save",
    "savesave",
    "share",
    "show",
    "show more options",
    "sign in",
}
NOISE_COMPANY_PREFIXES = {
    "actively reviewing applicants",
    "apply",
    "easy apply",
    "is this information helpful",
    "over 100 applicants",
    "people you can reach out to",
    "promoted by hirer",
    "recommended jobs",
    "reposted",
    "save",
    "school alumni",
    "share",
    "show match details",
    "show more options",
    "similar jobs",
    "your profile matches",
}
WEAK_OR_REVIEW_COMPANY_VALUES = {"confidential", "not provided", "unknown", "n/a", "na"}
NOISY_DOMAIN_LABELS = {
    "adzuna",
    "greenhouse",
    "indeed",
    "jooble",
    "lever",
    "linkedin",
    "myworkdayjobs",
    "workable",
}
ROLE_TITLE_WORDS = {
    "analyst",
    "associate",
    "consultant",
    "designer",
    "developer",
    "engineer",
    "fellow",
    "fellowship",
    "intern",
    "internship",
    "manager",
    "opportunities",
    "researcher",
    "scientist",
    "specialist",
}
LOCATION_WORDS = {
    "apac",
    "asia-pacific",
    "beijing",
    "boston",
    "california",
    "china",
    "hangzhou",
    "hybrid",
    "london",
    "new york",
    "remote",
    "san francisco",
    "seattle",
    "shanghai",
    "shenzhen",
    "singapore",
    "tokyo",
    "united kingdom",
    "united states",
}
BODY_PROSE_MARKERS = [
    " builds ",
    " building ",
    " connects ",
    " focused on ",
    " for ",
    " is hiring ",
    " looking for ",
    " open-source ",
    " opening ",
    " provides ",
    " we are ",
    " you will ",
]
ORG_SUFFIX_PATTERN = (
    r"(?:AI|Bank|Capital|Company|Corp\.?|Corporation|Foundation|Group|Holdings|Inc\.?|Investment|"
    r"Labs?|Limited|LLC|LP|Ltd\.?|Pte\.?\s+Ltd\.?|Research|Systems|Technologies|University)"
)


def utc_timestamp() -> str:
    """Return a compact local timestamp for confirmation metadata."""
    return datetime.now().replace(microsecond=0).isoformat()


def clean_one_line(value: object) -> str:
    """Collapse whitespace and strip punctuation that often wraps metadata."""
    return " ".join(str(value or "").replace("\n", " ").replace("_", " ").split()).strip(" \t\r\n\"'.,;:|·•-–—")


def canonical_company_key(value: str) -> str:
    """Normalize company text for deduplication and candidate merging."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9&.+#-]+", " ", str(value or "").lower())).strip()


def looks_like_date_or_timestamp(value: str) -> bool:
    """Return True for browser/OCR date and timestamp fragments."""
    clean_value = clean_one_line(value)
    if re.fullmatch(r"\d{4}[/-]\d{1,2}[/-]\d{1,2}(?:\s+\d{1,2}:\d{2})?", clean_value):
        return True
    if re.fullmatch(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}(?:\s+\d{1,2}:\d{2})?", clean_value):
        return True
    if re.fullmatch(r"\d{1,2}:\d{2}(?:\s*[AP]M)?", clean_value, flags=re.IGNORECASE):
        return True
    return False


def strip_leading_browser_timestamp(value: str) -> str:
    """Remove leading screenshot/browser timestamps while preserving the name."""
    value = clean_one_line(value)
    value = re.sub(r"^\d{4}[/-]\d{1,2}[/-]\d{1,2}\s+\d{1,2}:\d{2}\s*", "", value)
    value = re.sub(r"^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\s+\d{1,2}:\d{2}\s*", "", value)
    return value.strip(" ,;:|·•-–—")


def phrase_has_role_shape(value: str) -> bool:
    """Detect role titles accidentally stored as company names."""
    normalized = f" {canonical_company_key(value)} "
    return any(f" {word} " in normalized for word in ROLE_TITLE_WORDS)


def phrase_has_location_shape(value: str) -> bool:
    """Detect exact or obvious location names accidentally stored as companies."""
    normalized = canonical_company_key(value)
    if normalized in LOCATION_WORDS:
        return True
    return any(normalized == canonical_company_key(location) for location in LOCATION_WORDS)


def split_title_like_company(value: str) -> str:
    """Keep the company side of page titles such as 'Makimoto - About ...'."""
    for separator in [" — ", " – ", " - ", " | ", "•", "·"]:
        if separator not in value:
            continue
        left, right = [part.strip() for part in value.split(separator, 1)]
        right_lower = f" {right.lower()} "
        if left and (
            len(right.split()) > 3
            or any(marker in right_lower for marker in BODY_PROSE_MARKERS)
            or phrase_has_role_shape(right)
            or phrase_has_location_shape(right)
            or right.lower() in NOISE_COMPANY_VALUES
        ):
            return left
    return value


def normalize_company_name(raw_company: str) -> str:
    """Return a clean company name, or an empty string when the value is noise.

    Normalization removes browser timestamps, page-title wrappers, job-board UI
    labels, and obvious role/location/prose strings. Rejection is intentional:
    generation code must not silently use weak company text in cover letters.
    """
    value = strip_leading_browser_timestamp(raw_company)
    value = re.sub(r"^(?:company|employer|organization|hiring company)\s*[:\-]\s*", "", value, flags=re.IGNORECASE)
    value = re.sub(r"^apply\s+to\s+", "", value, flags=re.IGNORECASE).strip()
    value = re.sub(r"\s+careers?$", "", value, flags=re.IGNORECASE).strip()
    value = split_title_like_company(value)
    value = clean_one_line(value)

    lowered = value.lower()
    if not value or lowered in NOISE_COMPANY_VALUES or looks_like_date_or_timestamp(value):
        return ""
    if any(lowered.startswith(prefix) for prefix in NOISE_COMPANY_PREFIXES):
        return ""
    if "://" in lowered or lowered.startswith(("www.", "http")):
        return ""
    if len(value) > 90 or len(value.split()) > 8:
        return ""
    if value.endswith(".") and len(value.split()) > 2:
        return ""
    padded = f" {lowered} "
    if any(marker in padded for marker in [" for ", " focused on ", " open source ", " open-source "]) and len(value.split()) > 3:
        return ""
    if any(marker in padded for marker in BODY_PROSE_MARKERS) and not re.search(ORG_SUFFIX_PATTERN, value):
        return ""
    if phrase_has_role_shape(value) and not re.search(ORG_SUFFIX_PATTERN, value):
        return ""
    if phrase_has_location_shape(value):
        return ""
    return smart_company_case(value)


def smart_company_case(value: str) -> str:
    """Preserve existing casing while fixing all-lower/all-upper weak clues."""
    value = clean_one_line(value)
    if not value:
        return ""
    acronyms = {"ai", "api", "apac", "hk", "llc", "lp", "ml", "nlp", "pte", "uk", "us"}
    if value.islower() or value.isupper():
        fixed = []
        for word in value.split():
            clean_word = word.strip()
            suffix = ""
            if clean_word.endswith("."):
                clean_word, suffix = clean_word[:-1], "."
            if clean_word.lower() in acronyms:
                fixed.append(clean_word.upper() + suffix)
            else:
                fixed.append(clean_word[:1].upper() + clean_word[1:].lower() + suffix)
        return " ".join(fixed)
    return value


def company_issue_list(company: str, job: dict[str, Any] | None = None) -> list[str]:
    """Return concrete reasons a company value needs review."""
    job = job or {}
    issues = []
    normalized = normalize_company_name(company)
    lowered = clean_one_line(company).lower()
    if not normalized:
        issues.append("Company is missing or looks like job-board/OCR noise.")
        return issues
    if lowered in WEAK_OR_REVIEW_COMPANY_VALUES:
        issues.append("Company is a weak placeholder.")
    role = clean_one_line(job.get("role") or job.get("title"))
    location = clean_one_line(job.get("location"))
    if role and canonical_company_key(normalized) == canonical_company_key(role):
        issues.append("Company matches the role title.")
    if location and canonical_company_key(normalized) == canonical_company_key(location):
        issues.append("Company matches the location.")
    if phrase_has_role_shape(normalized) and not re.search(ORG_SUFFIX_PATTERN, normalized):
        issues.append("Company looks like a role title.")
    if phrase_has_location_shape(normalized):
        issues.append("Company looks like a location.")
    return issues


def candidate(company: str, confidence: str, evidence: str, source: str = "") -> dict[str, Any]:
    """Build one normalized company candidate with short evidence."""
    normalized = normalize_company_name(company)
    if not normalized:
        return {}
    return {
        "company": company,
        "normalized_company": normalized,
        "confidence": confidence if confidence in CONFIDENCE_ORDER else "low",
        "evidence": [evidence] if evidence else [],
        "source": source,
    }


def merge_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate company candidates while preserving strongest evidence."""
    merged: dict[str, dict[str, Any]] = {}
    for item in candidates:
        normalized = normalize_company_name(str(item.get("normalized_company") or item.get("company") or ""))
        if not normalized:
            continue
        key = canonical_company_key(normalized)
        existing = merged.get(key)
        evidence = [str(line) for line in item.get("evidence", []) if str(line).strip()]
        confidence = str(item.get("confidence", "low")).lower()
        if existing is None:
            merged[key] = {**item, "normalized_company": normalized, "company": normalized, "evidence": evidence}
            continue
        if CONFIDENCE_ORDER.get(confidence, 0) > CONFIDENCE_ORDER.get(str(existing.get("confidence", "low")), 0):
            existing["confidence"] = confidence
            existing["source"] = item.get("source", existing.get("source", ""))
        for line in evidence:
            if line not in existing["evidence"]:
                existing["evidence"].append(line)
    return sorted(
        merged.values(),
        key=lambda item: (CONFIDENCE_ORDER.get(str(item.get("confidence", "low")), 0), len(str(item.get("normalized_company", "")))),
        reverse=True,
    )


def infer_company_candidates(job_text: str, metadata: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Infer company candidates from structured fields, headers, titles, and body.

    Each candidate carries short evidence so the UI can explain why it was
    suggested instead of presenting parser output as fact.
    """
    metadata = metadata or {}
    text = str(job_text or "")
    candidates: list[dict[str, Any]] = []

    for key, evidence, source, confidence in [
        ("structured_company", "Found in structured source field.", "structured", "high"),
        ("manual_company", "Found in saved manual company field.", "manual", "high"),
        ("company", "Found in company metadata field.", "metadata", "medium"),
    ]:
        value = str(metadata.get(key, "") or "").strip()
        if value:
            candidates.append(candidate(value, confidence, evidence, source))

    for label in ["Company", "Employer", "Organization", "Hiring company"]:
        match = re.search(rf"^\s*{re.escape(label)}\s*[:\-]\s*(.+)$", text, flags=re.IGNORECASE | re.MULTILINE)
        if match:
            candidates.append(candidate(match.group(1), "high", f"Found in explicit line: {label}: ...", "explicit_line"))

    for line in [line.strip() for line in text.splitlines()[:30] if line.strip()]:
        for separator in ["·", "•", " - ", " – ", " — "]:
            if separator not in line:
                continue
            left, right = [part.strip() for part in line.split(separator, 1)]
            if phrase_has_location_shape(right) or any(location in right.lower() for location in LOCATION_WORDS):
                candidates.append(candidate(left, "high", "Found in LinkedIn header.", "linkedin_header"))

    for title in metadata.get("metadata_titles", []) or []:
        normalized = normalize_company_name(str(title))
        if normalized:
            candidates.append(candidate(normalized, "medium", f"Found in PDF title: {shorten_evidence(str(title))}", "metadata_title"))

    for title in [metadata.get("page_title", ""), metadata.get("title", "")]:
        title_text = str(title or "")
        normalized = normalize_company_name(title_text)
        if normalized:
            evidence = "Cleaned from page title: " + shorten_evidence(title_text)
            candidates.append(candidate(normalized, "medium", evidence, "page_title"))

    filename = str(metadata.get("filename", "") or metadata.get("source_upload_filename", "") or "")
    if filename:
        filename_hint = re.sub(r"\.[a-z0-9]+$", "", Path(filename).name, flags=re.IGNORECASE)
        filename_hint = re.sub(r"\b(?:screenshot|screen shot|job|description|jd)\b", " ", filename_hint, flags=re.IGNORECASE)
        normalized = normalize_company_name(filename_hint)
        if normalized:
            candidates.append(candidate(normalized, "medium", f"Found in filename clue: {Path(filename).name}", "filename"))

    domain_candidate = infer_company_from_domain(str(metadata.get("job_url", "") or metadata.get("url", "")))
    if domain_candidate:
        candidates.append(candidate(domain_candidate, "medium", f"Found in company website domain: {domain_candidate}", "domain"))

    body_counts: dict[str, int] = {}
    body_evidence: dict[str, list[str]] = {}
    body_patterns = [
        rf"\b([A-Z][A-Za-z0-9&.+-]*(?:\s+[A-Z][A-Za-z0-9&.+-]*){{0,4}}\s+{ORG_SUFFIX_PATTERN})\b",
        r"\b([A-Z][A-Za-z0-9&.+-]*(?:\s+[A-Z][A-Za-z0-9&.+-]*){0,4})\s+is an equal opportunity employer\b",
        r"\b(?:at|join|why join)\s+([A-Z][A-Za-z0-9&.+-]*(?:\s+[A-Z][A-Za-z0-9&.+-]*){0,4})\b",
    ]
    for raw_line in text.splitlines():
        line = clean_one_line(raw_line)
        if not line or len(line) > 240:
            continue
        for pattern in body_patterns:
            for match in re.finditer(pattern, line):
                normalized = normalize_company_name(match.group(1))
                if not normalized:
                    continue
                key = canonical_company_key(normalized)
                body_counts[key] = body_counts.get(key, 0) + 1
                body_evidence.setdefault(key, [])
                if len(body_evidence[key]) < 2:
                    body_evidence[key].append(shorten_evidence(line))

    for key, count in body_counts.items():
        name = next((item for item in body_evidence if item == key), key)
        evidence_lines = body_evidence[key]
        company_name = smart_company_case(" ".join(key.split()))
        confidence = "high" if count >= 2 or any("equal opportunity employer" in line.lower() for line in evidence_lines) else "medium"
        evidence = "Repeated in body text: " + " | ".join(evidence_lines)
        candidates.append(candidate(company_name or name, confidence, evidence, "body"))

    return merge_candidates(candidates)


def infer_company_from_domain(url: str) -> str:
    """Infer a medium-confidence company clue from a clean employer domain."""
    if not str(url or "").strip():
        return ""
    netloc = urlsplit(str(url).strip()).netloc.lower()
    if not netloc:
        return ""
    parts = [part for part in netloc.split(".") if part and part not in {"www", "careers", "jobs"}]
    if not parts:
        return ""
    label = parts[0]
    if label in NOISY_DOMAIN_LABELS:
        return ""
    return smart_company_case(label.replace("-", " "))


def shorten_evidence(value: str, limit: int = 120) -> str:
    """Keep evidence snippets short enough for tables and captions."""
    cleaned = clean_one_line(value)
    return cleaned if len(cleaned) <= limit else cleaned[: limit - 3].rstrip() + "..."


def validate_company_name(company: str, job: dict[str, Any] | None = None) -> dict[str, Any]:
    """Validate company text and return confidence, review state, evidence, candidates."""
    job = job or {}
    raw_company = clean_one_line(company)
    normalized = normalize_company_name(raw_company)
    job_text = str(job.get("job_text") or job.get("description") or job.get("job_description") or "")
    metadata = dict(job.get("metadata", {}) or {})
    for key in ["job_url", "url", "source", "role", "title", "location", "filename", "page_title", "metadata_titles"]:
        if key in job and key not in metadata:
            metadata[key] = job[key]
    if raw_company and "company" not in metadata:
        metadata["company"] = raw_company

    candidates = infer_company_candidates(job_text, metadata)
    issues = company_issue_list(normalized or raw_company, job)
    evidence: list[str] = []
    confidence = "missing"
    confirmed_by_user = bool(job.get("company_confirmed_by_user"))

    if normalized:
        confidence = "medium"
        for item in candidates:
            if canonical_company_key(str(item.get("normalized_company", ""))) == canonical_company_key(normalized):
                item_confidence = str(item.get("confidence", "low"))
                if CONFIDENCE_ORDER[item_confidence] > CONFIDENCE_ORDER[confidence]:
                    confidence = item_confidence
                evidence.extend(str(line) for line in item.get("evidence", []) if str(line).strip())
        source_confidence = str(job.get("company_source_confidence") or "").lower()
        source_evidence = str(job.get("company_source_evidence") or "").strip()
        if source_confidence in CONFIDENCE_ORDER:
            confidence = max([confidence, source_confidence], key=lambda item: CONFIDENCE_ORDER[item])
        if source_evidence:
            evidence.append(source_evidence)
        if bool(job.get("manual_company_entered")):
            confidence = max([confidence, "high"], key=lambda item: CONFIDENCE_ORDER[item])
            evidence.append("Found in saved manual company field.")
        if confirmed_by_user:
            confidence = "high"
            evidence.append("Confirmed by user.")
    elif candidates:
        best = candidates[0]
        normalized = str(best.get("normalized_company", ""))
        confidence = str(best.get("confidence", "low"))
        evidence.extend(str(line) for line in best.get("evidence", []) if str(line).strip())
        issues.append("Company was inferred from candidates and needs confirmation.")

    if raw_company.lower() in WEAK_OR_REVIEW_COMPANY_VALUES:
        confidence = "low" if normalized else "missing"
    if not normalized:
        confidence = "missing"

    evidence = dedupe_strings(evidence)
    needs_review = confidence not in {"high"} and not confirmed_by_user
    if issues:
        needs_review = True

    return {
        "company": raw_company,
        "normalized_company": normalized,
        "confidence": confidence,
        "needs_review": needs_review,
        "issues": dedupe_strings(issues),
        "evidence": evidence,
        "candidates": candidates,
    }


def dedupe_strings(values: list[str]) -> list[str]:
    """Deduplicate non-empty strings in first-seen order."""
    seen = set()
    deduped = []
    for value in values:
        clean_value = str(value or "").strip()
        if clean_value and clean_value not in seen:
            seen.add(clean_value)
            deduped.append(clean_value)
    return deduped


def verification_status_label(verification: dict[str, Any]) -> str:
    """Return the compact status label used in dashboard tables."""
    if bool(verification.get("company_confirmed_by_user")) or bool(verification.get("confirmed_by_user")):
        return "Confirmed"
    confidence = str(verification.get("company_confidence") or verification.get("confidence") or "").lower()
    needs_review = bool(verification.get("company_needs_review", verification.get("needs_review", True)))
    normalized = str(
        verification.get("company_normalized")
        or verification.get("normalized_company")
        or normalize_company_name(str(verification.get("company") or verification.get("company_raw") or ""))
    ).strip()
    if not normalized or confidence == "missing":
        return "Missing"
    if confidence == "high" and not needs_review:
        return "High confidence"
    return "Needs review"


def company_verification_fields(
    raw_company: str,
    job: dict[str, Any] | None = None,
    *,
    confirmed_by_user: bool | None = None,
    confirmed_at: str | None = None,
) -> dict[str, Any]:
    """Return storage fields for a job record or Markdown metadata block."""
    job = dict(job or {})
    if confirmed_by_user is not None:
        job["company_confirmed_by_user"] = confirmed_by_user
    validation = validate_company_name(raw_company, job)
    normalized = str(validation.get("normalized_company", "") or "")
    if confirmed_by_user and normalized:
        validation["needs_review"] = False
        validation["confidence"] = "high"
        confirmed_at = confirmed_at or utc_timestamp()
    return {
        "company_raw": raw_company,
        "company_normalized": normalized,
        "company_confidence": validation["confidence"],
        "company_needs_review": bool(validation["needs_review"]),
        "company_evidence": validation["evidence"],
        "company_candidates": validation["candidates"],
        "company_confirmed_by_user": bool(confirmed_by_user if confirmed_by_user is not None else job.get("company_confirmed_by_user")),
        "company_confirmed_at": confirmed_at or str(job.get("company_confirmed_at", "") or ""),
    }


def markdown_metadata_from_verification(fields: dict[str, Any]) -> dict[str, str]:
    """Flatten verification fields into simple Markdown metadata lines."""
    candidates = [
        str(item.get("normalized_company", "") or item.get("company", "")).strip()
        for item in fields.get("company_candidates", []) or []
    ]
    return {
        "Company Raw": str(fields.get("company_raw", "") or ""),
        "Company Normalized": str(fields.get("company_normalized", "") or ""),
        "Company Confidence": str(fields.get("company_confidence", "") or ""),
        "Company Needs Review": str(bool(fields.get("company_needs_review"))),
        "Company Evidence": " | ".join(str(item) for item in fields.get("company_evidence", []) or []),
        "Company Candidates": ", ".join(dedupe_strings(candidates)),
        "Company Confirmed By User": str(bool(fields.get("company_confirmed_by_user"))),
        "Company Confirmed At": str(fields.get("company_confirmed_at", "") or ""),
    }


def upsert_markdown_fields(path: Path, fields: dict[str, str]) -> None:
    """Insert or update metadata fields before the first Markdown body heading."""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    body_index = next((index for index, line in enumerate(lines) if index > 0 and line.startswith("## ")), len(lines))
    existing = {
        line.split(":", 1)[0].strip().lower(): index
        for index, line in enumerate(lines[:body_index])
        if ":" in line
    }
    for field_name, value in fields.items():
        line = f"{field_name}: {value or 'Not provided'}"
        existing_index = existing.get(field_name.lower())
        if existing_index is None:
            lines.insert(body_index, line)
            body_index += 1
        else:
            lines[existing_index] = line
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def read_markdown_field(markdown_text: str, field_name: str, default: str = "") -> str:
    """Read a simple metadata field from Markdown."""
    prefix = f"{field_name}:"
    for line in markdown_text.splitlines():
        if line.lower().startswith(prefix.lower()):
            value = line.split(":", 1)[1].strip()
            if value and value.lower() != "not provided":
                return value
    return default


def parse_bool(value: Any) -> bool:
    """Parse booleans stored as Markdown strings."""
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def verification_from_markdown(path: Path) -> dict[str, Any]:
    """Load or backfill company verification metadata from a Markdown job file."""
    text = path.read_text(encoding="utf-8")
    company = read_markdown_field(text, "Company")
    source = read_markdown_field(text, "Source")
    role = read_markdown_field(text, "Role")
    location = read_markdown_field(text, "Location")
    confirmed = parse_bool(read_markdown_field(text, "Company Confirmed By User"))
    stored = {
        "company_raw": read_markdown_field(text, "Company Raw", company),
        "company_normalized": read_markdown_field(text, "Company Normalized"),
        "company_confidence": read_markdown_field(text, "Company Confidence"),
        "company_needs_review": parse_bool(read_markdown_field(text, "Company Needs Review", "true")),
        "company_evidence": [
            item.strip()
            for item in read_markdown_field(text, "Company Evidence").split("|")
            if item.strip()
        ],
        "company_candidates": [
            {"company": item.strip(), "normalized_company": item.strip(), "confidence": "medium", "evidence": []}
            for item in read_markdown_field(text, "Company Candidates").split(",")
            if item.strip()
        ],
        "company_confirmed_by_user": confirmed,
        "company_confirmed_at": read_markdown_field(text, "Company Confirmed At"),
    }
    if stored["company_normalized"] and stored["company_confidence"]:
        return stored

    source_confidence = "high" if source.lower() in {"adzuna", "jooble"} else "medium"
    fields = company_verification_fields(
        company,
        {
            "job_text": text,
            "source": source,
            "role": role,
            "location": location,
            "job_url": read_markdown_field(text, "Job URL"),
            "company_source_confidence": source_confidence if company else "",
            "company_source_evidence": "Found in existing Markdown company field." if company else "",
            "company_confirmed_by_user": confirmed,
            "company_confirmed_at": stored["company_confirmed_at"],
        },
    )
    return fields


def confirm_markdown_company(path: Path, company: str) -> dict[str, Any]:
    """Persist an edited/confirmed company name in a Markdown job file."""
    text = path.read_text(encoding="utf-8")
    fields = company_verification_fields(
        company,
        {
            "job_text": text,
            "role": read_markdown_field(text, "Role"),
            "location": read_markdown_field(text, "Location"),
            "job_url": read_markdown_field(text, "Job URL"),
        },
        confirmed_by_user=True,
        confirmed_at=utc_timestamp(),
    )
    upsert_markdown_fields(path, {"Company": fields["company_normalized"] or company, **markdown_metadata_from_verification(fields)})
    return fields


def cover_letter_company_is_verified(company: str, job: dict[str, Any] | None = None) -> tuple[bool, dict[str, Any]]:
    """Return whether cover-letter generation may use this company name."""
    validation = validate_company_name(company, job or {})
    confirmed = bool((job or {}).get("company_confirmed_by_user"))
    allowed = bool(validation["normalized_company"]) and (validation["confidence"] == "high" or confirmed) and not validation["issues"]
    return allowed, validation


def assert_cover_letter_company_verified(company: str, job: dict[str, Any] | None = None) -> dict[str, Any]:
    """Raise a clear blocking error unless the company is high-confidence/confirmed."""
    allowed, validation = cover_letter_company_is_verified(company, job or {})
    if allowed:
        return validation
    candidates = [
        str(item.get("normalized_company", "") or item.get("company", ""))
        for item in validation.get("candidates", [])[:5]
        if str(item.get("normalized_company", "") or item.get("company", "")).strip()
    ]
    detail = ""
    if candidates:
        detail = " Suggested candidates: " + ", ".join(dedupe_strings(candidates)) + "."
    raise ValueError(
        "Company name needs confirmation before generating a cover letter. "
        "This prevents using the wrong company name in your application."
        + detail
    )
