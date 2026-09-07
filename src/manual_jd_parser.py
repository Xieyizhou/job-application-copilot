"""Pure, reviewable suggestions from manually imported job descriptions."""

from __future__ import annotations
import re
from pathlib import Path
from typing import Any
from company_verification import (
    infer_company_candidates,
    normalize_company_name,
)
from ml.jd_quality import (
    jd_quality_warning_messages,
)


LINKEDIN_HEADER_SEPARATORS = ["·", "•", " - ", " – ", " — "]


SECTION_ALIASES = {
    "about": ["about the job", "about this job", "about the role", "overview"],
    "role": ["the role", "role overview"],
    "responsibilities": [
        "what you'll do",
        "what you’ll do",
        "what you will do",
        "what you will be doing",
        "responsibilities",
        "responsibility",
        "day to day",
    ],
    "requirements": [
        "what we look for",
        "what we're looking for",
        "what we’re looking for",
        "we'd love to hear from you if you have",
        "we’d love to hear from you if you have",
        "technical foundations",
        "ways of working",
        "requirements",
        "qualifications",
        "minimum qualifications",
        "skills",
    ],
    "preferred_qualifications": ["preferred qualifications", "preferred", "nice to have", "bonus", "plus"],
    "logistics": ["logistics", "location", "compensation", "salary", "benefits"],
    "why_join": ["why join", "why join us", "why us"],
    "ai_note": ["a note on ai", "note on ai", "ai policy"],
    "equal_opportunity": ["equal opportunity", "accommodations", "equal opportunity and accommodations"],
    "visa_note": ["visa", "work authorization", "work authorisation", "sponsorship", "right to work"],
}


SECTION_HEADING_KEYWORDS = [alias for aliases in SECTION_ALIASES.values() for alias in aliases]


JOB_SECTION_HINTS = [
    "about the job",
    "the role",
    "what you'll do",
    "what you’ll do",
    "responsibilities",
    "requirements",
    "qualifications",
    "what we look for",
    "logistics",
    "visa",
    "work authorization",
]


INTERNAL_MARKER_PATTERNS = [
    r"^---\s*extracted text from file \d+:.*---$",
    r"^---\s*page\s+\d+\s*---$",
]


UI_NOISE_LINES = {
    "about the job",
    "actively reviewing applicants",
    "apply",
    "beta",
    "continue",
    "easy apply",
    "is this information helpful?",
    "log in",
    "login",
    "more",
    "options",
    "promoted",
    "promoted by hirer",
    "reposted",
    "save",
    "savesave",
    "saved",
    "school alumni",
    "share",
    "show match details",
    "show more",
    "show more options",
    "show less",
    "see how you compare",
    "keyboard_arrow_right",
    "linkedin",
    "sign in",
    "join now",
    "jobs",
    "people",
    "learning",
    "messaging",
    "notifications",
}


UI_NOISE_FRAGMENTS = [
    "actively reviewing applicants",
    "easy apply",
    "is this information helpful",
    "matches your job preferences",
    "over 100 applicants",
    "people you can reach out to",
    "promoted by hirer",
    "reposted",
    "school alumni",
    "see how you compare",
    "show match details",
    "show more options",
    "keyboard_arrow_right",
    "job collections",
    "similar jobs",
    "recommended for you",
    "recommended jobs",
    "your job alert",
    "set alert",
]


LINKEDIN_UI_NOISE_PREFIXES = tuple(sorted(UI_NOISE_LINES | {fragment for fragment in UI_NOISE_FRAGMENTS}))


STRICT_WORK_AUTHORIZATION_PHRASES = [
    "visa sponsorship",
    "no visa sponsorship",
    "work authorization",
    "work authorisation",
    "authorized to work",
    "authorised to work",
    "eligibility to work",
    "eligible to work",
    "work visa",
    "employment pass",
    "sponsorship",
    "work permit",
    "must have the right to work",
    "right to work",
    "existing work authorization",
    "existing work authorisation",
]


ROLE_TITLE_WORDS = [
    "engineer",
    "researcher",
    "analyst",
    "scientist",
    "developer",
    "intern",
    "internship",
    "fellow",
    "fellowship",
    "manager",
    "associate",
    "designer",
    "specialist",
    "consultant",
    "opportunities",
]


ROLE_MODIFIER_WORDS = [
    "software",
    "machine learning",
    "ml",
    "ai",
    "applied ai",
    "data",
    "research",
    "quantitative",
    "developer relations",
    "engineering",
    "platform",
    "infrastructure",
    "product",
    "prototyping",
    "speech",
    "community",
    "systems",
    "robotics",
    "backend",
    "frontend",
    "full stack",
    "strategy",
    "alpha",
]


TITLE_BODY_REJECTION_TERMS = [
    " builds ",
    " building ",
    " provides ",
    " opening ",
    " looking for ",
    " we are ",
    " you'll ",
    " you will ",
    " from the ",
    " runtime ",
    " underneath ",
    " focused on ",
    " designed for ",
]


KNOWN_LOCATION_NAMES = [
    "San Francisco",
    "Ho Chi Minh City",
    "Singapore",
    "Beijing",
    "Shanghai",
    "Shenzhen",
    "Hangzhou",
    "London",
    "New York",
    "Seattle",
    "Boston",
    "Tokyo",
    "China",
    "United States",
    "United Kingdom",
    "Asia-Pacific",
    "APAC",
]


def normalize_job_title(title: str) -> str:
    """Remove OCR/UI punctuation at title boundaries while preserving title text."""
    title = re.sub(r"\s+", " ", title or "").strip()
    title = title.removeprefix("#").strip()
    title = re.sub(r"^---\s*extracted text from file \d+:\s*", "", title, flags=re.IGNORECASE).strip()
    title = re.sub(r"\s*---$", "", title).strip()
    title = re.sub(r"^(job title|title|role|position)\s*[:\-]\s*", "", title, flags=re.IGNORECASE)
    title = re.sub(r"^\d{4}[/-]\d{1,2}[/-]\d{1,2}\s+\d{1,2}:\d{2}\s*", "", title).strip()
    title = re.sub(r"^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\s+\d{1,2}:\d{2}\s*", "", title).strip()
    title = re.sub(r"\s*\((?:china|singapore|remote|apac|asia-pacific|united states|united kingdom)\)\s*$", "", title, flags=re.IGNORECASE)
    title = re.sub(r"\s*[:|·•\-–—]+\s*$", "", title).strip()
    return title


def source_filenames_from_text(text: str) -> list[str]:
    """Read upload filenames from internal extraction separators."""
    filenames = []
    for line in text.splitlines():
        match = re.match(r"^---\s*extracted text from file \d+:\s*(.+?)\s*---$", line.strip(), flags=re.IGNORECASE)
        if match:
            filenames.append(match.group(1).strip())
    return filenames


def readable_filename_text(filename: str) -> str:
    """Convert an upload filename into lightweight title/company clues."""
    stem = Path(filename).stem
    stem = re.sub(r"\b(?:screenshot|screen shot|截屏)\b", " ", stem, flags=re.IGNORECASE)
    stem = re.sub(r"\b\d{4}[_-]\d{2}[_-]\d{2}(?:t\d{2}[_-]\d{2}[_-]\d{2})?\b", " ", stem, flags=re.IGNORECASE)
    stem = re.sub(r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b", " ", stem)
    stem = re.sub(r"\b\d{1,2}[_-]\d{1,2}[_-]\d{1,2}\b", " ", stem)
    stem = re.sub(r"\b\d{5,}\b", " ", stem)
    stem = stem.replace("_", " ").replace("-", " ")
    stem = re.sub(r"\s+", " ", stem)
    return stem.strip()


def has_role_title_word(text: str) -> bool:
    """Return True if a phrase contains a role-title noun."""
    normalized = normalize_job_text_for_keywords(text)
    return any(f" {word} " in normalized for word in ROLE_TITLE_WORDS)


def has_role_modifier_word(text: str) -> bool:
    """Return True if a phrase contains a domain/modifier usually used in titles."""
    normalized = normalize_job_text_for_keywords(text)
    return any(f" {word} " in normalized for word in ROLE_MODIFIER_WORDS)


def has_body_sentence_shape(text: str) -> bool:
    """Reject descriptive prose that happens to contain role-like words."""
    normalized = f" {text.lower()} "
    if any(term in normalized for term in TITLE_BODY_REJECTION_TERMS):
        return True
    if text.count(",") >= 1 or len(text.split()) > 9:
        return True
    return False


def extract_role_title_phrases(text: str) -> list[str]:
    """Find compact role-title phrases inside headers, filenames, and metadata."""
    normalized_text = " ".join(str(text or "").replace("_", " ").split())
    if not normalized_text:
        return []

    title_pattern = re.compile(
        r"\b((?:(?:senior|staff|principal|lead|junior|graduate|early[- ]career|applied|machine learning|"
        r"software|data|ai|ml|quantitative|research|developer relations|engineering|platform|infrastructure|product|"
        r"prototyping|speech|community|systems|robotics|backend|frontend|full[- ]stack|strategy|alpha)"
        r"(?:\s*/\s*|\s+|\s*&\s*)?){0,5}"
        r"(?:engineer|researcher|scientist|analyst|intern|internship|fellow|fellowship|developer|designer|"
        r"manager|specialist|associate|opportunities?)(?:\s*\([^)]+\))?)\b",
        flags=re.IGNORECASE,
    )
    phrases: list[str] = []
    for match in title_pattern.finditer(normalized_text):
        phrase = normalize_job_title(match.group(1))
        if phrase and is_plausible_job_title_line(phrase) and phrase.lower() not in {item.lower() for item in phrases}:
            phrases.append(smart_title_case(phrase))
    return phrases[:5]


def smart_title_case(value: str) -> str:
    """Title-case filename-derived phrases while preserving common acronyms."""
    words = []
    acronyms = {"ai", "ml", "nlp", "llm", "api", "apac"}
    for word in value.split():
        stripped = word.strip()
        if stripped.lower() in acronyms:
            words.append(stripped.upper())
        elif stripped.isupper() and len(stripped) <= 4:
            words.append(stripped)
        else:
            words.append(stripped[:1].upper() + stripped[1:])
    return " ".join(words)


def looks_like_date_or_url_line(line: str) -> bool:
    """Reject browser headers, page footers, and URLs as metadata values."""
    stripped = line.strip()
    lowered = stripped.lower()
    if "://" in lowered or lowered.startswith(("www.", "http")):
        return True
    if re.search(r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b", stripped):
        return True
    if re.search(r"\b\d{1,2}:\d{2}\b", stripped):
        return True
    if re.search(r"\b\d+/\d+\b", stripped) and len(stripped.split()) <= 4:
        return True
    return False


def looks_like_noise_line(line: str) -> bool:
    """Return True for short job-board UI lines and obvious OCR debris."""
    if is_internal_extraction_marker(line):
        return False
    normalized = " ".join(line.lower().split())
    compact = re.sub(r"[^a-z0-9]+", "", normalized)
    if not normalized:
        return False
    if normalized in UI_NOISE_LINES:
        return True
    if any(fragment in normalized for fragment in UI_NOISE_FRAGMENTS):
        return True
    if compact in {"eee", "ee", "oooo", "lll"}:
        return True
    if re.fullmatch(r"[\W_]{2,}", line.strip()):
        return True
    if re.fullmatch(r"(.)\1{2,}", compact):
        return True
    return False


def is_linkedin_ui_noise(line: str) -> bool:
    """Reject LinkedIn PDF/export chrome before field inference.

    LinkedIn PDF text often interleaves company/title/location with one-word UI
    actions. Those actions must never become company, title, location, or visa
    values.
    """
    normalized = " ".join(str(line or "").lower().split()).strip(" .,:;|-–—")
    if not normalized:
        return False
    compact = re.sub(r"[^a-z0-9]+", "", normalized)
    if normalized in UI_NOISE_LINES or compact in {re.sub(r"[^a-z0-9]+", "", item) for item in UI_NOISE_LINES}:
        return True
    return any(normalized.startswith(prefix) for prefix in LINKEDIN_UI_NOISE_PREFIXES)


def clean_linkedin_preference_artifacts(line: str) -> str:
    """Remove LinkedIn 'Matches your job preferences' text from header values."""
    value = " ".join(str(line or "").split())
    value = re.sub(r"Matches your job preferences,?\s*", " ", value, flags=re.IGNORECASE)
    value = re.sub(r"workplace type is\s*", " ", value, flags=re.IGNORECASE)
    value = re.sub(r"job type is\s*", " ", value, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", value).strip(" .,:;|-–—")


def is_internal_extraction_marker(line: str) -> bool:
    """Return True for internal extraction/page separators, not JD content."""
    stripped = line.strip()
    return any(re.match(pattern, stripped, flags=re.IGNORECASE) for pattern in INTERNAL_MARKER_PATTERNS)


def is_filename_like_line(line: str) -> bool:
    """Avoid inferring titles from upload filenames or metadata lines."""
    stripped = line.strip().lower()
    return bool(re.fullmatch(r"[\w .\-()]+\.(pdf|png|jpe?g|webp|txt|md)", stripped))


def clean_extracted_job_text(raw_text: str) -> str:
    """Clean OCR text conservatively, mostly by dropping standalone UI lines.

    The goal is to remove LinkedIn/job-board chrome without deleting real job
    content. For that reason, most filtering is line-based and exact/near-exact.
    """
    text = (raw_text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("{{:}}", " ").replace("{:}", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\.{4,}", "...", text)

    cleaned_lines = []
    previous_line = ""
    for raw_line in text.splitlines():
        line = raw_line.strip()
        line = re.sub(r"\s+", " ", line)
        line = re.sub(r"\s+([:;,.)])", r"\1", line)
        line = re.sub(r"([({])\s+", r"\1", line)
        if is_internal_extraction_marker(line):
            cleaned_lines.append(line)
            previous_line = line
            continue
        if looks_like_noise_line(line):
            continue
        if line and line == previous_line:
            continue
        cleaned_lines.append(line)
        if line:
            previous_line = line

    cleaned_text = "\n".join(cleaned_lines)
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)
    return cleaned_text.strip()


def infer_field_from_labeled_line(text: str, labels: list[str]) -> str:
    """Find a value after common labels such as Company: or Location:."""
    for line in text.splitlines()[:80]:
        clean_line = line.strip().strip("-*")
        for label in labels:
            match = re.match(rf"^{re.escape(label)}\s*[:\-]\s*(.+)$", clean_line, re.IGNORECASE)
            if match:
                return match.group(1).strip()
    return ""


def normalize_heading_text(line: str) -> str:
    """Normalize candidate section headings for alias matching."""
    line = line.strip().strip("#:-–—").strip()
    line = re.sub(r"\s+", " ", line)
    return line.lower()


def canonical_section_key(line: str) -> str:
    """Return the canonical section key for a recognized heading line."""
    if is_internal_extraction_marker(line) or is_filename_like_line(line):
        return ""
    normalized = normalize_heading_text(line)
    if not normalized or len(normalized) > 90:
        return ""
    for key, aliases in SECTION_ALIASES.items():
        if any(normalized == alias or normalized.startswith(f"{alias}:") for alias in aliases):
            return key
        if key == "why_join" and any(normalized.startswith(f"{alias} ") for alias in aliases):
            return key
    return ""


def parse_structured_sections(text: str) -> dict[str, list[str]]:
    """Parse common PDF/job-post sections while preserving line order.

    This uses explicit heading aliases so ambiguous text stays in the full
    editable description instead of becoming overconfident structured data.
    """
    sections: dict[str, list[str]] = {}
    current_key = ""
    for raw_line in text.splitlines():
        line = normalize_bullet_line(raw_line)
        if not line or is_internal_extraction_marker(line) or is_filename_like_line(line):
            continue
        section_key = canonical_section_key(line)
        if section_key:
            current_key = section_key
            sections.setdefault(current_key, [])
            continue
        if current_key and not looks_like_noise_line(line):
            sections.setdefault(current_key, []).append(line)
    return {key: lines for key, lines in sections.items() if lines}


def section_lines_to_text(lines: list[str], limit: int = 12) -> str:
    """Convert parsed section lines to compact newline text."""
    cleaned = []
    seen = set()
    for line in lines:
        clean_line = normalize_bullet_line(line)
        key = clean_line.lower()
        if clean_line and key not in seen:
            seen.add(key)
            cleaned.append(clean_line)
        if len(cleaned) >= limit:
            break
    return "\n".join(cleaned)


def split_linkedin_company_location(line: str) -> tuple[str, str]:
    """Split LinkedIn-style header lines into company and location."""
    clean_line = " ".join(line.strip().split())
    if looks_like_date_or_url_line(clean_line):
        return "", ""
    for separator in LINKEDIN_HEADER_SEPARATORS:
        if separator not in clean_line:
            continue
        left, right = clean_line.split(separator, 1)
        company = left.strip(" -·•")
        location = clean_location_value(right.strip(" -·•"))
        if (
            company
            and location
            and len(company) <= 100
            and len(location) <= 120
            and not looks_like_date_or_url_line(company)
            and looks_like_location_phrase(location)
        ):
            return company, location
    return "", ""


def parse_linkedin_pdf_header(lines: list[str]) -> dict[str, Any]:
    """Parse LinkedIn PDF/export header fields before generic heuristics run.

    Expected shape is company, UI controls, title, location/repost/applicants,
    then workplace type and job type. The parser intentionally ignores LinkedIn
    UI lines before deciding what can be a company or title.
    """
    meaningful_lines = []
    raw_top_lines = [
        " ".join(str(line or "").split())
        for line in lines[:40]
        if str(line or "").strip()
        and not is_internal_extraction_marker(str(line))
        and not is_filename_like_line(str(line))
    ]
    for line in raw_top_lines:
        if is_linkedin_ui_noise(line):
            continue
        meaningful_lines.append(line)

    result: dict[str, Any] = {
        "company": "",
        "job_title": "",
        "location": "",
        "job_type": "",
        "source": "",
        "confidence": {},
        "evidence": {},
    }
    if not meaningful_lines:
        return result

    # Priority 2: LinkedIn sometimes repeats title above "Company · Location".
    for index, line in enumerate(meaningful_lines[:16]):
        company, location = split_linkedin_company_location(line)
        if company and location:
            title = ""
            for previous_line in reversed(meaningful_lines[:index]):
                if is_plausible_job_title_line(previous_line) and normalize_job_title(previous_line) != clean_company_candidate(company):
                    title = normalize_job_title(previous_line)
                    break
            result.update(
                {
                    "company": clean_company_candidate(company),
                    "job_title": title,
                    "location": location,
                    "source": "LinkedIn",
                }
            )
            result["confidence"] = {"company": "high", "job_title": "high" if title else "low", "location": "high"}
            result["evidence"] = {
                "company": "Found in LinkedIn company/location header.",
                "job_title": "Found in LinkedIn header before company/location line." if title else "",
                "location": f"Found in LinkedIn company/location header: {line}",
            }
            break

    if not result["company"]:
        first_line = meaningful_lines[0]
        if looks_like_company_candidate(first_line):
            result["company"] = clean_company_candidate(first_line)
            result["source"] = "LinkedIn"
            result["confidence"]["company"] = "high"
            result["evidence"]["company"] = "Found as first meaningful line before LinkedIn UI controls."

    title_counts: dict[str, int] = {}
    for line in meaningful_lines[:18]:
        normalized_title = normalize_job_title(line)
        if (
            normalized_title
            and normalized_title != result.get("company")
            and is_plausible_job_title_line(normalized_title)
            and not looks_like_company_candidate(normalized_title)
        ):
            title_counts[normalized_title] = title_counts.get(normalized_title, 0) + 1
    if title_counts and not result["job_title"]:
        title = sorted(title_counts, key=lambda item: (title_counts[item], len(item)), reverse=True)[0]
        result["job_title"] = title
        result["confidence"]["job_title"] = "high" if title_counts[title] >= 2 else "high"
        result["evidence"]["job_title"] = "Found in LinkedIn header after UI controls."

    location_value = result.get("location", "")
    workplace_type = ""
    for line in meaningful_lines[:18]:
        if not location_value:
            location_candidate = parse_linkedin_location_line(line)
            if location_candidate:
                location_value = location_candidate
        workplace_candidate = parse_linkedin_workplace_type(line)
        if workplace_candidate:
            workplace_type = workplace_candidate
        job_type = parse_linkedin_job_type(line)
        if job_type:
            result["job_type"] = job_type

    if location_value:
        if workplace_type and f"({workplace_type})" not in location_value:
            location_value = f"{location_value} ({workplace_type})"
        result["location"] = location_value
        result["confidence"]["location"] = "high"
        result["evidence"]["location"] = f"Found in LinkedIn header line with {location_value}."
    if result["job_type"]:
        result["evidence"]["job_type"] = f"Found LinkedIn job type: {result['job_type']}."
    return result


def parse_linkedin_location_line(line: str) -> str:
    """Extract the location part from LinkedIn repost/applicant header lines."""
    clean_line = clean_linkedin_preference_artifacts(line)
    if is_linkedin_ui_noise(clean_line):
        return ""
    first_part = clean_line.split("·", 1)[0].strip()
    first_part = re.sub(r"\b(?:reposted|over\s+\d+\s+applicants?).*$", "", first_part, flags=re.IGNORECASE).strip()
    if "·" not in clean_line and "," not in first_part and not looks_like_location_only(first_part):
        return ""
    if looks_like_location_phrase(first_part) and not has_role_title_word(first_part):
        return clean_location_value(first_part)
    return ""


def parse_linkedin_workplace_type(line: str) -> str:
    """Extract Hybrid/Remote/On-site from LinkedIn preference artifact lines."""
    cleaned = clean_linkedin_preference_artifacts(line)
    match = re.search(r"\b(Hybrid|Remote|On-site|Onsite)\b", cleaned, flags=re.IGNORECASE)
    if not match:
        return ""
    value = match.group(1).replace("Onsite", "On-site")
    return value[:1].upper() + value[1:].lower()


def parse_linkedin_job_type(line: str) -> str:
    """Extract job type from LinkedIn preference artifact lines."""
    cleaned = clean_linkedin_preference_artifacts(line)
    match = re.search(r"\b(Full-time|Full time|Part-time|Part time|Contract|Temporary|Internship)\b", cleaned, flags=re.IGNORECASE)
    if not match:
        return ""
    value = match.group(1).replace("Full time", "Full-time").replace("Part time", "Part-time")
    return value[:1].upper() + value[1:].lower()


def infer_title_company_location_from_header(lines: list[str]) -> tuple[str, str, str]:
    """Infer title/company/location from the first plausible LinkedIn header."""
    candidate_lines = [
        line
        for line in lines[:30]
        if line
        and not looks_like_noise_line(line)
        and not is_internal_extraction_marker(line)
        and not is_filename_like_line(line)
    ]
    for index, line in enumerate(candidate_lines[:12]):
        company, location = split_linkedin_company_location(line)
        if not company or not location:
            continue

        title = ""
        for previous_line in reversed(candidate_lines[:index]):
            normalized_title = normalize_job_title(previous_line)
            if (
                normalized_title
                and is_plausible_job_title_line(normalized_title)
                and not split_linkedin_company_location(previous_line)[0]
                and not canonical_section_key(previous_line)
            ):
                title = normalized_title
                break
        return title, company, location

    title = ""
    for line in candidate_lines[:8]:
        if split_linkedin_company_location(line)[0]:
            continue
        if canonical_section_key(line):
            continue
        if is_plausible_job_title_line(line):
            title = normalize_job_title(line)
            break
    return title, "", ""


def is_plausible_job_title_line(line: str) -> bool:
    """Return True for short lines that look like role titles, not prose."""
    clean_line = normalize_job_title(line)
    if not clean_line or len(clean_line) > 100 or clean_line.endswith("."):
        return False
    if is_linkedin_ui_noise(clean_line):
        return False
    if looks_like_date_or_url_line(clean_line):
        return False
    if has_body_sentence_shape(clean_line) and not (has_role_title_word(clean_line) and len(clean_line.split()) <= 10):
        return False
    if is_internal_extraction_marker(clean_line) or is_filename_like_line(clean_line) or canonical_section_key(clean_line):
        return False
    if len(clean_line.split()) == 1 and clean_line.lower() not in {"intern", "internship", "fellow", "fellowship"}:
        return False
    return has_role_title_word(clean_line) and (has_role_modifier_word(clean_line) or len(clean_line.split()) <= 4)


def title_confidence_for(title: str, evidence: str) -> str:
    """Assign confidence for title inference; avoid body prose titles."""
    clean_title = normalize_job_title(title)
    if not clean_title:
        return "low"
    if not is_plausible_job_title_line(clean_title):
        return "low"
    if any(
        marker in evidence.lower()
        for marker in ["linkedin-style header", "linkedin pdf header", "linkedin header", "explicit field", "top document line", "filename clue", "pdf title metadata"]
    ):
        return "high"
    return "medium"


def infer_title_from_source_clues(text: str, metadata_titles: list[str] | None = None) -> tuple[str, str, str]:
    """Use filenames/PDF titles as title clues without trusting long page titles."""
    candidates: list[tuple[str, str]] = []
    for filename in source_filenames_from_text(text):
        for phrase in extract_role_title_phrases(readable_filename_text(filename)):
            candidates.append((phrase, f"Detected from filename clue: {filename}"))
    for metadata_title in metadata_titles or []:
        for phrase in extract_role_title_phrases(metadata_title):
            candidates.append((phrase, f"Detected from PDF title metadata: {metadata_title}"))

    for title, evidence in candidates:
        confidence = title_confidence_for(title, evidence)
        if confidence in {"high", "medium"}:
            return title, confidence, evidence
    return "", "low", ""


def infer_title_from_role_context(sections: dict[str, list[str]], text: str) -> tuple[str, str, str]:
    """Infer medium-confidence fellowship/internship titles from early role prose."""
    role_text = " ".join(sections.get("role", [])[:4])
    early_text = role_text or " ".join(
        line.strip()
        for line in text.splitlines()[:25]
        if line.strip() and not looks_like_noise_line(line) and not is_internal_extraction_marker(line)
    )
    normalized = early_text.lower()
    if "fellowship" in normalized and any(term in normalized for term in ["software engineering", "applied ai", "research"]):
        return "AI / Engineering Fellowship", "medium", "Detected from early role text mentioning a fellowship."
    if "internship" in normalized and any(term in normalized for term in ["software engineering", "engineering", "research", "ai"]):
        return "Engineering Internship", "medium", "Detected from early role text mentioning an internship."
    if "early-career roles" in normalized and "engineering" in normalized and "ai" in normalized:
        return "AI & Engineering Opportunities", "medium", "Detected from early text describing multiple AI/engineering opportunities."
    return "", "low", ""


def clean_company_candidate(candidate: str) -> str:
    """Normalize company candidates from headers, body statements, and filenames."""
    candidate = " ".join(str(candidate or "").replace("_", " ").split()).strip(" .,:;|-–—")
    candidate = re.sub(r"^\d{4}[/-]\d{1,2}[/-]\d{1,2}\s+\d{1,2}:\d{2}\s*", "", candidate).strip()
    return candidate


def looks_like_company_candidate(candidate: str) -> bool:
    """Return True for compact organization names, not browser titles or prose."""
    candidate = clean_company_candidate(candidate)
    lowered = f" {candidate.lower()} "
    if not candidate or len(candidate) > 80 or len(candidate.split()) > 5:
        return False
    if canonical_section_key(candidate):
        return False
    if is_linkedin_ui_noise(candidate):
        return False
    if looks_like_date_or_url_line(candidate):
        return False
    if has_role_title_word(candidate) or looks_like_location_only(candidate):
        return False
    if any(term in lowered for term in [" log in ", " careers ", " privacy ", " terms ", " open source ", " open-source "]):
        return False
    return bool(re.search(r"[A-Za-z]", candidate))


def infer_company_from_top_lines(lines: list[str], text: str) -> tuple[str, str, str]:
    """Use repeated top-of-document organization lines as company evidence."""
    body = normalize_job_text_for_keywords(text)
    for line in lines[:14]:
        candidate = clean_company_candidate(line)
        if not looks_like_company_candidate(candidate):
            continue
        normalized_candidate = normalize_job_text_for_keywords(candidate).strip()
        if normalized_candidate and body.count(f" {normalized_candidate} ") >= 2:
            return candidate, "high", f"Detected from repeated top document company line: {line}"
    return "", "low", ""


def infer_company_from_source_clues(text: str) -> tuple[str, str, str]:
    """Infer company from upload filenames when a role phrase separates the name."""
    for filename in source_filenames_from_text(text):
        readable = readable_filename_text(filename)
        title_phrases = extract_role_title_phrases(readable)
        if not title_phrases:
            continue
        first_title = title_phrases[0].lower()
        readable_lower = readable.lower()
        index = readable_lower.find(first_title.lower())
        if index <= 0:
            continue
        candidate = smart_title_case(readable[:index].strip())
        if looks_like_company_candidate(candidate):
            return candidate, "medium", f"Detected from filename before title phrase: {filename}"
    return "", "low", ""


def infer_company_from_body(text: str) -> tuple[str, str, str]:
    """Infer company from repeated organization-like mentions in PDF body text."""
    evidence_lines: list[str] = []
    candidate_counts: dict[str, int] = {}
    org_suffixes = r"(?:Research|Labs?|AI|Technologies|Systems|Capital|Investment|Inc\.?|LLC|LP|Ltd\.?|Corporation)"
    patterns = [
        rf"\b([A-Z][A-Za-z0-9&.-]*(?:\s+[A-Z][A-Za-z0-9&.-]*){{0,3}}\s+{org_suffixes})\b",
        r"\b([A-Z][A-Za-z0-9&.-]{2,})\s+(?:builds|provides|is an equal opportunity employer|is a|offers|is opening|is building)\b",
        r"\bWhy Join\s+([A-Z][A-Za-z0-9&.-]*(?:\s+[A-Z][A-Za-z0-9&.-]*){0,3})\b",
    ]
    for line in text.splitlines():
        if is_internal_extraction_marker(line) or canonical_section_key(line):
            continue
        for pattern in patterns:
            for match in re.finditer(pattern, line):
                candidate = clean_company_candidate(match.group(1))
                if len(candidate) < 3 or candidate.lower() in {"about", "the role", "logistics", "ai"}:
                    continue
                if candidate.lower().startswith(("use ", "note ")) or not looks_like_company_candidate(candidate):
                    continue
                candidate_counts[candidate] = candidate_counts.get(candidate, 0) + 1
                if len(evidence_lines) < 4:
                    evidence_lines.append(line.strip())

    if not candidate_counts:
        return "", "low", ""

    # Prefer organization-like full names, so "Menlo Research" wins over shorter
    # repeated aliases like "Menlo" when both appear in the PDF body.
    def company_rank(candidate: str) -> tuple[int, int, int]:
        has_org_suffix = int(
            bool(
                re.search(
                    r"\b(?:Research|Labs?|AI|Technologies|Systems|Capital|Investment|Inc\.?|LLC|LP|Ltd\.?|Corporation)\b",
                    candidate,
                )
            )
        )
        return (has_org_suffix, candidate_counts[candidate], len(candidate))

    best = sorted(candidate_counts, key=company_rank, reverse=True)[0]
    confidence = "medium" if candidate_counts[best] >= 1 else "low"
    if candidate_counts[best] >= 2:
        confidence = "high"
    evidence = "Detected from repeated organization mentions: " + " | ".join(deduplicate_preserving_order(evidence_lines)[:3])
    return best, confidence, evidence


def deduplicate_preserving_order(values: list[str]) -> list[str]:
    """Deduplicate evidence snippets without changing their first-seen order."""
    seen = set()
    deduped = []
    for value in values:
        if value not in seen:
            seen.add(value)
            deduped.append(value)
    return deduped


def clean_location_value(location: str) -> str:
    """Clean location text while preserving multi-location meaning."""
    value = " ".join(str(location or "").replace("\n", " ").split())
    value = re.sub(r"\b(?:paid role|full[- ]time role|apply now|careers?)\b.*$", "", value, flags=re.IGNORECASE).strip()
    value = re.sub(r"https?://\S+", "", value).strip()
    value = value.strip(" ,;:|·•-–—")
    value = re.sub(r"\.{2,}$", "", value).strip()
    return value


def detected_known_locations(text: str) -> list[str]:
    """Return known city/region names in first-seen order."""
    detected = []
    for name in KNOWN_LOCATION_NAMES:
        pattern = r"\b" + re.escape(name).replace(r"\ ", r"\s+") + r"\b"
        if re.search(pattern, text, flags=re.IGNORECASE):
            normalized_name = "Asia-Pacific" if name.upper() == "APAC" else name
            if normalized_name not in detected:
                detected.append(normalized_name)
    return detected


def looks_like_location_phrase(value: str) -> bool:
    """Identify exact, remote, or regional location text."""
    cleaned = clean_location_value(value)
    lowered = cleaned.lower()
    if not cleaned or looks_like_date_or_url_line(cleaned):
        return False
    if any(term in lowered for term in ["remote", "hybrid", "office", "based in", "work model"]):
        return True
    if detected_known_locations(cleaned):
        if "open-source" in lowered or "conversational ai" in lowered:
            return False
        return True
    return False


def looks_like_location_only(value: str) -> bool:
    """Return True for pure location strings, not organizations with place words."""
    cleaned = clean_location_value(value)
    lowered = cleaned.lower()
    if not cleaned:
        return False
    if lowered in {"remote", "hybrid", "on-site", "onsite"}:
        return True
    if re.fullmatch(r"[A-Za-z .'-]+,\s*[A-Za-z .'-]+(?:\s*\([^)]+\))?", cleaned):
        return bool(detected_known_locations(cleaned))
    known_locations = detected_known_locations(cleaned)
    return len(known_locations) == 1 and lowered == known_locations[0].lower()


def split_location_options(raw_locations: str) -> list[str]:
    """Split office location prose into readable options."""
    cleaned = clean_location_value(raw_locations)
    cleaned = re.sub(r"\b(?:one of our offices|our offices|offices|office locations?)\b", "", cleaned, flags=re.IGNORECASE)
    known_locations = detected_known_locations(cleaned)
    if known_locations:
        return known_locations[:8]
    if "remote" in cleaned.lower():
        return ["Remote"]
    cleaned = cleaned.replace(" and ", ", ").replace(" or ", ", ")
    parts = [clean_location_value(part) for part in cleaned.split(",")]
    deduped = []
    for option in parts:
        if not option or option.lower() in {"based in", "location", "locations"}:
            continue
        if option and option not in deduped and len(option) <= 80:
            deduped.append(option)
    return deduped[:8]


def infer_locations_from_logistics(sections: dict[str, list[str]], text: str) -> tuple[str, list[str], str, str]:
    """Infer office locations from Logistics/location prose without choosing one arbitrarily."""
    logistics_lines = sections.get("logistics", [])
    search_text = " ".join(logistics_lines[:8]) or text
    evidence = ""
    location_options: list[str] = []

    for index, line in enumerate(logistics_lines[:4]):
        candidate_line = line
        if index + 1 < len(logistics_lines) and (line.rstrip().endswith(",") or logistics_lines[index + 1].lower().startswith(("or ", "and "))):
            candidate_line = f"{line} {logistics_lines[index + 1]}"
        options = split_location_options(candidate_line)
        if options and looks_like_location_phrase(candidate_line):
            evidence = candidate_line.strip()
            location_options = options
            break

    location_patterns = [
        r"(?:offices?|office locations?|based in one of our offices?)[:\s]+(.+?)(?:\.|$)",
        r"(?:location|locations)[:\s]+(.+?)(?:\.|$)",
        r"based in\s+(.+?)(?:\.|$)",
        r"([A-Z][A-Za-z -]+-based)",
        r"(Remote\s*-\s*[A-Z][A-Za-z ,]+)",
    ]
    if not location_options:
        for pattern in location_patterns:
            match = re.search(pattern, search_text, flags=re.IGNORECASE)
            if not match:
                continue
            evidence = match.group(0).strip()
            raw_locations = match.group(1) if match.lastindex else match.group(0)
            location_options = split_location_options(raw_locations)
            if location_options:
                break

    if not location_options:
        known_locations = detected_known_locations(search_text)
        if known_locations:
            evidence = search_text[:220]
            location_options = known_locations

    if not location_options:
        return "", [], "low", ""
    if location_options == ["Asia-Pacific"]:
        return "Region: Asia-Pacific", location_options, "medium", f"Found regional location text: {evidence}"
    if len(location_options) > 1:
        return (
            "Multiple offices: " + ", ".join(location_options),
            location_options,
            "medium",
            f"Found in Logistics/location text: {evidence}",
        )
    return location_options[0], location_options, "medium", f"Found in Logistics/location text: {evidence}"


def visa_evidence_for(text: str) -> tuple[str, str, str]:
    """Return concise visa note, confidence, and supporting evidence."""
    lines = split_sponsorship_sentences(text)
    evidence = " ".join(lines[:2])
    note = concise_visa_note(text)
    if not note:
        return "", "low", ""
    normalized = evidence.lower()
    high_signal = any(phrase in normalized for phrase in ["work authorization", "visa sponsorship", "sponsorship"])
    return note, "high" if high_signal else "medium", evidence


def split_sponsorship_sentences(text: str) -> list[str]:
    """Collect full sentences/lines that mention authorization or sponsorship."""
    candidates = []
    joined_text = join_wrapped_text_lines(text)
    for line in joined_text.splitlines():
        clean_line = line.strip()
        if not clean_line:
            continue
        for sentence in re.split(r"(?<=[.!?])\s+", clean_line):
            lowered = sentence.lower()
            if not has_strict_work_authorization_phrase(lowered):
                continue
            if looks_like_consent_or_background_check_text(lowered):
                continue
            if is_linkedin_ui_noise(sentence):
                continue
            if len(sentence.split()) > 80:
                continue
            candidates.append(sentence.strip())
    return candidates[:5]


def line_is_heading(line: str) -> bool:
    """Identify compact section heading lines from job descriptions."""
    return bool(canonical_section_key(line))


def normalize_bullet_line(line: str) -> str:
    """Clean one likely requirement/responsibility bullet."""
    return line.strip().strip("-*•·").strip()


def extract_section(text: str, heading_keywords: list[str]) -> str:
    """Extract likely section lines until the next heading-like line."""
    lines = text.splitlines()
    captured: list[str] = []
    collecting = False
    for line in lines:
        stripped = line.strip()
        lower = stripped.lower()
        is_heading = line_is_heading(stripped)
        if any(keyword in lower for keyword in heading_keywords) and is_heading:
            collecting = True
            continue
        if collecting and is_heading:
            break
        if collecting:
            clean_line = normalize_bullet_line(stripped)
            if clean_line and not looks_like_noise_line(clean_line):
                captured.append(clean_line)
    return "\n".join(captured[:12])


def collect_keyword_lines(text: str, keywords: list[str], limit: int = 8) -> list[str]:
    """Collect concrete lines that mention useful job-content keywords."""
    results = []
    seen = set()
    for index, raw_line in enumerate(text.splitlines()):
        line = normalize_bullet_line(raw_line)
        lower = line.lower()
        if not line or looks_like_noise_line(line) or line_is_heading(line):
            continue
        # Early screenshot lines are often title/company/location headers. Avoid
        # treating them as concrete requirements just because they contain words
        # like "quantitative", "research", or "machine learning".
        if index < 5 and len(line) <= 120 and raw_line[:1] not in {"-", "*", "•"} and not line.endswith("."):
            continue
        if split_linkedin_company_location(line)[0]:
            continue
        if not any(keyword in lower for keyword in keywords):
            continue
        if lower in seen:
            continue
        seen.add(lower)
        results.append(line)
        if len(results) >= limit:
            break
    return results


def merge_section_and_keyword_lines(section_text: str, keyword_lines: list[str], limit: int = 8) -> str:
    """Combine heading-based extraction with keyword matches without duplicates."""
    merged = []
    seen = set()
    for line in [*section_text.splitlines(), *keyword_lines]:
        clean_line = normalize_bullet_line(line)
        key = clean_line.lower()
        if clean_line and key not in seen:
            seen.add(key)
            merged.append(clean_line)
        if len(merged) >= limit:
            break
    return "\n".join(merged)


def find_authorization_lines(text: str) -> str:
    """Return lines mentioning visa, sponsorship, citizenship, or work authorization."""
    lines = []
    for line in join_wrapped_text_lines(text).splitlines():
        stripped = line.strip()
        lower = stripped.lower()
        if (
            stripped
            and has_strict_work_authorization_phrase(lower)
            and not looks_like_consent_or_background_check_text(lower)
        ):
            lines.append(stripped)
    return "\n".join(lines[:8])


def has_strict_work_authorization_phrase(text: str) -> bool:
    """Require explicit visa/work-authorization wording before filling visa note."""
    normalized = " ".join(str(text or "").lower().split())
    return any(phrase in normalized for phrase in STRICT_WORK_AUTHORIZATION_PHRASES)


def looks_like_consent_or_background_check_text(text: str) -> bool:
    """Avoid treating legal consent/data verification text as work authorization."""
    normalized = " ".join(str(text or "").lower().split())
    legal_markers = ["consent", "personal data", "verify", "verification", "background check", "privacy", "share my data"]
    return any(marker in normalized for marker in legal_markers) and not any(
        phrase in normalized
        for phrase in [
            "visa sponsorship",
            "work authorization",
            "work authorisation",
            "authorized to work",
            "authorised to work",
            "right to work",
            "work permit",
            "employment pass",
        ]
    )


def join_wrapped_text_lines(text: str) -> str:
    """Join PDF-wrapped prose lines so sentence-level checks see full clauses."""
    joined_lines = []
    buffer = ""
    for raw_line in text.splitlines():
        line = normalize_bullet_line(raw_line)
        if not line:
            if buffer:
                joined_lines.append(buffer)
                buffer = ""
            continue
        if is_internal_extraction_marker(line) or canonical_section_key(line):
            if buffer:
                joined_lines.append(buffer)
                buffer = ""
            joined_lines.append(line)
            continue
        if buffer and not buffer.endswith((".", "!", "?", ":")) and raw_line[:1] not in {"-", "*", "•", "¢"}:
            buffer = f"{buffer} {line}"
        else:
            if buffer:
                joined_lines.append(buffer)
            buffer = line
    if buffer:
        joined_lines.append(buffer)
    return "\n".join(joined_lines)


def concise_visa_note(text: str) -> str:
    """Summarize common visa/work authorization language."""
    authorization_text = find_authorization_lines(text)
    normalized = authorization_text.lower()
    if not authorization_text:
        return ""
    no_sponsorship = any(phrase in normalized for phrase in ["no visa sponsorship", "don't provide visa sponsorship", "do not provide visa sponsorship", "no sponsorship"])
    existing_required = any(phrase in normalized for phrase in ["existing work authorization", "authorized to work", "right to work", "work authorization required"])
    if no_sponsorship and existing_required:
        return "Existing work authorization required; no visa sponsorship."
    if no_sponsorship:
        return "No visa sponsorship."
    if existing_required:
        return "Existing work authorization required."
    return authorization_text


def extract_keywords(text: str) -> list[str]:
    """Return a short deterministic keyword list for compact parser display."""
    keyword_catalog = [
        "Python",
        "PyTorch",
        "TensorFlow",
        "LLM",
        "NLP",
        "speech recognition",
        "debugging",
        "AI",
        "machine learning",
        "systems",
        "infrastructure",
        "ownership",
        "research",
        "production",
        "coding",
        "statistics",
        "finance",
        "robotics",
        "open-source",
        "developer relations",
        "work authorization",
        "visa sponsorship",
    ]
    normalized = normalize_job_text_for_keywords(text)
    keywords = []
    for keyword in keyword_catalog:
        if normalize_job_text_for_keywords(keyword).strip() in normalized:
            keywords.append(keyword)
    return keywords[:8]


def normalize_job_text_for_keywords(text: str) -> str:
    """Normalize text for parser keyword matching."""
    return " " + re.sub(r"[^a-z0-9+#]+", " ", text.lower()).strip() + " "


def summarize_lines(lines: list[str], fallback: str = "") -> str:
    """Create a compact non-fabricated summary from parsed lines."""
    if not lines:
        return fallback
    joined = " ".join(lines[:4])
    lowered = joined.lower()
    themes = []
    if any(word in lowered for word in ["own", "ownership", "end to end"]):
        themes.append("ownership")
    if any(word in lowered for word in ["production", "ship", "build", "implement", "systems", "infrastructure"]):
        themes.append("production-grade engineering")
    if any(word in lowered for word in ["ai", "machine learning", "research", "model"]):
        themes.append("AI/research work")
    if any(word in lowered for word in ["debug", "debugging", "codebase", "coding"]):
        themes.append("coding and debugging")
    if any(word in lowered for word in ["collaborate", "researcher", "team"]):
        themes.append("collaboration")
    if themes:
        return "Emphasizes " + ", ".join(themes[:4]) + "."
    return lines[0][:220]


def section_summaries_for(sections: dict[str, list[str]]) -> dict[str, str]:
    """Build compact summaries for parsed sections."""
    return {
        "responsibilities": summarize_lines(sections.get("responsibilities", []), "No responsibilities section detected."),
        "requirements": summarize_lines(sections.get("requirements", []), "No requirements section detected."),
        "logistics": summarize_lines(sections.get("logistics", []), "No logistics section detected."),
    }


def infer_employment_type(text: str) -> tuple[str, str]:
    """Detect internship/fellowship/full-time status from common role language."""
    normalized = normalize_job_text_for_keywords(text)
    has_fellowship = " fellowship " in normalized or " fellow " in normalized
    has_internship = " internship " in normalized or " intern " in normalized
    if has_fellowship and has_internship:
        return "Fellowship / Internship", "Fellowship and internship language detected."
    if has_fellowship:
        return "Fellowship", "Fellowship language detected."
    if has_internship:
        return "Internship", "Internship language detected."
    if " full time " in normalized or " full-time " in text.lower():
        return "Full-time", "Full-time language detected."
    if " part time " in normalized or " part-time " in text.lower():
        return "Part-time", "Part-time language detected."
    if " contract " in normalized or " contractor " in normalized:
        return "Contract", "Contract language detected."
    if " early career " in normalized or " early-career " in text.lower():
        return "Early-career", "Early-career language detected."
    return "Unknown", ""


def build_role_summary(
    *,
    title: str,
    company: str,
    employment_type: str,
    keywords: list[str],
    sections: dict[str, list[str]],
) -> str:
    """Create one user-facing, non-fabricated sentence from detected facts."""
    title_part = title or "This role"
    company_part = f" at {company}" if company else ""
    type_part = "" if employment_type == "Unknown" else f" ({employment_type.lower()})"
    focus_keywords = [keyword for keyword in keywords if keyword not in {"work authorization", "visa sponsorship"}][:3]
    if focus_keywords:
        return f"{title_part}{company_part}{type_part} appears focused on {', '.join(focus_keywords)}."
    responsibility_summary = summarize_lines(sections.get("responsibilities", []), "")
    if responsibility_summary:
        return f"{title_part}{company_part}{type_part}: {responsibility_summary}"
    return f"{title_part}{company_part}{type_part}; review the full description before generating a package."


def parse_job_description_suggestions(text: str, source_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    """Infer useful fields and sections with simple keyword heuristics."""
    source_metadata = source_metadata or {}
    metadata_titles = [
        str(title).strip()
        for title in source_metadata.get("metadata_titles", [])
        if str(title).strip()
    ]
    stripped_lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not is_internal_extraction_marker(line) and not is_filename_like_line(line)
    ]
    structured_sections = parse_structured_sections(text)
    linkedin_header = parse_linkedin_pdf_header(stripped_lines)
    title = infer_field_from_labeled_line(text, ["Role", "Title", "Job Title", "Position"])
    company = infer_field_from_labeled_line(text, ["Company", "Employer", "Organization"])
    location = infer_field_from_labeled_line(text, ["Location", "Office", "Work location"])
    header_title, header_company, header_location = infer_title_company_location_from_header(stripped_lines)
    title_evidence = ""
    company_evidence = ""
    location_evidence = ""
    company_confidence = "low"
    location_confidence = "low"
    location_options: list[str] = []

    # Improved field inference: explicit fields and clean top headers are strong,
    # while filename/PDF-title/context clues remain reviewable suggestions.
    if linkedin_header.get("job_title"):
        title = str(linkedin_header["job_title"])
        title_evidence = str((linkedin_header.get("evidence") or {}).get("job_title", "Found in LinkedIn PDF header."))
    elif title:
        title_evidence = "Detected from explicit title/role field."
    elif header_title:
        title = header_title
        title_evidence = "Detected from LinkedIn-style header or top document line."
    else:
        source_title, source_title_confidence, source_title_evidence = infer_title_from_source_clues(text, metadata_titles)
        if source_title:
            title = source_title
            title_evidence = source_title_evidence
        else:
            context_title, source_title_confidence, source_title_evidence = infer_title_from_role_context(structured_sections, text)
            if context_title:
                title = context_title
                title_evidence = source_title_evidence

    if linkedin_header.get("company"):
        company = str(linkedin_header["company"])
        company_evidence = str((linkedin_header.get("evidence") or {}).get("company", "Found in LinkedIn PDF header."))
        company_confidence = str((linkedin_header.get("confidence") or {}).get("company", "high"))
    elif not company:
        if header_company:
            company = header_company
            company_evidence = "Detected from LinkedIn-style header."
            company_confidence = "high"
        else:
            company = infer_field_from_labeled_line(text, ["Hiring company"])
            company_evidence = "Detected from hiring company field." if company else ""
            company_confidence = "high" if company else "low"
    else:
        company_evidence = "Detected from explicit company/employer field."
        company_confidence = "high"
    if company and not looks_like_company_candidate(company):
        company = ""
        company_evidence = ""
        company_confidence = "low"

    if linkedin_header.get("location"):
        location = str(linkedin_header["location"])
        location_evidence = str((linkedin_header.get("evidence") or {}).get("location", "Found in LinkedIn PDF header."))
        location_confidence = str((linkedin_header.get("confidence") or {}).get("location", "high"))
    elif not location:
        if header_location:
            location = header_location
            location_evidence = "Detected from LinkedIn-style header."
            location_confidence = "high"
    else:
        location = clean_location_value(location)
        location_evidence = "Detected from explicit location/office field."
        location_confidence = "high" if looks_like_location_phrase(location) else "medium"

    if not company:
        company, company_confidence, company_evidence = infer_company_from_top_lines(stripped_lines, text)
    if not company:
        company, company_confidence, company_evidence = infer_company_from_body(text)
    if not company:
        company, company_confidence, company_evidence = infer_company_from_source_clues(text)
    company_candidates = infer_company_candidates(
        text,
        {
            "company": company,
            "metadata_titles": metadata_titles,
            "source": source_metadata.get("source", ""),
            "job_url": source_metadata.get("job_url", ""),
            "filename": (source_filenames_from_text(text) or [""])[0],
        },
    )
    if company:
        normalized_company = normalize_company_name(company)
        if normalized_company:
            company = normalized_company
    elif company_candidates:
        company = str(company_candidates[0].get("normalized_company", "") or "")
        company_confidence = str(company_candidates[0].get("confidence", "low"))
        evidence_lines = company_candidates[0].get("evidence", []) or []
        company_evidence = str(evidence_lines[0]) if evidence_lines else ""

    if not location:
        location, location_options, location_confidence, location_evidence = infer_locations_from_logistics(structured_sections, text)
    else:
        location = clean_location_value(location)
        location_options = [location]
        if not location_confidence:
            location_confidence = "high" if "header" in location_evidence.lower() or "explicit" in location_evidence.lower() else "medium"

    normalized_title = normalize_job_title(title)
    job_title_confidence = title_confidence_for(normalized_title, title_evidence)
    if job_title_confidence == "low":
        normalized_title = ""
    visa_note, visa_confidence, visa_evidence = visa_evidence_for(text)
    structured_requirements = section_lines_to_text(structured_sections.get("requirements", []))
    structured_responsibilities = section_lines_to_text(structured_sections.get("responsibilities", []))
    requirements = structured_requirements or merge_section_and_keyword_lines(
        extract_section(text, ["requirement", "qualification", "what you bring", "skills"]),
        collect_keyword_lines(
            text,
            [
                "degree",
                "bs",
                "ms",
                "phd",
                "python",
                "programming",
                "statistics",
                "experience",
                "required",
            ],
        ),
    )
    responsibilities = structured_responsibilities or merge_section_and_keyword_lines(
        extract_section(text, ["responsibil", "what you will do", "the role"]),
        collect_keyword_lines(
            text,
            [
                "develop",
                "design",
                "implement",
                "optimize",
                "parse",
                "collaborate",
                "experiment",
                "simulation",
                "research",
                "model",
                "strategy",
                "signals",
                "alpha",
            ],
        ),
    )
    preferred_qualifications = merge_section_and_keyword_lines(
        section_lines_to_text(structured_sections.get("preferred_qualifications", [])) or extract_section(text, ["preferred", "nice to have", "bonus", "plus"]),
        collect_keyword_lines(text, ["preferred", "nice to have", "bonus", "plus"], limit=5),
        limit=5,
    )
    sections_as_text = {key: section_lines_to_text(lines, limit=20) for key, lines in structured_sections.items()}
    summaries = section_summaries_for(structured_sections)
    keywords = extract_keywords(text)
    employment_type, employment_type_evidence = infer_employment_type(text)
    if linkedin_header.get("job_type"):
        employment_type = str(linkedin_header["job_type"])
        employment_type_evidence = str((linkedin_header.get("evidence") or {}).get("job_type", "Found in LinkedIn PDF header."))
    source = str(linkedin_header.get("source") or "")
    role_summary = build_role_summary(
        title=normalized_title,
        company=company,
        employment_type=employment_type,
        keywords=keywords,
        sections=structured_sections,
    )

    return {
        "company": company,
        "company_confidence": company_confidence,
        "company_evidence": company_evidence,
        "company_candidates": company_candidates,
        "title": normalized_title,
        "job_title": normalized_title,
        "job_title_confidence": job_title_confidence,
        "job_title_evidence": title_evidence,
        "location": location,
        "location_options": location_options,
        "location_confidence": location_confidence,
        "location_evidence": location_evidence,
        "source": source,
        "source_confidence": "high" if source == "LinkedIn" else "",
        "source_evidence": "Detected LinkedIn PDF/export header." if source == "LinkedIn" else "",
        "employment_type": employment_type,
        "employment_type_evidence": employment_type_evidence,
        "internship_fellowship_status": employment_type if employment_type in {"Internship", "Fellowship", "Fellowship / Internship"} else "",
        "about": section_lines_to_text(structured_sections.get("about", []), limit=8),
        "requirements": requirements,
        "responsibilities": responsibilities,
        "preferred_qualifications": preferred_qualifications,
        "logistics": section_lines_to_text(structured_sections.get("logistics", []), limit=8),
        "visa_note": visa_note,
        "visa_confidence": visa_confidence,
        "visa_evidence": visa_evidence,
        "keywords": keywords,
        "role_summary": role_summary,
        "section_summaries": summaries,
        "parsed_sections": sections_as_text,
    }


def job_description_quality_warnings(
    *,
    company: str,
    title: str,
    location: str,
    url: str,
    job_description: str,
) -> list[str]:
    """Return save-time warnings for missing metadata or suspicious OCR text."""
    warnings = []
    if not company.strip():
        warnings.append("Company name is empty.")
    elif not normalize_company_name(company) or is_linkedin_ui_noise(company):
        warnings.append("Company name looks incorrect. Please confirm before saving or generating a cover letter.")
    if not title.strip():
        warnings.append("Job title is empty.")
    if not location.strip():
        warnings.append("Location is empty.")
    if not url.strip():
        warnings.append("Job URL was not found in the upload. Please paste the original job link manually if available.")
    warnings.extend(jd_quality_warning_messages(job_description))
    if any(fragment in job_description.lower() for fragment in UI_NOISE_FRAGMENTS) or any(
        looks_like_noise_line(line) for line in job_description.splitlines()
    ):
        warnings.append("Job description may still contain OCR/UI noise.")
    return warnings
