"""Canonical job-description structure shared by scoring and presentation.

The parser is deliberately conservative: it reorganizes employer text and keeps
the original source span, but never invents a missing field.
"""

from __future__ import annotations

import re
from typing import Any, Literal, TypedDict

from ml.evidence import clean_source_line


RequirementKind = Literal[
    "skill", "experience", "education", "eligibility", "behavior", "responsibility"
]
RequirementDemand = Literal["required", "preferred", "responsibility", "constraint"]


class StructuredRequirement(TypedDict):
    text: str
    type: RequirementDemand
    category: RequirementKind
    source_span: str
    section: str
    extraction_confidence: float


class StructuredJob(TypedDict):
    title: str
    company: str
    location: str
    work_mode: str
    employment_type: str
    salary: str
    overview: str
    responsibilities: list[str]
    required_qualifications: list[str]
    preferred_qualifications: list[str]
    skills_tools: list[str]
    experience: list[str]
    education: list[str]
    eligibility: list[str]
    benefits: list[str]
    about_company: list[str]
    requirements: list[StructuredRequirement]
    source_text: str


METADATA_FIELDS = {
    "company", "company raw", "company normalized", "role", "location", "job url",
    "source", "source job id", "created at", "first seen at", "last seen at",
    "description source", "jd fetch status", "salary", "employment type", "work mode",
    "company confidence", "company confirmed by user",
}
SECTION_ALIASES = {
    "responsibilities": (
        "responsibilities", "what you will do", "what you'll do", "what you will be doing",
        "what you'll be doing", "what success looks like", "duties", "the role",
    ),
    "required": (
        "requirements", "required qualifications", "qualifications", "what you bring",
        "who you are", "who you probably are", "stack",
    ),
    "preferred": ("preferred qualifications", "nice to have", "bonus", "preferred", "additional skills"),
    "benefits": ("benefits", "what we offer", "what we can offer", "perks"),
    "company": ("about us", "about the company", "who we are", "we are"),
}
MODAL_REQUIREMENT = re.compile(
    r"\b(must|required|requires?|proficien(?:t|cy)|experience with|knowledge of|"
    r"familiarity with|ability to|strong (?:background|foundation)|you have)\b",
    re.I,
)
STANDALONE_REQUIREMENT = re.compile(
    r"^(?:must|required|requires?|proficien(?:t|cy)|experience with|knowledge of|"
    r"familiarity with|ability to|strong (?:background|foundation)|you have|you must)\b|"
    r"\byou (?:have|bring|need|know|write|communicate|are)\b|"
    r"\b(?:degree|years? of experience)\b.{0,60}\brequired\b",
    re.I,
)
PREFERRED_SIGNAL = re.compile(r"\b(preferred|nice to have|bonus|plus|desired)\b", re.I)
RESPONSIBILITY_SIGNAL = re.compile(
    r"^(?:design|build|develop|implement|analy[sz]e|manage|lead|create|maintain|"
    r"collaborate|support|deliver|own|conduct|monitor|optimi[sz]e|write)\b",
    re.I,
)
NEGATED_REQUIREMENT = re.compile(
    r"\b(?:no|not|without)\b.{0,140}\b(?:required|requirement|necessary|needed)\b",
    re.I,
)


def _metadata(job_text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in job_text.splitlines():
        if line.startswith("## "):
            break
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        normalized = key.strip().lower()
        if normalized in METADATA_FIELDS:
            values[normalized] = value.strip()
    heading = re.search(r"(?m)^#\s+(.+?)\s*$", job_text)
    if heading and "role" not in values:
        values["role"] = heading.group(1).strip()
    return values


def _description_body(job_text: str) -> str:
    marker = re.search(r"(?im)^##\s+Job Description\s*$", job_text)
    return job_text[marker.end() :].strip() if marker else job_text.strip()


def _section_name(value: str) -> str:
    normalized = re.sub(r"[^a-z ]+", " ", value.lower()).strip()
    for canonical, aliases in SECTION_ALIASES.items():
        if any(alias == normalized or alias in normalized for alias in aliases):
            return canonical
    return ""


def _segments(body: str) -> list[tuple[str, str, float]]:
    """Return source-preserving text segments with their nearest section."""
    logical_lines: list[str] = []
    paragraph: list[str] = []
    for raw in body.replace("\r", "\n").splitlines():
        stripped = raw.strip()
        metadata_key = stripped.split(":", 1)[0].strip().lower() if ":" in stripped else ""
        section_candidate = stripped.lstrip("#").strip().rstrip(":")
        section_heading = (
            bool(_section_name(section_candidate))
            and len(section_candidate.split()) <= 7
            and not re.search(r"[.!?]", section_candidate)
        )
        standalone = (
            not stripped
            or stripped.startswith(("#", "-", "*", "•"))
            or metadata_key in METADATA_FIELDS
            or section_heading
        )
        if standalone:
            if paragraph:
                logical_lines.append(" ".join(paragraph))
                paragraph = []
            if stripped:
                logical_lines.append(stripped)
        else:
            paragraph.append(stripped)
    if paragraph:
        logical_lines.append(" ".join(paragraph))
    prepared = "\n".join(logical_lines)
    prepared = re.sub(r"\s*[•·▪◦]\s*", "\n- ", prepared)
    flat_markers = {
        "what you'll be doing": "Responsibilities",
        "what you will be doing": "Responsibilities",
        "what success looks like": "Responsibilities",
        "who you probably are": "Required Qualifications",
        "who you are": "Required Qualifications",
        "stack": "Required Qualifications",
    }
    for marker, heading in flat_markers.items():
        prepared = re.sub(
            rf"(?i)(?<![a-z]){re.escape(marker)}(?=\s|:)",
            f"\n## {heading}\n",
            prepared,
        )
    # Provider payloads often flatten headings into the same line.
    heading_terms = sorted({alias for aliases in SECTION_ALIASES.values() for alias in aliases}, key=len, reverse=True)
    for heading in heading_terms:
        prepared = re.sub(
            rf"(?i)(?<![a-z]){re.escape(heading)}\s*:",
            lambda match: f"\n## {match.group(0).rstrip(':').strip()}\n",
            prepared,
        )
    rows: list[tuple[str, str, float]] = []
    current_section = ""
    for raw in prepared.splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        heading = stripped.lstrip("#").strip().rstrip(":")
        recognized = _section_name(heading)
        heading_only = recognized and len(heading.split()) <= 7 and not re.search(r"[.!?]", heading)
        if stripped.startswith("#") or heading_only:
            current_section = recognized or heading.lower()
            continue
        cleaned = clean_source_line(stripped)
        if not cleaned or (":" in cleaned and cleaned.split(":", 1)[0].strip().lower() in METADATA_FIELDS):
            continue
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", cleaned)
        for part in parts:
            part = part.strip().rstrip(".;")
            if len(part.split()) >= 3 or (current_section in {"required", "preferred"} and part):
                confidence = 0.96 if stripped.startswith(("-", "*", "•")) else 0.82 if current_section else 0.68
                rows.append((current_section, part, confidence))
    return rows


def _category(text: str) -> RequirementKind:
    lower = text.lower()
    if re.search(r"\b(bachelor|master|ph\.?d|degree|diploma|education)\b", lower):
        return "education"
    if re.search(r"\b(years? of experience|senior|junior|entry.level|professional experience)\b", lower):
        return "experience"
    if re.search(r"\b(visa|sponsorship|citizen|work authori[sz]ation|security clearance|right to work)\b", lower):
        return "eligibility"
    if re.search(r"\b(communication|collaboration|leadership|stakeholder|team|interpersonal)\b", lower):
        return "behavior"
    return "skill"


def _atomic_requirement_texts(text: str) -> list[str]:
    """Split compact provider lists into independently reviewable requirements."""
    value = text.strip()
    if " — " in value:
        lead, tail = value.split(" — ", 1)
        if tail.count(",") >= 2 and len(lead.split()) <= 8:
            items = [item.strip(" .") for item in re.split(r",\s*|\s+and\s+", tail)]
            return [lead.strip(" ."), *(item for item in items if len(item.split()) <= 8)]
    if value.count(",") >= 2 and len(value.split()) <= 14 and not re.search(r"[.!?]", value):
        items = [item.strip(" .") for item in re.split(r",\s*|\s+and\s+", value)]
        return [item for item in items if item]
    return [value]


def _requirement(section: str, text: str, confidence: float) -> StructuredRequirement | None:
    preferred = section == "preferred" or bool(PREFERRED_SIGNAL.search(text))
    responsibility = section == "responsibilities" or bool(RESPONSIBILITY_SIGNAL.search(text))
    explicit = bool(MODAL_REQUIREMENT.search(text))
    if NEGATED_REQUIREMENT.search(text):
        return None
    if not section and explicit and not STANDALONE_REQUIREMENT.search(text):
        return None
    if not (preferred or responsibility or explicit or section == "required"):
        return None
    category = "responsibility" if responsibility and not (preferred or explicit) else _category(text)
    demand: RequirementDemand = "preferred" if preferred else "responsibility" if category == "responsibility" else "constraint" if category == "eligibility" else "required"
    return {
        "text": text,
        "type": demand,
        "category": category,
        "source_span": text,
        "section": section or "job description",
        "extraction_confidence": round(confidence, 2),
    }


def structure_job_description(job_text: str) -> StructuredJob:
    """Normalize one raw or saved JD into a stable, source-backed schema."""
    metadata = _metadata(job_text)
    body = _description_body(job_text)
    rows = _segments(body)
    grouped: dict[str, list[str]] = {key: [] for key in ("responsibilities", "required", "preferred", "benefits", "company")}
    requirements: list[StructuredRequirement] = []
    seen: set[str] = set()
    overview_parts: list[str] = []
    for section, text, confidence in rows:
        if section in grouped:
            grouped[section].append(text)
        elif len(overview_parts) < 2:
            overview_parts.append(text)
        for atomic_text in _atomic_requirement_texts(text):
            record = _requirement(section, atomic_text, confidence)
            if record and record["text"].casefold() not in seen:
                record["source_span"] = text
                requirements.append(record)
                seen.add(record["text"].casefold())

    required = [row["text"] for row in requirements if row["type"] in {"required", "constraint"}]
    preferred = [row["text"] for row in requirements if row["type"] == "preferred"]
    responsibilities = list(dict.fromkeys(grouped["responsibilities"] + [row["text"] for row in requirements if row["type"] == "responsibility"]))
    experience = [row["text"] for row in requirements if row["category"] == "experience"]
    education = [row["text"] for row in requirements if row["category"] == "education"]
    eligibility = [row["text"] for row in requirements if row["category"] == "eligibility"]
    work_mode = metadata.get("work mode", "")
    location = metadata.get("location", "")
    if not work_mode:
        work_mode = "Remote" if "remote" in location.lower() or re.search(r"\bremote\b", body, re.I) else ""
    return {
        "title": metadata.get("role", ""),
        "company": metadata.get("company normalized", metadata.get("company", "")),
        "location": location,
        "work_mode": work_mode,
        "employment_type": metadata.get("employment type", ""),
        "salary": metadata.get("salary", ""),
        "overview": " ".join(overview_parts[:2]),
        "responsibilities": responsibilities,
        "required_qualifications": list(dict.fromkeys(required)),
        "preferred_qualifications": list(dict.fromkeys(preferred)),
        "skills_tools": [row["text"] for row in requirements if row["category"] == "skill"],
        "experience": experience,
        "education": education,
        "eligibility": eligibility,
        "benefits": list(dict.fromkeys(grouped["benefits"])),
        "about_company": list(dict.fromkeys(grouped["company"])),
        "requirements": requirements,
        "source_text": body,
    }


def requirement_records(job: StructuredJob) -> list[dict[str, Any]]:
    """Translate the canonical contract into the evidence retriever contract."""
    return [
        {
            "text": row["text"],
            "demand": "preferred" if row["type"] == "preferred" else "required",
            "section": row["section"],
            "classification": row["type"].title(),
            "category": row["category"],
            "source_span": row["source_span"],
            "probability": row["extraction_confidence"],
        }
        for row in job["requirements"]
    ]


def pipeline_trace(job_text: str, structured: StructuredJob, evidence: dict[str, Any] | None = None) -> dict[str, Any]:
    """Expose which JD stage succeeded instead of collapsing all failures into no score."""
    evidence = evidence or {}
    word_count = len(_description_body(job_text).split())
    requirement_count = len(structured["requirements"])
    matched_count = int(evidence.get("accepted_count", 0) or 0)
    return {
        "fetch": {"status": "ready" if word_count >= 50 else "incomplete", "word_count": word_count},
        "structure": {"status": "ready" if requirement_count else "failed", "requirement_count": requirement_count},
        "evidence": {"status": "ready" if requirement_count else "blocked", "matched_count": matched_count},
        "score": {"status": "ready" if requirement_count else "blocked"},
    }


def pipeline_diagnostic_message(trace: dict[str, Any]) -> str:
    """Return the first actionable failure instead of a generic unavailable score."""
    fetch = dict(trace.get("fetch", {}) or {})
    structure = dict(trace.get("structure", {}) or {})
    evidence = dict(trace.get("evidence", {}) or {})
    if fetch.get("status") != "ready":
        return "The saved posting is incomplete; obtain the full JD before scoring."
    if structure.get("status") != "ready":
        return "The JD was fetched, but no source-backed requirements could be structured."
    if not int(evidence.get("matched_count", 0) or 0):
        return "Requirements were extracted, but no reliable resume evidence was accepted."
    return "JD acquisition, requirement structuring, evidence retrieval, and scoring completed."
