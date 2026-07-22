"""Requirement-to-resume evidence retrieval for local review and cover letters."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from ml.inference import DEFAULT_MODEL_PATH, portable_text_similarities, portable_text_similarity


MIN_ACCEPTED_SIMILARITY = 0.42
USEFUL_TOKEN_PATTERN = re.compile(r"[a-z0-9+#.-]{2,}")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "have",
    "in", "into", "is", "it", "of", "on", "or", "our", "that", "the", "their",
    "this", "to", "using", "we", "who", "will", "with", "work", "role", "job",
    "candidate", "responsible", "required", "requirement", "requirements", "must",
    "preferred", "experience", "data",
}
ACTION_PATTERN = re.compile(
    r"^(?:built|created|developed|designed|implemented|analyzed|evaluated|led|improved|"
    r"reduced|increased|delivered|automated|researched|supported|produced|managed|"
    r"coordinated|conducted|presented|wrote|deployed|optimized)\b",
    re.IGNORECASE,
)
CONCEPT_ALIASES = {
    "data_pipeline": (
        "data pipeline", "data pipelines", "etl", "data ingestion", "data integration",
        "data workflow", "data workflows", "processing pipeline", "processing pipelines",
    ),
    "analytics": (
        "data analysis", "data analytics", "analyze data", "analysed data", "analytics",
        "statistical analysis", "insights",
    ),
    "dashboard_reporting": (
        "dashboard", "dashboards", "reporting", "business intelligence", "bi report",
        "visualization", "visualisation", "tableau",
    ),
    "automation": ("automate", "automated", "automation", "workflow automation"),
    "machine_learning": (
        "machine learning", "classification", "predictive model", "predictive models",
        "model training", "scikit-learn", "sklearn",
    ),
    "model_evaluation": (
        "model evaluation", "cross-validation", "cross validation", "f1", "confusion matrix",
        "precision", "recall", "roc auc",
    ),
    "communication": (
        "communicate", "communication", "presented", "presentation", "stakeholder",
        "stakeholders", "documentation", "documented", "technical writing",
    ),
    "software_delivery": (
        "production system", "production systems", "deployed", "deployment", "api",
        "service", "services", "software development",
    ),
    "delivery_automation": (
        "continuous integration", "continuous delivery", "ci cd", "cicd", "ci/cd",
        "github actions", "build pipeline", "build pipelines",
    ),
    "database": ("sql", "database", "databases", "postgres", "mysql", "warehouse"),
    "python": ("python", "pandas", "numpy"),
    "cloud": ("aws", "azure", "gcp", "cloud platform", "cloud services"),
    "java_enterprise": ("java", "j2ee", "jee", "spring mvc", "spring framework"),
    "spreadsheet_analysis": (
        "excel", "pivot table", "pivot tables", "pivottable", "pivottables", "vlookup",
    ),
    "collaboration": (
        "collaborated", "collaboration", "cross-functional", "cross functional", "teamwork",
    ),
}
METADATA_PREFIXES = {
    "company", "role", "location", "job url", "source", "created at", "first seen at",
    "last seen at", "description source", "jd fetch status", "company confidence",
    "company evidence", "company confirmed by user",
}
COVER_LETTER_EXCLUDED_TERMS = (
    "work authorization", "work authorisation", "visa", "sponsorship", "citizenship",
    "permanent residency", "security clearance", "bachelor", "master", "phd", "ph.d",
    "doctorate", "degree required",
)
YEAR_WORDS = {
    "one": 1.0,
    "two": 2.0,
    "three": 3.0,
    "four": 4.0,
    "five": 5.0,
    "six": 6.0,
    "seven": 7.0,
    "eight": 8.0,
    "nine": 9.0,
    "ten": 10.0,
}


def clean_source_line(raw_line: str) -> str:
    """Clean Markdown decoration while preserving factual source wording."""
    line = raw_line.strip()
    line = re.sub(r"^[-*•]+\s*", "", line)
    line = re.sub(r"^\d+[.)]\s*", "", line)
    line = re.sub(r"\*\*([^*]+)\*\*", r"\1", line)
    line = re.sub(r"`([^`]+)`", r"\1", line)
    line = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", line)
    return re.sub(r"\s+", " ", line).strip()


def useful_tokens(text: str) -> set[str]:
    """Return specific lexical terms used by the transparent similarity layer."""
    return {
        token.strip(".-")
        for token in USEFUL_TOKEN_PATTERN.findall(text.lower())
        if token.strip(".-") and token.strip(".-") not in STOPWORDS
    }


def concept_tags(text: str) -> set[str]:
    """Map common job/resume paraphrases to reviewable canonical concepts."""
    normalized = " " + re.sub(r"[^a-z0-9+#]+", " ", text.lower()).strip() + " "
    return {
        concept
        for concept, aliases in CONCEPT_ALIASES.items()
        if any(f" {alias} " in normalized for alias in aliases)
    }


def requirement_allowed_in_cover_letter(requirement: str) -> bool:
    """Keep eligibility and sensitive personal-status claims out of CL prose."""
    lowered = requirement.lower()
    return not any(term in lowered for term in COVER_LETTER_EXCLUDED_TERMS)


def stated_years(text: str) -> list[float]:
    """Extract numeric or short word-form years-of-experience statements."""
    lowered = text.lower()
    values = [
        float(value)
        for value in re.findall(r"\b(\d+(?:\.\d+)?)\+?\s*(?:years|yrs)\b", lowered)
    ]
    values.extend(
        number
        for word, number in YEAR_WORDS.items()
        if re.search(rf"\b{word}\s+(?:years|yrs)\b", lowered)
    )
    return values


def extract_requirement_records(job_text: str) -> list[dict[str, str]]:
    """Extract ordered required/preferred statements from a saved JD."""
    records: list[dict[str, str]] = []
    current_section = ""
    source_lines = job_text.splitlines()
    for line_index, raw_line in enumerate(source_lines):
        stripped = raw_line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            current_section = clean_source_line(stripped.lstrip("#")).lower()
            continue
        line = clean_source_line(stripped)
        if not line or ":" in line and line.split(":", 1)[0].strip().lower() in METADATA_PREFIXES:
            continue
        lower = line.lower()
        section_label = lower.rstrip(":").strip()
        if stripped.endswith(":") and not re.match(r"^[-*•]\s+", stripped) and len(section_label.split()) <= 5 and any(
            term in section_label
            for term in ("requirements", "qualifications", "responsibilities", "skills", "preferred", "nice to have")
        ):
            current_section = section_label
            continue
        requirement_section = any(
            term in current_section
            for term in ("requirement", "qualification", "responsibilit", "skill", "what you")
        )
        preferred_section = any(
            term in current_section
            for term in ("preferred", "nice to have", "bonus", "plus")
        )
        preferred_signal = any(term in lower for term in ("preferred", "nice to have", "bonus", "a plus"))
        required_signal = any(
            term in lower
            for term in ("required", "must", "experience with", "responsible for", "proficiency in")
        )
        previous_line = clean_source_line(source_lines[line_index - 1]).lower() if line_index else ""
        negation_context = f"{previous_line} {lower}"
        negated_requirement = bool(
            re.search(
                r"\b(?:no|not|does not|is not|without)\b.{0,200}\b(?:required|must)\b",
                negation_context,
            )
        )
        is_bullet = bool(re.match(r"^[-*•]\s+", stripped))
        if negated_requirement or not (required_signal or preferred_signal or (requirement_section and is_bullet)):
            continue
        demand = "preferred" if preferred_section or preferred_signal else "required"
        if line not in {record["text"] for record in records}:
            records.append({"text": line, "demand": demand, "section": current_section or "job description"})
    return sorted(records, key=lambda item: item["demand"] == "preferred")


def extract_resume_evidence_records(resume_text: str) -> list[dict[str, Any]]:
    """Extract factual resume statements with their original section names."""
    records: list[dict[str, Any]] = []
    current_section = "Resume evidence"
    for line_index, raw_line in enumerate(resume_text.splitlines()):
        stripped = raw_line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            heading = clean_source_line(stripped.lstrip("#"))
            if heading:
                current_section = heading
            continue
        is_bullet = bool(re.match(r"^[-*•]\s+", stripped))
        line = clean_source_line(stripped)
        if not line or "@" in line or line.lower().startswith(("http://", "https://")):
            continue
        word_count = len(line.split())
        if word_count < 4 or word_count > 80:
            continue
        if not is_bullet and current_section.lower() in {
            "resume evidence", "contact", "summary", "profile", "education", "skills",
        }:
            continue
        records.append({"text": line, "section": current_section, "line_index": line_index})
    return records


def score_evidence_pair(
    requirement: str,
    evidence: str,
    *,
    model_path: Path = DEFAULT_MODEL_PATH,
) -> dict[str, Any]:
    """Score one pair using portable TF-IDF plus auditable concept expansion."""
    model_similarity = portable_text_similarity(requirement, evidence, model_path=model_path)
    return _score_evidence_pair(requirement, evidence, model_similarity=model_similarity)


def _score_evidence_pair(
    requirement: str,
    evidence: str,
    *,
    model_similarity: float | None,
) -> dict[str, Any]:
    """Combine a precomputed model similarity with transparent evidence features."""
    requirement_tokens = useful_tokens(requirement)
    evidence_tokens = useful_tokens(evidence)
    shared_tokens = requirement_tokens & evidence_tokens
    requirement_coverage = len(shared_tokens) / len(requirement_tokens) if requirement_tokens else 0.0
    lexical_f1 = (
        2 * len(shared_tokens) / (len(requirement_tokens) + len(evidence_tokens))
        if requirement_tokens and evidence_tokens
        else 0.0
    )
    lexical_signal = max(lexical_f1, requirement_coverage * 0.85, model_similarity or 0.0)
    requirement_concepts = concept_tags(requirement)
    evidence_concepts = concept_tags(evidence)
    shared_concepts = requirement_concepts & evidence_concepts
    concept_coverage = len(shared_concepts) / len(requirement_concepts) if requirement_concepts else 0.0
    concrete_bonus = 0.1 if ACTION_PATTERN.search(evidence) or re.search(r"\b\d+(?:[.,]\d+)?%?\b", evidence) else 0.0
    similarity = min(1.0, 0.5 * lexical_signal + 0.4 * concept_coverage + concrete_bonus)
    if not shared_tokens and not shared_concepts:
        similarity = min(similarity, 0.24)
    required_years = stated_years(requirement)
    evidence_years = stated_years(evidence)
    numeric_constraint_supported = (
        not required_years
        or bool(evidence_years) and max(evidence_years) >= max(required_years)
    )
    compound_requirement_supported = not (
        len(requirement_concepts) >= 2
        and concept_coverage < 0.75
        and requirement_coverage < 0.5
    )
    accepted = (
        similarity >= MIN_ACCEPTED_SIMILARITY
        and bool(shared_tokens or shared_concepts)
        and numeric_constraint_supported
        and compound_requirement_supported
    )
    if accepted and (len(shared_tokens) >= 2 or requirement_coverage >= 0.5):
        match_type = "Direct support"
    elif accepted and shared_concepts:
        match_type = "Semantic support"
    else:
        match_type = "Insufficient evidence"
    return {
        "similarity": round(similarity, 4),
        "accepted": accepted,
        "match_type": match_type,
        "shared_terms": sorted(shared_tokens),
        "shared_concepts": sorted(shared_concepts),
        "model_similarity": round(model_similarity, 4) if model_similarity is not None else None,
        "numeric_constraint_supported": numeric_constraint_supported,
        "compound_requirement_supported": compound_requirement_supported,
    }


def build_semantic_evidence_index(
    job_text: str,
    resume_text: str,
    *,
    model_path: Path = DEFAULT_MODEL_PATH,
    max_requirements: int = 8,
) -> dict[str, Any]:
    """Map each important requirement to its strongest truthful resume statement."""
    requirements = extract_requirement_records(job_text)[:max_requirements]
    evidence_records = extract_resume_evidence_records(resume_text)
    pair_keys = [
        (requirement["text"], evidence["text"])
        for requirement in requirements
        for evidence in evidence_records
    ]
    model_similarities = portable_text_similarities(pair_keys, model_path=model_path)
    similarity_by_pair = dict(zip(pair_keys, model_similarities))
    matches: list[dict[str, Any]] = []
    for requirement in requirements:
        candidates = []
        for evidence in evidence_records:
            score = _score_evidence_pair(
                requirement["text"],
                evidence["text"],
                model_similarity=similarity_by_pair[(requirement["text"], evidence["text"])],
            )
            candidates.append({**evidence, **score})
        best = max(candidates, key=lambda item: (item["similarity"], -item["line_index"]), default=None)
        if best and best["accepted"]:
            matches.append({**requirement, **best})
        else:
            matches.append(
                {
                    "requirement": requirement["text"],
                    "demand": requirement["demand"],
                    "requirement_section": requirement["section"],
                    "evidence": "",
                    "similarity": float(best["similarity"]) if best else 0.0,
                    "accepted": False,
                    "match_type": "Insufficient evidence",
                    "shared_terms": [],
                    "shared_concepts": [],
                    "model_similarity": best.get("model_similarity") if best else None,
                    "section_evidence": "",
                    "line_index": -1,
                    "cover_letter_eligible": requirement_allowed_in_cover_letter(requirement["text"]),
                }
            )
            continue
        matches[-1]["evidence"] = matches[-1].pop("text")
        matches[-1]["section_evidence"] = matches[-1].pop("section")
        matches[-1]["requirement"] = requirement["text"]
        matches[-1]["requirement_section"] = requirement["section"]
        matches[-1]["cover_letter_eligible"] = requirement_allowed_in_cover_letter(requirement["text"])

    accepted_matches = [match for match in matches if match["accepted"]]
    return {
        "method": "portable TF-IDF + concept expansion" if model_path.is_file() else "concept + lexical fallback",
        "threshold": MIN_ACCEPTED_SIMILARITY,
        "requirement_count": len(requirements),
        "accepted_count": len(accepted_matches),
        "cover_letter_eligible_count": sum(
            1 for match in accepted_matches if match["cover_letter_eligible"]
        ),
        "matches": matches,
        "accepted_matches": accepted_matches,
        "unmatched_requirements": [
            match["requirement"]
            for match in matches
            if not match["accepted"]
        ],
    }
