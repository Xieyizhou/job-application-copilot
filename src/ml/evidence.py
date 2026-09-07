"""Requirement-to-resume evidence retrieval for local review and cover letters."""

from __future__ import annotations

from ml.evidence_text import useful_tokens, concept_tags, stated_years, clean_source_line

from ml.evidence_text import ACTION_PATTERN

from pathlib import Path
import re
from typing import Any

from ml.inference import DEFAULT_MODEL_PATH, portable_text_similarities, portable_text_similarity
from jd_text import normalize_jd_boundaries


MIN_ACCEPTED_SIMILARITY = 0.42
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


def _atomic_requirement_fragments(raw_line: str) -> list[str]:
    """Split compressed preview text while discarding ellipsis-truncated tails."""
    cleaned = clean_source_line(raw_line)
    if not cleaned:
        return []
    pieces = re.split(r"(\.{3,}|[•~])", cleaned)
    fragments: list[str] = []
    for index in range(0, len(pieces), 2):
        chunk = pieces[index].strip()
        preceding_boundary = pieces[index - 1] if index else ""
        following_boundary = pieces[index + 1] if index + 1 < len(pieces) else ""
        if not chunk:
            continue
        chunk = re.sub(
            r"^(?:role description|description|qualifications?|requirements?)\s*:?​?\s*",
            "",
            chunk,
            flags=re.IGNORECASE,
        ).strip()
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", chunk)
        for sentence_index, sentence in enumerate(sentences):
            sentence = sentence.strip()
            is_final_sentence = sentence_index == len(sentences) - 1
            if (
                preceding_boundary.startswith("...")
                and sentence_index == 0
                and sentence[:1].islower()
            ):
                continue
            if (
                following_boundary.startswith("...")
                and is_final_sentence
                and not re.search(r"[.!?]$", sentence)
            ):
                continue
            sentence = sentence.rstrip(".!?").strip()
            if sentence:
                fragments.append(sentence)
    return fragments


def requirement_allowed_in_cover_letter(requirement: str) -> bool:
    """Keep eligibility and sensitive personal-status claims out of CL prose."""
    lowered = requirement.lower()
    return not any(term in lowered for term in COVER_LETTER_EXCLUDED_TERMS)


def extract_requirement_records(job_text: str) -> list[dict[str, str]]:
    """Extract ordered required/preferred statements from a saved JD."""
    records: list[dict[str, str]] = []
    current_section = ""
    source_lines = normalize_jd_boundaries(job_text).splitlines()
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
        is_bullet = bool(re.match(r"^[-*•]\s+", stripped))
        has_inline_list = bool(re.search(r"[•~]", raw_line))
        previous_line = clean_source_line(source_lines[line_index - 1]).lower() if line_index else ""
        fragments = _atomic_requirement_fragments(raw_line)
        for fragment in fragments:
            fragment_lower = fragment.lower()
            preferred_signal = any(
                term in fragment_lower
                for term in ("preferred", "nice to have", "bonus", "a plus")
            )
            required_signal = any(
                term in fragment_lower
                for term in (
                    "required",
                    "requires",
                    "must",
                    "experience with",
                    "responsible for",
                    "proficiency in",
                    "strong background in",
                    "strong foundation in",
                    "familiarity with",
                )
            )
            negation_context = f"{previous_line} {fragment_lower}"
            negated_requirement = bool(
                re.search(
                    r"\b(?:no|not|does not|is not|without)\b.{0,200}\b(?:required|must)\b",
                    negation_context,
                )
            )
            if negated_requirement or not (
                required_signal
                or preferred_signal
                or (requirement_section and is_bullet)
                or (requirement_section and has_inline_list)
            ):
                continue
            demand = "preferred" if preferred_section or preferred_signal else "required"
            if fragment not in {record["text"] for record in records}:
                records.append(
                    {
                        "text": fragment,
                        "demand": demand,
                        "section": current_section or "job description",
                    }
                )
    return sorted(records, key=lambda item: item["demand"] == "preferred")


def extract_resume_evidence_records(resume_text: str) -> list[dict[str, Any]]:
    """Extract factual resume statements with their original section names."""
    records: list[dict[str, Any]] = []
    current_section = "Resume evidence"
    pending: dict[str, Any] | None = None

    def flush_pending() -> None:
        nonlocal pending
        if pending is not None:
            records.append(pending)
            pending = None

    for line_index, raw_line in enumerate(resume_text.splitlines()):
        stripped = raw_line.strip()
        if not stripped:
            flush_pending()
            continue
        if stripped.startswith("#"):
            flush_pending()
            heading = clean_source_line(stripped.lstrip("#"))
            if heading:
                current_section = heading
            continue
        plain_heading = (
            bool(re.fullmatch(r"[A-Z][A-Z &/+\-]{2,}", stripped))
            and len(stripped.split()) <= 6
        )
        if plain_heading:
            flush_pending()
            current_section = stripped.title()
            continue
        is_bullet = bool(re.match(r"^[-*•]\s+", stripped))
        if is_bullet:
            flush_pending()
        line = clean_source_line(stripped)
        if not line or "@" in line or line.lower().startswith(("http://", "https://")):
            flush_pending()
            continue
        continuation = (
            pending is not None
            and not is_bullet
            and not str(pending["text"]).rstrip().endswith((".", "!", "?"))
        )
        if continuation and pending is not None:
            combined = f"{pending['text']} {line}".strip()
            if len(combined.split()) <= 120:
                pending["text"] = combined
                continue
            flush_pending()
        elif pending is not None:
            flush_pending()
        word_count = len(line.split())
        if word_count < 4 or word_count > 120:
            continue
        if not is_bullet and current_section.lower() in {
            "resume evidence", "contact", "summary", "profile", "education", "skills",
        }:
            continue
        pending = {"text": line, "section": current_section, "line_index": line_index}
    flush_pending()
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


def score_transparent_evidence_pair(
    requirement: str,
    evidence: str,
) -> dict[str, Any]:
    """Score one pair without loading or blending a saved model artifact."""
    return _score_evidence_pair(requirement, evidence, model_similarity=None)


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
    from ml.evidence_label_guard import label_checks

    rejected, label_reasons = label_checks(requirement, evidence)
    accepted = accepted and not rejected
    if accepted and not label_reasons and (len(shared_tokens) >= 2 or requirement_coverage >= 0.5):
        match_type = "Direct support"
    elif accepted:
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
        "label_reasons": label_reasons,
    }


def build_semantic_evidence_index(
    job_text: str,
    resume_text: str,
    *,
    model_path: Path = DEFAULT_MODEL_PATH,
    max_requirements: int = 8,
    requirement_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Map each important requirement to its strongest truthful resume statement."""
    requirements = (requirement_records or extract_requirement_records(job_text))[:max_requirements]
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
        best = max(candidates, key=lambda item: (item["accepted"], item["match_type"] == "Direct support", item["similarity"], -item["line_index"]), default=None)
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
                    "label_reasons": best.get("label_reasons", []) if best else [],
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
