"""Generate an employer-facing cover letter from local evidence.

The generator is deterministic and local:
- reads the selected workspace experience bank or tracked generic fallback
- reads the selected job description Markdown file
- uses keyword matching and evidence blocks
- keeps candidate evidence and document generation local
"""

from __future__ import annotations

from ml.evidence_text import clean_source_line as clean_resume_evidence_line

from document_text import format_bullets

from document_text import clean_duplicated_punctuation

import argparse
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from company_verification import (
    COMPANY_CONFIRMATION_PLACEHOLDER,
    assert_cover_letter_company_verified,
    parse_bool,
    validate_company_name,
)
from ml.evidence import build_semantic_evidence_index
from ml.jd_quality import assert_cover_letter_jd_ready
from output_paths import application_package_dir
from workspace import Workspace, WorkspaceError, personal_workspace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET_WORD_COUNT_MIN = 150
TARGET_WORD_COUNT_MAX = 230
MAX_COVER_LETTER_EVIDENCE_LINES = 3
DEFAULT_THEME_KEYWORDS = {
    "machine_learning": ["machine learning", "ml", "classification", "model", "prediction"],
    "python_data": ["python", "pandas", "numpy", "scikit-learn", "sklearn", "data analysis"],
    "model_evaluation": ["model evaluation", "metrics", "accuracy", "f1", "confusion matrix"],
    "data_visualization": ["visualization", "matplotlib", "plot", "chart", "dashboard"],
    "uav_robotics_sensor": ["uav", "drone", "robotics", "sensor", "thermal", "route planning"],
    "game_ai": ["game ai", "npc", "reinforcement learning", "neural network", "cnn", "gameplay"],
    "econometrics_statistics": ["econometrics", "statistics", "regression", "causal"],
    "communication": ["communication", "collaboration", "presentation", "teaching", "documentation"],
}
GENERIC_PHRASES = [
    "I am writing to apply",
    "I believe I am a good fit",
    "my recent graduate background",
]
INTERNAL_WORKFLOW_PHRASES = [
    "uploaded resume",
    "evidence threshold",
    "automatically generated",
    "insufficient proof point",
    "claim plan",
]
SENSITIVE_IDENTITY_PHRASES = [
    "citizenship",
    "permanent resident",
    "permanent residency",
    "security clearance",
    "visa sponsorship",
    "work authorization",
]
DEFAULT_BANNED_PHRASES = [
    "automatically generated",
    "Tailoring Notes",
    "internal notes",
    "undergraduate graduate",
    "master's",
    "Master's",
    "MS",
    "M.S.",
]
THEME_LABELS = {
    "machine_learning": "machine learning",
    "python_data": "Python-based data analysis",
    "model_evaluation": "model evaluation",
    "data_visualization": "data visualization",
    "uav_robotics_sensor": "UAV inspection and sensor workflows",
    "game_ai": "game AI",
    "econometrics_statistics": "statistics and econometrics",
    "communication": "technical communication",
}


class CoverLetterGenerationError(ValueError):
    """Raised when a truthful employer-facing draft cannot be produced."""

    def __init__(self, message: str, *, reasons: list[str] | None = None, gaps: list[str] | None = None) -> None:
        super().__init__(message)
        self.reasons = reasons or []
        self.gaps = gaps or []


@dataclass(frozen=True)
class CoverLetterClaim:
    """One selected requirement-to-resume proof used by the draft."""

    claim_id: str
    requirement: str
    demand: str
    support_label: str
    evidence: str
    source_section: str
    rank_score: float


@dataclass(frozen=True)
class CoverLetterParagraph:
    """Rendered paragraph with explicit claim provenance."""

    kind: str
    text: str
    claim_ids: tuple[str, ...] = ()


@dataclass
class CoverLetterPlan:
    """Versioned local generation plan persisted beside the employer draft."""

    schema_version: int
    company: str
    role: str
    candidate_name: str
    claims: list[CoverLetterClaim]
    paragraphs: list[CoverLetterParagraph]
    word_count: int = 0
    validation_status: str = "pending"
    validation_errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Return a stable JSON representation for local traceability."""
        return asdict(self)


def load_experience_bank(path: Path) -> dict[str, Any]:
    """Load the local evidence bank."""
    if not path.exists():
        raise FileNotFoundError(f"Experience bank was not found: {path}")

    with path.open(encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    if not isinstance(data, dict):
        raise ValueError("experience_bank.yaml must contain a YAML mapping.")
    return data


def normalize_text(text: str) -> str:
    """Normalize text for simple keyword matching."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9+#./-]+", " ", text)
    return f" {text} "


def contains_phrase(text: str, phrase: str) -> bool:
    """Return True when a phrase appears in normalized text."""
    return f" {normalize_text(phrase).strip()} " in normalize_text(text)


def extract_markdown_field(job_text: str, field_name: str, default: str) -> str:
    """Extract a simple 'Field: value' line from a Markdown job description."""
    prefix = f"{field_name}:"
    for line in job_text.splitlines():
        if line.lower().startswith(prefix.lower()):
            value = line.split(":", 1)[1].strip()
            if value and value.lower() != "not provided":
                return value
    return default


def extract_job_title(job_text: str) -> str:
    """Use the Role field, first heading, or a fallback role label."""
    role = extract_markdown_field(job_text, "Role", "")
    if role:
        return role

    for line in job_text.splitlines():
        if line.startswith("# "):
            return line.removeprefix("# ").strip()
    return "the role"


def extract_company(job_text: str) -> str:
    """Use the Company field when available."""
    company = extract_markdown_field(job_text, "Company", "")
    validation = validate_company_name(
        company,
        {
            "job_text": job_text,
            "company_confirmed_by_user": parse_bool(extract_markdown_field(job_text, "Company Confirmed By User", "")),
            "company_source_confidence": extract_markdown_field(job_text, "Company Confidence", ""),
            "company_source_evidence": extract_markdown_field(job_text, "Company Evidence", ""),
        },
    )
    if validation["normalized_company"] and (
        validation["confidence"] == "high" or parse_bool(extract_markdown_field(job_text, "Company Confirmed By User", ""))
    ):
        return str(validation["normalized_company"])
    return COMPANY_CONFIRMATION_PLACEHOLDER


def extract_requirements(job_text: str) -> tuple[list[str], list[str]]:
    """Extract likely required and preferred lines from a job description."""
    required = []
    preferred = []
    current_section = ""

    for raw_line in job_text.splitlines():
        line = raw_line.strip().strip("-*").strip()
        if not line:
            continue

        lower_line = line.lower()
        if lower_line.startswith("#"):
            current_section = lower_line
            continue

        is_requirement_section = any(
            word in current_section
            for word in ["requirement", "qualification", "responsibilit", "skill"]
        )
        is_preferred_section = any(word in current_section for word in ["preferred", "plus", "nice"])
        has_required_signal = any(
            word in lower_line
            for word in ["required", "requirement", "must", "responsible", "experience with"]
        )
        has_preferred_signal = any(
            word in lower_line
            for word in ["preferred", "plus", "nice to have", "bonus"]
        )

        if is_preferred_section or has_preferred_signal:
            preferred.append(line)
        elif is_requirement_section or has_required_signal:
            required.append(line)

    return required, preferred


def detect_themes(job_text: str, experience_bank: dict[str, Any] | None = None) -> list[str]:
    """Detect JD themes using keywords from the experience bank."""
    bank = experience_bank or {}
    theme_keywords = bank.get("theme_keywords") or DEFAULT_THEME_KEYWORDS
    detected = []

    for theme, keywords in theme_keywords.items():
        if any(contains_phrase(job_text, str(keyword)) for keyword in keywords):
            detected.append(theme)

    return detected


EVIDENCE_STOPWORDS = {
    "and", "the", "with", "for", "from", "that", "this", "into", "using", "used",
    "role", "work", "team", "job", "your", "our", "you", "are", "will", "have",
    "has", "was", "were", "their", "they", "but", "not", "all", "can", "who",
}


def evidence_terms(text: str) -> set[str]:
    """Return useful lexical terms for transparent JD-to-resume evidence ranking."""
    return {
        token
        for token in re.findall(r"[a-z0-9+#.-]{3,}", text.lower())
        if token not in EVIDENCE_STOPWORDS
    }


def extract_candidate_name(resume_text: str) -> str:
    """Use the first plausible resume heading or short line as the sign-off name."""
    for raw_line in resume_text.splitlines():
        line = clean_resume_evidence_line(raw_line.lstrip("#").strip())
        if not line or "@" in line or "http" in line.lower() or "|" in line:
            continue
        words = line.split()
        if 2 <= len(words) <= 5 and not any(char.isdigit() for char in line):
            return line
    return ""


def score_resume_evidence_line(
    line: str,
    job_text: str,
    detected_themes: list[str],
    experience_bank: dict[str, Any],
) -> tuple[int, list[str]]:
    """Rank a resume line by theme support, JD overlap, and concrete outcomes."""
    theme_keywords = experience_bank.get("theme_keywords") or DEFAULT_THEME_KEYWORDS
    matched_themes = [
        theme
        for theme in detected_themes
        if any(contains_phrase(line, str(keyword)) for keyword in theme_keywords.get(theme, []))
    ]
    overlap = evidence_terms(line).intersection(evidence_terms(job_text))
    concrete_bonus = 2 if re.search(r"\b\d+(?:[.,]\d+)?%?\b", line) else 0
    action_bonus = 1 if re.match(
        r"^(built|created|developed|designed|implemented|analyzed|evaluated|led|improved|reduced|increased|delivered|automated|researched|supported)\b",
        line,
        flags=re.IGNORECASE,
    ) else 0
    return len(matched_themes) * 5 + min(len(overlap), 8) + concrete_bonus + action_bonus, matched_themes


def select_resume_evidence_blocks(
    resume_text: str,
    job_text: str,
    detected_themes: list[str],
    experience_bank: dict[str, Any],
    max_experiences: int = 2,
    evidence_index: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Select only threshold-passing proof mapped to explicit JD requirements."""
    evidence_index = evidence_index or build_semantic_evidence_index(job_text, resume_text)
    ranked_matches = sorted(
        [
            match
            for match in evidence_index["accepted_matches"]
            if match.get("cover_letter_eligible", True)
        ],
        key=lambda item: (item["demand"] == "preferred", -float(item["similarity"])),
    )
    sections: dict[str, dict[str, Any]] = {}
    seen_evidence: set[str] = set()
    for match in ranked_matches:
        evidence = str(match["evidence"])
        if evidence in seen_evidence:
            continue
        seen_evidence.add(evidence)
        section = str(match["section_evidence"])
        block = sections.setdefault(
            section,
            {
                "score": 0,
                "experience": {
                    "id": re.sub(r"[^a-z0-9]+", "_", section.lower()).strip("_") or "resume_evidence",
                    "name": section,
                    "category": "resume",
                    "evidence": [],
                    "safe_phrases": [],
                },
                "matched_themes": [],
                "selected_evidence": [],
                "semantic_matches": [],
            },
        )
        _, themes = score_resume_evidence_line(evidence, job_text, detected_themes, experience_bank)
        block["score"] = max(int(block["score"]), round(float(match["similarity"]) * 100))
        block["matched_themes"] = sorted(set(block["matched_themes"]) | set(themes))
        block["selected_evidence"].append(evidence)
        block["experience"]["evidence"].append(evidence)
        block["semantic_matches"].append(match)
        if len(seen_evidence) >= MAX_COVER_LETTER_EVIDENCE_LINES or (
            len(sections) >= max_experiences and all(
            len(item["selected_evidence"]) >= 1 for item in sections.values()
            )
        ):
            break
    return list(sections.values())[:max_experiences]


def first_person_evidence_sentence(line: str) -> str:
    """Turn a resume bullet into restrained first-person prose without adding facts."""
    sentence = clean_resume_evidence_line(line).rstrip(".")
    if not sentence:
        return ""
    if sentence.lower().startswith(("i ", "my ")):
        return sentence + "."
    lower = sentence.lower()
    if lower.startswith("experience in ") or lower.startswith("experience with "):
        return f"I have {lower}."
    if lower.startswith(("proficient in ", "skilled in ", "familiar with ")):
        return f"I am {lower}."
    action_verbs = (
        "built", "created", "developed", "designed", "implemented", "analyzed", "evaluated",
        "led", "improved", "reduced", "increased", "delivered", "automated", "researched",
        "supported", "produced", "managed", "coordinated", "conducted", "presented", "wrote",
    )
    if lower.startswith(action_verbs):
        first_word, separator, rest = sentence.partition(" ")
        normalized = f"{first_word.lower()} {rest}" if separator else first_word.lower()
        return f"I {normalized}."
    if "," in sentence or " · " in sentence or " | " in sentence:
        return f"My resume lists {sentence}."
    return f"My resume describes {sentence[0].lower() + sentence[1:]}."


def top_job_priority(job_text: str, detected_themes: list[str]) -> str:
    """Return one concise, source-grounded description of the role's priority."""
    required, preferred = extract_requirements(job_text)
    candidates = required or preferred
    if candidates:
        priority = re.sub(
            r"^(required|requirements?|must|responsible for|experience with)[:\s-]*",
            "",
            clean_resume_evidence_line(candidates[0]),
            flags=re.IGNORECASE,
        )
        words = priority.rstrip(".").split()
        if words:
            return " ".join(words[:18])
    return build_focus_summary(detected_themes[:2])


def build_focus_summary(detected_themes: list[str]) -> str:
    """Describe the JD focus in one short phrase."""
    labels = [THEME_LABELS.get(theme, theme.replace("_", " ")) for theme in detected_themes]
    if not labels:
        return "applied technical problem solving"
    if len(labels) == 1:
        return labels[0]
    return ", ".join(labels[:2]) + (" and " + labels[2] if len(labels) == 3 else "")


def build_primary_evidence_paragraph(selection: dict[str, Any]) -> str:
    """Build a source-grounded proof paragraph from the strongest resume block."""
    experience = selection["experience"]
    bullets = selection.get("selected_evidence", [])
    name = str(experience.get("name", "this experience"))

    if not bullets:
        return ""

    sentences = [first_person_evidence_sentence(bullet) for bullet in bullets[:2]]
    if name.lower() in {"skills", "technical skills", "core competencies", "summary", "profile"}:
        return " ".join(sentences)
    return f"In {name}, " + " ".join(sentences)


def build_secondary_evidence_paragraph(selected: list[dict[str, Any]]) -> str:
    """Add one concise, independently sourced proof when another block is relevant."""
    if len(selected) > 1:
        secondary = selected[1]
        evidence = list(secondary.get("selected_evidence", []))
        if evidence:
            sentence = first_person_evidence_sentence(evidence[0])
            name = str(secondary["experience"].get("name", "another resume experience"))
            if name.lower() in {"skills", "technical skills", "core competencies", "summary", "profile"}:
                return sentence
            return f"In {name}, {sentence}"
    return ""


def concise_requirement(text: str, *, limit: int = 18) -> str:
    """Return a compact source-grounded requirement phrase for prose."""
    cleaned = re.sub(
        r"^(?:required|requirements?|must|responsible for|experience with|proficiency in)[:\s-]*",
        "",
        clean_resume_evidence_line(text),
        flags=re.IGNORECASE,
    ).rstrip(".:")
    words = cleaned.split()
    return " ".join(words[:limit])


def requirement_sentence_fragment(text: str) -> str:
    """Lowercase only generic leading verbs while preserving names and acronyms."""
    phrase = concise_requirement(text)
    first_word = phrase.split(maxsplit=1)[0].casefold() if phrase else ""
    if first_word in {
        "build",
        "communicate",
        "create",
        "develop",
        "design",
        "implement",
        "lead",
        "manage",
        "monitor",
        "optimize",
        "support",
        "use",
        "write",
    }:
        return phrase[:1].lower() + phrase[1:]
    return phrase


def select_cover_letter_claims(evidence_index: dict[str, Any], *, limit: int = 3) -> list[CoverLetterClaim]:
    """Select diverse, cover-letter-safe proof using one shared evidence index."""
    eligible = [
        match
        for match in evidence_index["accepted_matches"]
        if match.get("cover_letter_eligible", True)
        and str(match.get("evidence", "")).strip()
        and str(match.get("requirement", "")).strip()
    ]
    ranked = sorted(
        eligible,
        key=lambda match: (
            str(match.get("demand", "required")) == "preferred",
            str(match.get("match_type", "")) != "Direct support",
            -float(match.get("similarity", 0.0)),
            int(match.get("line_index", 0)),
        ),
    )
    chosen: list[dict[str, Any]] = []
    seen_requirements: set[str] = set()
    seen_evidence: set[str] = set()
    seen_sections: set[str] = set()

    for prefer_new_section in (True, False):
        for match in ranked:
            requirement = str(match["requirement"]).strip()
            evidence = str(match["evidence"]).strip()
            section = str(match.get("section_evidence", "Resume evidence")).strip()
            if requirement in seen_requirements or evidence in seen_evidence:
                continue
            if prefer_new_section and section in seen_sections:
                continue
            chosen.append(match)
            seen_requirements.add(requirement)
            seen_evidence.add(evidence)
            seen_sections.add(section)
            if len(chosen) == limit:
                break
        if len(chosen) == limit:
            break

    return [
        CoverLetterClaim(
            claim_id=f"claim_{index}",
            requirement=str(match["requirement"]),
            demand=str(match.get("demand", "required")),
            support_label=(
                "Direct" if str(match.get("match_type")) == "Direct support" else "Partial"
            ),
            evidence=str(match["evidence"]),
            source_section=str(match.get("section_evidence", "Resume evidence")),
            rank_score=round(float(match.get("similarity", 0.0)), 4),
        )
        for index, match in enumerate(chosen, start=1)
    ]


def assert_cover_letter_claim_gate(
    claims: list[CoverLetterClaim],
    evidence_index: dict[str, Any],
    *,
    candidate_name: str,
) -> None:
    """Block employer-facing generation until truthful proof is sufficient."""
    reasons: list[str] = []
    if not candidate_name.strip():
        reasons.append("Confirm your name on the Resume page.")
    if len({claim.evidence for claim in claims}) < 2:
        reasons.append("At least two distinct resume evidence statements are required.")
    if len({claim.requirement for claim in claims}) < 2:
        reasons.append("The evidence must cover at least two job requirements.")
    if not any(claim.support_label == "Direct" for claim in claims):
        reasons.append("At least one job requirement needs Direct resume support.")
    if reasons:
        raise CoverLetterGenerationError(
            "Not enough grounded evidence to generate a credible cover letter.",
            reasons=reasons,
            gaps=list(evidence_index.get("unmatched_requirements", []))[:3],
        )


def render_claim_paragraph(claim: CoverLetterClaim) -> str:
    """Render one proof paragraph without expanding beyond the selected evidence."""
    evidence_sentence = first_person_evidence_sentence(claim.evidence)
    requirement = requirement_sentence_fragment(claim.requirement)
    section = claim.source_section.strip()
    prefix = "" if section.casefold() in {"skills", "technical skills", "summary", "profile"} else f"In {section}, "
    secondary_variant = claim.claim_id.endswith("2")
    if claim.support_label == "Direct" and secondary_variant:
        bridge = (
            f"This evidence aligns directly with the position's need for {requirement}. "
            "It is a second, distinct example of how I have applied relevant skills to concrete work and communicated the result."
        )
    elif claim.support_label == "Direct":
        bridge = (
            f"This directly supports the position's emphasis on {requirement}. "
            "It gives me a concrete basis for contributing to similar work while adapting to the team's tools and operating context."
        )
    elif secondary_variant:
        bridge = (
            f"This experience is related to the position's need for {requirement}. "
            "Although it is not the same setting, it offers a documented foundation I could transfer while learning the team's specific approach."
        )
    else:
        bridge = (
            f"This experience is adjacent to the position's need for {requirement}. "
            "It provides a practical foundation I could transfer to the work while learning the role's specific methods and standards."
        )
    return f"{prefix}{evidence_sentence} {bridge}"


def render_cover_letter_plan(plan: CoverLetterPlan) -> str:
    """Render the employer-facing Markdown from a validated provenance plan."""
    body = "\n\n".join(paragraph.text for paragraph in plan.paragraphs)
    return clean_duplicated_punctuation(
        f"Dear Hiring Team,\n\n{body}\n\nSincerely,\n\n{plan.candidate_name}\n"
    )


def validate_cover_letter_plan(plan: CoverLetterPlan, cover_letter: str, experience_bank: dict[str, Any]) -> list[str]:
    """Return hard generation failures for content, length, and provenance."""
    errors = validate_cover_letter(cover_letter, experience_bank)
    claim_ids = {claim.claim_id for claim in plan.claims}
    used_claim_ids = {
        claim_id
        for paragraph in plan.paragraphs
        for claim_id in paragraph.claim_ids
    }
    if used_claim_ids != claim_ids:
        errors.append("not every selected claim has paragraph provenance")
    if any(paragraph.kind.startswith("evidence") and not paragraph.claim_ids for paragraph in plan.paragraphs):
        errors.append("an evidence paragraph is missing provenance")
    normalized_paragraphs = [re.sub(r"\s+", " ", paragraph.text).strip().casefold() for paragraph in plan.paragraphs]
    if len(normalized_paragraphs) != len(set(normalized_paragraphs)):
        errors.append("duplicate paragraph detected")
    return errors


def build_cover_letter_plan(
    resume_text: str,
    job_text: str,
    experience_bank: dict[str, Any] | None = None,
    *,
    candidate_name: str | None = None,
) -> CoverLetterPlan:
    """Build and validate a versioned evidence-first local writing plan."""
    bank = experience_bank or {"theme_keywords": DEFAULT_THEME_KEYWORDS}
    resolved_name = (candidate_name if candidate_name is not None else extract_candidate_name(resume_text)).strip()
    evidence_index = build_semantic_evidence_index(job_text, resume_text)
    # Two distinct proof points are enough for a concise one-page letter.  The
    # selector still supports a third claim for future layouts, but the default
    # draft deliberately favors depth and provenance over evidence stuffing.
    claims = select_cover_letter_claims(evidence_index, limit=2)
    assert_cover_letter_claim_gate(claims, evidence_index, candidate_name=resolved_name)
    primary, secondary = claims[:2]
    company = extract_company(job_text)
    role = extract_job_title(job_text)
    opening = (
        f"The {role} position at {company} centers on {requirement_sentence_fragment(primary.requirement)}, "
        f"with {requirement_sentence_fragment(secondary.requirement)} also important to the work. "
        "I am interested in the role because these priorities connect to specific projects and experience in my background, including the examples below."
    )
    claim_ids = tuple(claim.claim_id for claim in claims)
    paragraphs = [
        CoverLetterParagraph(kind="opening", text=opening, claim_ids=claim_ids),
        CoverLetterParagraph(
            kind="evidence_primary",
            text=render_claim_paragraph(primary),
            claim_ids=(primary.claim_id,),
        ),
        CoverLetterParagraph(
            kind="evidence_secondary",
            text=render_claim_paragraph(secondary),
            claim_ids=(secondary.claim_id,),
        ),
    ]
    paragraphs.append(
        CoverLetterParagraph(
            kind="closing",
            text=(
                "Together, these examples represent the work in my background that is most relevant to this position. "
                f"I would welcome the opportunity to discuss how this experience could support the {role} team at {company}. "
                "Thank you for your time and consideration."
            ),
            claim_ids=claim_ids,
        )
    )
    plan = CoverLetterPlan(
        schema_version=1,
        company=company,
        role=role,
        candidate_name=resolved_name,
        claims=claims,
        paragraphs=paragraphs,
    )
    cover_letter = render_cover_letter_plan(plan)
    plan.word_count = len(cover_letter.split())
    plan.validation_errors = validate_cover_letter_plan(plan, cover_letter, bank)
    plan.validation_status = "passed" if not plan.validation_errors else "failed"
    if plan.validation_errors:
        raise CoverLetterGenerationError(
            "The grounded cover letter failed the employer-facing quality gate.",
            reasons=plan.validation_errors,
        )
    return plan


def build_cover_letter(
    resume_text: str,
    job_text: str,
    experience_bank: dict[str, Any] | None = None,
) -> str:
    """Compatibility wrapper around the evidence-first v2 plan builder."""
    plan = build_cover_letter_plan(resume_text, job_text, experience_bank)
    return render_cover_letter_plan(plan)


def find_missing_or_weak_areas(job_text: str, detected_themes: list[str], selected: list[dict[str, Any]]) -> list[str]:
    """Flag terms that need human review before relying on the letter."""
    weak_areas = []
    selected_theme_set = {
        theme
        for selection in selected
        for theme in selection.get("matched_themes", [])
    }

    for term in [
        "robotics perception",
        "citizenship",
        "permanent residency",
        "work authorization",
        "visa",
        "sponsorship",
        "senior",
        "staff",
        "principal",
        "manager",
        "phd",
        "ph.d.",
        "master's",
    ]:
        if contains_phrase(job_text, term):
            weak_areas.append(f"{term}: review manually before submitting")

    for theme in detected_themes:
        if theme not in selected_theme_set:
            weak_areas.append(f"{theme}: no strong evidence block selected")

    if not weak_areas:
        weak_areas.append("No major missing or weak areas detected by keyword matching.")

    return weak_areas


def build_internal_note_context(
    resume_text: str,
    job_text: str,
    experience_bank: dict[str, Any],
) -> dict[str, Any]:
    """Collect the reusable evidence trace used by internal notes."""
    detected_themes = detect_themes(job_text, experience_bank)
    evidence_index = build_semantic_evidence_index(job_text, resume_text)
    selected = select_resume_evidence_blocks(
        resume_text,
        job_text,
        detected_themes,
        experience_bank,
        evidence_index=evidence_index,
    )
    required, preferred = extract_requirements(job_text)

    selected_experience_lines = [
        f"{item['experience'].get('name', 'Unnamed experience')} "
        f"(themes: {', '.join(item.get('matched_themes', [])) or 'semantic evidence'})"
        for item in selected
    ]
    selected_evidence_lines = [
        bullet
        for item in selected
        for bullet in item.get("selected_evidence", [])
    ]
    evidence_map_lines = [
        (
            f"{match['demand'].title()} — {match['requirement']} => "
            f"{match['evidence']} ({float(match['similarity']):.0%}, {match['match_type']}, "
            f"section: {match['section_evidence']}, "
            f"CL use: {'eligible' if match.get('cover_letter_eligible', True) else 'excluded'})"
        )
        for match in evidence_index["accepted_matches"]
    ]
    rejected_requirement_lines = [
        f"{requirement} — rejected because no resume statement passed the "
        f"{float(evidence_index['threshold']):.0%} evidence threshold"
        for requirement in evidence_index["unmatched_requirements"]
    ]

    return {
        "detected_themes": detected_themes,
        "selected": selected,
        "required": required,
        "preferred": preferred,
        "selected_experience_lines": selected_experience_lines,
        "selected_evidence_lines": selected_evidence_lines,
        "evidence_map_lines": evidence_map_lines,
        "rejected_requirement_lines": rejected_requirement_lines,
        "evidence_index": evidence_index,
    }


def render_internal_notes(
    context: dict[str, Any],
    *,
    job_text: str,
    cover_letter: str,
) -> str:
    """Render the internal review document from a prepared evidence trace."""
    detected_themes = list(context["detected_themes"])
    evidence_index = dict(context["evidence_index"])
    return "\n".join(
        [
            "# Internal Cover Letter Notes",
            "",
            "## Draft Quality",
            "",
            f"- Word count: {len(cover_letter.split())} (target: {TARGET_WORD_COUNT_MIN}-{TARGET_WORD_COUNT_MAX})",
            f"- Primary JD priority: {top_job_priority(job_text, detected_themes)}",
            "- Employer-facing claims are selected only from the uploaded resume text.",
            "- A personal experience bank may expand matching vocabulary but cannot introduce new claims.",
            "",
            "## Detected JD Themes",
            "",
            format_bullets(context["detected_themes"]),
            "",
            "## Required Signals Found",
            "",
            format_bullets(context["required"]),
            "",
            "## Preferred Signals Found",
            "",
            format_bullets(context["preferred"]),
            "",
            "## Selected Resume Sections",
            "",
            format_bullets(context["selected_experience_lines"]),
            "",
            "## Claim Trace — Exact Resume Evidence",
            "",
            format_bullets(context["selected_evidence_lines"]),
            "",
            "## Requirement-to-Resume Evidence Map",
            "",
            f"- Method: {evidence_index['method']}",
            f"- Accepted: {evidence_index['accepted_count']} of {evidence_index['requirement_count']} requirements",
            format_bullets(context["evidence_map_lines"]),
            "",
            "## Rejected Requirement Evidence",
            "",
            format_bullets(context["rejected_requirement_lines"]),
            "",
            "## Weak Or Missing Areas",
            "",
            format_bullets(find_missing_or_weak_areas(job_text, detected_themes, context["selected"])),
            "",
            "## Human Review Warnings",
            "",
            "- Review the employer-facing cover letter before submitting.",
            "- Confirm degree, visa, work authorization, citizenship, sponsorship, seniority, and location requirements manually.",
            "- Do not submit this internal notes file to employers.",
            "- Confirm the candidate's degree level before relying on education-related statements.",
            "- The uploaded resume is not rewritten or regenerated by this workflow.",
        ]
    ) + "\n"


def build_internal_notes(
    resume_text: str,
    job_text: str,
    experience_bank: dict[str, Any],
    _examples: list[str],
    cover_letter: str,
) -> str:
    """Build a source trace, gap audit, and quality check for human review."""
    context = build_internal_note_context(resume_text, job_text, experience_bank)
    return render_internal_notes(context, job_text=job_text, cover_letter=cover_letter)


def validate_cover_letter(cover_letter: str, experience_bank: dict[str, Any]) -> list[str]:
    """Return hard employer-facing content and length failures."""
    found = []
    banned_phrases = list(experience_bank.get("banned_phrases") or DEFAULT_BANNED_PHRASES)
    for phrase in banned_phrases + GENERIC_PHRASES + INTERNAL_WORKFLOW_PHRASES:
        if phrase.casefold() in cover_letter.casefold():
            found.append(phrase)
    word_count = len(cover_letter.split())
    if word_count < TARGET_WORD_COUNT_MIN:
        found.append(f"word count {word_count} is below {TARGET_WORD_COUNT_MIN}")
    if word_count > TARGET_WORD_COUNT_MAX:
        found.append(f"word count {word_count} exceeds {TARGET_WORD_COUNT_MAX}")
    return found


def validate_manual_cover_letter_draft(cover_letter: str) -> tuple[list[str], list[str]]:
    """Validate a user-edited draft and separate hard failures from review warnings."""
    errors: list[str] = []
    warnings: list[str] = []
    folded = cover_letter.casefold()
    for phrase in INTERNAL_WORKFLOW_PHRASES:
        if phrase.casefold() in folded:
            errors.append(f"Remove internal workflow phrase: {phrase}")
    for placeholder in ("candidate name", "your name", "example@email", "example.com/in/"):
        if placeholder in folded:
            errors.append(f"Replace placeholder content: {placeholder}")
    word_count = len(cover_letter.split())
    if not TARGET_WORD_COUNT_MIN <= word_count <= TARGET_WORD_COUNT_MAX:
        errors.append(
            f"Keep the draft between {TARGET_WORD_COUNT_MIN} and {TARGET_WORD_COUNT_MAX} words "
            f"(currently {word_count})."
        )
    for phrase in SENSITIVE_IDENTITY_PHRASES:
        if phrase in folded:
            warnings.append(f"Verify this sensitive identity statement before export: {phrase}")
    return errors, warnings


def save_generated_file(
    job_description_path: Path,
    content: str,
    filename: str,
    workspace: Workspace,
    package_dir: Path | None = None,
) -> Path:
    """Save a generated Markdown file in a structured application folder."""
    if package_dir is None:
        package_dir = application_package_dir(workspace.generated_dir, job_description_path.stem)

    package_dir.mkdir(parents=True, exist_ok=True)
    output_path = package_dir / filename
    output_path.write_text(content, encoding="utf-8")
    return output_path


def load_cover_letter_inputs(job_description_path: Path, workspace: Workspace) -> dict[str, Any]:
    """Load generation inputs without mixing validation or persistence concerns."""
    assert workspace.resume_source_path is not None
    return {
        "resume_text": workspace.resume_source_path.read_text(encoding="utf-8"),
        "job_text": job_description_path.read_text(encoding="utf-8"),
        "experience_bank": (
            load_experience_bank(workspace.experience_bank_path)
            if workspace.experience_bank_path
            else {"theme_keywords": DEFAULT_THEME_KEYWORDS, "banned_phrases": DEFAULT_BANNED_PHRASES}
        ),
        "candidate_name": workspace.candidate_profile.name,
    }


def validate_cover_letter_inputs(job_text: str) -> None:
    """Enforce employer identity and complete-JD gates before drafting."""
    company = extract_markdown_field(job_text, "Company", "")
    assert_cover_letter_company_verified(
        company,
        {
            "job_text": job_text,
            "role": extract_job_title(job_text),
            "company_confirmed_by_user": parse_bool(extract_markdown_field(job_text, "Company Confirmed By User", "")),
            "company_source_confidence": extract_markdown_field(job_text, "Company Confidence", ""),
            "company_source_evidence": extract_markdown_field(job_text, "Company Evidence", ""),
        },
    )
    assert_cover_letter_jd_ready(job_text)


def save_cover_letter_outputs(
    job_description_path: Path,
    workspace: Workspace,
    package_dir: Path | None,
    cover_letter: str,
    internal_notes: str,
) -> tuple[Path, Path]:
    """Persist the employer-facing draft and separate internal notes."""
    cover_letter_path = save_generated_file(
        job_description_path, cover_letter, "cover_letter.md", workspace, package_dir
    )
    notes_path = save_generated_file(
        job_description_path, internal_notes, "cover_letter_notes.md", workspace, package_dir
    )
    return cover_letter_path, notes_path


def save_cover_letter_plan(
    job_description_path: Path,
    workspace: Workspace,
    package_dir: Path | None,
    plan: CoverLetterPlan,
) -> Path:
    """Persist the versioned local provenance plan beside the draft."""
    return save_generated_file(
        job_description_path,
        json.dumps(plan.as_dict(), indent=2, ensure_ascii=False) + "\n",
        "cover_letter_plan.json",
        workspace,
        package_dir,
    )


def generate_cover_letter(
    job_description_path: Path,
    workspace: Workspace,
    package_dir: Path | None = None,
) -> tuple[str, Path, str, Path]:
    """Read inputs, generate the cover letter and notes, and save both files."""
    workspace.require_writable()
    inputs = load_cover_letter_inputs(job_description_path, workspace)
    validate_cover_letter_inputs(inputs["job_text"])
    plan = build_cover_letter_plan(
        inputs["resume_text"],
        inputs["job_text"],
        inputs["experience_bank"],
        candidate_name=inputs["candidate_name"],
    )
    cover_letter = render_cover_letter_plan(plan)
    internal_notes = build_internal_notes(
        inputs["resume_text"], inputs["job_text"], inputs["experience_bank"], [], cover_letter
    )

    cover_letter_path, notes_path = save_cover_letter_outputs(
        job_description_path, workspace, package_dir, cover_letter, internal_notes
    )
    save_cover_letter_plan(job_description_path, workspace, package_dir, plan)
    return cover_letter, cover_letter_path, internal_notes, notes_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate a Markdown cover letter from local evidence."
    )
    parser.add_argument(
        "job_description",
        help="Path to a Markdown or text job description file.",
    )
    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""
    args = parse_args()
    job_description_path = Path(args.job_description).expanduser()

    if not job_description_path.is_absolute():
        job_description_path = PROJECT_ROOT / job_description_path

    if not job_description_path.exists():
        raise FileNotFoundError(f"Job description file was not found: {job_description_path}")

    workspace = personal_workspace()
    try:
        workspace.require_ready()
    except WorkspaceError as error:
        raise SystemExit(str(error)) from None
    cover_letter, cover_letter_path, _, notes_path = generate_cover_letter(job_description_path, workspace)
    print(cover_letter)
    print(f"Cover letter saved to: {cover_letter_path}")
    print(f"Internal notes saved to: {notes_path}")


if __name__ == "__main__":
    main()
