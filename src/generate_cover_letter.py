"""Generate an employer-facing cover letter from local evidence.

The generator is deterministic and local:
- reads the selected workspace experience bank or tracked generic fallback
- reads the selected job description Markdown file
- uses keyword matching and evidence blocks
- keeps candidate evidence and document generation local
"""

from __future__ import annotations

import argparse
import re
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
COVER_LETTER_EXAMPLES_DIR = PROJECT_ROOT / "data" / "cover_letter_examples_md"

TARGET_WORD_COUNT_MIN = 120
TARGET_WORD_COUNT_MAX = 250
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


def load_experience_bank(path: Path) -> dict[str, Any]:
    """Load the local evidence bank."""
    if not path.exists():
        raise FileNotFoundError(f"Experience bank was not found: {path}")

    with path.open(encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    if not isinstance(data, dict):
        raise ValueError("experience_bank.yaml must contain a YAML mapping.")
    return data


def load_cover_letter_examples() -> list[str]:
    """Read local examples for tone reference when the folder exists."""
    if not COVER_LETTER_EXAMPLES_DIR.exists():
        return []

    examples = []
    for path in sorted(COVER_LETTER_EXAMPLES_DIR.glob("*")):
        if path.is_file() and path.suffix.lower() in {".md", ".txt"}:
            examples.append(path.read_text(encoding="utf-8"))
    return examples


def normalize_text(text: str) -> str:
    """Normalize text for simple keyword matching."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9+#./-]+", " ", text)
    return f" {text} "


def contains_phrase(text: str, phrase: str) -> bool:
    """Return True when a phrase appears in normalized text."""
    return f" {normalize_text(phrase).strip()} " in normalize_text(text)


def clean_duplicated_punctuation(text: str) -> str:
    """Fix duplicated periods caused by company names ending in periods."""
    replacements = {
        "Pte..": "Pte.",
        "Ltd..": "Ltd.",
        "Inc..": "Inc.",
        "Company..": "Company.",
    }
    for bad_text, clean_text in replacements.items():
        text = text.replace(bad_text, clean_text)
    return text


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


def tag_matches_theme(tag: str, theme: str) -> bool:
    """Return True when an experience tag supports a detected theme."""
    normalized_tag = tag.lower().replace("-", "_")
    if normalized_tag == theme:
        return True

    theme_to_tags = {
        "machine_learning": {"machine_learning", "classification"},
        "python_data": {"python", "pandas", "numpy", "scikit_learn", "data_cleaning", "data_validation"},
        "model_evaluation": {"model_evaluation", "classification", "robustness", "pca"},
        "data_visualization": {"data_visualization", "matplotlib"},
        "uav_robotics_sensor": {
            "uav",
            "route_planning",
            "obstacle_avoidance",
            "flight_stability",
            "thermal_data",
            "aerial_thermal_data",
            "inspection",
            "warning_system",
            "temperature_difference",
        },
        "game_ai": {"game_ai", "reinforcement_learning", "cnn", "csharp", "gameplay", "npc_behavior"},
        "econometrics_statistics": {
            "econometrics",
            "statistics",
            "regression",
            "fixed_effects",
            "instrumental_variables",
            "iv_2sls",
            "difference_in_differences",
            "causal_inference",
        },
        "communication": {"communication", "teaching", "technical_explanation", "peer_instruction", "collaboration"},
    }
    return normalized_tag in theme_to_tags.get(theme, set())


def score_experience(experience: dict[str, Any], detected_themes: list[str], job_text: str) -> tuple[int, list[str]]:
    """Score one experience by theme/tag and direct text overlap."""
    tags = [str(tag) for tag in experience.get("tags", [])]
    matched_themes = []
    score = 0

    for theme in detected_themes:
        if any(tag_matches_theme(tag, theme) for tag in tags):
            matched_themes.append(theme)
            score += 3

    evidence_text = " ".join(str(item) for item in experience.get("evidence", []))
    for tag in tags:
        readable_tag = tag.replace("_", " ")
        if contains_phrase(job_text, readable_tag):
            score += 1

    return score, matched_themes


def select_evidence_blocks(
    experience_bank: dict[str, Any],
    detected_themes: list[str],
    job_text: str,
) -> list[dict[str, Any]]:
    """Select the strongest experience blocks for the detected JD themes."""
    max_experiences = int(
        experience_bank.get("selection_rules", {}).get("max_experiences_per_letter", 3)
    )
    scored = []

    for experience in experience_bank.get("experiences", []):
        score, matched_themes = score_experience(experience, detected_themes, job_text)
        if score > 0:
            scored.append((score, experience, matched_themes))

    scored.sort(key=lambda item: (item[0], len(item[2])), reverse=True)

    selected = []
    covered_themes: set[str] = set()
    for score, experience, matched_themes in scored:
        selected.append(
            {
                "score": score,
                "experience": experience,
                "matched_themes": matched_themes,
                "selected_evidence": select_evidence_bullets(experience, matched_themes),
            }
        )
        covered_themes.update(matched_themes)
        if len(selected) >= max_experiences:
            break

    if selected:
        return selected

    # Fallback stays local and resume-backed when the JD has sparse keywords.
    for experience in experience_bank.get("experiences", [])[:1]:
        selected.append(
            {
                "score": 0,
                "experience": experience,
                "matched_themes": [],
                "selected_evidence": select_evidence_bullets(experience, []),
            }
        )
    return selected


def select_evidence_bullets(experience: dict[str, Any], matched_themes: list[str]) -> list[str]:
    """Choose the most useful evidence bullets from an experience block."""
    evidence = [str(item) for item in experience.get("evidence", [])]
    if len(evidence) <= 3:
        return evidence

    tags = [str(tag).lower().replace("-", "_") for tag in experience.get("tags", [])]
    selected = [
        bullet
        for bullet in evidence
        if any(tag_matches_theme(tag, theme) for tag in tags for theme in matched_themes)
    ]

    if not selected:
        selected = evidence[:3]
    return selected[:3]


EVIDENCE_STOPWORDS = {
    "and", "the", "with", "for", "from", "that", "this", "into", "using", "used",
    "role", "work", "team", "job", "your", "our", "you", "are", "will", "have",
    "has", "was", "were", "their", "they", "but", "not", "all", "can", "who",
}


def clean_resume_evidence_line(raw_line: str) -> str:
    """Normalize one resume line while preserving the candidate's factual wording."""
    line = raw_line.strip()
    line = re.sub(r"^[-*•]+\s*", "", line)
    line = re.sub(r"^\d+[.)]\s*", "", line)
    line = re.sub(r"\*\*([^*]+)\*\*", r"\1", line)
    line = re.sub(r"`([^`]+)`", r"\1", line)
    line = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", line)
    return re.sub(r"\s+", " ", line).strip()


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
    return "Candidate Name"


def extract_resume_evidence_blocks(
    resume_text: str,
    experience_bank: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build factual evidence blocks directly from the uploaded resume text.

    The experience bank may expand theme vocabulary, but employer-facing claims
    always come from the resume itself.
    """
    bank = experience_bank or {}
    theme_keywords = bank.get("theme_keywords") or DEFAULT_THEME_KEYWORDS
    current_heading = "Resume evidence"
    grouped: dict[str, list[str]] = {}

    for raw_line in resume_text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            heading = clean_resume_evidence_line(stripped.lstrip("#").strip())
            if heading and heading != extract_candidate_name(resume_text):
                current_heading = heading
            continue

        is_bullet = bool(re.match(r"^[-*•]\s+", stripped))
        line = clean_resume_evidence_line(stripped)
        if not line or "@" in line or line.lower().startswith(("http://", "https://")):
            continue
        word_count = len(line.split())
        if word_count < 4 or word_count > 80:
            continue
        if not is_bullet and current_heading.lower() in {"resume evidence", "contact", "summary"}:
            continue
        grouped.setdefault(current_heading, []).append(line)

    blocks = []
    for heading, evidence in grouped.items():
        combined = " ".join(evidence)
        tags = [
            theme
            for theme, keywords in theme_keywords.items()
            if any(contains_phrase(combined, str(keyword)) for keyword in keywords)
        ]
        blocks.append(
            {
                "id": re.sub(r"[^a-z0-9]+", "_", heading.lower()).strip("_") or "resume_evidence",
                "name": heading,
                "category": "resume",
                "tags": tags,
                "evidence": evidence,
                "safe_phrases": [],
            }
        )
    return blocks


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


def build_evidence_sentence(selection: dict[str, Any]) -> str:
    """Turn one selected evidence block into employer-facing prose."""
    experience = selection["experience"]
    safe_phrases = [str(item) for item in experience.get("safe_phrases", [])]
    if safe_phrases:
        return safe_phrases[0]

    evidence = selection.get("selected_evidence", [])
    if evidence:
        return evidence[0]

    return f"I bring relevant experience from {experience.get('name', 'a related project')}."


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


def build_cover_letter(
    resume_text: str,
    job_text: str,
    experience_bank: dict[str, Any] | None = None,
) -> str:
    """Build a concise employer-facing CL from JD priorities and resume proof."""
    bank = experience_bank or {"theme_keywords": DEFAULT_THEME_KEYWORDS}
    detected_themes = detect_themes(job_text, bank)
    evidence_index = build_semantic_evidence_index(job_text, resume_text)
    selected = select_resume_evidence_blocks(
        resume_text,
        job_text,
        detected_themes,
        bank,
        evidence_index=evidence_index,
    )
    candidate_name = extract_candidate_name(resume_text)
    job_title = extract_job_title(job_text)
    company = extract_company(job_text)
    focus_summary = build_focus_summary(detected_themes[:2])
    priority = top_job_priority(job_text, detected_themes)

    paragraphs = [
        "Dear Hiring Team,",
        (
            f"The {job_title} role at {company} emphasizes {priority}. "
            f"My resume shows hands-on work in {focus_summary}, and the evidence most relevant to this role "
            "comes from projects and experience where I applied those skills to concrete work."
        ),
    ]

    if selected:
        proof_sentences = [build_primary_evidence_paragraph(selected[0])]
        secondary = build_secondary_evidence_paragraph(selected)
        if secondary:
            proof_sentences.append(secondary)
        paragraphs.append(" ".join(sentence for sentence in proof_sentences if sentence))
    else:
        paragraphs.append(
            "My uploaded resume does not contain a sufficiently specific proof point for the main JD priority, "
            "so I would review this draft and add only a truthful example already supported by the resume."
        )

    paragraphs.extend(
        [
            (
                f"I would welcome a conversation about how I could bring this experience to {company}'s "
                f"{focus_summary} work."
            ),
            f"Sincerely,\n\n{candidate_name}",
        ]
    )

    cover_letter = "\n\n".join(paragraphs) + "\n"
    if len(cover_letter.split()) > TARGET_WORD_COUNT_MAX and selected:
        compact_selection = dict(selected[0])
        compact_selection["selected_evidence"] = list(selected[0].get("selected_evidence", []))[:1]
        paragraphs[2] = build_primary_evidence_paragraph(compact_selection)
        cover_letter = "\n\n".join(paragraphs) + "\n"
    return clean_duplicated_punctuation(cover_letter)


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
    examples: list[str],
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
            "## Example Files Used For Structure",
            "",
            f"- {len(examples)} local example file(s) available; examples are used only as tone references and are not copied.",
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
    examples: list[str],
    cover_letter: str,
) -> str:
    """Build a source trace, gap audit, and quality check for human review."""
    context = build_internal_note_context(resume_text, job_text, experience_bank)
    return render_internal_notes(context, job_text=job_text, examples=examples, cover_letter=cover_letter)


def validate_cover_letter(cover_letter: str, experience_bank: dict[str, Any]) -> list[str]:
    """Return forbidden phrases found in employer-facing content."""
    found = []
    banned_phrases = list(experience_bank.get("banned_phrases") or DEFAULT_BANNED_PHRASES)
    for phrase in banned_phrases + GENERIC_PHRASES:
        if phrase in cover_letter:
            found.append(phrase)
    word_count = len(cover_letter.split())
    if word_count > TARGET_WORD_COUNT_MAX:
        found.append(f"word count {word_count} exceeds {TARGET_WORD_COUNT_MAX}")
    return found


def format_bullets(items: list[str]) -> str:
    """Format a list as Markdown bullets."""
    if not items:
        return "- None found"
    return "\n".join(f"- {item}" for item in items)


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
        "examples": load_cover_letter_examples(),
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


def generate_cover_letter(
    job_description_path: Path,
    workspace: Workspace,
    package_dir: Path | None = None,
) -> tuple[str, Path, str, Path]:
    """Read inputs, generate the cover letter and notes, and save both files."""
    workspace.require_writable()
    inputs = load_cover_letter_inputs(job_description_path, workspace)
    validate_cover_letter_inputs(inputs["job_text"])
    cover_letter = build_cover_letter(inputs["resume_text"], inputs["job_text"], inputs["experience_bank"])
    internal_notes = build_internal_notes(
        inputs["resume_text"], inputs["job_text"], inputs["experience_bank"], inputs["examples"], cover_letter
    )

    forbidden_phrases = validate_cover_letter(cover_letter, inputs["experience_bank"])
    if forbidden_phrases:
        print(
            "WARNING: employer-facing cover letter contains forbidden phrase(s): "
            + ", ".join(forbidden_phrases)
        )

    cover_letter_path, notes_path = save_cover_letter_outputs(
        job_description_path, workspace, package_dir, cover_letter, internal_notes
    )
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
