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
from output_paths import application_package_dir
from workspace import GENERIC_EXPERIENCE_BANK_PATH, Workspace, WorkspaceError, personal_workspace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COVER_LETTER_EXAMPLES_DIR = PROJECT_ROOT / "data" / "cover_letter_examples_md"

TARGET_WORD_COUNT_MIN = 230
TARGET_WORD_COUNT_MAX = 320
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
    "AI-generated",
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
    """Build a detailed paragraph from the strongest selected evidence."""
    experience = selection["experience"]
    bullets = selection.get("selected_evidence", [])
    name = str(experience.get("name", "this experience"))

    if not bullets:
        return build_evidence_sentence(selection)

    return f"I bring relevant experience from {name}: " + " ".join(bullets)


def build_secondary_evidence_paragraph(selected: list[dict[str, Any]]) -> str:
    """Build a concise second evidence paragraph when additional matches exist."""
    if len(selected) > 1:
        return " ".join(build_evidence_sentence(selection) for selection in selected[1:3])

    primary = selected[0]["experience"] if selected else {}
    if "machine_learning" in primary.get("tags", []):
        return (
            "That work is relevant because it required turning raw information "
            "into structured features, comparing model outputs with multiple "
            "metrics, and explaining what evaluation results meant rather than "
            "relying on a single score. I would bring the same careful, "
            "evidence-based approach to model development and review."
        )

    return ""


def build_cover_letter(
    resume_text: str,
    job_text: str,
    experience_bank: dict[str, Any] | None = None,
) -> str:
    """Build the employer-facing cover letter from local evidence blocks."""
    _ = resume_text
    bank = experience_bank or load_experience_bank()
    detected_themes = detect_themes(job_text, bank)
    selected = select_evidence_blocks(bank, detected_themes, job_text)
    profile = bank.get("profile", {})
    positioning = str(
        profile.get(
            "positioning",
            "early-career candidate with experience in Python, data analysis, model evaluation, and technical communication.",
        )
    )
    candidate_name = str(profile.get("name", "Candidate Name") or "Candidate Name").strip()
    job_title = extract_job_title(job_text)
    company = extract_company(job_text)
    focus_summary = build_focus_summary(detected_themes)

    evidence_sentences = [build_evidence_sentence(selection) for selection in selected[:3]]
    first_evidence = evidence_sentences[0] if evidence_sentences else ""
    other_evidence = evidence_sentences[1:3]

    paragraphs = [
        "Dear Hiring Team,",
        (
            f"I am excited to apply for the {job_title} position at {company}. "
            f"I am a {positioning[0].lower() + positioning[1:]} This role's focus on {focus_summary} connects closely "
            "with the project and internship work I would bring to the team."
        ),
    ]

    if selected:
        paragraphs.append(build_primary_evidence_paragraph(selected[0]))

    secondary_paragraph = build_secondary_evidence_paragraph(selected)
    if secondary_paragraph:
        paragraphs.append(secondary_paragraph)

    contribution_themes = [THEME_LABELS.get(theme, theme.replace("_", " ")) for theme in detected_themes[:2]]
    if not contribution_themes:
        contribution_text = "applied technical work"
    elif len(contribution_themes) == 1:
        contribution_text = contribution_themes[0]
    else:
        contribution_text = f"{contribution_themes[0]} and {contribution_themes[1]}"

    paragraphs.extend(
        [
            (
                f"I would welcome the opportunity to discuss how my experience in "
                f"{contribution_text} could contribute to the {job_title} role. "
                "Thank you for your time and consideration."
            ),
            f"Sincerely,\n\n{candidate_name}",
        ]
    )

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

    if contains_phrase(job_text, "robotics") and not contains_phrase(job_text, "uav"):
        weak_areas.append("robotics: resume support is strongest through UAV inspection algorithms and route planning")

    if not weak_areas:
        weak_areas.append("No major missing or weak areas detected by keyword matching.")

    return weak_areas


def build_internal_notes(
    job_text: str,
    experience_bank: dict[str, Any],
    examples: list[str],
) -> str:
    """Build separate internal notes for human review."""
    detected_themes = detect_themes(job_text, experience_bank)
    selected = select_evidence_blocks(experience_bank, detected_themes, job_text)
    required, preferred = extract_requirements(job_text)

    selected_experience_lines = [
        f"{item['experience'].get('name', 'Unnamed experience')} "
        f"(themes: {', '.join(item.get('matched_themes', [])) or 'fallback'})"
        for item in selected
    ]
    selected_evidence_lines = [
        bullet
        for item in selected
        for bullet in item.get("selected_evidence", [])
    ]

    return "\n".join(
        [
            "# Internal Cover Letter Notes",
            "",
            "## Detected JD Themes",
            "",
            format_bullets(detected_themes),
            "",
            "## Required Signals Found",
            "",
            format_bullets(required),
            "",
            "## Preferred Signals Found",
            "",
            format_bullets(preferred),
            "",
            "## Selected Experiences",
            "",
            format_bullets(selected_experience_lines),
            "",
            "## Selected Evidence Bullets",
            "",
            format_bullets(selected_evidence_lines),
            "",
            "## Weak Or Missing Areas",
            "",
            format_bullets(find_missing_or_weak_areas(job_text, detected_themes, selected)),
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
            "- Review the generated tailored resume before submitting.",
        ]
    ) + "\n"


def validate_cover_letter(cover_letter: str, experience_bank: dict[str, Any]) -> list[str]:
    """Return forbidden phrases found in employer-facing content."""
    found = []
    banned_phrases = list(experience_bank.get("banned_phrases") or DEFAULT_BANNED_PHRASES)
    for phrase in banned_phrases + GENERIC_PHRASES:
        if phrase in cover_letter:
            found.append(phrase)
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


def generate_cover_letter(
    job_description_path: Path,
    workspace: Workspace,
    package_dir: Path | None = None,
) -> tuple[str, Path, str, Path]:
    """Read inputs, generate the cover letter and notes, and save both files."""
    workspace.require_writable()
    assert workspace.resume_source_path is not None
    resume_text = workspace.resume_source_path.read_text(encoding="utf-8")
    job_text = job_description_path.read_text(encoding="utf-8")
    experience_bank = load_experience_bank(workspace.experience_bank_path or GENERIC_EXPERIENCE_BANK_PATH)
    examples = load_cover_letter_examples()
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

    cover_letter = build_cover_letter(resume_text, job_text, experience_bank)
    internal_notes = build_internal_notes(job_text, experience_bank, examples)

    forbidden_phrases = validate_cover_letter(cover_letter, experience_bank)
    if forbidden_phrases:
        print(
            "WARNING: employer-facing cover letter contains forbidden phrase(s): "
            + ", ".join(forbidden_phrases)
        )

    cover_letter_path = save_generated_file(
        job_description_path,
        cover_letter,
        "cover_letter.md",
        workspace,
        package_dir,
    )
    notes_path = save_generated_file(
        job_description_path,
        internal_notes,
        "cover_letter_notes.md",
        workspace,
        package_dir,
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
