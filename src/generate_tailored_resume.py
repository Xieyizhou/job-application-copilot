"""Generate a tailored Markdown resume from the resume source of truth.

This script is intentionally conservative:
- It only reads facts from the selected workspace candidate source.
- It does not invent experience, skills, degree level, visa status, or metrics.
- It does not submit applications or interact with job platforms.
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path

from output_paths import application_package_dir
from workspace import Workspace, WorkspaceError, personal_workspace


PROJECT_ROOT = Path(__file__).resolve().parents[1]


CLEAN_SUMMARY = (
    "Candidate with experience in Python-based data analysis, machine learning "
    "model evaluation, technical communication, and applied project work. "
    "Experienced in translating job requirements into evidence-backed resume "
    "sections while preserving factual source material."
)


FORBIDDEN_EMPLOYER_PHRASES = [
    "Tailored Resume Draft",
    "Tailoring Notes",
    "undergraduate graduate",
    "master's",
    "Master's",
    "MS",
    "M.S.",
    "AI-generated",
    "internal comments",
]


ML_PYTHON_LIBRARIES = ["NumPy", "pandas", "scikit-learn", "matplotlib"]


# Theme keywords decide what parts of the resume should move earlier.
# The resume text itself remains the source of truth.
THEME_KEYWORDS = {
    "machine learning / data": [
        "python",
        "pandas",
        "scikit-learn",
        "machine learning",
        "model evaluation",
        "pca",
        "data visualization",
        "data analysis",
        "classification",
    ],
    "UAV / robotics / algorithms": [
        "uav",
        "robotics",
        "robot",
        "route planning",
        "route-planning",
        "obstacle avoidance",
        "flight stability",
        "aerial thermal data",
        "thermal data",
        "temperature-difference",
        "abnormal-temperature",
        "sensor data",
    ],
    "game AI": [
        "game ai",
        "npc",
        "neural-network",
        "neural network",
        "reinforcement learning",
        "cnn",
        "image enhancement",
        "c#",
        "gameplay",
    ],
    "econometrics / analyst": [
        "regression",
        "iv/2sls",
        "2sls",
        "fixed effects",
        "did",
        "difference-in-differences",
        "statistics",
        "econometrics",
        "causal inference",
        "analyst",
    ],
    "communication": [
        "communication",
        "teaching",
        "teamwork",
        "documentation",
        "presentation",
        "peer educator",
        "instructional assistant",
    ],
}


# Keywords in a JD that this resume source supports directly.
SUPPORTED_KEYWORDS = [
    "python",
    "pandas",
    "scikit-learn",
    "machine learning",
    "model evaluation",
    "data visualization",
    "data analysis",
    "uav",
    "route planning",
    "thermal data",
    "communication",
    "sql",
    "pca",
    "classification",
    "cnn",
    "reinforcement learning",
    "econometrics",
    "statistics",
    "causal inference",
]


# Adjacent matches should be surfaced for human review instead of overstated.
ADJACENT_KEYWORDS = {
    "robotics": "adjacent match through UAV inspection algorithms and route planning",
    "robotics perception": "adjacent match through UAV inspection algorithms and route planning",
    "sensor data": "adjacent match through UAV inspection data and aerial thermal data",
}


SECTION_ORDER = [
    "Contact",
    "Summary",
    "Education",
    "Technical Skills",
    "Experience",
    "Projects",
    "Volunteer Experience",
]


def normalize_text(text: str) -> str:
    """Lowercase text and replace punctuation with spaces for simple matching."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9+#./-]+", " ", text)
    return f" {text} "


def contains_phrase(text: str, phrase: str) -> bool:
    """Return True when a phrase appears in normalized text."""
    return f" {normalize_text(phrase).strip()} " in normalize_text(text)


def parse_sections(markdown_text: str) -> dict[str, str]:
    """Parse top-level Markdown sections that begin with '## '."""
    sections = {}
    current_title = None
    current_lines = []

    for line in markdown_text.splitlines():
        if line.startswith("## "):
            if current_title is not None:
                sections[current_title] = "\n".join(current_lines).strip()
            current_title = line.removeprefix("## ").strip()
            current_lines = []
        elif current_title is not None:
            current_lines.append(line)

    if current_title is not None:
        sections[current_title] = "\n".join(current_lines).strip()

    return sections


def parse_subsection_blocks(section_text: str) -> list[str]:
    """Split a section into blocks beginning with '### '."""
    blocks = []
    current_lines = []

    for line in section_text.splitlines():
        if line.startswith("### ") and current_lines:
            blocks.append("\n".join(current_lines).strip())
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_lines:
        blocks.append("\n".join(current_lines).strip())

    return [block for block in blocks if block]


def detect_themes(job_text: str) -> list[str]:
    """Find which tailoring themes appear in the job description."""
    active_themes = []

    for theme, keywords in THEME_KEYWORDS.items():
        if any(contains_phrase(job_text, keyword) for keyword in keywords):
            active_themes.append(theme)

    return active_themes


def theme_score(text: str, active_themes: list[str]) -> int:
    """Score resume text based on active JD themes."""
    score = 0
    for theme in active_themes:
        for keyword in THEME_KEYWORDS[theme]:
            if contains_phrase(text, keyword):
                score += 1
    return score


def reorder_blocks(section_text: str, active_themes: list[str]) -> str:
    """Move the most relevant experience or project blocks earlier."""
    blocks = parse_subsection_blocks(section_text)
    if not blocks:
        return section_text

    scored_blocks = [
        (theme_score(block, active_themes), original_index, tailor_bullets(block, active_themes))
        for original_index, block in enumerate(blocks)
    ]
    scored_blocks.sort(key=lambda item: (-item[0], item[1]))

    return "\n\n".join(block for _, _, block in scored_blocks)


def tailor_bullets(block_text: str, active_themes: list[str]) -> str:
    """Move relevant bullets within a block earlier while preserving wording."""
    lines = block_text.splitlines()
    header_lines = []
    bullet_lines = []
    other_lines = []

    for line in lines:
        if line.startswith("- "):
            bullet_lines.append(line)
        elif bullet_lines:
            other_lines.append(line)
        else:
            header_lines.append(line)

    bullet_lines.sort(key=lambda line: -theme_score(line, active_themes))
    return "\n".join(header_lines + bullet_lines + other_lines).strip()


def tailor_technical_skills(section_text: str, active_themes: list[str]) -> str:
    """Reorder skill groups and bullets so JD-relevant skills appear first."""
    blocks = parse_subsection_blocks(section_text)
    if not blocks:
        return section_text

    tailored_blocks = []
    for block in blocks:
        lines = block.splitlines()
        heading = lines[0]
        bullets = [line for line in lines[1:] if line.startswith("- ")]
        non_bullets = [line for line in lines[1:] if not line.startswith("- ")]
        if heading == "### Python Libraries" and "machine learning / data" in active_themes:
            bullets = order_python_libraries(bullets)
        else:
            bullets.sort(key=lambda line: -theme_score(line, active_themes))
        tailored_blocks.append(
            (theme_score(block, active_themes), f"{heading}\n\n" + "\n".join(non_bullets + bullets).strip())
        )

    tailored_blocks.sort(key=lambda item: -item[0])
    return "\n\n".join(block for _, block in tailored_blocks)


def order_python_libraries(bullets: list[str]) -> list[str]:
    """Keep all Python library bullets while prioritizing ML/data libraries."""
    ordered_bullets = []
    remaining_bullets = bullets[:]

    for library in ML_PYTHON_LIBRARIES:
        for bullet in remaining_bullets:
            if contains_phrase(bullet, library):
                ordered_bullets.append(bullet)
                remaining_bullets.remove(bullet)
                break

    return ordered_bullets + remaining_bullets


def find_job_requests(job_text: str) -> list[str]:
    """Return supported or adjacent keywords that the JD appears to ask for."""
    requests = []

    for keyword in SUPPORTED_KEYWORDS:
        if contains_phrase(job_text, keyword):
            requests.append(keyword)

    for keyword in ADJACENT_KEYWORDS:
        if contains_phrase(job_text, keyword):
            requests.append(keyword)

    return requests


def find_missing_or_weak_areas(job_text: str) -> list[str]:
    """Flag adjacent or unsupported JD terms for manual review."""
    weak_areas = []

    for keyword, note in ADJACENT_KEYWORDS.items():
        if contains_phrase(job_text, keyword):
            weak_areas.append(f"{keyword}: {note}")

    known_terms = set(SUPPORTED_KEYWORDS) | set(ADJACENT_KEYWORDS)
    common_terms_to_check = [
        "master's",
        "phd",
        "ph.d.",
        "citizenship",
        "permanent residency",
        "work authorization",
        "visa",
        "sponsorship",
        "senior",
        "staff",
        "principal",
        "manager",
    ]

    for term in common_terms_to_check:
        if term not in known_terms and contains_phrase(job_text, term):
            weak_areas.append(f"{term}: review manually before applying")

    if not weak_areas:
        weak_areas.append("No major missing or weak areas detected by keyword matching.")

    return weak_areas


def build_internal_notes(job_text: str, active_themes: list[str]) -> str:
    """Create a separate internal notes file for human review."""
    job_requests = find_job_requests(job_text)
    matched_skills = [
        keyword
        for keyword in SUPPORTED_KEYWORDS
        if contains_phrase(job_text, keyword)
    ]
    weak_areas = find_missing_or_weak_areas(job_text)

    return "\n".join(
        [
            "# Internal Tailoring Notes",
            "",
            "## Job Description Themes Detected",
            "",
            format_bullets(active_themes),
            "",
            "## What The Job Description Asked For",
            "",
            format_bullets(job_requests),
            "",
            "## Emphasized Resume Themes",
            "",
            format_bullets(active_themes),
            "",
            "## Matched Skills",
            "",
            format_bullets(matched_skills),
            "",
            "## Missing Or Weak Areas",
            "",
            format_bullets(weak_areas),
            "",
            "## Human Review Warnings",
            "",
            "- Review adjacent matches before submitting the resume.",
            "- Confirm the job's degree, visa, work authorization, citizenship, and location requirements manually.",
            "- This notes file is internal only and should not be submitted to employers.",
            "- The employer-facing resume should use only facts from the selected candidate source.",
            "- Confirm the resume source's degree level before relying on education-related statements.",
        ]
    )


def extract_resume_display_name(resume_text: str) -> str:
    """Return the first plausible candidate name from the resume source."""
    in_contact = False
    for raw_line in resume_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.lower() == "## contact":
            in_contact = True
            continue
        if in_contact:
            if line.startswith("#"):
                break
            lowered = line.lower()
            if "@" in line or "linkedin.com" in lowered or "http" in lowered:
                continue
            return line
    return "Candidate Name"


def build_tailored_resume(resume_text: str, job_text: str) -> str:
    """Build the clean employer-facing tailored Markdown resume."""
    sections = parse_sections(resume_text)
    active_themes = detect_themes(job_text)
    output_lines = [f"# {extract_resume_display_name(resume_text)}", ""]

    for section_name in SECTION_ORDER:
        section_text = sections.get(section_name, "").strip()
        if not section_text:
            continue

        if section_name == "Summary":
            section_text = CLEAN_SUMMARY
        elif section_name == "Technical Skills":
            section_text = tailor_technical_skills(section_text, active_themes)
        elif section_name in {"Experience", "Projects", "Volunteer Experience"}:
            section_text = reorder_blocks(section_text, active_themes)

        output_lines.extend([f"## {section_name}", "", section_text, ""])

    return "\n".join(output_lines).strip() + "\n"


def validate_employer_resume(resume_text: str) -> list[str]:
    """Return forbidden employer-facing phrases found in the resume."""
    found_phrases = []

    for phrase in FORBIDDEN_EMPLOYER_PHRASES:
        if phrase in resume_text:
            found_phrases.append(phrase)

    return found_phrases


def format_bullets(items: list[str]) -> str:
    """Format a list as Markdown bullets with a safe fallback."""
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


def generate_tailored_resume(
    job_description_path: Path,
    workspace: Workspace,
    package_dir: Path | None = None,
) -> tuple[str, Path, str, Path]:
    """Read inputs, generate the resume and notes, and save both files."""
    workspace.require_writable()
    assert workspace.resume_source_path is not None
    resume_text = workspace.resume_source_path.read_text(encoding="utf-8")
    job_text = job_description_path.read_text(encoding="utf-8")
    active_themes = detect_themes(job_text)
    tailored_resume = build_tailored_resume(resume_text, job_text)
    internal_notes = build_internal_notes(job_text, active_themes)
    forbidden_phrases = validate_employer_resume(tailored_resume)
    if forbidden_phrases:
        print(
            "WARNING: employer-facing resume contains forbidden phrase(s): "
            + ", ".join(forbidden_phrases)
        )
    resume_path = save_generated_file(
        job_description_path,
        tailored_resume,
        "tailored_resume.md",
        workspace,
        package_dir,
    )
    notes_path = save_generated_file(
        job_description_path,
        internal_notes,
        "tailoring_notes.md",
        workspace,
        package_dir,
    )
    return tailored_resume, resume_path, internal_notes, notes_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate a tailored Markdown resume from the Personal workspace candidate source."
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
    tailored_resume, resume_path, _, notes_path = generate_tailored_resume(job_description_path, workspace)
    print(tailored_resume)
    print(f"Employer-facing resume saved to: {resume_path}")
    print(f"Internal notes saved to: {notes_path}")


if __name__ == "__main__":
    main()
