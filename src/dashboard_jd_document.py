"""Readable presentation helpers for saved job-description text."""

from __future__ import annotations

import re


SECTION_LABELS = (
    "Education and Experience Required",
    "Personal & Professional Development",
    "What We Can Offer You",
    "Unconditional Inclusion",
    "Let's Stay Connected",
    "Job Description",
    "Additional Skills",
    "Health & Wellbeing",
    "Recruitment Fraud Alert",
    "Who We Are",
    "We Are",
)


def _drop_repeated_trailing_sections(text: str) -> str:
    """Discard provider-added sections that repeat prose already present above them."""
    parts = re.split(r"(?m)(?=^##\s+)", text)
    if len(parts) < 2:
        return text
    kept: list[str] = []
    prior_text = ""
    for part in parts:
        section_body = re.sub(r"(?m)^##\s+[^\n]+\n*", "", part, count=1).strip()
        normalized_body = re.sub(r"\s+", " ", section_body).casefold()
        normalized_prior = re.sub(r"\s+", " ", prior_text).casefold()
        fingerprint = normalized_body[:160]
        if len(fingerprint) >= 80 and fingerprint in normalized_prior:
            continue
        kept.append(part)
        prior_text = "\n".join(kept)
    return "".join(kept).strip()


def _paragraphize(line: str, target_length: int = 380) -> list[str]:
    """Split an imported wall of prose into readable, sentence-safe paragraphs."""
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", line.strip())
    if len(sentences) < 2:
        return [line.strip()]
    paragraphs: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip()
        if current and len(candidate) > target_length:
            paragraphs.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        paragraphs.append(current)
    return paragraphs


def format_job_description_body(body: str, role: str = "") -> str:
    """Turn API-delivered plain text into restrained Markdown without changing its words."""
    text = body.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return ""
    text = _drop_repeated_trailing_sections(text)
    if role and text.casefold().startswith(role.strip().casefold()):
        text = text[len(role.strip()) :].lstrip(" :-")
    text = re.sub(r"[\t ]+", " ", text)
    text = re.sub(r"\s*•\s*", "\n- ", text)
    structured_lines: list[str] = []
    section_pattern = "|".join(
        re.escape(label) for label in SECTION_LABELS if label not in {"Who We Are", "We Are"}
    )
    for source_line in text.splitlines():
        if source_line.lstrip().startswith("#"):
            structured_lines.append(source_line)
            continue
        source_line = re.sub(
            rf"\s*(?P<label>{section_pattern})\s*:?[\t ]*",
            lambda match: f"\n\n### {match.group('label')}\n\n",
            source_line,
            flags=re.IGNORECASE,
        )
        source_line = re.sub(
            r"\s*(?P<label>Who We Are|We Are)\s*:[\t ]*",
            lambda match: f"\n\n### {match.group('label')}\n\n",
            source_line,
            flags=re.IGNORECASE,
        )
        structured_lines.append(source_line)
    text = "\n".join(structured_lines)

    rendered: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if rendered and rendered[-1] != "":
                rendered.append("")
            continue
        if re.match(r"(?:[-*]\s+|#{1,6}\s+)", line):
            rendered.append(line)
        else:
            for paragraph in _paragraphize(line):
                rendered.extend([paragraph, ""])
    return "\n".join(rendered).strip()
