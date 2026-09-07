"""Shared Markdown metadata and document formatting primitives."""

from pathlib import Path


def read_markdown_field(markdown_text: str, field_name: str, default: str = "") -> str:
    """Read a simple Markdown metadata field."""
    prefix = f"{field_name}:"
    for line in markdown_text.splitlines():
        if line.lower().startswith(prefix.lower()):
            value = line.split(":", 1)[1].strip()
            if value and value.lower() != "not provided":
                return value
    return default


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


def format_bullets(items: list[str]) -> str:
    """Format a list as Markdown bullets."""
    if not items:
        return "- None found"
    return "\n".join(f"- {item}" for item in items)


def read_text_file(path: Path) -> str:
    """Read a saved Markdown preview without raising for missing files."""
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""
