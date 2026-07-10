"""Shared path helpers for generated job and application files."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path


def safe_slug(value: str) -> str:
    """Convert text into a lowercase filesystem-safe slug."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    return slug or "unknown"


def first_slug_word(value: str) -> str:
    """Return the first slug word, useful for compact fetched job filenames."""
    slug = safe_slug(value)
    return slug.split("_", 1)[0] if slug else "unknown"


def timestamp_slug() -> str:
    """Return the current timestamp used for application package folders."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def date_slug() -> str:
    """Return the current date used for fetched job folders."""
    return datetime.now().strftime("%Y%m%d")


def application_package_dir(output_root: Path, family: str, timestamp: str | None = None) -> Path:
    """Build a structured generated application output directory."""
    return output_root / safe_slug(family) / (timestamp or timestamp_slug())
