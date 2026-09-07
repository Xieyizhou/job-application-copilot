"""Shared path helpers for generated job and application files."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def safe_slug(value: str) -> str:
    """Convert text into a lowercase filesystem-safe slug."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    return slug or "unknown"


def timestamp_slug() -> str:
    """Return the current timestamp used for cover-letter bundle folders."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def date_slug() -> str:
    """Return the current date used for fetched job folders."""
    return datetime.now().strftime("%Y%m%d")


def application_package_dir(output_root: Path, family: str, timestamp: str | None = None) -> Path:
    """Build a structured generated application output directory."""
    return output_root / safe_slug(family) / (timestamp or timestamp_slug())


def relative_path(path: Path, root: Path = PROJECT_ROOT) -> str:
    """Return project-relative paths for JSON sidecar storage."""
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def local_timestamp() -> str:
    """Return the existing local ISO timestamp without fractional seconds."""
    return datetime.now().replace(microsecond=0).isoformat()
