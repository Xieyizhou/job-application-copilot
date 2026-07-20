"""Safe filesystem and ZIP helpers for generated cover-letter bundles."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

from output_paths import safe_slug


INTERNAL_PACKAGE_FILES = {
    "analysis.md",
    "cover_letter.md",
    "cover_letter.docx",
    "cover_letter_notes.md",
}

PACKAGE_ZIP_FILE_ORDER = [
    "cover_letter.docx",
    "cover_letter.md",
    "analysis.md",
    "cover_letter_notes.md",
]


def existing_package_files(package_dir: Path, names: list[str]) -> list[Path]:
    """Return allowlisted files directly inside one cover-letter bundle folder."""
    try:
        resolved_package_dir = package_dir.resolve()
    except OSError:
        return []

    files = []
    for name in names:
        candidate = package_dir / name
        try:
            if (
                candidate.exists()
                and candidate.is_file()
                and candidate.resolve().parent == resolved_package_dir
                and candidate.name in INTERNAL_PACKAGE_FILES
            ):
                files.append(candidate)
        except OSError:
            continue
    return files


def build_application_package_zip(package_dir: Path) -> tuple[bytes, list[Path]]:
    """Create an in-memory ZIP containing only allowlisted bundle files."""
    package_files = existing_package_files(package_dir, PACKAGE_ZIP_FILE_ORDER)
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for path in package_files:
            zip_file.write(path, arcname=path.name)
    return zip_buffer.getvalue(), package_files


def package_zip_filename(package_dir: Path) -> str:
    """Build a readable ZIP download name for one selected bundle."""
    family = package_dir.parent.name if package_dir.parent != package_dir else "application"
    base_name = safe_slug(f"{family}_{package_dir.name}") or "application_package"
    return f"{base_name}.zip"


def readiness_status(
    *,
    source_exists: bool,
    docx_exists: bool | None = None,
    optional: bool = False,
    read_only_sample: bool = False,
) -> str:
    """Return a short user-facing bundle-readiness status."""
    if docx_exists is True:
        return "Ready"
    if docx_exists is False:
        if read_only_sample:
            return "Unavailable"
        return "Can generate" if source_exists else "Missing source"
    if source_exists:
        return "Ready"
    return "Optional" if optional else "Missing"
