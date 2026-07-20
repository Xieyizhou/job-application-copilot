"""Helpers for cleaning up old generated application files."""

from __future__ import annotations

import re
from pathlib import Path


TIMESTAMP_DIR_PATTERN = r"\d{8}_\d{6}"


def cleanup_old_outputs(output_dir: str, keep: int = 3) -> list[Path]:
    """Keep only the latest generated cover-letter bundle folders per family.

    The cleanup is intentionally narrow:
    - It only operates inside a directory named ``generated_applications``.
    - It only deletes timestamp-named folders inside family folders.
    - It never deletes ``.gitkeep`` or files outside the output directory.
    """
    output_path = Path(output_dir).resolve()

    if keep < 1:
        raise ValueError("keep must be at least 1")

    if output_path.name != "generated_applications":
        raise ValueError(
            "cleanup_old_outputs only deletes files inside generated_applications/"
        )

    if not output_path.exists():
        return []

    deleted_count = 0
    deleted_paths: list[Path] = []

    for family_dir in output_path.iterdir():
        if not family_dir.is_dir():
            continue
        if family_dir.name == ".gitkeep":
            continue

        timestamp_dirs = [
            child
            for child in family_dir.iterdir()
            if child.is_dir() and is_timestamp_dir(child)
        ]
        sorted_dirs = sorted(
            timestamp_dirs,
            key=lambda path: (path.name, path.stat().st_mtime),
            reverse=True,
        )
        old_dirs = sorted_dirs[keep:]

        for old_dir in old_dirs:
            delete_directory_tree(old_dir)
            deleted_count += 1
            deleted_paths.append(old_dir)
            print(f"Cleanup: deleted old generated output folder: {old_dir}")

    if deleted_count == 0:
        print("Cleanup complete: no old generated outputs to delete.")

    return deleted_paths


def is_timestamp_dir(path: Path) -> bool:
    """Return True for folders named like 20260701_101854."""
    return bool(re.fullmatch(TIMESTAMP_DIR_PATTERN, path.name))


def delete_directory_tree(directory: Path) -> None:
    """Delete a generated package directory and its files."""
    for child in directory.iterdir():
        if child.is_dir():
            delete_directory_tree(child)
        else:
            child.unlink()
    directory.rmdir()
