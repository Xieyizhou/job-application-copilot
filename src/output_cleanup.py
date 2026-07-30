"""Helpers for cleaning up old generated application files."""

from __future__ import annotations

from pathlib import Path

def delete_directory_tree(directory: Path) -> None:
    """Delete a generated package directory and its files."""
    for child in directory.iterdir():
        if child.is_dir():
            delete_directory_tree(child)
        else:
            child.unlink()
    directory.rmdir()
