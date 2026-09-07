"""Recoverable deletion for locally saved job Markdown files."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping


def old_saved_jobs(jobs: Iterable[Mapping[str, Any]], *, today: date | None = None) -> tuple[list[Mapping[str, Any]], int]:
    """Select by first save date, never by publication or last refresh date."""
    cutoff = (today or date.today()) - timedelta(days=30)
    eligible: list[Mapping[str, Any]] = []
    unknown = 0
    for job in jobs:
        raw = str(job.get("first_seen_at") or job.get("created_at") or "")
        try:
            saved = datetime.fromisoformat(raw.replace("Z", "+00:00")).date()
        except ValueError:
            unknown += 1
            continue
        if saved < cutoff:
            eligible.append(job)
    return eligible, unknown


def archive_job_paths(paths: Iterable[Path], *, jobs_dir: Path, trash_dir: Path) -> tuple[list[Path], dict[str, str]]:
    """Archive an explicit selection in one batch; report individual failures."""
    batch_root = Path(trash_dir).resolve() / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    archived: list[Path] = []
    failed: dict[str, str] = {}
    for path in dict.fromkeys(paths):
        try:
            target, root = _validated_job_path(path, jobs_dir)
            destination = batch_root / target.relative_to(root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(target), str(destination))
            archived.append(destination)
        except (ValueError, OSError) as error:
            failed[str(path)] = str(error)
    return archived, failed


class SavedJobDeletionError(ValueError):
    """Raised when a deletion target is outside the active jobs directory."""


def _validated_job_path(path: Path, jobs_dir: Path) -> tuple[Path, Path]:
    root = Path(jobs_dir).resolve()
    target = Path(path).resolve()
    if not target.is_relative_to(root) or target.suffix.lower() != ".md":
        raise SavedJobDeletionError("Only Markdown jobs inside the active workspace can be deleted.")
    if not target.is_file():
        raise SavedJobDeletionError("The saved job no longer exists.")
    return target, root


def archive_saved_job(path: Path, *, jobs_dir: Path, trash_dir: Path) -> Path:
    """Move one saved job to a timestamped local Trash directory."""
    target, root = _validated_job_path(path, jobs_dir)
    relative = target.relative_to(root)
    batch = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    destination = Path(trash_dir).resolve() / batch / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(target), str(destination))
    return destination


def archive_all_saved_jobs(*, jobs_dir: Path, trash_dir: Path) -> list[Path]:
    """Move every saved job to one recoverable local Trash batch."""
    root = Path(jobs_dir).resolve()
    paths = sorted(path for path in root.rglob("*.md") if path.is_file())
    if not paths:
        return []
    batch = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    batch_root = Path(trash_dir).resolve() / batch
    archived: list[Path] = []
    for path in paths:
        target, _ = _validated_job_path(path, root)
        destination = batch_root / target.relative_to(root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(target), str(destination))
        archived.append(destination)
    return archived
