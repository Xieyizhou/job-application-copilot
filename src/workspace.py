"""Centralized Demo and Personal workspace path resolution."""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from candidate_document import (
    CandidateDocumentError,
    CandidateDocumentResult,
    SUPPORTED_CANDIDATE_EXTENSIONS,
    parse_candidate_document,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMO_ROOT = PROJECT_ROOT / "data" / "demo"
LOCAL_WORKSPACE_ROOT = PROJECT_ROOT / "data" / "local_workspace"
GENERIC_RESUME_PATH = PROJECT_ROOT / "data" / "resume" / "resume_source.example.md"
GENERIC_EXPERIENCE_BANK_PATH = PROJECT_ROOT / "data" / "experience_bank.example.yaml"
GENERIC_COVER_LETTER_TEMPLATE_PATH = PROJECT_ROOT / "data" / "templates" / "cover_letter_template.docx"
MANIFEST_NAME = "workspace.json"
SUPPORTED_RESUME_EXTENSIONS = SUPPORTED_CANDIDATE_EXTENSIONS
SUPPORTED_EXPERIENCE_BANK_EXTENSIONS = {".yaml", ".yml"}
SUPPORTED_COVER_LETTER_TEMPLATE_EXTENSIONS = {".docx"}


class WorkspaceError(RuntimeError):
    """Raised when a workspace operation is invalid or unsafe."""


@dataclass(frozen=True)
class Workspace:
    """Resolved paths and readiness for one local application workspace."""

    mode: Literal["demo", "personal"]
    root: Path
    resume_source_path: Path | None
    experience_bank_path: Path | None
    cover_letter_template_path: Path | None
    jobs_dir: Path
    generated_dir: Path
    tracker_database_path: Path | None
    manifest_path: Path | None
    ready: bool
    missing_inputs: tuple[str, ...] = ()
    read_only: bool = False
    candidate_original_extension: str | None = None
    candidate_extraction_method: str | None = None
    candidate_pdf_page_count: int | None = None

    def require_ready(self) -> None:
        """Reject candidate workflows until required workspace inputs exist."""
        if not self.ready or self.resume_source_path is None:
            raise WorkspaceError(
                "Personal workspace is not configured. Add a candidate source in Candidate Workspace Setup."
            )

    def require_writable(self) -> None:
        """Reject writes from the bundled Demo workspace."""
        if self.read_only:
            raise WorkspaceError("Demo workspace is read-only.")
        self.require_ready()


def demo_workspace() -> Workspace:
    """Return the read-only workspace backed only by sanitized tracked data."""
    missing = []
    if not GENERIC_RESUME_PATH.is_file():
        missing.append("demo candidate source")
    if not (DEMO_ROOT / "jobs").is_dir():
        missing.append("demo jobs")
    if not (DEMO_ROOT / "sample_package").is_dir():
        missing.append("demo sample package")
    return Workspace(
        mode="demo",
        root=DEMO_ROOT,
        resume_source_path=GENERIC_RESUME_PATH if GENERIC_RESUME_PATH.is_file() else None,
        experience_bank_path=GENERIC_EXPERIENCE_BANK_PATH if GENERIC_EXPERIENCE_BANK_PATH.is_file() else None,
        cover_letter_template_path=(
            GENERIC_COVER_LETTER_TEMPLATE_PATH if GENERIC_COVER_LETTER_TEMPLATE_PATH.is_file() else None
        ),
        jobs_dir=DEMO_ROOT / "jobs",
        generated_dir=DEMO_ROOT / "sample_package",
        tracker_database_path=None,
        manifest_path=None,
        ready=not missing,
        missing_inputs=tuple(missing),
        read_only=True,
    )


def _safe_manifest_file(root: Path, relative_value: object, allowed_dir: str) -> Path | None:
    if not isinstance(relative_value, str) or not relative_value.strip():
        return None
    relative_path = Path(relative_value)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        return None
    candidate = (root / relative_path).resolve()
    allowed_root = (root / allowed_dir).resolve()
    if candidate.parent != allowed_root:
        return None
    return candidate if candidate.is_file() else None


def personal_workspace(root: Path = LOCAL_WORKSPACE_ROOT) -> Workspace:
    """Load Personal workspace paths without opening any candidate document."""
    root = root.resolve()
    manifest_path = root / MANIFEST_NAME
    manifest: dict[str, object] = {}
    if manifest_path.is_file():
        try:
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                manifest = loaded
        except (OSError, json.JSONDecodeError):
            manifest = {}

    resume_path = _safe_manifest_file(root, manifest.get("resume_source"), "candidate")
    if resume_path is not None and resume_path.name != "candidate_source.md":
        resume_path = None
    experience_path = _safe_manifest_file(root, manifest.get("experience_bank"), "candidate")
    template_path = _safe_manifest_file(root, manifest.get("cover_letter_template"), "templates")
    missing = () if resume_path else ("candidate source",)
    return Workspace(
        mode="personal",
        root=root,
        resume_source_path=resume_path,
        experience_bank_path=experience_path,
        cover_letter_template_path=template_path,
        jobs_dir=root / "jobs",
        generated_dir=root / "generated",
        tracker_database_path=root / "applications.db",
        manifest_path=manifest_path,
        ready=manifest_path.is_file() and resume_path is not None,
        missing_inputs=missing,
        read_only=False,
        candidate_original_extension=(
            str(manifest.get("candidate_original_extension"))
            if isinstance(manifest.get("candidate_original_extension"), str)
            else None
        ),
        candidate_extraction_method=(
            str(manifest.get("candidate_extraction_method"))
            if isinstance(manifest.get("candidate_extraction_method"), str)
            else None
        ),
        candidate_pdf_page_count=(
            int(manifest["candidate_pdf_page_count"])
            if isinstance(manifest.get("candidate_pdf_page_count"), int)
            else None
        ),
    )


def resolve_workspace(mode: str) -> Workspace:
    """Resolve a user-facing workspace mode."""
    normalized = mode.strip().lower()
    if normalized == "demo":
        return demo_workspace()
    if normalized == "personal":
        return personal_workspace()
    raise WorkspaceError(f"Unsupported workspace mode: {mode}")


def sanitize_upload_filename(filename: str, allowed_extensions: set[str]) -> str:
    """Return a basename-only safe upload filename with an allowed extension."""
    basename = Path(filename.replace("\\", "/")).name
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(basename).stem).strip("._-")
    extension = Path(basename).suffix.lower()
    if extension not in allowed_extensions:
        allowed = ", ".join(sorted(allowed_extensions))
        raise WorkspaceError(f"Unsupported file type. Use one of: {allowed}.")
    if not stem:
        stem = "uploaded_file"
    return f"{stem[:80]}{extension}"


def _write_upload(root: Path, directory: str, filename: str, content: bytes, allowed: set[str]) -> str:
    if not content:
        raise WorkspaceError("Uploaded files must not be empty.")
    safe_name = sanitize_upload_filename(filename, allowed)
    destination_dir = (root / directory).resolve()
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = (destination_dir / safe_name).resolve()
    if destination.parent != destination_dir:
        raise WorkspaceError("Upload destination is outside the local workspace.")
    destination.write_bytes(content)
    return destination.relative_to(root).as_posix()


def _candidate_directory(root: Path) -> Path:
    candidate_dir = (root / "candidate").resolve()
    if candidate_dir.parent != root:
        raise WorkspaceError("Candidate storage is outside the local workspace.")
    return candidate_dir


def _write_candidate_file_atomically(candidate_dir: Path, filename: str, content: bytes) -> Path:
    destination = (candidate_dir / filename).resolve()
    if destination.parent != candidate_dir:
        raise WorkspaceError("Candidate storage is outside the local workspace.")
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=candidate_dir, prefix=".candidate-", delete=False) as temporary_file:
            temporary_file.write(content)
            temporary_path = Path(temporary_file.name)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return destination


def initialize_personal_workspace(
    resume_filename: str,
    resume_content: bytes,
    experience_bank: tuple[str, bytes] | None = None,
    cover_letter_template: tuple[str, bytes] | None = None,
    root: Path = LOCAL_WORKSPACE_ROOT,
) -> Workspace:
    """Safely replace the canonical candidate source and update local metadata."""
    root = root.resolve()
    try:
        parsed_candidate = parse_candidate_document(resume_filename, resume_content)
    except CandidateDocumentError as error:
        raise WorkspaceError(str(error)) from None

    candidate_dir = _candidate_directory(root)
    (root / "candidate").mkdir(parents=True, exist_ok=True)
    (root / "templates").mkdir(parents=True, exist_ok=True)
    (root / "jobs").mkdir(parents=True, exist_ok=True)
    (root / "generated").mkdir(parents=True, exist_ok=True)

    existing = personal_workspace(root)
    original_name = f"original_resume{parsed_candidate.original_extension}"
    canonical_name = "candidate_source.md"
    _write_candidate_file_atomically(candidate_dir, original_name, resume_content)
    _write_candidate_file_atomically(candidate_dir, canonical_name, parsed_candidate.markdown.encode("utf-8"))
    resume_relative = f"candidate/{canonical_name}"
    experience_relative = (
        existing.experience_bank_path.relative_to(root).as_posix()
        if existing.experience_bank_path
        else None
    )
    template_relative = (
        existing.cover_letter_template_path.relative_to(root).as_posix()
        if existing.cover_letter_template_path
        else None
    )
    if experience_bank is not None:
        experience_relative = _write_upload(
            root, "candidate", experience_bank[0], experience_bank[1], SUPPORTED_EXPERIENCE_BANK_EXTENSIONS
        )
    if cover_letter_template is not None:
        template_relative = _write_upload(
            root, "templates", cover_letter_template[0], cover_letter_template[1],
            SUPPORTED_COVER_LETTER_TEMPLATE_EXTENSIONS,
        )

    manifest = {
        "version": 1,
        "mode": "personal",
        "resume_source": resume_relative,
        "experience_bank": experience_relative,
        "cover_letter_template": template_relative,
        "candidate_original_extension": parsed_candidate.original_extension,
        "candidate_extraction_method": parsed_candidate.extraction_method,
        "candidate_pdf_page_count": parsed_candidate.page_count,
        "configured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    manifest_temporary = root / f".{MANIFEST_NAME}.tmp"
    manifest_temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(manifest_temporary, root / MANIFEST_NAME)
    for extension in SUPPORTED_RESUME_EXTENSIONS:
        obsolete_original = candidate_dir / f"original_resume{extension}"
        if obsolete_original.name != original_name and obsolete_original.is_file():
            obsolete_original.unlink()
    return personal_workspace(root)


def generic_cover_letter_template(workspace: Workspace) -> Path:
    """Resolve an explicit user template or the tracked generic fallback."""
    if workspace.cover_letter_template_path:
        return workspace.cover_letter_template_path
    if GENERIC_COVER_LETTER_TEMPLATE_PATH.is_file():
        return GENERIC_COVER_LETTER_TEMPLATE_PATH
    raise WorkspaceError("No cover-letter template is available.")
