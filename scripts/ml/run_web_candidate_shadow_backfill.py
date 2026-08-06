"""Backfill and summarize the MiniLM candidate on current personal Web jobs."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
SHADOW_ROOT = PROJECT_ROOT / "data" / "ml" / "shadow" / "web_candidate"
SUMMARY_PATH = SHADOW_ROOT / "backfill_summary.json"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml.web_candidate_shadow import (  # noqa: E402
    record_web_candidate_shadow,
    summarize_web_candidate_shadow_reports,
)
from workspace import personal_workspace  # noqa: E402


def _write_private_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
        os.replace(temporary, path)
        path.chmod(0o600)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    workspace = personal_workspace()
    workspace.require_ready()
    assert workspace.resume_source_path is not None
    resume_text = workspace.resume_source_path.read_text(encoding="utf-8")
    jobs = sorted(path for path in workspace.jobs_dir.rglob("*.md") if path.is_file())
    discovered = [(path, path.read_text(encoding="utf-8")) for path in jobs]
    report_paths: list[Path] = []
    for _, job_text in discovered:
        report = record_web_candidate_shadow(
            job_text,
            resume_text,
            workspace_mode=workspace.mode,
        )
        if report is not None:
            report_paths.append(report)
    summary = summarize_web_candidate_shadow_reports(
        report_paths,
        discovered_jobs=len(jobs),
        parseable_jobs=len(report_paths),
    )
    summary["created_at"] = datetime.now(timezone.utc).isoformat()
    summary["input_paths_recorded"] = False
    _write_private_json(SUMMARY_PATH, summary)
    print(json.dumps(summary, sort_keys=True))
    print(f"Summary: {SUMMARY_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
