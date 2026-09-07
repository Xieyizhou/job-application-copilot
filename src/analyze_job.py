"""CLI for deterministic job-fit analysis; domain APIs live in scoring modules."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scoring_report import analyze_job, analyze_job_structured
from workspace import WorkspaceError, demo_workspace, personal_workspace


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze a job description against the selected candidate workspace."
    )
    parser.add_argument(
        "job_description",
        help="Path to a Markdown or text job description file.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Analyze with sanitized Demo candidate data without writing files.",
    )
    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""
    args = parse_args()
    job_description_path = Path(args.job_description).expanduser()

    if not job_description_path.is_absolute():
        job_description_path = PROJECT_ROOT / job_description_path

    if not job_description_path.exists():
        raise FileNotFoundError(f"Job description file was not found: {job_description_path}")

    if args.demo:
        workspace = demo_workspace()
        workspace.require_ready()
        assert workspace.resume_source_path is not None
        analysis = analyze_job_structured(
            job_description_path.read_text(encoding="utf-8"),
            workspace.resume_source_path.read_text(encoding="utf-8"),
        )
        print(json.dumps(analysis, indent=2))
        return

    workspace = personal_workspace()
    try:
        workspace.require_ready()
    except WorkspaceError as error:
        raise SystemExit(str(error)) from None
    report, report_path = analyze_job(job_description_path, workspace)
    print(report)
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
