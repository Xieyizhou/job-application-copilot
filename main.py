"""Unified command-line entry point for the local Job Application Toolkit.

This file is a thin convenience wrapper. The existing scripts in src/ still
work directly; main.py only makes the daily workflow shorter.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from apply_package import create_application_package, resolve_metadata  # noqa: E402
from fetch_jobs import fetch_and_save_jobs  # noqa: E402
from tracker import delete_application, list_applications, show_application, update_status  # noqa: E402
from workspace import WorkspaceError, personal_workspace  # noqa: E402


def run_fetch(args: argparse.Namespace) -> None:
    """Fetch jobs through the existing fetch_jobs.py logic."""
    result = fetch_and_save_jobs(args)
    saved_paths = list(result.get("saved_paths", []))
    fetch_run = dict(result.get("fetch_run", {}))

    print("")
    print("Fetch Summary")
    print("-------------")
    print(f"Fetch run: {fetch_run.get('fetch_run_id', 'unknown')}")
    print(f"Returned: {fetch_run.get('total_jobs_returned', 0)} job(s)")
    print(f"New jobs: {fetch_run.get('new_jobs_count', len(saved_paths))}")
    print(f"Previously seen: {fetch_run.get('duplicate_jobs_count', 0)}")
    print(f"Created {len(saved_paths)} job description file(s).")
    print("")
    if saved_paths:
        print("Generated file(s):")
        for path in saved_paths:
            print(f"- {path}")
    else:
        print("No new job description files were created.")
    print("")
    print("Next step: run `python3 main.py cover-letter <job.md>`.")


def run_package(args: argparse.Namespace) -> None:
    """Generate a cover-letter bundle through apply_package.py logic."""
    job_description_path = Path(args.job_description).expanduser()
    if not job_description_path.is_absolute():
        job_description_path = PROJECT_ROOT / job_description_path

    if not job_description_path.exists():
        raise FileNotFoundError(f"Job description file was not found: {job_description_path}")

    metadata = resolve_metadata(args, job_description_path)
    workspace = personal_workspace()
    try:
        workspace.require_ready()
    except WorkspaceError as error:
        raise SystemExit(str(error)) from None
    summary = create_application_package(
        job_description_path=job_description_path,
        workspace=workspace,
        company=metadata["company"],
        role=metadata["role"],
        location=metadata["location"],
        job_url=metadata["job_url"],
    )

    print("\nCover Letter Bundle Summary")
    print("---------------------------")
    print(f"Parsed company: {summary['company']}")
    print(f"Parsed role: {summary['role']}")
    print(f"Parsed location: {summary['location']}")
    print(f"Parsed job URL: {summary['job_url']}")
    print(f"Match score: {summary['match_score']}/100")
    print(f"Recommendation: {summary['recommendation']}")
    print(f"Generated bundle folder: {summary['package_dir']}")
    print(f"Analysis report file: {summary['analysis_path']}")
    print(f"Cover letter Markdown file: {summary['cover_letter_path']}")
    print(f"Cover letter DOCX file: {summary['cover_letter_docx_path']}")
    print(f"Tracker id: {summary['tracker_id']}")
    print("Status: ready")
    if summary["export_warnings"]:
        print("Validation warnings:")
        for warning in summary["export_warnings"]:
            print(f"- {warning}")
    else:
        print("Validation warnings: none")
    print("Reminder: no application was submitted automatically.")


def build_parser() -> argparse.ArgumentParser:
    """Build the unified CLI parser."""
    parser = argparse.ArgumentParser(description="Local Job Application Toolkit CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch_parser = subparsers.add_parser("fetch", help="Fetch jobs from a supported source.")
    fetch_parser.add_argument("--source", choices=["jsearch", "adzuna", "jooble"], default="jsearch")
    fetch_parser.add_argument("--query", required=True)
    fetch_parser.add_argument("--country", default="sg")
    fetch_parser.add_argument("--location", default="")
    fetch_parser.add_argument("--max-results", type=int, default=5)

    package_parser = subparsers.add_parser(
        "cover-letter",
        aliases=["package"],
        help="Generate a resume-grounded cover-letter bundle.",
    )
    package_parser.add_argument("job_description")
    package_parser.add_argument("--company", default="")
    package_parser.add_argument("--role", default="")
    package_parser.add_argument("--location", default="")
    package_parser.add_argument("--job-url", default="")

    subparsers.add_parser("list", help="List tracked applications.")

    show_parser = subparsers.add_parser("show", help="Show one tracked application.")
    show_parser.add_argument("--id", type=int, required=True)

    applied_parser = subparsers.add_parser("applied", help="Mark an application as applied.")
    applied_parser.add_argument("--id", type=int, required=True)

    delete_parser = subparsers.add_parser("delete", help="Delete one tracker record.")
    delete_parser.add_argument("--id", type=int, required=True)

    return parser


def main() -> None:
    """Run the selected command."""
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "fetch":
        run_fetch(args)
    elif args.command in {"cover-letter", "package"}:
        run_package(args)
    elif args.command == "list":
        list_applications()
    elif args.command == "show":
        show_application(args.id)
    elif args.command == "applied":
        update_status(args.id, "applied")
    elif args.command == "delete":
        delete_application(args.id)
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
