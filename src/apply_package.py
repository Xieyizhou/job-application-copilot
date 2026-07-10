"""Generate a complete local application package for one job description.

This script runs the local workflow only:
1. Analyze the job.
2. Generate a tailored resume.
3. Generate a cover letter.
4. Add a local SQLite tracker record.

It does not submit applications or interact with job platforms.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from analyze_job import UK_ALREADY_AUTHORIZED_WARNING, UK_HPI_MANUAL_REVIEW_WARNING, UK_HPI_NOTE, analyze_job
from company_verification import assert_cover_letter_company_verified, parse_bool
from export_documents import export_application_package
from generate_cover_letter import generate_cover_letter
from generate_tailored_resume import generate_tailored_resume
from output_paths import application_package_dir, safe_slug, timestamp_slug
from tracker import add_application
from workspace import Workspace, WorkspaceError, personal_workspace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_METADATA_FIELDS = ["company", "role", "location", "job_url"]
MARKDOWN_FIELD_LABELS = {
    "company": "Company",
    "role": "Role",
    "location": "Location",
    "job_url": "Job URL",
}
TRACKING_QUERY_PARAMETERS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "app_id",
    "app_key",
    "aztt",
}


def parse_analysis_summary(report_text: str) -> tuple[int, str]:
    """Extract match score and recommendation from the analysis report."""
    score_match = re.search(r"Match score:\s+\*\*(\d+)/100\*\*", report_text)
    recommendation_match = re.search(r"Recommendation:\s+\*\*(.+?)\*\*", report_text)

    if not score_match:
        raise ValueError("Could not find match score in the analysis report.")
    if not recommendation_match:
        raise ValueError("Could not find recommendation in the analysis report.")

    score = int(score_match.group(1))
    recommendation = recommendation_match.group(1)
    return score, recommendation


def build_tracker_notes(score: int, recommendation: str) -> str:
    """Create short tracker notes without inventing extra facts."""
    return (
        f"Generated local application package. Match score: {score}/100. "
        f"Recommendation: {recommendation}. Review all files before applying."
    )


def collect_uk_review_notes(analysis_report: str) -> list[str]:
    """Return UK work authorization notes that should be visible in summaries."""
    notes = []
    for note in [UK_HPI_NOTE, UK_HPI_MANUAL_REVIEW_WARNING, UK_ALREADY_AUTHORIZED_WARNING]:
        if note in analysis_report:
            notes.append(note)
    return notes


def relative_path(path: Path) -> str:
    """Return a project-relative path when possible for cleaner tracker records."""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def parse_job_metadata(job_description_path: Path) -> dict[str, str]:
    """Extract common metadata fields from a Markdown job description.

    Supported lines look like:
    Company: Example Company
    Role: Machine Learning Intern
    Location: Remote
    Job URL: https://example.com/job
    """
    job_text = job_description_path.read_text(encoding="utf-8")
    metadata = {field: "" for field in REQUIRED_METADATA_FIELDS}

    for line in job_text.splitlines():
        for field, label in MARKDOWN_FIELD_LABELS.items():
            prefix = f"{label}:"
            if line.lower().startswith(prefix.lower()):
                value = line.split(":", 1)[1].strip()
                if value and value.lower() != "not provided":
                    if field == "job_url":
                        value = sanitize_job_url(value)
                    metadata[field] = value

    return metadata


def sanitize_job_url(job_url: str) -> str:
    """Remove tracking parameters and canonicalize common Adzuna redirect URLs."""
    if not job_url:
        return ""

    split_url = urlsplit(job_url)

    # Older fetched files may contain Adzuna redirect links. Convert them to the
    # cleaner details URL before storing tracker records.
    land_ad_match = re.match(r"^/land/ad/(\d+)", split_url.path)
    if "adzuna." in split_url.netloc.lower() and land_ad_match:
        return urlunsplit(
            (
                split_url.scheme,
                split_url.netloc,
                f"/details/{land_ad_match.group(1)}",
                "",
                "",
            )
        )

    safe_query_pairs = [
        (key, value)
        for key, value in parse_qsl(split_url.query, keep_blank_values=True)
        if key.lower() not in TRACKING_QUERY_PARAMETERS
    ]
    safe_query = urlencode(safe_query_pairs)
    return urlunsplit(
        (
            split_url.scheme,
            split_url.netloc,
            split_url.path,
            safe_query,
            split_url.fragment,
        )
    )


def resolve_metadata(args: argparse.Namespace, job_description_path: Path) -> dict[str, str]:
    """Combine CLI metadata with Markdown metadata.

    Command-line values always win. Missing values are filled from the Markdown
    file. If anything is still missing, the user gets a clear error.
    """
    parsed_metadata = parse_job_metadata(job_description_path)
    resolved_metadata = {}

    for field in REQUIRED_METADATA_FIELDS:
        cli_value = getattr(args, field, "") or ""
        value = cli_value.strip() or parsed_metadata[field]
        if field == "job_url":
            value = sanitize_job_url(value)
        resolved_metadata[field] = value

    missing_fields = [
        field
        for field, value in resolved_metadata.items()
        if not value
    ]
    if missing_fields:
        missing_args = ", ".join(f"--{field.replace('_', '-')}" for field in missing_fields)
        raise ValueError(
            "Could not find required metadata in the Markdown file. "
            f"Please provide: {missing_args}"
        )

    return resolved_metadata


def create_application_package(
    job_description_path: Path,
    workspace: Workspace,
    company: str,
    role: str,
    location: str,
    job_url: str,
) -> dict[str, object]:
    """Run the full local package workflow and return summary details."""
    workspace.require_writable()
    assert workspace.tracker_database_path is not None
    job_text = job_description_path.read_text(encoding="utf-8")
    assert_cover_letter_company_verified(
        company,
        {
            "job_text": job_text,
            "role": role,
            "location": location,
            "job_url": job_url,
            "company_confirmed_by_user": parse_bool(parse_markdown_value(job_text, "Company Confirmed By User")),
            "company_source_confidence": parse_markdown_value(job_text, "Company Confidence"),
            "company_source_evidence": parse_markdown_value(job_text, "Company Evidence"),
            "metadata": {
                "structured_company": company if parse_markdown_value(job_text, "Source").lower() in {"adzuna", "jooble"} else "",
                "job_url": job_url,
            },
        },
    )
    family = safe_slug(f"{company}_{role}")
    package_dir = application_package_dir(workspace.generated_dir, family, timestamp_slug())

    analysis_report, analysis_path = analyze_job(job_description_path, workspace, package_dir)
    match_score, recommendation = parse_analysis_summary(analysis_report)
    uk_review_notes = collect_uk_review_notes(analysis_report)

    _, resume_path, _, tailoring_notes_path = generate_tailored_resume(job_description_path, workspace, package_dir)
    _, cover_letter_path, _, cover_letter_notes_path = generate_cover_letter(job_description_path, workspace, package_dir)
    resume_docx_path, cover_letter_docx_path, export_warnings = export_application_package(package_dir, workspace)

    tracker_args = SimpleNamespace(
        company=company,
        role=role,
        location=location,
        job_url=job_url,
        match_score=match_score,
        recommendation=recommendation,
        status="ready",
        resume_file=relative_path(resume_path),
        cover_letter_file=relative_path(cover_letter_docx_path),
        notes=build_tracker_notes(match_score, recommendation),
    )
    tracker_id = add_application(tracker_args, workspace.tracker_database_path)

    return {
        "company": company,
        "role": role,
        "location": location,
        "job_url": job_url,
        "match_score": match_score,
        "recommendation": recommendation,
        "package_dir": package_dir,
        "analysis_path": analysis_path,
        "resume_path": resume_path,
        "resume_docx_path": resume_docx_path,
        "cover_letter_path": cover_letter_path,
        "cover_letter_docx_path": cover_letter_docx_path,
        "tailoring_notes_path": tailoring_notes_path,
        "cover_letter_notes_path": cover_letter_notes_path,
        "export_warnings": export_warnings,
        "tracker_id": tracker_id,
        "uk_review_notes": uk_review_notes,
    }


def parse_markdown_value(job_text: str, field_name: str) -> str:
    """Read optional metadata without making old Markdown files invalid."""
    prefix = f"{field_name}:"
    for line in job_text.splitlines():
        if line.lower().startswith(prefix.lower()):
            value = line.split(":", 1)[1].strip()
            return "" if value.lower() == "not provided" else value
    return ""


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate a local application package for one job description."
    )
    parser.add_argument("job_description", help="Path to a Markdown or text job description.")
    parser.add_argument("--company", default="")
    parser.add_argument("--role", default="")
    parser.add_argument("--location", default="")
    parser.add_argument("--job-url", default="")
    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""
    args = parse_args()
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

    print("\nApplication Package Summary")
    print("---------------------------")
    print(f"Parsed company: {summary['company']}")
    print(f"Parsed role: {summary['role']}")
    print(f"Parsed location: {summary['location']}")
    print(f"Parsed job URL: {summary['job_url']}")
    print(f"Match score: {summary['match_score']}/100")
    print(f"Recommendation: {summary['recommendation']}")
    print(f"Generated package folder: {summary['package_dir']}")
    print(f"Analysis report file: {summary['analysis_path']}")
    print(f"Resume DOCX file: {summary['resume_docx_path']}")
    print(f"Cover letter Markdown file: {summary['cover_letter_path']}")
    print(f"Cover letter DOCX file: {summary['cover_letter_docx_path']}")
    print(f"Tracker id: {summary['tracker_id']}")
    if summary["uk_review_notes"]:
        print("UK work authorization review:")
        for note in summary["uk_review_notes"]:
            print(f"- {note}")
    print("Status: ready")
    if summary["export_warnings"]:
        print("Validation warnings:")
        for warning in summary["export_warnings"]:
            print(f"- {warning}")
    else:
        print("Validation warnings: none")
    print("Reminder: no application was submitted automatically.")


if __name__ == "__main__":
    main()
