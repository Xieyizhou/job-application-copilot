"""Local SQLite application tracker.

This script is only for local organization. It does not connect to external
websites, submit applications, or store sensitive eligibility details unless
the user explicitly types them into notes.
"""

from __future__ import annotations

import argparse
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from workspace import WorkspaceError, personal_workspace

VALID_STATUSES = {
    "saved",
    "ready",
    "applied",
    "interview",
    "rejected",
    "archived",
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


def connect_to_database(database_path: Path) -> sqlite3.Connection:
    """Open a SQLite connection and return rows that behave like dictionaries."""
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database(database_path: Path) -> None:
    """Create the applications table if it does not already exist."""
    with connect_to_database(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS applications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                company TEXT NOT NULL,
                role TEXT NOT NULL,
                location TEXT DEFAULT '',
                job_url TEXT DEFAULT '',
                match_score INTEGER,
                recommendation TEXT DEFAULT '',
                status TEXT NOT NULL,
                resume_file TEXT DEFAULT '',
                cover_letter_file TEXT DEFAULT '',
                notes TEXT DEFAULT '',
                created_at TEXT NOT NULL,
                applied_date TEXT
            )
            """
        )
        connection.commit()

    print(f"Tracker database ready: {database_path}")


def validate_status(status: str) -> None:
    """Raise a helpful error if the status is not supported."""
    if status not in VALID_STATUSES:
        valid_statuses = ", ".join(sorted(VALID_STATUSES))
        raise ValueError(f"Unsupported status '{status}'. Use one of: {valid_statuses}")


def add_application(args: argparse.Namespace, database_path: Path) -> int:
    """Add one application record from command-line arguments."""
    validate_status(args.status)
    initialize_database(database_path)
    clean_job_url = sanitize_job_url(args.job_url)

    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    applied_date = created_at if args.status == "applied" else None

    with connect_to_database(database_path) as connection:
        existing_row = find_existing_application_by_url(connection, clean_job_url)
        if existing_row is None and args.job_url and clean_job_url != args.job_url:
            existing_row = find_existing_application_by_url(connection, args.job_url)
        if existing_row is not None:
            existing_id = int(existing_row["id"])
            print(
                f"Application already exists as #{existing_id}: "
                f"{existing_row['company']} - {existing_row['role']}"
            )
            return existing_id

        cursor = connection.execute(
            """
            INSERT INTO applications (
                company,
                role,
                location,
                job_url,
                match_score,
                recommendation,
                status,
                resume_file,
                cover_letter_file,
                notes,
                created_at,
                applied_date
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                args.company,
                args.role,
                args.location,
                clean_job_url,
                args.match_score,
                args.recommendation,
                args.status,
                args.resume_file,
                args.cover_letter_file,
                args.notes,
                created_at,
                applied_date,
            ),
        )
        connection.commit()

    application_id = int(cursor.lastrowid)
    print(f"Added application #{application_id}: {args.company} - {args.role}")
    return application_id


def find_existing_application_by_url(
    connection: sqlite3.Connection,
    job_url: str,
) -> sqlite3.Row | None:
    """Return an existing tracker row with the same job URL, if any."""
    if not job_url:
        return None

    return connection.execute(
        """
        SELECT id, company, role
        FROM applications
        WHERE job_url = ?
        LIMIT 1
        """,
        (job_url,),
    ).fetchone()


def sanitize_job_url(job_url: str) -> str:
    """Remove tracking parameters and canonicalize common Adzuna redirect URLs."""
    if not job_url:
        return ""

    split_url = urlsplit(job_url)
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


def list_applications(database_path: Path) -> None:
    """Print a compact table of all application records."""
    initialize_database(database_path)

    with connect_to_database(database_path) as connection:
        rows = connection.execute(
            """
            SELECT
                id,
                company,
                role,
                location,
                match_score,
                recommendation,
                status,
                applied_date
            FROM applications
            ORDER BY id
            """
        ).fetchall()

    if not rows:
        print("No applications found.")
        return

    print("ID | Status | Score | Recommendation | Company | Role | Location | Applied Date")
    print("-" * 92)
    for row in rows:
        print(
            f"{row['id']} | {row['status']} | {format_optional(row['match_score'])} | "
            f"{format_optional(row['recommendation'])} | {row['company']} | "
            f"{row['role']} | {format_optional(row['location'])} | "
            f"{format_optional(row['applied_date'])}"
        )


def update_status(application_id: int, status: str, database_path: Path) -> None:
    """Update the status for one application."""
    validate_status(status)
    initialize_database(database_path)

    applied_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S") if status == "applied" else None

    with connect_to_database(database_path) as connection:
        if status == "applied":
            cursor = connection.execute(
                """
                UPDATE applications
                SET status = ?, applied_date = COALESCE(applied_date, ?)
                WHERE id = ?
                """,
                (status, applied_date, application_id),
            )
        else:
            cursor = connection.execute(
                """
                UPDATE applications
                SET status = ?
                WHERE id = ?
                """,
                (status, application_id),
            )
        connection.commit()

    if cursor.rowcount == 0:
        print(f"No application found with id {application_id}.")
    else:
        print(f"Updated application #{application_id} status to {status}.")


def show_application(application_id: int, database_path: Path) -> None:
    """Print all details for one application."""
    initialize_database(database_path)

    with connect_to_database(database_path) as connection:
        row = connection.execute(
            """
            SELECT
                id,
                company,
                role,
                location,
                job_url,
                match_score,
                recommendation,
                status,
                resume_file,
                cover_letter_file,
                notes,
                created_at,
                applied_date
            FROM applications
            WHERE id = ?
            """,
            (application_id,),
        ).fetchone()

    if row is None:
        print(f"No application found with id {application_id}.")
        return

    print(f"Application #{row['id']}")
    print(f"Company: {row['company']}")
    print(f"Role: {row['role']}")
    print(f"Location: {format_optional(row['location'])}")
    print(f"Job URL: {format_optional(row['job_url'])}")
    print(f"Match score: {format_optional(row['match_score'])}")
    print(f"Recommendation: {format_optional(row['recommendation'])}")
    print(f"Status: {row['status']}")
    print(f"Resume file: {format_optional(row['resume_file'])}")
    print(f"Cover letter file: {format_optional(row['cover_letter_file'])}")
    print(f"Notes: {format_optional(row['notes'])}")
    print(f"Created at: {row['created_at']}")
    print(f"Applied date: {format_optional(row['applied_date'])}")


def delete_application(application_id: int, database_path: Path) -> None:
    """Delete one application record without touching generated files."""
    initialize_database(database_path)

    with connect_to_database(database_path) as connection:
        row = connection.execute(
            """
            SELECT id, company, role
            FROM applications
            WHERE id = ?
            """,
            (application_id,),
        ).fetchone()

        if row is None:
            print(f"No application found with id {application_id}. Nothing was deleted.")
            return

        print(
            f"Warning: deleting tracker record #{row['id']} for "
            f"{row['company']} - {row['role']}."
        )
        print("Generated cover-letter and report files will not be deleted.")

        connection.execute(
            """
            DELETE FROM applications
            WHERE id = ?
            """,
            (application_id,),
        )
        connection.commit()

    print(f"Deleted application #{application_id} from the tracker.")


def format_optional(value: object) -> str:
    """Display empty database values consistently."""
    if value is None or value == "":
        return "-"
    return str(value)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser and subcommands."""
    parser = argparse.ArgumentParser(description="Track job applications locally.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("init", help="Initialize the SQLite database.")

    add_parser = subparsers.add_parser("add", help="Add a new application.")
    add_parser.add_argument("--company", required=True)
    add_parser.add_argument("--role", required=True)
    add_parser.add_argument("--location", default="")
    add_parser.add_argument("--job-url", default="")
    add_parser.add_argument("--match-score", type=int)
    add_parser.add_argument("--recommendation", default="")
    add_parser.add_argument("--status", default="saved", choices=sorted(VALID_STATUSES))
    add_parser.add_argument("--resume-file", default="")
    add_parser.add_argument("--cover-letter-file", default="")
    add_parser.add_argument("--notes", default="")

    subparsers.add_parser("list", help="List all applications.")

    update_parser = subparsers.add_parser("update-status", help="Update application status.")
    update_parser.add_argument("--id", type=int, required=True)
    update_parser.add_argument("--status", required=True, choices=sorted(VALID_STATUSES))

    show_parser = subparsers.add_parser("show", help="Show one application by id.")
    show_parser.add_argument("--id", type=int, required=True)

    delete_parser = subparsers.add_parser("delete", help="Delete one application by id.")
    delete_parser.add_argument("--id", type=int, required=True)

    return parser


def main() -> None:
    """Run the selected tracker command."""
    parser = build_parser()
    args = parser.parse_args()
    workspace = personal_workspace()
    try:
        workspace.require_ready()
    except WorkspaceError as error:
        raise SystemExit(str(error)) from None
    assert workspace.tracker_database_path is not None
    database_path = workspace.tracker_database_path

    if args.command == "init":
        initialize_database(database_path)
    elif args.command == "add":
        add_application(args, database_path)
    elif args.command == "list":
        list_applications(database_path)
    elif args.command == "update-status":
        update_status(args.id, args.status, database_path)
    elif args.command == "show":
        show_application(args.id, database_path)
    elif args.command == "delete":
        delete_application(args.id, database_path)
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
