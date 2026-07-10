"""Scan public-release candidate files for privacy risks."""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


LOCAL_TERMS_FILE = Path("privacy_terms.local.txt")

SKIP_SUFFIXES = {
    ".docx",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".pdf",
    ".pyc",
}

SKIP_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".cache",
}

LOCAL_PATH_ROOTS = ("/" + "Users/", "/" + "home/")

GENERIC_PATTERNS = {
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "phone_like": re.compile(r"(?<![\w.])(?:\+?\d[\d .()/-]{8,}\d)(?![\w.])"),
    "linkedin": re.compile(r"\blinkedin\.com/in/[A-Za-z0-9_.%-]+/?", re.I),
    "absolute_path": re.compile(
        r"(?<!\w)(?:" + "|".join(re.escape(root) for root in LOCAL_PATH_ROOTS) + r")[^\s'\"`<>)]*"
    ),
}

ALLOWED_EMAILS = {
    "candidate@example.com",
    "you@example.com",
}

ALLOWED_LINKEDIN_PATHS = {
    "linkedin.com/in/example-profile",
    "linkedin.com/in/your-profile",
}


@dataclass(frozen=True)
class Finding:
    """A redacted privacy finding."""

    category: str
    path: Path
    line_number: int | None
    message: str

    def format(self) -> str:
        location = str(self.path)
        if self.line_number is not None:
            location = f"{location}:{self.line_number}"
        return f"[{self.category}] {location} - {self.message}"


class AuditError(RuntimeError):
    """Raised when the privacy audit cannot run correctly."""


def public_candidate_files() -> list[Path]:
    """Return tracked, staged, and non-ignored untracked files from Git."""
    try:
        result = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        stderr = ""
        if isinstance(exc, subprocess.CalledProcessError):
            stderr = exc.stderr.decode("utf-8", errors="replace").strip()
        detail = f": {stderr}" if stderr else ""
        raise AuditError(f"could not list Git public-release candidate files{detail}") from exc

    paths: list[Path] = []
    seen: set[Path] = set()
    for raw_item in result.stdout.split(b"\0"):
        if not raw_item:
            continue
        path = Path(raw_item.decode("utf-8", errors="replace"))
        if path in seen:
            continue
        seen.add(path)
        paths.append(path)
    return paths


def should_skip(path: Path) -> bool:
    """Return True for file types and directories the audit should not read."""
    if any(part in SKIP_DIRS for part in path.parts):
        return True
    return path.suffix.lower() in SKIP_SUFFIXES


def is_probably_text(path: Path) -> bool:
    """Return True when a candidate file can be safely decoded as text."""
    if should_skip(path):
        return False
    try:
        path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return False
    return True


def load_local_terms(path: Path = LOCAL_TERMS_FILE) -> list[str]:
    """Load optional local literal terms without exposing them in output."""
    if not path.exists():
        return []

    terms: list[str] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            term = line.strip()
            if not term or term.startswith("#"):
                continue
            key = term.casefold()
            if key in seen:
                continue
            seen.add(key)
            terms.append(term)
    return terms


def is_allowed_placeholder(category: str, value: str) -> bool:
    """Return True for documented fictional placeholder values."""
    normalized = value.strip().rstrip("/")
    lowered = normalized.casefold()
    if category == "email":
        return lowered in ALLOWED_EMAILS
    if category == "linkedin":
        normalized_path = lowered.removeprefix("https://www.")
        normalized_path = normalized_path.removeprefix("http://www.")
        normalized_path = normalized_path.removeprefix("https://")
        normalized_path = normalized_path.removeprefix("http://")
        return normalized_path in ALLOWED_LINKEDIN_PATHS
    return False


def is_likely_phone(value: str) -> bool:
    """Reduce false positives from versions and plain long numbers."""
    digits = re.sub(r"\D", "", value)
    if len(digits) < 10 or len(digits) > 15:
        return False
    return bool(re.search(r"[ +()./-]", value.strip()))


def scan_file(path: Path, local_terms: list[str]) -> list[Finding]:
    """Return redacted privacy findings for one text file."""
    findings: list[Finding] = []
    text = path.read_text(encoding="utf-8")
    folded_terms = [(term, term.casefold()) for term in local_terms]

    for line_number, line in enumerate(text.splitlines(), start=1):
        folded_line = line.casefold()
        for _term, folded_term in folded_terms:
            if folded_term in folded_line:
                findings.append(
                    Finding(
                        "personal_term",
                        path,
                        line_number,
                        "possible local personal identifier",
                    )
                )

        for category, pattern in GENERIC_PATTERNS.items():
            for match in pattern.finditer(line):
                value = match.group(0)
                if is_allowed_placeholder(category, value):
                    continue
                if category == "phone_like" and not is_likely_phone(value):
                    continue
                findings.append(
                    Finding(
                        category,
                        path,
                        line_number,
                        f"possible {category.replace('_', ' ')}",
                    )
                )
    return findings


def main() -> int:
    """Run the audit and return a shell-friendly exit code."""
    try:
        candidates = public_candidate_files()
        local_terms = load_local_terms()
    except AuditError as exc:
        print(f"Privacy audit could not run: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"Privacy audit could not run: {exc}", file=sys.stderr)
        return 2

    scanned_count = 0
    findings: list[Finding] = []
    for path in candidates:
        if is_probably_text(path):
            scanned_count += 1
            findings.extend(scan_file(path, local_terms))

    if scanned_count == 0:
        print("Privacy audit could not run: scanned 0 public-release candidate files.", file=sys.stderr)
        return 2

    if findings:
        print(f"Privacy audit findings: scanned {scanned_count} public-release candidate files.")
        for finding in findings:
            print(finding.format())
        return 1

    print(f"Privacy audit passed: scanned {scanned_count} public-release candidate files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
