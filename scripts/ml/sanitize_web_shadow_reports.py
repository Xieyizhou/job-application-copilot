"""Remove raw requirement and resume evidence text from existing Shadow reports."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
DEFAULT_SHADOW_ROOT = PROJECT_ROOT / "data" / "ml" / "shadow" / "web_candidate"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml.web_candidate_shadow import privacy_safe_shadow_report  # noqa: E402


def rewrite_private(path: Path, payload: dict[str, object]) -> None:
    """Atomically replace one local report with a permission-600 safe payload."""
    temporary = path.with_suffix(path.suffix + ".sanitizing")
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
        os.replace(temporary, path)
        path.chmod(0o600)
    finally:
        temporary.unlink(missing_ok=True)


def sanitize_directory(shadow_root: Path) -> dict[str, int]:
    """Sanitize observation files without reading or rewriting the aggregate summary."""
    scanned = 0
    rewritten = 0
    for path in sorted(shadow_root.glob("*.json")):
        if path.name == "backfill_summary.json":
            continue
        scanned += 1
        report = json.loads(path.read_text(encoding="utf-8"))
        safe = privacy_safe_shadow_report(report)
        if safe != report:
            rewrite_private(path, safe)
            rewritten += 1
    return {"scanned": scanned, "rewritten": rewritten}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shadow-root", type=Path, default=DEFAULT_SHADOW_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.shadow_root.is_dir():
        raise SystemExit("Shadow root does not exist.")
    print(json.dumps(sanitize_directory(args.shadow_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
