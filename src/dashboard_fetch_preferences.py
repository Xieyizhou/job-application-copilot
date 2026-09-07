"""Private local preferences for the job-discovery form."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FETCH_PREFERENCES_PATH = PROJECT_ROOT / "data" / "ui_state" / "fetch_preferences.json"
FETCH_SOURCES = ("company_ats", "jsearch", "adzuna", "jooble")


def load_fetch_sources(default: list[str]) -> list[str]:
    """Load the last non-empty provider selection, or return a safe default."""
    try:
        payload = json.loads(FETCH_PREFERENCES_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return list(default)
    raw_sources = payload.get("sources", []) if isinstance(payload, dict) else []
    sources = [source for source in raw_sources if source in FETCH_SOURCES]
    return list(dict.fromkeys(sources)) or list(default)


def save_fetch_sources(sources: list[str]) -> None:
    """Atomically remember a validated provider selection on this device."""
    validated = list(dict.fromkeys(source for source in sources if source in FETCH_SOURCES))
    if not validated:
        return
    FETCH_PREFERENCES_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = FETCH_PREFERENCES_PATH.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps({"sources": validated}, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.chmod(0o600)
    temporary_path.replace(FETCH_PREFERENCES_PATH)
    FETCH_PREFERENCES_PATH.chmod(0o600)
