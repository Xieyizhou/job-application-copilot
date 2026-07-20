"""Location and source filtering helpers for the dashboard."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from output_paths import safe_slug


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RECENT_REGIONS_PATH = PROJECT_ROOT / "data" / "ui_state" / "recent_regions.json"
COMMON_HIGH_LEVEL_REGIONS = ["United States", "Canada", "Australia", "Remote", "Other"]


def _normalize_match_text(text: str) -> str:
    return " " + " ".join(text.lower().replace("-", " ").split()) + " "


def path_contains_slug(path: Path, slug: str) -> bool:
    """Return True when a normalized path segment contains a location slug."""
    normalized_parts = [part.lower().replace("-", "_") for part in path.parts]
    return slug in normalized_parts


def infer_location_from_path(path: Path) -> str:
    """Infer a readable location for older files without Location metadata."""
    if path_contains_slug(path, "london"):
        return "London"
    if path_contains_slug(path, "singapore") or path_contains_slug(path, "sg"):
        return "Singapore"
    if path_contains_slug(path, "united_kingdom") or path_contains_slug(path, "uk") or path_contains_slug(path, "gb"):
        return "United Kingdom"
    if path_contains_slug(path, "remote"):
        return "Remote"
    if path_contains_slug(path, "tokyo"):
        return "Tokyo"
    if path_contains_slug(path, "shenzhen"):
        return "Shenzhen"
    if path_contains_slug(path, "hangzhou"):
        return "Hangzhou"
    return "Unknown"


def normalize_location(location: str) -> str:
    """Clean display and filter locations from metadata, paths, and OCR text."""
    value = " ".join(str(location or "").replace("\n", " ").split())
    value = re.sub(r"\.{2,}$", "", value).strip()
    value = value.strip(" ,;:|·•-–—")
    aliases = {
        "uk": "United Kingdom",
        "u.k.": "United Kingdom",
        "gb": "United Kingdom",
        "great britain": "United Kingdom",
        "usa": "United States",
        "u.s.": "United States",
        "us": "United States",
        "united states of america": "United States",
    }
    return aliases.get(value.lower(), value)


def infer_high_level_region(location: str) -> str:
    """Map a normalized location to a broad region used by dashboard filters."""
    normalized = _normalize_match_text(normalize_location(location))
    if " remote " in normalized:
        return "Remote"
    if any(marker in normalized for marker in [" china ", " beijing ", " shanghai ", " shenzhen ", " hangzhou "]):
        return "China"
    if " singapore " in normalized:
        return "Singapore"
    if any(marker in normalized for marker in [" united kingdom ", " london ", " england ", " scotland ", " wales "]):
        return "United Kingdom"
    if any(
        marker in normalized
        for marker in [
            " united states ",
            " usa ",
            " california ",
            " ca ",
            " new york ",
            " washington ",
            " boston ",
            " seattle ",
        ]
    ):
        return "United States"
    return "Other"


def region_option_key(option_type: str, value: str) -> str:
    """Build a stable key for a region option."""
    if option_type == "all":
        return "all"
    return f"{option_type}:{safe_slug(value)}"


def region_label(option: dict[str, Any], include_count: bool = True) -> str:
    """Build display text while keeping option keys stable internally."""
    if option["type"] == "all":
        base = "all"
    elif option["type"] == "high_level":
        base = f"High-level: {option['value']}"
    else:
        base = f"Exact: {option['value']}"
    return f"{base} ({option['count']})" if include_count else base


def region_search_blob(option: dict[str, Any]) -> str:
    """Return searchable text including aliases such as UK."""
    aliases = []
    value = str(option.get("value", ""))
    high_level_value = value if option.get("type") == "high_level" else infer_high_level_region(value)
    if high_level_value == "United Kingdom":
        aliases.extend(["uk", "u.k.", "gb", "great britain", "london"])
    if high_level_value == "United States":
        aliases.extend(["us", "u.s.", "usa", "america"])
    if high_level_value == "China":
        aliases.extend(["cn", "beijing", "shanghai", "shenzhen", "hangzhou"])
    return " ".join([region_label(option, include_count=False), value, *aliases]).lower()


def load_recent_region_keys() -> list[str]:
    """Read recent region keys; missing or malformed state is treated as empty."""
    try:
        data = json.loads(RECENT_REGIONS_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return []
    return [str(item) for item in data if isinstance(item, str)][:10] if isinstance(data, list) else []


def build_region_options(jobs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Build all searchable region options from currently loaded jobs."""
    options: dict[str, dict[str, Any]] = {
        "all": {"key": "all", "label": "all", "type": "all", "value": "all", "count": len(jobs)}
    }
    for job in jobs:
        normalized_location = normalize_location(str(job.get("location", "")))
        if not normalized_location or normalized_location.lower() in {"unknown", "not provided"}:
            normalized_location = "Other"
        high_level = infer_high_level_region(normalized_location)
        for option_type, value in [("high_level", high_level), ("exact", normalized_location)]:
            key = region_option_key(option_type, value)
            if key not in options:
                options[key] = {"key": key, "label": "", "type": option_type, "value": value, "count": 0}
            options[key]["count"] += 1
    for option in options.values():
        option["label"] = region_label(option)
    return options


def default_region_option_keys(options_by_key: dict[str, dict[str, Any]]) -> list[str]:
    """Return the short default region list before the user searches."""
    keys = ["all", *[key for key in load_recent_region_keys() if key in options_by_key][:5]]
    for region in COMMON_HIGH_LEVEL_REGIONS:
        key = region_option_key("high_level", region)
        if key in options_by_key:
            keys.append(key)
    exact_options = [
        option
        for option in options_by_key.values()
        if option["type"] == "exact" and option["value"] not in {"Other", "Unknown", "Not provided"}
    ]
    exact_options.sort(key=lambda option: (-int(option["count"]), str(option["value"]).lower()))
    keys.extend(option["key"] for option in exact_options[:5])
    return list(dict.fromkeys(key for key in keys if key in options_by_key))


def filtered_region_option_keys(options_by_key: dict[str, dict[str, Any]], query: str) -> list[str]:
    """Search all region options by label, value, and common aliases."""
    cleaned_query = " ".join(query.lower().split())
    if not cleaned_query:
        return default_region_option_keys(options_by_key)
    matches = [
        option
        for option in options_by_key.values()
        if option["key"] == "all" or cleaned_query in region_search_blob(option)
    ]
    matches.sort(key=lambda option: (option["key"] != "all", option["type"] != "high_level", -int(option["count"])))
    return [option["key"] for option in matches] or ["all"]


def job_matches_region_option(job: dict[str, Any], selected_option: dict[str, Any]) -> bool:
    """Filter jobs by exact normalized location or inferred high-level region."""
    if selected_option["type"] == "all":
        return True
    if selected_option["type"] == "exact":
        return normalize_location(str(job.get("location", ""))).lower() == str(selected_option["value"]).lower()
    return str(job.get("high_level_region", "")) == selected_option["value"]


def source_display_name(source: str) -> str:
    """Normalize source labels for dynamic source filtering."""
    cleaned = " ".join(str(source or "unknown").replace("_", " ").split()).strip()
    aliases = {
        "company website": "Company Website",
        "manual": "Manual",
        "linkedin": "LinkedIn",
        "indeed": "Indeed",
        "handshake": "Handshake",
        "jooble": "Jooble",
        "adzuna": "Adzuna",
        "jsearch": "JSearch · Full JD",
    }
    return aliases.get(cleaned.lower(), cleaned.title() if cleaned else "Unknown")


def dynamic_source_options(jobs: list[dict[str, Any]]) -> list[str]:
    """Build source filter options from the currently loaded jobs."""
    discovered = sorted({source_display_name(str(job.get("source", ""))) for job in jobs})
    preferred = ["LinkedIn", "Jooble", "Adzuna", "Company Website", "Indeed", "Handshake", "Manual"]
    ordered = ["all"]
    for source in preferred + discovered:
        if source in discovered and source not in ordered:
            ordered.append(source)
    return ordered
