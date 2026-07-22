"""Provider orchestration and live status for the Find Jobs page."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable

import streamlit as st

from dashboard_regions import source_display_name
from fetch_jobs import fetch_and_save_jobs


@dataclass
class FetchSearchOutcome:
    """Collected provider results without hiding partial failures."""

    saved_paths: list[Any] = field(default_factory=list)
    runs: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    backend_outputs: list[str] = field(default_factory=list)


def run_job_search(
    *,
    sources: list[str],
    query: str,
    country: str,
    adzuna_location: str,
    jooble_location: str,
    adzuna_supported: bool,
    limit_per_source: int,
    run_with_captured_output: Callable[..., tuple[Any, str]],
    relocate_fetched_jobs_to_workspace: Callable[..., list[Any]],
) -> FetchSearchOutcome:
    """Search selected sources while keeping progress and failures visible."""
    outcome = FetchSearchOutcome()
    with st.status("Searching configured job sources…", expanded=True) as search_status:
        for source in sources:
            display_name = source_display_name(source)
            st.write(f"Searching {display_name}…")
            if source == "adzuna" and not adzuna_supported:
                message = f"{display_name}: this region is not supported."
                outcome.errors.append(message)
                outcome.backend_outputs.append(f"[adzuna] {message}")
                continue

            location = adzuna_location if source in {"adzuna", "jsearch"} else jooble_location
            args = SimpleNamespace(
                source=source,
                country=country,
                query=query,
                location=location,
                max_results=limit_per_source,
            )
            try:
                result, output = run_with_captured_output(fetch_and_save_jobs, args)
                saved_paths = list(result.get("saved_paths", []) if isinstance(result, dict) else [])
                saved_paths = relocate_fetched_jobs_to_workspace(saved_paths, source)
                run_record = dict(result.get("fetch_run", {}) if isinstance(result, dict) else {})
                outcome.saved_paths.extend(saved_paths)
                if run_record:
                    outcome.runs.append(run_record)
                    st.session_state["latest_fetch_run_id"] = run_record.get("fetch_run_id", "")
                if output:
                    outcome.backend_outputs.append(f"[{source}]\n{output}")
            except Exception as error:  # noqa: BLE001
                outcome.errors.append(f"{display_name}: {error}")
                outcome.backend_outputs.append(f"[{source}] {error}")

        if outcome.runs and outcome.errors:
            search_status.update(label="Search completed with source issues.", state="complete")
        elif outcome.runs:
            search_status.update(label="Search complete.", state="complete", expanded=False)
        else:
            search_status.update(label="Search could not be completed.", state="error")
    return outcome
