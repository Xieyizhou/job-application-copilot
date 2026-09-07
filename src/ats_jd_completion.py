"""Batch-complete saved jobs through stable public ATS posting APIs."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable
from uuid import uuid4

from company_ats import learn_ats_board, load_ats_boards, search_configured_ats_boards
from document_text import read_markdown_field
from jd_enrichment import (
    AMBIGUITY_MARGIN,
    _rank_safe_candidates,
    enrich_saved_job_description_from_url,
    replace_saved_job_description_from_candidate,
)
from job_page_fetch import public_ats_provider
from ml.jd_quality import classify_jd_quality


EnrichFunction = Callable[[Path], dict[str, Any]]
MAX_BATCH_SIZE = 10


def _path_record(path: Path) -> tuple[str, str, bool]:
    text = path.read_text(encoding="utf-8")
    url = read_markdown_field(text, "Job URL")
    provider = public_ats_provider(url) or ""
    ready = bool(classify_jd_quality(text)["reliable_scoring_ready"])
    return url, provider, ready


def public_ats_completion_candidates(
    paths: Iterable[Path], *, include_resolvable: bool = False
) -> list[Path]:
    """Return incomplete direct-ATS jobs, optionally including resolvable previews."""
    candidates: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        try:
            _url, provider, ready = _path_record(path)
        except (OSError, UnicodeError):
            continue
        if not ready and (
            provider
            or (
                include_resolvable
                and bool(read_markdown_field(path.read_text(encoding="utf-8"), "Company"))
                and bool(read_markdown_field(path.read_text(encoding="utf-8"), "Role"))
            )
        ):
            candidates.append(path)
    return candidates


def _workspace_root(paths: list[Path]) -> Path | None:
    for path in paths:
        jobs_root = next((parent for parent in path.parents if parent.name == "jobs"), None)
        if jobs_root is not None:
            return jobs_root.parent
    return None


def _attempts_path(paths: list[Path]) -> Path | None:
    root = _workspace_root(paths)
    return root / "jd_recovery_attempts.json" if root else None


def _load_attempts(paths: list[Path]) -> dict[str, dict[str, Any]]:
    path = _attempts_path(paths)
    if path is None:
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {
        str(key): value
        for key, value in payload.items()
        if isinstance(value, dict)
    } if isinstance(payload, dict) else {}


def _write_attempts(paths: list[Path], attempts: dict[str, dict[str, Any]]) -> None:
    path = _attempts_path(paths)
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(json.dumps(attempts, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def ats_completion_progress(paths: Iterable[Path]) -> dict[str, int]:
    """Return persistent pending/failed counts for the batch recovery controls."""
    all_paths = [Path(path) for path in paths]
    incomplete = public_ats_completion_candidates(all_paths, include_resolvable=True)
    attempts = _load_attempts(all_paths)
    pending = [path for path in incomplete if str(path) not in attempts]
    failed = [
        path
        for path in incomplete
        if str(path) in attempts and attempts[str(path)].get("status") != "updated"
    ]
    return {
        "incomplete": len(incomplete),
        "pending": len(pending),
        "failed": len(failed),
        "direct_ats": sum(1 for path in incomplete if public_ats_provider(_path_record(path)[0])),
    }


def _target(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    return {
        "company": read_markdown_field(text, "Company Normalized")
        or read_markdown_field(text, "Company"),
        "role": read_markdown_field(text, "Role"),
        "location": read_markdown_field(text, "Location"),
        "job_url": read_markdown_field(text, "Job URL"),
        "description": text.split("## Job Description", 1)[-1].strip(),
    }


def _resolve_from_catalog(path: Path, catalog: list[dict[str, Any]]) -> dict[str, Any]:
    ranked = _rank_safe_candidates(_target(path), catalog)
    if not ranked:
        return {"updated": False, "status": "no_ats_board_match", "message": "No configured employer board contained a safe exact match."}
    if len(ranked) > 1 and ranked[0][0] - ranked[1][0] < AMBIGUITY_MARGIN:
        first_id = str(ranked[0][1].get("source_job_id", ""))
        second_id = str(ranked[1][1].get("source_job_id", ""))
        if not first_id or not second_id or first_id != second_id:
            return {"updated": False, "status": "ambiguous_match", "message": "Multiple similar employer postings were found, so the saved JD was not changed."}
    score, candidate, _scores, _quality = ranked[0]
    return replace_saved_job_description_from_candidate(
        path,
        candidate,
        enriched_by="Configured employer ATS board",
        match_score=score,
    )


def complete_public_ats_jobs(
    paths: Iterable[Path],
    *,
    enrich: EnrichFunction = enrich_saved_job_description_from_url,
    resolve: EnrichFunction | None = None,
    max_jobs: int = MAX_BATCH_SIZE,
    retry_failed: bool = False,
) -> dict[str, Any]:
    """Resolve aggregator records, then complete verified saved JDs."""
    summary: dict[str, Any] = {
        "eligible": 0,
        "updated": 0,
        "already_ready": 0,
        "unsupported": 0,
        "failed": 0,
        "providers": {},
        "updated_jobs": [],
        "resolved_links": 0,
        "failures": [],
        "attempted": 0,
        "deferred": 0,
        "batch_limit": max_jobs,
    }
    all_paths = [Path(path) for path in paths]
    incomplete = public_ats_completion_candidates(all_paths, include_resolvable=True)
    attempts = _load_attempts(all_paths)
    pool = (
        [
            path
            for path in incomplete
            if str(path) in attempts and attempts[str(path)].get("status") != "updated"
        ]
        if retry_failed
        else [path for path in incomplete if str(path) not in attempts]
    )
    ordered = sorted(
        pool,
        key=lambda path: (
            0 if retry_failed or str(path) not in attempts else 1,
            0 if public_ats_provider(_path_record(path)[0]) else 1,
            str(path),
        ),
    )
    selected = ordered[: max(1, min(int(max_jobs), MAX_BATCH_SIZE))]
    summary["deferred"] = max(0, len(ordered) - len(selected))
    workspace_root = _workspace_root(all_paths)
    catalog: list[dict[str, Any]] = []
    if workspace_root and any(board.enabled for board in load_ats_boards(workspace_root)):
        catalog, catalog_errors = search_configured_ats_boards(
            workspace_root, query="", location="", max_results=1000
        )
        summary["board_errors"] = catalog_errors

    for path in selected:
        try:
            _url, provider, ready = _path_record(path)
        except (OSError, UnicodeError) as error:
            summary["failed"] += 1
            summary["failures"].append({"path": str(path), "message": str(error)})
            continue
        if ready:
            summary["already_ready"] += 1
            continue
        if not provider and resolve is not None:
            resolved = _resolve_from_catalog(path, catalog) if catalog else {"updated": False}
            if not resolved.get("updated"):
                resolved = resolve(path)
            if resolved.get("updated"):
                summary["updated"] += 1
                summary["resolved_links"] += 1
                summary["updated_jobs"].append(str(path))
                source_url = str(resolved.get("source_url", "") or "")
                if workspace_root and source_url:
                    learn_ats_board(
                        workspace_root,
                        source_url,
                        company=_target(path)["company"],
                    )
                summary["attempted"] += 1
                attempts[str(path)] = {
                    "status": "updated",
                    "attempted_at": datetime.now().replace(microsecond=0).isoformat(sep=" "),
                }
                continue
            summary["unsupported"] += 1
            summary["failed"] += 1
            summary["failures"].append(
                {
                    "path": str(path),
                    "status": str(resolved.get("status", "not_resolved")),
                    "message": str(resolved.get("message", "No verified employer posting was found.")),
                }
            )
            summary["attempted"] += 1
            attempts[str(path)] = {
                "status": str(resolved.get("status", "not_resolved")),
                "message": str(resolved.get("message", "")),
                "attempted_at": datetime.now().replace(microsecond=0).isoformat(sep=" "),
            }
            continue
        if not provider:
            summary["unsupported"] += 1
            continue
        summary["eligible"] += 1
        provider_summary = summary["providers"].setdefault(
            provider, {"eligible": 0, "updated": 0, "failed": 0}
        )
        provider_summary["eligible"] += 1
        result = enrich(path)
        if result.get("updated"):
            summary["updated"] += 1
            provider_summary["updated"] += 1
            summary["updated_jobs"].append(str(path))
            if workspace_root:
                learn_ats_board(
                    workspace_root,
                    _path_record(path)[0],
                    company=_target(path)["company"],
                )
        else:
            summary["failed"] += 1
            provider_summary["failed"] += 1
            summary["failures"].append(
                {
                    "path": str(path),
                    "provider": provider,
                    "status": str(result.get("status", "not_updated")),
                    "message": str(result.get("message", "The JD was not updated.")),
                }
            )
        summary["attempted"] += 1
        attempts[str(path)] = {
            "status": str(result.get("status", "updated" if result.get("updated") else "not_updated")),
            "message": str(result.get("message", "")),
            "attempted_at": datetime.now().replace(microsecond=0).isoformat(sep=" "),
        }
    _write_attempts(all_paths, attempts)
    return summary
