"""Local employer-board registry and public ATS job discovery."""

from __future__ import annotations

import html
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote, urlsplit
from uuid import uuid4

import requests


SUPPORTED_ATS = ("greenhouse", "lever", "ashby", "smartrecruiters")
REQUEST_HEADERS = {
    "Accept": "application/json",
    "User-Agent": "JobCopilot/1.0 (+local employer job-board reader)",
}
MAX_RESPONSE_BYTES = 3_000_000
PAGE_SIZE = 100
MAX_PAGES = 10


class ATSBoardError(RuntimeError):
    """Raised when a configured public job board cannot be read safely."""


@dataclass(frozen=True)
class ATSBoardConfig:
    """One validated employer-hosted public job board."""

    provider: str
    company: str
    board_url: str
    board_token: str
    enabled: bool = True
    api_region: str = "global"


def ats_boards_path(workspace_root: Path) -> Path:
    return Path(workspace_root) / "ats_boards.json"


def _plain(value: object) -> str:
    text = html.unescape(str(value or ""))
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_ats_board_url(url: str, *, company: str = "") -> ATSBoardConfig:
    """Recognize a supported board or posting URL and return its board identity."""
    raw = str(url or "").strip()
    parsed = urlsplit(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username:
        raise ATSBoardError("Enter a public HTTPS employer job-board URL.")
    host = parsed.hostname.lower()
    parts = [part for part in parsed.path.split("/") if part]
    provider = ""
    token = ""
    region = "global"
    board_url = ""
    if host in {"boards.greenhouse.io", "job-boards.greenhouse.io"} and parts:
        provider, token = "greenhouse", parts[0]
        board_url = f"https://{host}/{quote(token, safe='')}"
    elif host == "boards-api.greenhouse.io" and len(parts) >= 3 and parts[:2] == ["v1", "boards"]:
        provider, token = "greenhouse", parts[2]
        board_url = f"https://boards.greenhouse.io/{quote(token, safe='')}"
    elif host in {"jobs.lever.co", "jobs.eu.lever.co"} and parts:
        provider, token = "lever", parts[0]
        region = "eu" if host == "jobs.eu.lever.co" else "global"
        board_url = f"https://{host}/{quote(token, safe='')}"
    elif host in {"api.lever.co", "api.eu.lever.co"} and len(parts) >= 3 and parts[:2] == ["v0", "postings"]:
        provider, token = "lever", parts[2]
        region = "eu" if host == "api.eu.lever.co" else "global"
        job_host = "jobs.eu.lever.co" if region == "eu" else "jobs.lever.co"
        board_url = f"https://{job_host}/{quote(token, safe='')}"
    elif host == "jobs.ashbyhq.com" and parts:
        provider, token = "ashby", parts[0]
        board_url = f"https://jobs.ashbyhq.com/{quote(token, safe='')}"
    elif host == "api.ashbyhq.com" and len(parts) >= 3 and parts[:2] == ["posting-api", "job-board"]:
        provider, token = "ashby", parts[2]
        board_url = f"https://jobs.ashbyhq.com/{quote(token, safe='')}"
    elif host in {"jobs.smartrecruiters.com", "careers.smartrecruiters.com"} and parts:
        provider, token = "smartrecruiters", parts[0]
        board_url = f"https://jobs.smartrecruiters.com/{quote(token, safe='')}"
    elif host == "api.smartrecruiters.com" and len(parts) >= 4 and parts[:2] == ["v1", "companies"]:
        provider, token = "smartrecruiters", parts[2]
        board_url = f"https://jobs.smartrecruiters.com/{quote(token, safe='')}"
    if not provider or not token:
        raise ATSBoardError(
            "Supported boards are Greenhouse, Lever, Ashby, and SmartRecruiters."
        )
    return ATSBoardConfig(
        provider=provider,
        company=str(company or token).strip(),
        board_url=board_url,
        board_token=token,
        api_region=region,
    )


def load_ats_boards(workspace_root: Path) -> list[ATSBoardConfig]:
    """Load valid local board entries; malformed entries fail closed."""
    try:
        payload = json.loads(ats_boards_path(workspace_root).read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return []
    raw_boards = payload.get("boards", []) if isinstance(payload, dict) else []
    boards: list[ATSBoardConfig] = []
    for item in raw_boards if isinstance(raw_boards, list) else []:
        if not isinstance(item, dict):
            continue
        try:
            parsed = parse_ats_board_url(
                str(item.get("board_url", "")), company=str(item.get("company", ""))
            )
        except ATSBoardError:
            continue
        boards.append(
            ATSBoardConfig(
                **{
                    **asdict(parsed),
                    "enabled": bool(item.get("enabled", True)),
                }
            )
        )
    return boards


def save_ats_boards(workspace_root: Path, boards: Iterable[ATSBoardConfig]) -> None:
    """Atomically persist a de-duplicated local board registry."""
    unique: dict[tuple[str, str, str], ATSBoardConfig] = {}
    for board in boards:
        unique[(board.provider, board.api_region, board.board_token.lower())] = board
    path = ats_boards_path(workspace_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(
            {"schema_version": 1, "boards": [asdict(board) for board in unique.values()]},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def add_ats_board(workspace_root: Path, url: str, *, company: str = "") -> ATSBoardConfig:
    """Validate and add one board without creating duplicates."""
    board = parse_ats_board_url(url, company=company)
    # A bounded live read proves both the token and the public endpoint.
    list_public_ats_jobs(board, query="", location="", max_results=1)
    boards = load_ats_boards(workspace_root)
    identity = (board.provider, board.api_region, board.board_token.lower())
    if any((b.provider, b.api_region, b.board_token.lower()) == identity for b in boards):
        raise ATSBoardError("This employer job board is already configured.")
    save_ats_boards(workspace_root, [*boards, board])
    return board


def remove_ats_board(workspace_root: Path, board: ATSBoardConfig) -> None:
    identity = (board.provider, board.api_region, board.board_token.lower())
    save_ats_boards(
        workspace_root,
        [
            item
            for item in load_ats_boards(workspace_root)
            if (item.provider, item.api_region, item.board_token.lower()) != identity
        ],
    )


def set_ats_board_enabled(
    workspace_root: Path, board: ATSBoardConfig, *, enabled: bool
) -> None:
    identity = (board.provider, board.api_region, board.board_token.lower())
    updated = []
    for item in load_ats_boards(workspace_root):
        item_identity = (item.provider, item.api_region, item.board_token.lower())
        updated.append(ATSBoardConfig(**{**asdict(item), "enabled": enabled}) if item_identity == identity else item)
    save_ats_boards(workspace_root, updated)


def learn_ats_board(workspace_root: Path, url: str, *, company: str = "") -> bool:
    """Remember a verified ATS URL without another network call."""
    try:
        board = parse_ats_board_url(url, company=company)
    except ATSBoardError:
        return False
    boards = load_ats_boards(workspace_root)
    identity = (board.provider, board.api_region, board.board_token.lower())
    if any((b.provider, b.api_region, b.board_token.lower()) == identity for b in boards):
        return False
    save_ats_boards(workspace_root, [*boards, board])
    return True


def _get_json(url: str) -> Any:
    try:
        response = requests.get(url, headers=REQUEST_HEADERS, timeout=(5, 20))
        response.raise_for_status()
    except requests.RequestException as error:
        raise ATSBoardError(f"The public ATS endpoint could not be reached: {error}") from error
    if len(response.content) > MAX_RESPONSE_BYTES:
        raise ATSBoardError("The ATS response was too large to process safely.")
    try:
        return response.json()
    except ValueError as error:
        raise ATSBoardError("The ATS endpoint did not return valid JSON.") from error


def _job(
    board: ATSBoardConfig,
    *,
    posting_id: object,
    role: object,
    location: object,
    url: object,
    description: object,
    salary: object = "",
) -> dict[str, Any]:
    return {
        "source_job_id": f"{board.provider}:{board.board_token}:{_plain(posting_id)}",
        "company": board.company,
        "role": _plain(role),
        "location": _plain(location),
        "job_url": str(url or "").strip(),
        "discovery_url": str(url or "").strip(),
        "description": _plain(description),
        "requirements": "",
        "salary": _plain(salary),
        "source": "company_ats",
        "description_source": "company_ats_public_api",
        "jd_fetch_status": "complete",
        "ats_provider": board.provider,
        "ats_company_token": board.board_token,
    }


def _greenhouse_jobs(board: ATSBoardConfig) -> list[dict[str, Any]]:
    payload = _get_json(
        f"https://boards-api.greenhouse.io/v1/boards/{quote(board.board_token, safe='')}/jobs?content=true"
    )
    raw = payload.get("jobs", []) if isinstance(payload, dict) else []
    return [
        _job(
            board,
            posting_id=item.get("id"),
            role=item.get("title"),
            location=(item.get("location") or {}).get("name", "") if isinstance(item.get("location"), dict) else "",
            url=item.get("absolute_url"),
            description=item.get("content"),
        )
        for item in raw
        if isinstance(item, dict)
    ]


def _lever_jobs(board: ATSBoardConfig) -> list[dict[str, Any]]:
    api_host = "api.eu.lever.co" if board.api_region == "eu" else "api.lever.co"
    jobs: list[dict[str, Any]] = []
    for page in range(MAX_PAGES):
        payload = _get_json(
            f"https://{api_host}/v0/postings/{quote(board.board_token, safe='')}?mode=json&skip={page * PAGE_SIZE}&limit={PAGE_SIZE}"
        )
        raw = payload if isinstance(payload, list) else []
        for item in raw:
            if not isinstance(item, dict):
                continue
            categories = item.get("categories", {})
            lists = item.get("lists", [])
            extras = " ".join(
                _plain(part.get("content", ""))
                for part in lists if isinstance(part, dict)
            ) if isinstance(lists, list) else ""
            jobs.append(
                _job(
                    board,
                    posting_id=item.get("id"),
                    role=item.get("text"),
                    location=categories.get("location", "") if isinstance(categories, dict) else "",
                    url=item.get("hostedUrl"),
                    description=f"{item.get('descriptionPlain', '')} {extras}",
                    salary=item.get("salaryDescription", ""),
                )
            )
        if len(raw) < PAGE_SIZE:
            break
    return jobs


def _ashby_jobs(board: ATSBoardConfig) -> list[dict[str, Any]]:
    payload = _get_json(
        f"https://api.ashbyhq.com/posting-api/job-board/{quote(board.board_token, safe='')}?includeCompensation=true"
    )
    raw = payload.get("jobs", []) if isinstance(payload, dict) else []
    return [
        _job(
            board,
            posting_id=item.get("id") or item.get("jobUrl"),
            role=item.get("title"),
            location=item.get("location"),
            url=item.get("jobUrl"),
            description=item.get("descriptionPlain") or item.get("descriptionHtml"),
            salary=item.get("compensation", ""),
        )
        for item in raw
        if isinstance(item, dict) and item.get("isListed", True)
    ]


def _smartrecruiters_jobs(board: ATSBoardConfig, query: str) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    offset = 0
    for _page in range(MAX_PAGES):
        suffix = f"&q={quote(query)}" if query.strip() else ""
        payload = _get_json(
            "https://api.smartrecruiters.com/v1/companies/"
            f"{quote(board.board_token, safe='')}/postings?limit={PAGE_SIZE}&offset={offset}{suffix}"
        )
        raw = payload.get("content", []) if isinstance(payload, dict) else []
        for summary in raw:
            if not isinstance(summary, dict) or not summary.get("id"):
                continue
            detail = _get_json(
                "https://api.smartrecruiters.com/v1/companies/"
                f"{quote(board.board_token, safe='')}/postings/{quote(str(summary['id']), safe='')}"
            )
            if not isinstance(detail, dict):
                continue
            sections = ((detail.get("jobAd") or {}).get("sections") or {}) if isinstance(detail.get("jobAd"), dict) else {}
            description = " ".join(
                _plain((sections.get(key) or {}).get("text", ""))
                for key in ("jobDescription", "qualifications", "additionalInformation")
                if isinstance(sections.get(key), dict)
            ) if isinstance(sections, dict) else ""
            location_data = detail.get("location", {})
            location = ", ".join(
                _plain(location_data.get(key))
                for key in ("city", "region", "country")
                if isinstance(location_data, dict) and location_data.get(key)
            )
            slug = _plain(detail.get("name")).lower().replace(" ", "-")
            url = detail.get("ref") or f"https://jobs.smartrecruiters.com/{board.board_token}/{detail.get('id')}-{slug}"
            jobs.append(
                _job(
                    board,
                    posting_id=detail.get("id"),
                    role=detail.get("name"),
                    location=location,
                    url=url,
                    description=description,
                )
            )
        offset += len(raw)
        if len(raw) < PAGE_SIZE:
            break
    return jobs


def _rank_job(job: dict[str, Any], query: str, location: str) -> float:
    query_tokens = set(re.findall(r"[a-z0-9+#]+", query.lower())) - {
        "and", "or", "the", "in", "at", "for", "jobs", "job"
    }
    title = str(job.get("role", "")).lower()
    description = str(job.get("description", "")).lower()
    matched = sum(3 if token in title else 1 if token in description else 0 for token in query_tokens)
    if query_tokens and not matched:
        return -1.0
    location_score = 1.0 if location and location.lower() in str(job.get("location", "")).lower() else 0.0
    return float(matched) + location_score


def list_public_ats_jobs(
    board: ATSBoardConfig,
    *,
    query: str,
    location: str,
    max_results: int,
) -> list[dict[str, Any]]:
    """Return normalized, filtered, complete public postings for one employer board."""
    if board.provider == "greenhouse":
        jobs = _greenhouse_jobs(board)
    elif board.provider == "lever":
        jobs = _lever_jobs(board)
    elif board.provider == "ashby":
        jobs = _ashby_jobs(board)
    elif board.provider == "smartrecruiters":
        jobs = _smartrecruiters_jobs(board, query)
    else:
        raise ATSBoardError(f"Unsupported ATS provider: {board.provider}")
    ranked = [(score, job) for job in jobs if (score := _rank_job(job, query, location)) >= 0]
    ranked.sort(key=lambda item: (-item[0], str(item[1].get("role", ""))))
    return [job for _score, job in ranked[: max(1, max_results)]]


def search_configured_ats_boards(
    workspace_root: Path,
    *,
    query: str,
    location: str,
    max_results: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Search enabled boards independently and retain partial successes."""
    jobs: list[dict[str, Any]] = []
    errors: list[str] = []
    for board in load_ats_boards(workspace_root):
        if not board.enabled:
            continue
        try:
            jobs.extend(
                list_public_ats_jobs(
                    board,
                    query=query,
                    location=location,
                    max_results=max_results,
                )
            )
        except ATSBoardError as error:
            errors.append(f"{board.company} ({board.provider}): {error}")
    ranked = [(score, job) for job in jobs if (score := _rank_job(job, query, location)) >= 0]
    ranked.sort(key=lambda item: (-item[0], str(item[1].get("company", ""))))
    return [job for _score, job in ranked[:max_results]], errors
