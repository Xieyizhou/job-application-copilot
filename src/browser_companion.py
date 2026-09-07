"""Local-only browser companion protocol for importing an opened job posting."""

from __future__ import annotations

import json
import os
import re
import secrets
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from fetch_history import canonicalize_job_url
from document_text import read_markdown_field
from jd_enrichment import replace_saved_job_description


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
MAX_REQUEST_BYTES = 2_000_000


@dataclass(frozen=True)
class CompanionConfig:
    """Runtime state shared by the loopback HTTP handler."""

    jobs_dir: Path
    token: str


@dataclass
class BrowserCompanionServer:
    """One in-process loopback server and its background thread."""

    server: ThreadingHTTPServer
    thread: threading.Thread
    token: str
    port: int

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


def _valid_public_url(value: str) -> bool:
    parsed = urlsplit(str(value or "").strip())
    return parsed.scheme in {"http", "https"} and bool(parsed.hostname) and not parsed.username


def load_or_create_token(path: Path) -> str:
    """Persist one ignored local token so the extension survives dashboard restarts."""
    path = Path(path)
    try:
        token = path.read_text(encoding="utf-8").strip()
    except OSError:
        token = ""
    if len(token) >= 24:
        return token
    path.parent.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(32)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(token + "\n", encoding="utf-8")
    os.chmod(temporary_path, 0o600)
    temporary_path.replace(path)
    return token


def _saved_job_paths(jobs_dir: Path) -> list[Path]:
    if not jobs_dir.is_dir():
        return []
    return sorted(path for path in jobs_dir.rglob("*.md") if path.is_file())


def _identity_tokens(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9+#]+", str(value or "").lower()))


def _field_similarity(left: str, right: str) -> float:
    left_tokens = _identity_tokens(left)
    right_tokens = _identity_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = len(left_tokens & right_tokens)
    return overlap / min(len(left_tokens), len(right_tokens))


def resolve_saved_job(
    jobs_dir: Path,
    *,
    page_url: str,
    title: str = "",
    company: str = "",
) -> Path | None:
    """Resolve one existing saved job without creating records from arbitrary pages."""
    canonical_page_url = canonicalize_job_url(page_url)
    ranked: list[tuple[float, Path]] = []
    for path in _saved_job_paths(jobs_dir):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        saved_url = canonicalize_job_url(read_markdown_field(text, "Job URL"))
        if canonical_page_url and saved_url and canonical_page_url == saved_url:
            return path
        saved_title = read_markdown_field(text, "Role")
        saved_company = read_markdown_field(text, "Company Normalized") or read_markdown_field(
            text, "Company"
        )
        title_score = _field_similarity(title, saved_title)
        company_score = _field_similarity(company, saved_company)
        if title_score >= 0.6 and company_score >= 0.75:
            ranked.append((title_score * 0.55 + company_score * 0.45, path))
    ranked.sort(key=lambda item: item[0], reverse=True)
    if len(ranked) == 1 or (len(ranked) > 1 and ranked[0][0] - ranked[1][0] >= 0.12):
        return ranked[0][1]
    return None


def import_browser_capture(config: CompanionConfig, payload: dict[str, Any]) -> dict[str, Any]:
    """Validate a browser capture and replace only its matching saved JD."""
    page_url = str(payload.get("url", "") or "").strip()
    title = str(payload.get("title", "") or "").strip()
    company = str(payload.get("company", "") or "").strip()
    description = str(payload.get("description", "") or "").strip()
    extractor = str(payload.get("extractor", "browser_visible_page") or "browser_visible_page")
    if not _valid_public_url(page_url):
        return {
            "ok": False,
            "status": "invalid_url",
            "message": "Open a public HTTP(S) job page first.",
        }
    if not description:
        return {
            "ok": False,
            "status": "empty_description",
            "message": "No job description was visible on this page.",
        }
    path = resolve_saved_job(config.jobs_dir, page_url=page_url, title=title, company=company)
    if path is None:
        return {
            "ok": False,
            "status": "saved_job_not_found",
            "message": "Save this job in JobCopilot first, then import the opened posting again.",
        }
    saved_text = path.read_text(encoding="utf-8")
    saved_title = read_markdown_field(saved_text, "Role")
    saved_company = read_markdown_field(saved_text, "Company Normalized") or read_markdown_field(
        saved_text, "Company"
    )
    if (title and _field_similarity(title, saved_title) < 0.35) or (
        company and _field_similarity(company, saved_company) < 0.65
    ):
        return {
            "ok": False,
            "status": "page_identity_mismatch",
            "message": "The opened page appears to show a different company or role.",
        }
    result = replace_saved_job_description(
        path,
        description,
        description_source="browser_companion",
        enriched_by="Local browser companion",
        extractor=extractor,
        source_url=page_url,
    )
    return {
        "ok": bool(result.get("updated")),
        "status": str(result.get("status", "unknown")),
        "message": str(result.get("message", "The browser capture could not be saved.")),
        "quality": result.get("quality", {}),
    }


def _handler_class(config: CompanionConfig) -> type[BaseHTTPRequestHandler]:
    class CompanionHandler(BaseHTTPRequestHandler):
        server_version = "JobCopilotCompanion/1.0"

        def _reply(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self._reply(200, {"ok": True, "service": "JobCopilot browser companion"})
                return
            self._reply(404, {"ok": False, "message": "Not found."})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/v1/import":
                self._reply(404, {"ok": False, "message": "Not found."})
                return
            if self.headers.get("X-JobCopilot-Token", "") != config.token:
                self._reply(
                    401,
                    {
                        "ok": False,
                        "status": "unauthorized",
                        "message": "Invalid companion token.",
                    },
                )
                return
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                content_length = 0
            if content_length <= 0 or content_length > MAX_REQUEST_BYTES:
                self._reply(
                    413,
                    {
                        "ok": False,
                        "status": "invalid_size",
                        "message": "The captured page is too large.",
                    },
                )
                return
            try:
                payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                self._reply(400, {"ok": False, "status": "invalid_json", "message": "Invalid capture payload."})
                return
            if not isinstance(payload, dict):
                self._reply(400, {"ok": False, "status": "invalid_json", "message": "Invalid capture payload."})
                return
            result = import_browser_capture(config, payload)
            self._reply(200 if result["ok"] else 422, result)

        def log_message(self, _format: str, *_args: object) -> None:
            return

    return CompanionHandler


def start_browser_companion(
    jobs_dir: Path,
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    token: str | None = None,
) -> BrowserCompanionServer:
    """Start the loopback import service for the current Personal workspace."""
    if host not in {"127.0.0.1", "localhost"}:
        raise ValueError("The browser companion may listen only on loopback.")
    resolved_token = token or secrets.token_urlsafe(24)
    config = CompanionConfig(jobs_dir=Path(jobs_dir).resolve(), token=resolved_token)
    server = ThreadingHTTPServer((host, port), _handler_class(config))
    thread = threading.Thread(target=server.serve_forever, daemon=True, name="jobcopilot-companion")
    thread.start()
    return BrowserCompanionServer(
        server=server,
        thread=thread,
        token=resolved_token,
        port=int(server.server_address[1]),
    )
