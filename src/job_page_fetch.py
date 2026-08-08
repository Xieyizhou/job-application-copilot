"""Fetch a complete job description from its original public posting URL."""

from __future__ import annotations

import html
import ipaddress
import json
import socket
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urljoin, urlsplit

import requests


MAX_REDIRECTS = 4
MAX_RESPONSE_BYTES = 3_000_000
REQUEST_HEADERS = {
    "Accept": "text/html,application/xhtml+xml",
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36 "
        "JobCopilot/1.0"
    ),
}
JOB_CONTAINER_MARKERS = (
    "job-description",
    "job_description",
    "jobdescription",
    "job-details",
    "job_details",
    "posting-description",
    "description-content",
)


class JobPageFetchError(RuntimeError):
    """Raised when a public job page cannot be read safely."""


@dataclass(frozen=True)
class FetchedJobPage:
    """A complete description extracted from one public job page."""

    description: str
    source_url: str
    extractor: str
    title: str = ""
    company: str = ""


def _public_http_url(url: str) -> bool:
    parsed = urlsplit(str(url or "").strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username:
        return False
    try:
        addresses = {
            result[4][0]
            for result in socket.getaddrinfo(parsed.hostname, parsed.port or 443)
        }
    except OSError:
        return False
    for address in addresses:
        ip = ipaddress.ip_address(address)
        if not ip.is_global:
            return False
    return bool(addresses)


def _clean_html_text(value: str) -> str:
    parser = _VisibleTextParser(capture_all=True)
    parser.feed(str(value or ""))
    return parser.text()


class _VisibleTextParser(HTMLParser):
    """Collect JSON-LD and likely job-description containers without dependencies."""

    def __init__(self, *, capture_all: bool = False) -> None:
        super().__init__(convert_charrefs=True)
        self.capture_all = capture_all
        self.json_ld: list[str] = []
        self.containers: list[list[str]] = []
        self._script_type = ""
        self._script_parts: list[str] = []
        self._skip_depth = 0
        self._capture_depths: list[int] = []
        self._depth = 0
        self._all_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._depth += 1
        attributes = {key.lower(): str(value or "") for key, value in attrs}
        if tag.lower() == "script":
            self._script_type = attributes.get("type", "").lower()
            self._script_parts = []
            self._skip_depth += 1
            return
        if tag.lower() in {"style", "svg", "noscript", "nav", "footer", "header"}:
            self._skip_depth += 1
        marker_text = " ".join(
            [attributes.get("id", ""), attributes.get("class", ""), attributes.get("data-testid", "")]
        ).lower()
        semantic_main = tag.lower() in {"main", "article"}
        if semantic_main or any(marker in marker_text for marker in JOB_CONTAINER_MARKERS):
            self._capture_depths.append(self._depth)
            self.containers.append([])
        if tag.lower() in {"p", "li", "br", "h1", "h2", "h3", "h4"}:
            self._append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "script":
            if "ld+json" in self._script_type and self._script_parts:
                self.json_ld.append("".join(self._script_parts))
            self._script_type = ""
            self._script_parts = []
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag.lower() in {"style", "svg", "noscript", "nav", "footer", "header"}:
            self._skip_depth = max(0, self._skip_depth - 1)
        if self._capture_depths and self._capture_depths[-1] == self._depth:
            self._capture_depths.pop()
        if tag.lower() in {"p", "li", "div", "section", "br", "h1", "h2", "h3", "h4"}:
            self._append("\n")
        self._depth = max(0, self._depth - 1)

    def handle_data(self, data: str) -> None:
        if self._script_type:
            self._script_parts.append(data)
            return
        if self._skip_depth:
            return
        self._append(data)

    def _append(self, value: str) -> None:
        if self.capture_all:
            self._all_parts.append(value)
        if self._capture_depths and self.containers:
            self.containers[-1].append(value)

    def text(self) -> str:
        return _normalize_text("".join(self._all_parts))


def _normalize_text(value: str) -> str:
    decoded = html.unescape(str(value or "")).replace("\xa0", " ")
    lines = [" ".join(line.split()) for line in decoded.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def _job_postings(value: Any) -> list[dict[str, Any]]:
    postings: list[dict[str, Any]] = []
    if isinstance(value, list):
        for item in value:
            postings.extend(_job_postings(item))
    elif isinstance(value, dict):
        raw_types = value.get("@type", [])
        types = [raw_types] if isinstance(raw_types, str) else raw_types
        if any(str(item).lower() == "jobposting" for item in types):
            postings.append(value)
        for key, item in value.items():
            if key != "@context":
                postings.extend(_job_postings(item))
    return postings


def _structured_description(posting: dict[str, Any]) -> str:
    sections: list[str] = []
    description = _clean_html_text(str(posting.get("description", "")))
    if description:
        sections.append(description)
    for heading, keys in (
        ("Responsibilities", ("responsibilities",)),
        ("Requirements", ("qualifications", "skills", "experienceRequirements", "educationRequirements")),
    ):
        values = [
            _clean_html_text(str(posting.get(key, "")))
            for key in keys
            if posting.get(key)
        ]
        combined = "\n".join(value for value in values if value)
        if combined and combined.lower() not in description.lower():
            sections.extend([heading, combined])
    return "\n\n".join(sections).strip()


def extract_job_page(html_text: str, source_url: str) -> FetchedJobPage:
    """Extract the strongest structured or page-level JD candidate."""
    parser = _VisibleTextParser()
    parser.feed(html_text)
    structured: list[tuple[int, FetchedJobPage]] = []
    for script in parser.json_ld:
        try:
            payload = json.loads(script)
        except (json.JSONDecodeError, TypeError):
            continue
        for posting in _job_postings(payload):
            description = _structured_description(posting)
            organization = posting.get("hiringOrganization", {})
            company = str(organization.get("name", "")) if isinstance(organization, dict) else ""
            if description:
                structured.append(
                    (
                        len(description.split()),
                        FetchedJobPage(
                            description=description,
                            source_url=source_url,
                            extractor="jobposting_json_ld",
                            title=str(posting.get("title", "")),
                            company=company,
                        ),
                    )
                )
    if structured:
        return max(structured, key=lambda item: item[0])[1]

    container_texts = [_normalize_text("".join(parts)) for parts in parser.containers]
    viable = [text for text in container_texts if len(text.split()) >= 50]
    if viable:
        return FetchedJobPage(
            description=max(viable, key=lambda text: len(text.split())),
            source_url=source_url,
            extractor="job_page_container",
        )
    raise JobPageFetchError(
        "The page loaded, but no complete structured job description was detected."
    )


def fetch_job_page(url: str) -> FetchedJobPage:
    """Fetch a public HTTP(S) job page with bounded, validated redirects."""
    current_url = str(url or "").strip()
    for _redirect in range(MAX_REDIRECTS + 1):
        if not _public_http_url(current_url):
            raise JobPageFetchError("The original job URL is invalid or is not publicly reachable.")
        try:
            response = requests.get(
                current_url,
                headers=REQUEST_HEADERS,
                timeout=(5, 20),
                allow_redirects=False,
            )
        except requests.RequestException as error:
            raise JobPageFetchError(f"The original job page could not be reached: {error}") from error
        if response.is_redirect or response.is_permanent_redirect:
            location = response.headers.get("Location", "")
            if not location:
                raise JobPageFetchError("The original job page returned an invalid redirect.")
            current_url = urljoin(current_url, location)
            continue
        if response.status_code in {401, 403, 429}:
            raise JobPageFetchError(
                "This employer page blocked the server fetch. Open it in your browser "
                "and use Paste full JD, or try Search provider."
            )
        try:
            response.raise_for_status()
        except requests.RequestException as error:
            raise JobPageFetchError(f"The original job page returned an error: {error}") from error
        content_type = response.headers.get("Content-Type", "").lower()
        if content_type and "html" not in content_type:
            raise JobPageFetchError("The original job URL did not return an HTML job page.")
        if len(response.content) > MAX_RESPONSE_BYTES:
            raise JobPageFetchError("The original job page was too large to process safely.")
        response.encoding = response.encoding or "utf-8"
        return extract_job_page(response.text, current_url)
    raise JobPageFetchError("The original job page redirected too many times.")
