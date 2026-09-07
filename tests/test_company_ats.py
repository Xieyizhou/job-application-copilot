"""Tests for local employer ATS board discovery and configuration."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from company_ats import (
    ATSBoardConfig,
    ATSBoardError,
    list_public_ats_jobs,
    load_ats_boards,
    parse_ats_board_url,
    save_ats_boards,
)


@pytest.mark.parametrize(
    ("url", "provider", "token", "region"),
    [
        ("https://boards.greenhouse.io/acme/jobs/123", "greenhouse", "acme", "global"),
        ("https://jobs.lever.co/acme/abc", "lever", "acme", "global"),
        ("https://jobs.eu.lever.co/acme/abc", "lever", "acme", "eu"),
        ("https://jobs.ashbyhq.com/acme/abc", "ashby", "acme", "global"),
        ("https://jobs.smartrecruiters.com/Acme/123-role", "smartrecruiters", "Acme", "global"),
    ],
)
def test_parse_supported_board_urls(url: str, provider: str, token: str, region: str) -> None:
    board = parse_ats_board_url(url, company="Acme")
    assert board.provider == provider
    assert board.board_token == token
    assert board.api_region == region


def test_parse_rejects_unknown_hosts() -> None:
    with pytest.raises(ATSBoardError):
        parse_ats_board_url("https://linkedin.com/jobs/123")


def test_registry_round_trip_and_deduplicates() -> None:
    board = parse_ats_board_url("https://jobs.ashbyhq.com/acme", company="Acme")
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        save_ats_boards(root, [board, board])
        loaded = load_ats_boards(root)
    assert loaded == [board]


def test_greenhouse_list_normalizes_full_postings() -> None:
    board = ATSBoardConfig("greenhouse", "Acme", "https://boards.greenhouse.io/acme", "acme")
    payload = {
        "jobs": [
            {
                "id": 7,
                "title": "Machine Learning Engineer",
                "location": {"name": "Singapore"},
                "absolute_url": "https://boards.greenhouse.io/acme/jobs/7",
                "content": "Build machine learning systems. " * 40,
            }
        ]
    }
    with patch("company_ats._get_json", return_value=payload):
        jobs = list_public_ats_jobs(
            board, query="machine learning", location="Singapore", max_results=20
        )
    assert len(jobs) == 1
    assert jobs[0]["source"] == "company_ats"
    assert jobs[0]["ats_provider"] == "greenhouse"
    assert jobs[0]["jd_fetch_status"] == "complete"


def test_each_adapter_returns_one_normalized_job() -> None:
    lever = ATSBoardConfig("lever", "Acme", "https://jobs.lever.co/acme", "acme")
    ashby = ATSBoardConfig("ashby", "Acme", "https://jobs.ashbyhq.com/acme", "acme")
    smart = ATSBoardConfig(
        "smartrecruiters", "Acme", "https://jobs.smartrecruiters.com/Acme", "Acme"
    )
    lever_payload = [{
        "id": "l1", "text": "Data Engineer", "categories": {"location": "Remote"},
        "hostedUrl": "https://jobs.lever.co/acme/l1", "descriptionPlain": "Build data systems " * 40,
        "lists": [],
    }]
    ashby_payload = {"jobs": [{
        "id": "a1", "title": "Data Engineer", "location": "Remote",
        "jobUrl": "https://jobs.ashbyhq.com/acme/a1", "descriptionPlain": "Build data systems " * 40,
        "isListed": True,
    }]}
    smart_list = {"content": [{"id": "s1"}]}
    smart_detail = {
        "id": "s1", "name": "Data Engineer", "location": {"country": "Singapore"},
        "ref": "https://jobs.smartrecruiters.com/Acme/s1-data-engineer",
        "jobAd": {"sections": {"jobDescription": {"text": "Build data systems " * 40}}},
    }
    with patch("company_ats._get_json", return_value=lever_payload):
        assert len(list_public_ats_jobs(lever, query="data", location="", max_results=5)) == 1
    with patch("company_ats._get_json", return_value=ashby_payload):
        assert len(list_public_ats_jobs(ashby, query="data", location="", max_results=5)) == 1
    with patch("company_ats._get_json", side_effect=[smart_list, smart_detail]):
        assert len(list_public_ats_jobs(smart, query="data", location="", max_results=5)) == 1
