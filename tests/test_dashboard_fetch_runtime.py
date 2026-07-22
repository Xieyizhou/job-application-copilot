"""Runtime tests for the injected job-discovery page boundary."""

from __future__ import annotations

import unittest
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import Mock, patch

import dashboard_fetch


class ContextBlock:
    def __init__(self, owner: "FakeStreamlit") -> None:
        self.owner = owner

    def __enter__(self) -> "ContextBlock":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def metric(self, label: str, value: object) -> None:
        self.owner.metrics[label] = value

    def button(self, label: str, **kwargs: object) -> bool:
        return False


@dataclass
class FakeStreamlit:
    demo: bool = False
    submitted: bool = True
    sources: list[str] = field(default_factory=lambda: ["jsearch"])
    session_state: dict[str, Any] = field(default_factory=dict)
    messages: list[tuple[str, str]] = field(default_factory=list)
    metrics: dict[str, object] = field(default_factory=dict)

    def __getattr__(self, name: str) -> Any:
        if name in {"caption", "info", "warning", "error", "success", "markdown", "text", "write"}:
            return lambda value, *args, **kwargs: self.messages.append((name, str(value)))
        if name in {"dataframe"}:
            return lambda *args, **kwargs: None
        raise AttributeError(name)

    def text_input(self, label: str, **kwargs: object) -> str:
        return str(kwargs.get("value", "Remote"))

    def selectbox(self, label: str, options: list[str], **kwargs: object) -> str:
        return "Remote"

    def form(self, key: str) -> ContextBlock:
        return ContextBlock(self)

    def slider(self, label: str, **kwargs: object) -> int:
        return int(kwargs["value"])

    def multiselect(self, *args: object, **kwargs: object) -> list[str]:
        return self.sources

    def form_submit_button(self, label: str) -> bool:
        return self.submitted

    def columns(self, count: int | list[float]) -> list[ContextBlock]:
        size = count if isinstance(count, int) else len(count)
        return [ContextBlock(self) for _ in range(size)]

    def expander(self, *args: object, **kwargs: object) -> ContextBlock:
        return ContextBlock(self)


class DashboardFetchRuntimeTests(unittest.TestCase):
    def services(self, *, demo: bool, run_result: dict[str, Any] | None = None) -> dashboard_fetch.FetchPageServices:
        run = Mock(return_value=(run_result or {}, "provider output"))
        return dashboard_fetch.FetchPageServices(
            demo_mode_enabled=lambda: demo,
            go_to_page=Mock(),
            relocate_fetched_jobs_to_workspace=lambda paths, source: list(paths),
            render_fetch_history_section=Mock(),
            render_fetch_run_job_cards=Mock(),
            render_fetch_run_job_table=Mock(),
            render_page_header=Mock(),
            run_with_captured_output=run,
            default_recommendation_limit=12,
            min_recommendation_limit=5,
            max_recommendation_limit=30,
        )

    def test_demo_submission_never_calls_provider(self) -> None:
        fake = FakeStreamlit()
        services = self.services(demo=True)
        with patch.object(dashboard_fetch, "st", fake), patch.object(
            dashboard_fetch, "jsearch_configured", return_value=True
        ):
            dashboard_fetch.fetch_jobs_tab(services)
        self.assertFalse(services.run_with_captured_output.called)
        self.assertTrue(any("does not call external" in message for _, message in fake.messages))

    def test_personal_search_renders_new_jobs_and_metrics(self) -> None:
        new_job = {"company": "Example", "role": "Analyst"}
        run_record = {
            "fetch_run_id": "run-7",
            "source": "jsearch",
            "total_jobs_returned": 3,
            "new_jobs_count": 1,
            "duplicate_jobs_count": 2,
            "skipped_jobs_count": 0,
            "full_descriptions_count": 3,
            "new_jobs": [new_job],
            "previously_seen_jobs": [],
        }
        fake = FakeStreamlit()
        services = self.services(
            demo=False,
            run_result={"saved_paths": ["job.md"], "fetch_run": run_record},
        )
        with patch.object(dashboard_fetch, "st", fake), patch.object(
            dashboard_fetch, "jsearch_configured", return_value=True
        ):
            dashboard_fetch.fetch_jobs_tab(services)
        self.assertEqual(fake.metrics["Returned"], 3)
        self.assertEqual(fake.metrics["New"], 1)
        self.assertEqual(fake.metrics["Full JDs"], 3)
        self.assertEqual(fake.session_state["recommendation_limit"], 12)
        self.assertEqual(fake.session_state["latest_fetch_run_id"], "run-7")
        services.render_fetch_run_job_cards.assert_called_once()

    def test_missing_source_is_reported_without_provider_call(self) -> None:
        fake = FakeStreamlit(sources=[])
        services = self.services(demo=False)
        with patch.object(dashboard_fetch, "st", fake), patch.object(
            dashboard_fetch, "jsearch_configured", return_value=False
        ):
            dashboard_fetch.fetch_jobs_tab(services)
        self.assertFalse(services.run_with_captured_output.called)
        self.assertIn(("error", "Select at least one source."), fake.messages)


if __name__ == "__main__":
    unittest.main()
