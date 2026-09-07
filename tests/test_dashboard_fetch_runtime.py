"""Runtime tests for the injected job-discovery page boundary."""

from __future__ import annotations

import unittest
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import dashboard_fetch
import dashboard_fetch_runner


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

    def update(self, **kwargs: object) -> None:
        self.owner.status_updates.append(dict(kwargs))


@dataclass
class FakeStreamlit:
    demo: bool = False
    submitted: bool = True
    sources: list[str] = field(default_factory=lambda: ["jsearch"])
    session_state: dict[str, Any] = field(default_factory=dict)
    messages: list[tuple[str, str]] = field(default_factory=list)
    metrics: dict[str, object] = field(default_factory=dict)
    status_updates: list[dict[str, object]] = field(default_factory=list)
    selectbox_value: str = "Remote"
    selectbox_kwargs: dict[str, object] = field(default_factory=dict)

    def __getattr__(self, name: str) -> Any:
        if name in {"caption", "info", "warning", "error", "success", "markdown", "text", "write"}:
            return lambda value, *args, **kwargs: self.messages.append((name, str(value)))
        if name in {"dataframe"}:
            return lambda *args, **kwargs: None
        raise AttributeError(name)

    def text_input(self, label: str, **kwargs: object) -> str:
        key = str(kwargs.get("key", ""))
        if key and key in self.session_state:
            return str(self.session_state[key])
        return str(kwargs.get("value", ""))

    def selectbox(self, label: str, options: list[str], **kwargs: object) -> str:
        self.selectbox_kwargs = dict(kwargs)
        return self.selectbox_value

    def form(self, key: str) -> ContextBlock:
        return ContextBlock(self)

    def container(self, **kwargs: object) -> ContextBlock:
        return ContextBlock(self)

    def slider(self, label: str, **kwargs: object) -> int:
        return int(kwargs["value"])

    def multiselect(self, *args: object, **kwargs: object) -> list[str]:
        return self.sources

    def form_submit_button(self, label: str, **kwargs: object) -> bool:
        return self.submitted

    def button(self, label: str, **kwargs: object) -> bool:
        return False

    def status(self, *args: object, **kwargs: object) -> ContextBlock:
        return ContextBlock(self)

    def columns(self, count: int | list[float], **kwargs: object) -> list[ContextBlock]:
        size = count if isinstance(count, int) else len(count)
        return [ContextBlock(self) for _ in range(size)]

    def expander(self, *args: object, **kwargs: object) -> ContextBlock:
        return ContextBlock(self)


class DashboardFetchRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.render_fetch_history_section = self.enterContext(patch("dashboard_fetch.render_fetch_history_section"))
        self.render_fetch_run_job_cards = self.enterContext(patch("dashboard_fetch.render_fetch_run_job_cards"))
        self.render_fetch_run_job_table = self.enterContext(patch("dashboard_fetch.render_fetch_run_job_table"))
        self.render_page_header = self.enterContext(patch("dashboard_fetch.render_page_header"))

    def services(self, *, demo: bool, run_result: dict[str, Any] | None = None) -> dashboard_fetch.FetchPageServices:
        run = Mock(return_value=(run_result or {}, "provider output"))
        return dashboard_fetch.FetchPageServices(
            current_workspace=lambda: SimpleNamespace(
                mode="personal",
                resume_source_path=Path("/missing/candidate_source.md"),
            ),
            demo_mode_enabled=lambda: demo,
            go_to_page=Mock(),
            relocate_fetched_jobs_to_workspace=lambda paths, source: list(paths),
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
        ), patch.object(dashboard_fetch_runner, "st", fake):
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
        ), patch.object(dashboard_fetch_runner, "st", fake):
            dashboard_fetch.fetch_jobs_tab(services)
        self.assertEqual(fake.metrics["Returned"], 3)
        self.assertEqual(fake.metrics["New"], 1)
        self.assertEqual(fake.metrics["Full JDs"], 3)
        self.assertEqual(fake.session_state["recommendation_limit"], 12)
        self.assertEqual(fake.session_state["latest_fetch_run_id"], "run-7")
        self.render_fetch_run_job_cards.assert_called_once()
        self.assertEqual(fake.status_updates[-1]["state"], "complete")

    def test_missing_source_is_reported_without_provider_call(self) -> None:
        fake = FakeStreamlit(sources=[])
        services = self.services(demo=False)
        with patch.object(dashboard_fetch, "st", fake), patch.object(
            dashboard_fetch, "jsearch_configured", return_value=False
        ), patch.object(dashboard_fetch_runner, "st", fake):
            dashboard_fetch.fetch_jobs_tab(services)
        self.assertFalse(services.run_with_captured_output.called)
        self.assertIn(("error", "Select at least one source."), fake.messages)

    def test_search_uses_resume_suggested_query(self) -> None:
        fake = FakeStreamlit()
        services = self.services(demo=False)
        profile = dashboard_fetch.search_profile_from_path(None)
        profile = type(profile)("Machine Learning Engineer", ("machine learning",), True)
        with patch.object(dashboard_fetch, "st", fake), patch.object(
            dashboard_fetch, "jsearch_configured", return_value=True
        ), patch.object(
            dashboard_fetch, "search_profile_from_path", return_value=profile
        ), patch.object(dashboard_fetch_runner, "st", fake):
            dashboard_fetch.fetch_jobs_tab(services)
        args = services.run_with_captured_output.call_args.args[1]
        self.assertEqual(args.query, "Machine Learning Engineer")

    def test_provider_failure_is_visible(self) -> None:
        fake = FakeStreamlit()
        services = self.services(demo=False)
        services.run_with_captured_output.side_effect = RuntimeError("temporary provider failure")
        with patch.object(dashboard_fetch, "st", fake), patch.object(
            dashboard_fetch, "jsearch_configured", return_value=True
        ), patch.object(dashboard_fetch_runner, "st", fake):
            dashboard_fetch.fetch_jobs_tab(services)
        self.assertEqual(fake.status_updates[-1]["state"], "error")
        self.assertTrue(any("temporary provider failure" in message for _, message in fake.messages))

    def test_singapore_and_united_kingdom_regions_are_supported(self) -> None:
        self.assertIn("Singapore", dashboard_fetch.REGION_OPTIONS)
        self.assertIn("United Kingdom", dashboard_fetch.REGION_OPTIONS)
        self.assertEqual(dashboard_fetch.REGION_CONFIG["Singapore"]["adzuna_country"], "sg")
        self.assertEqual(dashboard_fetch.REGION_CONFIG["United Kingdom"]["adzuna_country"], "gb")

    def test_region_control_suggests_markets_and_accepts_custom_locations(self) -> None:
        fake = FakeStreamlit(selectbox_value="Berlin")
        services = self.services(demo=False)
        with patch.object(dashboard_fetch, "st", fake):
            country, adzuna_location, jooble_location, supported = (
                dashboard_fetch.render_fetch_region_fields(services)
            )
        self.assertIn("China", dashboard_fetch.REGION_OPTIONS)
        self.assertIn("Canada", dashboard_fetch.REGION_OPTIONS)
        self.assertTrue(fake.selectbox_kwargs["accept_new_options"])
        self.assertIsNone(fake.selectbox_kwargs.get("index"))
        self.assertEqual(country, "de")
        self.assertEqual(adzuna_location, "Berlin")
        self.assertEqual(jooble_location, "Berlin")
        self.assertTrue(supported)

    def test_source_selection_is_remembered_locally_after_search(self) -> None:
        fake = FakeStreamlit(sources=["adzuna"])
        services = self.services(demo=False)
        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            dashboard_fetch, "st", fake
        ), patch.object(
            dashboard_fetch, "jsearch_configured", return_value=True
        ), patch(
            "dashboard_fetch_preferences.FETCH_PREFERENCES_PATH",
            Path(temp_dir) / "fetch_preferences.json",
        ):
            dashboard_fetch.render_fetch_options_form(
                services,
                adzuna_supported=True,
            )
            remembered = dashboard_fetch.load_fetch_sources(["jsearch"])
        self.assertEqual(remembered, ["adzuna"])

    def test_result_summary_keeps_provider_totals_separate(self) -> None:
        outcome = dashboard_fetch.FetchSearchOutcome(
            saved_paths=["one.md"],
            errors=["provider issue"],
            runs=[
                {
                    "total_jobs_returned": 4,
                    "new_jobs_count": 1,
                    "duplicate_jobs_count": 2,
                    "skipped_jobs_count": 1,
                    "full_descriptions_count": 3,
                    "new_jobs": [{"role": "Analyst"}],
                    "previously_seen_jobs": [{"role": "Scientist"}],
                }
            ],
        )
        summary = dashboard_fetch.summarize_fetch_outcome(outcome)
        self.assertEqual(summary.returned, 4)
        self.assertEqual(summary.new, 1)
        self.assertEqual(summary.already_seen, 2)
        self.assertEqual(summary.issues, 2)
        self.assertEqual(summary.full_descriptions, 3)
        self.assertEqual(len(summary.new_jobs), 1)
        self.assertEqual(len(summary.seen_jobs), 1)
        self.assertEqual(summary.skipped_jobs, [])

    def test_preview_only_results_are_visible_but_not_saved(self) -> None:
        preview = {
            "company": "Preview Company",
            "role": "Data Analyst",
            "source": "jooble",
        }
        outcome = dashboard_fetch.FetchSearchOutcome(
            runs=[
                {
                    "total_jobs_returned": 1,
                    "new_jobs_count": 0,
                    "duplicate_jobs_count": 0,
                    "skipped_jobs_count": 1,
                    "skipped_jobs": [preview],
                }
            ]
        )
        fake = FakeStreamlit()
        services = self.services(demo=False)
        request = dashboard_fetch.FetchSearchRequest(
            submitted=True,
            query="Data Analyst",
            sources=["jooble"],
            recommendation_limit=12,
            fetch_limit_per_source=20,
            adzuna_country="us",
            adzuna_location="Remote",
            jooble_location="Remote",
            adzuna_supported=True,
        )

        with patch.object(dashboard_fetch, "st", fake):
            dashboard_fetch.render_fetch_results(outcome, request, services)

        self.render_fetch_run_job_cards.assert_called_once_with(
            [preview],
            "No preview-only results in this search.",
        )
        self.assertEqual(fake.metrics["Skipped previews"], 1)


if __name__ == "__main__":
    unittest.main()
