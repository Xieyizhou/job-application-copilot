"""Runtime tests for the injected application-tracker page boundary."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import unittest
from unittest.mock import Mock, patch

import dashboard_tracker


class ContextBlock:
    def __init__(self, owner: "FakeStreamlit") -> None:
        self.owner = owner

    def __enter__(self) -> "ContextBlock":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def metric(self, label: str, value: object) -> None:
        self.owner.metrics[label] = value


@dataclass
class FakeStreamlit:
    pressed: set[str] = field(default_factory=set)
    confirm_delete: bool = False
    messages: list[tuple[str, str]] = field(default_factory=list)
    metrics: dict[str, object] = field(default_factory=dict)
    tables: list[object] = field(default_factory=list)

    def __getattr__(self, name: str) -> Any:
        if name in {"caption", "info", "warning", "error", "success", "markdown", "text", "write"}:
            return lambda value, *args, **kwargs: self.messages.append((name, str(value)))
        if name == "rerun":
            return lambda: None
        raise AttributeError(name)

    def columns(self, count: int | list[float], **kwargs: object) -> list[ContextBlock]:
        size = count if isinstance(count, int) else len(count)
        return [ContextBlock(self) for _ in range(size)]

    def metric(self, label: str, value: object) -> None:
        self.metrics[label] = value

    def expander(self, *args: object, **kwargs: object) -> ContextBlock:
        return ContextBlock(self)

    def multiselect(self, label: str, options: list[str], **kwargs: object) -> list[str]:
        return options

    def slider(self, label: str, **kwargs: object) -> int:
        return int(kwargs["value"])

    def text_input(self, label: str, **kwargs: object) -> str:
        return str(kwargs.get("value", ""))

    def selectbox(self, label: str, options: list[Any], **kwargs: object) -> Any:
        if label == "Move to stage" and "interview" in options:
            return "interview"
        return options[0]

    def checkbox(self, label: str, **kwargs: object) -> bool:
        return self.confirm_delete

    def button(self, label: str, **kwargs: object) -> bool:
        return label in self.pressed

    def dataframe(self, value: object, **kwargs: object) -> None:
        self.tables.append(value)

    def link_button(self, *args: object, **kwargs: object) -> None:
        return None


class DashboardTrackerRuntimeTests(unittest.TestCase):
    def services(
        self,
        *,
        demo: bool,
        rows: list[dict[str, Any]],
    ) -> dashboard_tracker.TrackerPageServices:
        load_rows = Mock(return_value=rows)
        workspace = type("Workspace", (), {"tracker_database_path": "tracker.db"})()
        return dashboard_tracker.TrackerPageServices(
            current_workspace=lambda: workspace,
            demo_mode_enabled=lambda: demo,
            load_tracker_rows=load_rows,
            render_action_callout=Mock(),
            render_page_header=Mock(),
            run_with_captured_output=Mock(return_value=(True, "updated")),
        )

    def test_demo_never_loads_personal_tracker(self) -> None:
        fake = FakeStreamlit()
        services = self.services(demo=True, rows=[])
        with patch.object(dashboard_tracker, "st", fake):
            dashboard_tracker.tracker_tab(services)
        self.assertFalse(services.load_tracker_rows.called)
        self.assertTrue(any("does not read" in message for _, message in fake.messages))

    def test_empty_tracker_renders_zero_metrics_and_guidance(self) -> None:
        fake = FakeStreamlit()
        services = self.services(demo=False, rows=[])
        with patch.object(dashboard_tracker, "st", fake):
            dashboard_tracker.tracker_tab(services)
        self.assertEqual(fake.metrics["Active"], 0)
        self.assertEqual(fake.metrics["Interviews"], 0)
        self.assertEqual(services.load_tracker_rows.call_count, 2)
        self.assertTrue(any("No tracker records" in message for _, message in fake.messages))

    def test_record_can_move_stage_through_injected_service(self) -> None:
        row = {
            "id": 7,
            "company": "Example Labs",
            "role": "Data Analyst",
            "location": "Remote",
            "job_url": "https://jobs.example.test/7",
            "match_score": 81,
            "recommendation": "Review",
            "status": "applied",
            "resume_file": "resume.pdf",
            "cover_letter_file": "cover-letter.docx",
            "notes": "Follow up next week",
            "created_at": "2026-07-01T00:00:00",
            "applied_date": "2026-07-02",
        }
        fake = FakeStreamlit(pressed={"Update Stage"})
        services = self.services(demo=False, rows=[row])
        with patch.object(dashboard_tracker, "st", fake):
            dashboard_tracker.tracker_tab(services)
        self.assertEqual(len(fake.tables), 1)
        services.render_action_callout.assert_called_once()
        services.run_with_captured_output.assert_called_once_with(
            dashboard_tracker.update_status,
            7,
            "interview",
            "tracker.db",
        )
        self.assertTrue(any("Updated application #7" in message for _, message in fake.messages))

    def test_delete_requires_confirmation(self) -> None:
        row = {
            "id": 3,
            "company": "Example",
            "role": "Analyst",
            "location": "Remote",
            "match_score": None,
            "recommendation": "",
            "status": "saved",
            "created_at": "2026-07-20T00:00:00",
        }
        fake = FakeStreamlit(pressed={"Delete Record"})
        services = self.services(demo=False, rows=[row])
        with patch.object(dashboard_tracker, "st", fake):
            dashboard_tracker.tracker_tab(services)
        self.assertFalse(services.run_with_captured_output.called)
        self.assertTrue(any("confirmation box" in message for _, message in fake.messages))


if __name__ == "__main__":
    unittest.main()
