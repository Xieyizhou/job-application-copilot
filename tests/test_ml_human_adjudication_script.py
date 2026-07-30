from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "ml" / "import_human_adjudications.py"


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_import_script_exports_resolved_human_adjudication_gold(tmp_path: Path) -> None:
    task = {
        "schema_version": 1,
        "task_id": "case-1",
        "role_family": "Data",
        "requirement": "Build recurring SQL reports for operations teams.",
        "candidates": [
            {
                "candidate_id": "candidate-a",
                "evidence": "Automated monthly SQL reporting for finance operations.",
            },
            {
                "candidate_id": "candidate-b",
                "evidence": "Recorded action items after stakeholder meetings.",
            },
        ],
    }
    queue_path = tmp_path / "queue.jsonl"
    events_path = tmp_path / "events.jsonl"
    cases_path = tmp_path / "cases.jsonl"
    output_path = tmp_path / "human_gold.jsonl"
    report_path = tmp_path / "report.json"
    _write_jsonl(queue_path, [task])
    _write_jsonl(
        events_path,
        [
            {
                "event_id": "event-1",
                "task_id": "case-1",
                "action": "label",
                "support_label": "Direct",
                "selected_candidate_id": "candidate-a",
                "cover_letter_safe": True,
            }
        ],
    )
    _write_jsonl(
        cases_path,
        [
            {
                "case_id": "case-1",
                "semantic_case_group_id": "sql-reporting-1",
                "source_kind": "synthetic",
                "requirement": task["requirement"],
                "candidates": task["candidates"],
            }
        ],
    )

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--queue-path",
            str(queue_path),
            "--events-path",
            str(events_path),
            "--cases-path",
            str(cases_path),
            "--output-path",
            str(output_path),
            "--report-path",
            str(report_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Accepted human adjudication gold tasks: 1" in result.stdout
    gold = json.loads(output_path.read_text(encoding="utf-8"))
    assert gold["decision_source"] == "human_adjudication"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["accepted_gold_tasks"] == 1
