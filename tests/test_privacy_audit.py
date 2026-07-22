"""Regression tests for public-release privacy and prompt safeguards."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from privacy_audit import agent_prompt_path_reason, scan_file


class AgentPromptGuardTests(unittest.TestCase):
    def test_reserved_agent_prompt_paths_are_blocked(self) -> None:
        blocked_paths = [
            Path("AGENTS.md"),
            Path(".codex/instructions.md"),
            Path("docs/system_prompt.txt"),
            Path("workflow.prompt.md"),
            Path("prompts/reviewer.txt"),
            Path("prompt.md"),
        ]

        for path in blocked_paths:
            with self.subTest(path=path):
                self.assertIsNotNone(agent_prompt_path_reason(path))

    def test_normal_project_files_are_allowed(self) -> None:
        for path in [Path("README.md"), Path("src/analyze_job.py"), Path("docs/USAGE.md")]:
            with self.subTest(path=path):
                self.assertIsNone(agent_prompt_path_reason(path))

    def test_tool_control_content_is_detected(self) -> None:
        restricted_text = "<collaboration_" + "mode>default</collaboration_" + "mode>"
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "notes.txt"
            path.write_text(restricted_text, encoding="utf-8")

            findings = scan_file(path, [])

        self.assertTrue(any(finding.category == "agent_prompt_content" for finding in findings))

    def test_regular_ai_job_text_is_not_detected_as_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "job.md"
            path.write_text("Role: AI / ML Analyst\nBuild and evaluate machine-learning models.", encoding="utf-8")

            findings = scan_file(path, [])

        self.assertFalse(any(finding.category == "agent_prompt_content" for finding in findings))


if __name__ == "__main__":
    unittest.main()
