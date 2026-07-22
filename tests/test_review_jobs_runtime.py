"""Isolated Streamlit regression test for Demo Review Jobs reruns."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_child_check() -> None:
    from streamlit.testing.v1 import AppTest

    local_workspace = PROJECT_ROOT / "data" / "local_workspace"
    recent_regions = PROJECT_ROOT / "data" / "ui_state" / "recent_regions.json"
    local_existed_before = local_workspace.exists()
    recent_existed_before = recent_regions.exists()
    recent_contents_before = recent_regions.read_bytes() if recent_existed_before else None

    app = AppTest.from_file(PROJECT_ROOT / "src" / "dashboard.py")
    app.session_state["workspace_mode"] = "Demo"
    app.run(timeout=30)
    app.radio[0].set_value("Review Jobs").run(timeout=30)
    assert list(app.exception) == []
    assert "Choose a job" in [selectbox.label for selectbox in app.selectbox]
    assert "Review" not in [button.label for button in app.button]

    app.text_input[0].set_value("no-such-fictional-job").run(timeout=30)
    assert list(app.exception) == []
    assert "No jobs match the current filters." in [info.value for info in app.info]
    assert local_workspace.exists() == local_existed_before
    assert recent_regions.exists() == recent_existed_before
    if recent_existed_before:
        assert recent_regions.read_bytes() == recent_contents_before


class ReviewJobsRuntimeTests(unittest.TestCase):
    def test_demo_review_jobs_empty_filter_rerun(self) -> None:
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--child"],
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    if "--child" in sys.argv:
        run_child_check()
    else:
        unittest.main()
