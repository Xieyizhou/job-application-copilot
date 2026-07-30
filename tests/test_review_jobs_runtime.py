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
    assert "Choose a job" not in [selectbox.label for selectbox in app.selectbox]
    assert "Number of recommendations" not in [slider.label for slider in app.slider]
    job_buttons = [button.label for button in app.button if " · " in button.label]
    assert len(job_buttons) == 3
    assert "Review" not in [button.label for button in app.button]
    assert not any(
        "Demo workspace uses" in str(info.value)
        for info in app.info
    )

    app.run(timeout=30)
    assert list(app.exception) == []
    app.radio[0].set_value("Add Target Job").run(timeout=30)
    assert list(app.exception) == []
    assert "Return to Personal Workspace" in [button.label for button in app.button]
    assert not any("Demo workspace uses" in str(info.value) for info in app.info)
    assert local_workspace.exists() == local_existed_before
    assert recent_regions.exists() == recent_existed_before
    if recent_existed_before:
        assert recent_regions.read_bytes() == recent_contents_before


class ReviewJobsRuntimeTests(unittest.TestCase):
    def test_demo_review_jobs_compact_table_rerun(self) -> None:
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
