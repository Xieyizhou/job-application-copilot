"""Focused tests for the stable dashboard launcher."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pyarrow

import run_dashboard


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class DashboardLauncherTests(unittest.TestCase):
    def test_configure_arrow_memory_pool_selects_system_backend(self) -> None:
        self.assertEqual(run_dashboard.configure_arrow_memory_pool(), "system")
        self.assertEqual(pyarrow.default_memory_pool().backend_name, "system")

    def test_build_streamlit_argv(self) -> None:
        self.assertEqual(
            run_dashboard.build_streamlit_argv(),
            ["streamlit", "run", str(PROJECT_ROOT / "src" / "dashboard.py")],
        )

    def test_build_streamlit_argv_preserves_extra_arguments(self) -> None:
        extras = ["--server.port", "8600", "--server.headless=true"]
        self.assertEqual(run_dashboard.build_streamlit_argv(extras)[3:], extras)

    def test_main_reports_arrow_initialization_failure(self) -> None:
        with (
            patch.object(
                run_dashboard,
                "configure_arrow_memory_pool",
                side_effect=RuntimeError("native allocator unavailable"),
            ),
            patch("builtins.print") as print_mock,
        ):
            self.assertEqual(run_dashboard.main([]), 1)
        self.assertIn("native allocator unavailable", print_mock.call_args.args[0])

    def test_smoke_test_uses_supported_entry_without_launching_server(self) -> None:
        with (
            patch.object(run_dashboard, "configure_arrow_memory_pool", return_value="system"),
            patch.object(run_dashboard, "run_smoke_test", return_value=0) as smoke_test,
        ):
            self.assertEqual(run_dashboard.main(["--smoke-test"]), 0)
        smoke_test.assert_called_once_with()

    def test_import_does_not_launch_streamlit(self) -> None:
        result = subprocess.run(
            [sys.executable, "-c", "import run_dashboard; print('imported')"],
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "imported")

    def test_main_starts_and_stops_browser_companion_around_streamlit(self) -> None:
        class Companion:
            port = 8765
            token = "test-token"

            def __init__(self) -> None:
                self.stopped = False

            def stop(self) -> None:
                self.stopped = True

        companion = Companion()
        with tempfile.TemporaryDirectory() as temp_dir:
            companion_dir = Path(temp_dir)
            with (
                patch.object(run_dashboard, "configure_arrow_memory_pool", return_value="system"),
                patch.object(run_dashboard, "COMPANION_DIR", companion_dir),
                patch("browser_companion.load_or_create_token", return_value="test-token"),
                patch("browser_companion.start_browser_companion", return_value=companion),
                patch("streamlit.web.cli.main", return_value=None),
            ):
                self.assertEqual(run_dashboard.main([]), 0)

            self.assertTrue(companion.stopped)
            self.assertFalse((companion_dir / "connection.json").exists())


if __name__ == "__main__":
    unittest.main()
