"""Protect page composition and domain boundaries without prescribing file sizes."""

from __future__ import annotations

import ast
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import dashboard_ui


class UIOrchestrationBoundaryTests(unittest.TestCase):
    def test_pages_only_compose_stateful_dependencies_and_dispatch(self) -> None:
        tree = ast.parse((PROJECT_ROOT / "src/dashboard.py").read_text())
        pages = {
            "dashboard_tab", "fetch_jobs_tab", "manual_job_target_tab",
            "job_descriptions_tab", "tracker_tab", "package_viewer_tab", "safety_notes_tab",
        }
        functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
        for name in pages:
            with self.subTest(page=name):
                body = functions[name].body
                self.assertEqual(len(body), 2)  # Documentation, then one renderer call.
                call = body[-1].value
                self.assertIsInstance(call, ast.Call)
                self.assertIsInstance(call.func, ast.Name)
                self.assertTrue(call.func.id.startswith("render_"))
                self.assertEqual(len(call.args), 1)
                services = call.args[0]
                self.assertIsInstance(services, ast.Call)
                self.assertTrue(services.func.id.endswith("PageServices"))
                self.assertTrue(all(isinstance(kw.value, ast.Name) for kw in services.keywords))

    def test_domain_services_do_not_import_presentation(self) -> None:
        modules = (
            "dashboard_repository", "dashboard_analysis_service", "manual_jobs",
            "job_document", "manual_jd_parser",
            "jd_enrichment", "fetch_jobs", "generate_cover_letter", "tracker", "workspace",
        )
        for name in modules:
            tree = ast.parse((PROJECT_ROOT / "src" / f"{name}.py").read_text())
            imports = {
                node.module for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom)
            } | {
                alias.name for node in ast.walk(tree)
                if isinstance(node, ast.Import) for alias in node.names
            }
            with self.subTest(module=name):
                self.assertFalse(imports & {"streamlit", "dashboard", "dashboard_ui"})

    def test_page_service_records_do_not_inject_stable_presentation(self) -> None:
        stable = {
            "read_text_file", "render_page_header", "render_action_callout",
            "sanitize_fit_text", "render_readiness_checklist", "company_generation_allowed",
        }
        for path in (PROJECT_ROOT / "src").glob("dashboard*.py"):
            for node in ast.parse(path.read_text()).body:
                if isinstance(node, ast.ClassDef) and node.name.endswith("PageServices"):
                    fields = {item.target.id for item in node.body if isinstance(item, ast.AnnAssign)}
                    with self.subTest(service=node.name):
                        self.assertFalse(fields & stable)

    def test_shared_header_escapes_user_text(self) -> None:
        with patch.object(dashboard_ui, "st") as ui:
            dashboard_ui.render_page_header("<title>", "<script>alert(1)</script>")
        rendered = ui.markdown.call_args.args[0]
        self.assertIn("&lt;title&gt;", rendered)
        self.assertNotIn("<script>", rendered)


if __name__ == "__main__":
    unittest.main()
