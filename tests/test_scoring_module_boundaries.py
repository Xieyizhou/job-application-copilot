"""Guard domain dependency direction and the supported analysis CLI."""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scoring_report import analyze_job_structured
from workspace import demo_workspace


def source_imports(path: Path) -> set[str]:
    result: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom) and node.module:
            result.add(node.module)
        elif isinstance(node, ast.Import):
            result.update(alias.name for alias in node.names)
    return result


class ScoringModuleBoundaryTests(unittest.TestCase):
    def test_domain_modules_do_not_depend_on_cli_or_ui(self) -> None:
        for path in (PROJECT_ROOT / "src").glob("scoring_*.py"):
            with self.subTest(module=path.stem):
                imports = source_imports(path)
                self.assertFalse(imports & {"analyze_job", "dashboard", "streamlit"})

    def test_source_import_graph_is_acyclic_including_local_imports(self) -> None:
        source = PROJECT_ROOT / "src"
        modules = {
            ".".join(path.relative_to(source).with_suffix("").parts): path
            for path in source.rglob("*.py")
        }
        graph = {name: source_imports(path) & modules.keys() for name, path in modules.items()}
        visited: set[str] = set()
        active: list[str] = []

        def visit(name: str) -> None:
            self.assertNotIn(name, active, f"Import cycle: {' -> '.join([*active, name])}")
            if name in visited:
                return
            active.append(name)
            for dependency in sorted(graph[name]):
                visit(dependency)
            active.pop()
            visited.add(name)

        for name in sorted(graph):
            visit(name)

    def test_demo_cli_matches_domain_analysis_without_writing(self) -> None:
        workspace = demo_workspace()
        assert workspace.resume_source_path is not None
        job = next(iter(sorted(workspace.jobs_dir.glob("*.md"))))
        before = {p: p.read_bytes() for p in workspace.root.rglob("*") if p.is_file()}
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "src/analyze_job.py"), str(job), "--demo"],
            cwd=PROJECT_ROOT,
            env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")},
            capture_output=True,
            text=True,
            check=True,
        )
        expected = analyze_job_structured(job.read_text(), workspace.resume_source_path.read_text())
        self.assertEqual(json.loads(result.stdout), expected)
        self.assertEqual(before, {p: p.read_bytes() for p in workspace.root.rglob("*") if p.is_file()})


if __name__ == "__main__":
    unittest.main()
