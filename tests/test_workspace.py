import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from analyze_job import analyze_job
from tracker import initialize_database
from src.workspace import (
    WorkspaceError,
    demo_workspace,
    initialize_personal_workspace,
    personal_workspace,
    sanitize_upload_filename,
)


class WorkspaceTests(unittest.TestCase):
    def test_demo_workspace_is_read_only_and_never_uses_personal_paths(self) -> None:
        workspace = demo_workspace()
        self.assertEqual(workspace.mode, "demo")
        self.assertTrue(workspace.read_only)
        self.assertIsNone(workspace.tracker_database_path)
        self.assertNotIn("local_workspace", str(workspace.root))
        with self.assertRaises(WorkspaceError):
            workspace.require_writable()

    def test_personal_workspace_requires_manifest_and_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            workspace = personal_workspace(Path(temporary_dir) / "local_workspace")
            self.assertFalse(workspace.ready)
            self.assertIsNone(workspace.resume_source_path)

    def test_initialize_personal_workspace_writes_content_free_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir) / "local_workspace"
            candidate_text = b"Sanitized Candidate\nPython and SQL"
            workspace = initialize_personal_workspace(
                "candidate profile.md", candidate_text, root=root
            )

            self.assertTrue(workspace.ready)
            self.assertIsNotNone(workspace.resume_source_path)
            assert workspace.resume_source_path is not None
            self.assertEqual(workspace.resume_source_path, (root / "candidate" / "candidate_source.md").resolve())
            self.assertEqual(workspace.resume_source_path.read_text(encoding="utf-8"), candidate_text.decode() + "\n")
            self.assertEqual((root / "candidate" / "original_resume.md").read_bytes(), candidate_text)
            manifest_text = (root / "workspace.json").read_text(encoding="utf-8")
            manifest = json.loads(manifest_text)
            self.assertNotIn(candidate_text.decode(), manifest_text)
            self.assertEqual(manifest["resume_source"], "candidate/candidate_source.md")
            self.assertEqual(manifest["candidate_original_extension"], ".md")
            self.assertEqual(manifest["candidate_extraction_method"], "markdown")
            self.assertEqual(workspace.jobs_dir, root.resolve() / "jobs")
            self.assertEqual(workspace.generated_dir, root.resolve() / "generated")
            self.assertEqual(workspace.tracker_database_path, root.resolve() / "applications.db")
            assert workspace.tracker_database_path is not None
            initialize_database(workspace.tracker_database_path)
            self.assertTrue(workspace.tracker_database_path.is_file())

    def test_upload_filename_cannot_escape_workspace(self) -> None:
        self.assertEqual(sanitize_upload_filename("../../profile.md", {".md"}), "profile.md")
        self.assertEqual(
            sanitize_upload_filename(r"..\\..\\profile.txt", {".txt"}), "profile.txt"
        )
        with self.assertRaises(WorkspaceError):
            sanitize_upload_filename("profile.pdf", {".md", ".txt"})

    def test_invalid_manifest_path_is_not_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir) / "local_workspace"
            root.mkdir()
            (root / "workspace.json").write_text(
                json.dumps({"resume_source": "../outside.md"}), encoding="utf-8"
            )
            self.assertFalse(personal_workspace(root).ready)

    def test_analysis_uses_personal_candidate_and_generated_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir) / "local_workspace"
            workspace = initialize_personal_workspace(
                "candidate.md", b"# Candidate\nSQL analytics", root=root
            )
            job_path = root / "jobs" / "sample_job.md"
            job_path.parent.mkdir(parents=True, exist_ok=True)
            job_path.write_text("# Data role\nRequirements:\n- SQL", encoding="utf-8")

            report, report_path = analyze_job(job_path, workspace)

            self.assertIn("SQL", report)
            self.assertTrue(report_path.is_relative_to(workspace.generated_dir))
            self.assertTrue(report_path.is_file())


if __name__ == "__main__":
    unittest.main()
