from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from saved_job_deletion import (
    SavedJobDeletionError,
    archive_all_saved_jobs,
    archive_saved_job,
)


class SavedJobDeletionTests(unittest.TestCase):
    def test_archive_one_preserves_relative_path(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            jobs = root / "jobs"
            target = jobs / "jooble" / "role.md"
            target.parent.mkdir(parents=True)
            target.write_text("job", encoding="utf-8")
            destination = archive_saved_job(target, jobs_dir=jobs, trash_dir=root / "trash")
            self.assertFalse(target.exists())
            self.assertEqual(destination.read_text(encoding="utf-8"), "job")
            self.assertEqual(destination.parts[-2:], ("jooble", "role.md"))

    def test_archive_all_moves_only_job_markdown(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            jobs = root / "jobs"
            (jobs / "a").mkdir(parents=True)
            (jobs / "a" / "one.md").write_text("one", encoding="utf-8")
            (jobs / "two.md").write_text("two", encoding="utf-8")
            (jobs / "keep.txt").write_text("keep", encoding="utf-8")
            archived = archive_all_saved_jobs(jobs_dir=jobs, trash_dir=root / "trash")
            self.assertEqual(len(archived), 2)
            self.assertTrue((jobs / "keep.txt").is_file())
            self.assertFalse((jobs / "two.md").exists())

    def test_rejects_path_outside_jobs_directory(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            jobs = root / "jobs"
            jobs.mkdir()
            outside = root / "outside.md"
            outside.write_text("outside", encoding="utf-8")
            with self.assertRaises(SavedJobDeletionError):
                archive_saved_job(outside, jobs_dir=jobs, trash_dir=root / "trash")


if __name__ == "__main__":
    unittest.main()
