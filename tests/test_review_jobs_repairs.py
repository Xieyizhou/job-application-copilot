from datetime import date

from dashboard_evidence_map import build_evidence_cards
from dashboard_review import sorted_review_jobs
from dashboard_review_components import visible_role_fit
from dashboard_review_styles import decision_field_html
from jd_text import normalize_jd_boundaries
from ml.jd_quality import classify_jd_quality
from saved_job_deletion import archive_job_paths, old_saved_jobs
from structured_jd import structure_job_description


def test_visible_scores_drive_sorting():
    def job(path, score, confidence="high", **extra):
        return dict(path=path, score=score, confidence={"level": confidence}, analysis_available=True, **extra)
    jobs = [job("hidden", 99, "low"), job("zero", 0), job("low", 28), job("high", 80), job("missing", None)]
    assert [j["path"] for j in sorted_review_jobs(jobs, "Role Fit high to low")] == ["high", "low", "zero", "hidden", "missing"]
    assert visible_role_fit(jobs[0]) == "—"
    assert visible_role_fit(jobs[1]) == "0/100"
    tied = [job("b", 28, first_seen_at="2026-08-02"), job("a", 28, first_seen_at="2026-08-02"), job("c", 28, first_seen_at="2026-08-01")]
    assert [j["path"] for j in sorted_review_jobs(tied, "Role Fit high to low")] == ["a", "b", "c"]
    timed = [job("a", 28, first_seen_at="2026-08-02T08:00:00Z"), job("b", 28, first_seen_at="2026-08-02T09:00:00Z")]
    assert [j["path"] for j in sorted_review_jobs(timed, "Role Fit high to low")] == ["b", "a"]


def test_score_and_confidence_colors():
    assert "review-decision-danger" in decision_field_html("Role Fit", "28/100")
    assert "review-decision-warning" in decision_field_html("Role Fit", "60/100")
    assert "review-decision-success" in decision_field_html("Role Fit", "80/100")
    assert "review-decision-neutral" in decision_field_html("Assessment confidence", "High")
    assert "not how well you match" in decision_field_html("Assessment confidence", "High")
    assert "review-decision-warning" in decision_field_html("JD Quality", "Requirements missing")


def test_evidence_order_is_stable():
    rows = [{"requirement": str(i), "prediction": status} for i, status in enumerate(["no_support", "partial", "direct", "direct", "partial"])]
    assert [c["requirement"] for c in build_evidence_cards({"semantic_evidence": {"matches": rows}})] == ["2", "3", "1", "4", "0"]


def test_old_jobs_use_first_save_and_strict_boundary():
    jobs = [
        {"path": "old", "first_seen_at": "2026-08-04", "last_seen_at": "2026-09-04"},
        {"path": "boundary", "first_seen_at": "2026-08-05"},
        {"path": "manual", "created_at": "2026-08-01T12:00:00"},
        {"path": "missing"}, {"path": "bad", "first_seen_at": "unknown"},
    ]
    selected, unknown = old_saved_jobs(jobs, today=date(2026, 9, 4))
    assert [j["path"] for j in selected] == ["old", "manual"]
    assert unknown == 2


def test_batch_archive_is_recoverable_and_reports_failures(tmp_path):
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    first, second = jobs / "a.md", jobs / "b.md"
    first.write_text("a")
    second.write_text("b")
    outside = tmp_path / "tracker.md"
    outside.write_text("keep")
    moved, failed = archive_job_paths([first, second, jobs / "missing.md", outside], jobs_dir=jobs, trash_dir=tmp_path / "trash")
    assert len(moved) == 2 and len(failed) == 2
    assert moved[0].parent == moved[1].parent
    assert [p.read_text() for p in moved] == ["a", "b"]
    assert outside.read_text() == "keep"


def test_flat_asterisk_jd_preserves_qualifications():
    body = (
        "Develop machine learning models for object detection and image analysis. "
        "* Implement robust anomaly detection to identify defects in printed images. "
        "* Create an interface for image upload, analysis and dashboarding. "
        "Key Deliverables and Success Criteria: * Annotated dataset of print images. "
        "* Trained machine learning models with documented performance evaluation results. "
        "Individuals who do well in this role at Example, usually possess: "
        "* Undergraduate in Computer Science, Artificial Intelligence or related studies. "
        "* Able to commit for a 5-6 months full time internship starting in January. "
        "* Familiarity with Machine Learning, Root Cause Analysis, Dashboarding. "
        "Sustainable impact is our commitment to create lasting change for our communities."
    )
    prefix = "Source: JSearch\nDescription Source: full_jd_api\n## Job Description\n"
    flat, multiline = prefix + body, prefix + body.replace(" * ", "\n* ")
    assert structure_job_description(flat)["requirements"] == structure_job_description(multiline)["requirements"]
    assert classify_jd_quality(flat)["requirement_statement_count"] >= 3
    assert not classify_jd_quality(flat)["appears_incomplete"]
    assert normalize_jd_boundaries("Compute x * y and **bold** text.") == "Compute x * y and **bold** text."


def test_cleanup_confirmation_and_cancel_ui(tmp_path):
    from streamlit.testing.v1 import AppTest

    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()
    target = jobs_dir / "old.md"
    target.write_text("temporary job")
    app = AppTest.from_string(f'''
from pathlib import Path
from types import SimpleNamespace
from dashboard_saved_job_management import render_saved_job_management
root = Path({str(tmp_path)!r})
services = SimpleNamespace(demo_mode_enabled=lambda: False,
    current_workspace=lambda: SimpleNamespace(root=root, jobs_dir=root / "jobs"))
jobs = [dict(path=str(root / "jobs" / "old.md"), role="Temporary job", company="Example", first_seen_at="2020-01-01")]
render_saved_job_management(jobs, jobs[0], services)
''').run()
    app.button(key="request_delete_old_jobs").click().run()
    assert target.exists()
    assert "first saved before" in app.warning[0].value
    app.button(key="cancel_saved_job_delete").click().run()
    assert target.exists() and not (tmp_path / "trash").exists()
    app.button(key="request_delete_old_jobs").click().run()
    app.button(key="confirm_saved_job_delete").click().run()
    assert not app.exception
    assert not target.exists()
    assert len(list((tmp_path / "trash").rglob("old.md"))) == 1
