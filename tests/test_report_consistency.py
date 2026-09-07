"""Exported reports and Tracker summaries must use the displayed fit decision."""

from pathlib import Path

from apply_package import parse_analysis_summary
from scoring_report import analyze_job, analyze_job_structured
from workspace import initialize_personal_workspace


def test_full_manual_jd_report_matches_dashboard_score(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    resume = (root / "data/resume/resume_source.example.md").read_bytes()
    workspace = initialize_personal_workspace("fictional.md", resume, root=tmp_path / "personal")
    job = workspace.jobs_dir / "fictional.md"
    body = (root / "data/demo/jobs/data_analyst.md").read_text(encoding="utf-8")
    job.write_text(
        "# Data Analyst\nSource: Manual\nDescription Source: full_jd_manual\n"
        "JD Fetch Status: complete\n\n## Job Description\n" + body,
        encoding="utf-8",
    )
    visible = analyze_job_structured(job.read_text(encoding="utf-8"), resume.decode())
    report, _ = analyze_job(job, workspace, tmp_path / "bundle")
    assert parse_analysis_summary(report) == (visible["score"], visible["recommendation"])
    assert "Structured requirement coverage" in report
