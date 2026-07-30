"""Guard the simplified Add Target Job and Cover Letter hierarchy."""

from __future__ import annotations

import tempfile
from pathlib import Path

from dashboard_cover_letter_components import cover_letter_artifacts
from dashboard_cover_letter_evidence import markdown_section_items


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_cover_letter_page_keeps_draft_and_docx_primary() -> None:
    source = (PROJECT_ROOT / "src" / "dashboard_cover_letter_components.py").read_text(encoding="utf-8")
    evidence_source = (
        PROJECT_ROOT / "src" / "dashboard_cover_letter_evidence.py"
    ).read_text(encoding="utf-8")
    assert source.index("render_evidence_and_gaps(artifacts, services)") < source.index(
        'st.markdown("**Cover letter draft**")'
    )
    assert "**Resume evidence**" in evidence_source
    assert "**Gaps to review**" in evidence_source
    assert source.index('st.markdown("**Cover letter draft**")') < source.index(
        '"Download Cover Letter DOCX"'
    )
    assert 'with st.expander("Supporting materials and details"' in source
    assert "Stored Role Fit" not in source


def test_add_job_page_is_jd_first_and_optional_fields_are_collapsed() -> None:
    source = (PROJECT_ROOT / "src" / "dashboard_manual_entry.py").read_text(encoding="utf-8")
    page_source = (PROJECT_ROOT / "src" / "dashboard_manual.py").read_text(encoding="utf-8")
    desktop_source = (
        PROJECT_ROOT / "src" / "dashboard_desktop_styles.py"
    ).read_text(encoding="utf-8")
    assert source.index("**1. Add the full job description**") < source.index(
        "**2. Verify the job details**"
    )
    assert 'with st.expander("More details"' in source
    assert "Fit results remain provisional" in source
    assert "st.columns([0.56, 0.44]" in page_source
    assert 'tab_labels = ["Add Job", "Saved Jobs"]' in page_source
    assert "render_manual_generate_package_tab" not in page_source
    assert "Demo workspace uses bundled sample jobs" not in page_source
    assert ".st-key-manual_verify_panel" in desktop_source
    assert "border:0 !important" in desktop_source


def test_cover_letter_artifacts_only_collect_known_files() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        package_dir = Path(temp_dir)
        (package_dir / "cover_letter_notes.md").write_text("Review gap", encoding="utf-8")
        (package_dir / "unrelated.txt").write_text("ignore", encoding="utf-8")
        artifacts = cover_letter_artifacts(package_dir)
    assert artifacts.markdown.name == "cover_letter.md"
    assert [path.name for path in artifacts.internal_notes] == ["cover_letter_notes.md"]


def test_markdown_section_items_extracts_only_requested_section() -> None:
    text = """
## Claim Trace — Exact Resume Evidence
- Built production ETL pipelines.
- Reduced processing time by 30%.

## Weak Or Missing Areas
- No direct cloud deployment evidence.
"""

    assert markdown_section_items(
        text,
        ("Claim Trace — Exact Resume Evidence",),
    ) == [
        "Built production ETL pipelines.",
        "Reduced processing time by 30%.",
    ]
