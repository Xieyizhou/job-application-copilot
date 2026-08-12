"""Contract tests for source-backed JD structuring and pipeline diagnostics."""

from structured_jd import pipeline_diagnostic_message, pipeline_trace, structure_job_description
from scoring_report import analyze_job_structured


JD = """# Data Platform Engineer
Company: Example Systems
Role: Data Platform Engineer
Location: Remote

## Job Description
About the company: We build trusted analytics products.
Responsibilities:
- Build and maintain batch data pipelines.
- Collaborate with analysts and product managers.
Required Qualifications:
- Must have experience with Python and SQL.
- Bachelor's degree in computer science or equivalent required.
Preferred Qualifications:
- Experience with Airflow is preferred.
Benefits:
- Flexible working hours.
"""


def test_structures_stable_sections_without_inventing_values() -> None:
    job = structure_job_description(JD)
    assert job["title"] == "Data Platform Engineer"
    assert job["work_mode"] == "Remote"
    assert job["salary"] == ""
    assert "Build and maintain batch data pipelines" in job["responsibilities"]
    assert any(row["category"] == "education" for row in job["requirements"])
    assert any(row["type"] == "preferred" for row in job["requirements"])
    assert all(row["source_span"] == row["text"] for row in job["requirements"])


def test_pipeline_trace_distinguishes_extraction_and_evidence_failures() -> None:
    job = structure_job_description(JD)
    trace = pipeline_trace(JD, job, {"accepted_count": 0})
    assert trace["fetch"]["status"] == "ready"
    assert trace["structure"]["status"] == "ready"
    assert "no reliable resume evidence" in pipeline_diagnostic_message(trace).lower()


def test_flat_provider_text_is_recovered_into_requirements() -> None:
    flat = (
        "Role: Analyst\n## Job Description\n"
        "Responsibilities: Analyze operational datasets and present findings. "
        "Requirements: Must have experience with SQL and dashboard reporting. "
        "Preferred Qualifications: Python experience is preferred."
    )
    job = structure_job_description(flat)
    assert len(job["requirements"]) == 3
    assert job["required_qualifications"]
    assert job["preferred_qualifications"]


def test_structured_evidence_fills_keyword_catalog_blind_spot() -> None:
    job = """# Platform Engineer
## Job Description
Responsibilities:
- Build reliable distributed event streaming services.
Requirements:
- Must have experience with Apache Kafka and schema registries.
- Must implement idempotent consumers and monitor message lag.
Preferred Qualifications:
- Experience with stream processing is preferred.
"""
    resume = """# Experience
- Built Apache Kafka consumers with schema validation and idempotent retry handling.
- Monitored consumer lag and production streaming reliability.
"""
    result = analyze_job_structured(job, resume)
    assert result["scoring_method"] == "structured_requirement_evidence_v1"
    assert result["legacy_score"] == 0
    assert result["confidence"]["active_requirement_count"] >= 3
    assert result["analysis_available"] if "analysis_available" in result else True


def test_full_source_structured_takeover_marks_recovered_jd_scoring_ready() -> None:
    job = """# ML Platform Engineer
Source: JSearch
Description Source: full_jd
JD Fetch Status: complete

## Job Description
What you'll be doing • Build training workflows • Deploy inference services
• Evaluate model quality • Maintain data pipelines
Who you probably are • Python • PyTorch • Production ML systems • Monitoring
"""
    resume = """# Experience
- Built Python training and evaluation pipelines with PyTorch.
- Deployed monitored inference services for production ML systems.
"""

    result = analyze_job_structured(job, resume)

    assert result["scoring_method"] == "structured_requirement_evidence_v1"
    assert result["confidence"]["level"] == "high"
    assert result["jd_quality"]["display_label"] == "Scoring-ready"
    assert result["jd_quality"]["reliable_scoring_ready"] is True


def test_structurer_joins_wrapped_paragraphs_and_hides_internal_metadata() -> None:
    source = """# Data Analyst
Company: Example Co
Company Confidence: High
Company Confirmed By User: true

Example Co needs an analyst to support recurring reporting. The analyst
will clean structured datasets and explain findings to stakeholders.

Requirements:
- Python
- SQL

No specific degree is required for this role.
"""

    structured = structure_job_description(source)

    assert "Company Confidence" not in structured["overview"]
    assert "will clean structured datasets" in structured["overview"]
    assert structured["required_qualifications"] == ["Python", "SQL"]
    assert all("degree" not in item.lower() for item in structured["required_qualifications"])


def test_flat_ats_markers_recover_responsibilities_and_candidate_requirements() -> None:
    source = """# Machine Learning Specialist
Source: Jsearch
Description Source: full_jd_api
JD Fetch Status: complete

## Job Description
Technical Lead, Machine Learning The role We turn research into production systems.
What you'll be doing • Owning ML systems end-to-end — data pipelines, training, and inference.
• Building evaluation pipelines for safe and robust models.
Stack Python, PyTorch/JAX, GPU-based training and inference systems.
Who you probably are • You've shipped ML systems real people use.
• You know large models well enough to know how they break.
"""

    structured = structure_job_description(source)

    assert len(structured["requirements"]) >= 5
    assert any("Python" in item for item in structured["required_qualifications"])
    assert any("shipped ML systems" in item for item in structured["required_qualifications"])
    assert any("Owning ML systems" in item for item in structured["responsibilities"])


def test_company_boilerplate_is_not_a_requirement() -> None:
    source = """# Machine Learning Graduate
## Job Description
We help companies connect, protect, analyze, and act on their data at the speed required.
Requirements:
- Python
- Machine learning
"""

    structured = structure_job_description(source)

    assert all("We help companies" not in row["text"] for row in structured["requirements"])
