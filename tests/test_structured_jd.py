"""Contract tests for source-backed JD structuring and pipeline diagnostics."""

from structured_jd import (
    pipeline_diagnostic_message,
    pipeline_trace,
    requirement_records,
    structure_job_description,
)
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


def test_colonless_flat_full_jd_uses_requirement_evidence_instead_of_keyword_score() -> None:
    job = """# AI / Machine Learning Intern
Company: Example Capital
Role: AI / Machine Learning Intern
Location: Singapore
Source: JSearch
Description Source: full_jd_api
JD Fetch Status: complete

## Job Description
We build responsible data products for investment teams.
Responsibilities Working knowledge of AI and machine learning frameworks Collaborate with
engineering teams Implement retrieval workflows Write clean and maintainable code
Requirements Familiarity with orchestration frameworks for multi-agent systems Knowledge of
embeddings, semantic search, and document processing Prior internship or project experience
in production AI applications Pursuing a Bachelor's Degree in Computer Science or related fields
Development knowledge and experience in Python, SQL, Pandas Strong communication skills
Ability to work with minimal supervision Please state your availability clearly in your resume.
"""
    resume = """# Candidate
- Built Python and SQL data pipelines with pandas.
- Evaluated a small machine-learning classifier in an academic project.
"""

    result = analyze_job_structured(job, resume)
    requirements = requirement_records(structure_job_description(job))

    assert len(requirements) >= 8
    assert result["scoring_method"] == "structured_requirement_evidence_v1"
    assert result["score"] <= result["coverage_score"]
    assert result["score"] < result["legacy_score"]
    assert not any(row["text"].lower().startswith("and/or") for row in requirements)


def test_all_missing_evidence_can_never_keep_a_high_keyword_score() -> None:
    job = """# AI Intern
Location: Singapore
Source: JSearch
Description Source: full_jd_api
JD Fetch Status: complete
## Job Description
Requirements:
- Must have production experience with a proprietary orchestration platform.
"""
    resume = "Candidate has unrelated retail operations experience."

    result = analyze_job_structured(job, resume)

    assert result["semantic_evidence"]["accepted_count"] == 0
    assert result["score"] == 0
    assert result["coverage_score"] == 0
    assert result["scoring_method"] == "semantic_evidence_consistency_guard_v1"


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


def test_flat_provider_sections_exclude_company_benefits_and_legal_boilerplate() -> None:
    source = """# Machine Learning Graduate
## Job Description
Who We Are: We help companies connect and act on their data at the speed required.
Job Description: • Design, develop, and implement machine learning models.
• Collaborate with product teams to ship ML systems.
Education and Experience Required: • Must have a Bachelor's degree.
• Possess good knowledge of machine learning techniques.
Additional Skills: Python, SQL, model evaluation.
What We Can Offer You: Health benefits and flexible working hours.
Recruitment Fraud Alert: We never charge candidates a registration fee.
"""

    structured = structure_job_description(source)
    texts = [row["text"] for row in structured["requirements"]]

    assert any(text.startswith("Design, develop, and implement") for text in texts)
    assert not any(text in {"Design", "develop"} for text in texts)
    assert not any("connect and act" in text for text in texts)
    assert not any("Health benefits" in text for text in texts)
    assert not any("registration fee" in text for text in texts)


def test_scoring_contract_prioritizes_at_most_twelve_real_requirements() -> None:
    responsibilities = "\n".join(
        f"- Build production workflow number {index}." for index in range(15)
    )
    source = f"""# Platform Engineer
## Job Description
Responsibilities:
{responsibilities}
Required Qualifications:
- Must have experience with Python.
- Bachelor's degree is required.
Preferred Qualifications:
- Experience with Airflow is preferred.
"""

    records = requirement_records(structure_job_description(source))

    assert len(records) == 12
    assert any("Python" in row["text"] for row in records)
    assert any("Bachelor" in row["text"] for row in records)
    assert any(row["classification"] == "Responsibility" for row in records)


def test_repeated_provider_requirement_suffix_is_not_rescored() -> None:
    repeated = (
        "We provide benefits and flexible working arrangements for everyone in the "
        "company while supporting long term professional development goals."
    )
    source = f"""# Engineer
## Job Description
Benefits:
{repeated}
Requirements:
- Must have experience with Python.

## Requirements
{repeated}
"""

    records = requirement_records(structure_job_description(source))

    assert [row["text"] for row in records] == ["Must have experience with Python"]
