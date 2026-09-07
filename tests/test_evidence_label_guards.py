from pathlib import Path

import pytest

from ml.evidence import score_transparent_evidence_pair, build_semantic_evidence_index


@pytest.mark.parametrize("requirement,evidence", [
    ("PhD or Master's in Machine Learning required", "Bachelor of Science in Machine Learning and Neural Computation"),
    ("Five years of Python experience", "Built Python applications for two years"),
    ("Master's degree in Computer Science", "Pursuing a Master's degree in Computer Science"),
    ("Experience with Python", "I have no experience with Python"),
])
def test_unsupported_constraints_are_not_accepted(requirement, evidence):
    assert not score_transparent_evidence_pair(requirement, evidence)["accepted"]


@pytest.mark.parametrize("requirement,evidence", [
    ("Develop machine learning models", "Bachelor of Science in Machine Learning and Neural Computation"),
    ("Python and SQL", "Built Python applications for data analysis"),
    ("Deploy machine learning models", "Coursework included machine learning models and deployment"),
    ("Develop machine learning models", "Delivered instructional sessions on machine learning models"),
])
def test_related_knowledge_is_never_direct(requirement, evidence):
    result = score_transparent_evidence_pair(requirement, evidence)
    assert result["match_type"] != "Direct support"
    assert result["label_reasons"]


def test_degree_alternatives_accept_explicit_masters():
    result = score_transparent_evidence_pair("PhD or Master's in Computer Science", "Master's in Computer Science")
    assert result["match_type"] == "Direct support"


def test_real_application_stays_direct():
    assert score_transparent_evidence_pair("Python and SQL", "Built Python and SQL applications for reporting")["match_type"] == "Direct support"


def test_index_selects_qualified_evidence_and_keeps_provenance():
    result = build_semantic_evidence_index(
        "## Requirements\n- Develop machine learning models",
        "## Education\n- Bachelor of Science in machine learning models\n\n## Projects\n- Developed machine learning models for image classification.",
        model_path=Path("/missing/model.json"),
    )
    assert result["matches"][0]["match_type"] == "Direct support"
    assert result["matches"][0]["section_evidence"] == "Projects"
