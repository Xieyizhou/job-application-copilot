"""Tests for auditable evidence constraint features."""

from __future__ import annotations

from ml.evidence_constraints import (
    constraint_features,
    hard_constraints_supported,
    named_terms,
)


def test_named_tools_distinguish_adjacent_but_unsupported_evidence() -> None:
    features = constraint_features(
        "Working experience with Spring Boot and Spring Data",
        "Implemented an ETL process using Airflow and BigQuery.",
    )

    assert {"spring", "spring boot"} <= named_terms(
        "Working experience with Spring Boot and Spring Data"
    )
    assert features["named_term_coverage"] == 0.0
    assert features["missing_named_term"] == 1.0
    assert features["named_constraint_supported"] == 0.0


def test_product_analytics_tools_are_reviewable_named_terms() -> None:
    requirement = named_terms(
        "Experience with Amplitude, Appsflyer, Firebase, or Looker."
    )
    evidence = named_terms("Built recurring dashboards in Looker Studio.")

    assert {"amplitude", "appsflyer", "firebase", "looker"} <= requirement
    assert {"looker", "looker studio"} <= evidence
    assert requirement & evidence == {"looker"}


def test_named_term_at_sentence_end_is_not_lost_to_punctuation() -> None:
    assert named_terms("Built production services with Python.") == {"python"}


def test_acronym_requirement_needs_matching_evidence_signal() -> None:
    unsupported = constraint_features(
        "Use an ATS to manage candidate interactions",
        "Worked with product marketing on brand messaging.",
    )
    supported = constraint_features(
        "Use an ATS to manage candidate interactions",
        "Managed recruiting workflows and candidate records in an ATS.",
    )

    assert unsupported["named_term_coverage"] == 0.0
    assert supported["named_term_coverage"] == 1.0
    assert unsupported["named_constraint_supported"] == 0.0
    assert supported["named_constraint_supported"] == 1.0


def test_adjacent_named_tools_can_support_the_same_skill_family() -> None:
    vision = constraint_features(
        "Experience with ML and image processing",
        "Implemented computer vision research in a production workflow.",
    )
    mobile = constraint_features(
        "Native mobile development on Android or iOS",
        "Developed an application using SwiftUI.",
    )

    assert vision["named_constraint_supported"] == 1.0
    assert mobile["named_constraint_supported"] == 1.0


def test_explicit_year_and_degree_constraints_remain_visible_for_partial_support() -> None:
    features = constraint_features(
        "Master's degree and at least 5 years of Python experience",
        "Built Python services for three years after a bachelor's degree.",
    )

    assert features["numeric_constraint_supported"] == 0.0
    assert features["degree_constraint_supported"] == 0.0
    assert hard_constraints_supported(features) is True


def test_negated_delivery_is_not_accepted_as_support() -> None:
    features = constraint_features(
        "Deploy machine learning models to production",
        "Trained several models but did not deploy them to production.",
    )

    assert features["explicit_negation"] == 1.0
    assert hard_constraints_supported(features) is False


def test_unrelated_concept_is_a_hard_support_failure() -> None:
    features = constraint_features(
        "Apply software development lifecycle expertise",
        "Designed a database for monthly statistics.",
    )

    assert features["concept_constraint_supported"] == 0.0
    assert hard_constraints_supported(features) is False
