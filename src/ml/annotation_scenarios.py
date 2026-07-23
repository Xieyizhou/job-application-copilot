"""Build varied fictional scenarios for annotation calibration."""

from __future__ import annotations

import hashlib
import random
from typing import Any


ROLE_CONCEPTS = {
    "Data": [
        ("SQL", "Excel", "Java"),
        ("Python data analysis", "R analysis", "sales negotiation"),
        ("Tableau", "Power BI", "backend APIs"),
        ("Power BI", "Tableau", "unit testing"),
        ("ETL pipelines", "data cleaning", "content marketing"),
        ("A/B testing", "survey analysis", "Docker"),
        ("statistical modeling", "descriptive statistics", "CRM administration"),
        ("Pandas", "Excel formulas", "JavaScript"),
        ("data validation", "quality reporting", "product roadmaps"),
        ("analytics dashboards", "spreadsheet reports", "Kubernetes"),
        ("data modeling", "database reporting", "customer support"),
        ("forecasting", "historical reporting", "REST APIs"),
    ],
    "ML": [
        ("PyTorch", "TensorFlow", "financial reporting"),
        ("TensorFlow", "PyTorch", "CRM administration"),
        ("scikit-learn", "statistical modeling", "frontend styling"),
        ("feature engineering", "data cleaning", "sales forecasting"),
        ("model deployment", "REST API development", "market research"),
        ("MLOps", "CI/CD", "customer onboarding"),
        ("natural language processing", "text analytics", "budget planning"),
        ("computer vision", "image processing", "stakeholder interviews"),
        ("model monitoring", "dashboard monitoring", "content writing"),
        ("experiment tracking", "project documentation", "account management"),
        ("ML data pipelines", "ETL pipelines", "roadmap planning"),
        ("ML experiment design", "A/B testing", "Java backend services"),
    ],
    "Software": [
        ("Java", "Python", "financial modeling"),
        ("REST APIs", "GraphQL APIs", "customer segmentation"),
        ("Docker", "virtual machines", "market research"),
        ("Git", "document versioning", "forecasting"),
        ("relational databases", "spreadsheet storage", "content strategy"),
        ("unit testing", "manual testing", "sales operations"),
        ("CI/CD", "release checklists", "stakeholder interviews"),
        ("JavaScript", "HTML and CSS", "statistical modeling"),
        ("backend services", "internal automation", "brand marketing"),
        ("cloud deployment", "on-premise deployment", "CRM reporting"),
        ("system design", "process mapping", "survey research"),
        ("Kubernetes", "Docker Compose", "financial reconciliation"),
    ],
    "Business": [
        ("Intercom", "Zendesk", "PyTorch"),
        ("Salesforce", "HubSpot", "computer vision"),
        ("stakeholder interviews", "customer surveys", "Docker"),
        ("process mapping", "workflow documentation", "TensorFlow"),
        ("KPI reporting", "operational reporting", "Java services"),
        ("financial modeling", "budget reporting", "frontend development"),
        ("market research", "customer interviews", "Kubernetes"),
        ("CRM administration", "customer support tools", "ETL pipelines"),
        ("requirements gathering", "meeting facilitation", "model deployment"),
        ("product roadmaps", "project planning", "SQL optimization"),
        ("Excel forecasting", "spreadsheet reporting", "REST APIs"),
        ("customer segmentation", "audience research", "unit testing"),
    ],
}

ROLE_TITLES = {
    "Data": "Data Analyst",
    "ML": "Machine Learning Engineer",
    "Software": "Software Engineer",
    "Business": "Business Analyst",
}

REQUIREMENT_TEMPLATES = (
    "The role calls for hands-on experience with {skill}.",
    "Candidates should be able to apply {skill} in practical work.",
    "Demonstrated capability in {skill} is expected.",
    "This position requires working knowledge of {skill}.",
    "Success in this role depends on using {skill} effectively.",
    "Practical familiarity with {skill} is essential for this position.",
)

DIRECT_EVIDENCE_TEMPLATES = (
    "Implemented a project workflow using {skill}.",
    "Applied {skill} while delivering a documented project.",
    "Used {skill} to complete a practical assignment.",
    "Built and evaluated work that relied on {skill}.",
    "The project record describes hands-on use of {skill}.",
    "Delivered a working project with {skill} as a core tool.",
)

STRATA = ("direct", "adjacent", "mention_only", "compound_partial", "unsupported")


def _render(template: str, skill: str) -> str:
    return template.format(skill=skill)


def _direct_evidence(skill: str, index: int) -> str:
    return _render(DIRECT_EVIDENCE_TEMPLATES[index % len(DIRECT_EVIDENCE_TEMPLATES)], skill)


def _generic_evidence(skill: str, index: int) -> str:
    templates = (
        "Completed related work using {skill}.",
        "Supported a team project that involved {skill}.",
        "Produced a documented assignment with {skill}.",
        "Maintained a workflow based on {skill}.",
    )
    return _render(templates[index % len(templates)], skill)


def _scenario_text(
    primary: str,
    adjacent: str,
    unrelated: str,
    stratum: str,
    index: int,
) -> tuple[str, list[str]]:
    template = REQUIREMENT_TEMPLATES[index % len(REQUIREMENT_TEMPLATES)]
    if stratum == "compound_partial":
        requirement = (
            f"The role requires practical use of both {primary} and {adjacent} in the same workflow."
        )
    else:
        requirement = _render(template, primary)
    if stratum == "direct":
        evidence = [
            _direct_evidence(primary, index),
            _generic_evidence(adjacent, index + 1),
            f"Reviewed introductory material about {primary} without using it in project work.",
            _generic_evidence(unrelated, index + 2),
        ]
    elif stratum == "adjacent":
        evidence = [
            _generic_evidence(adjacent, index),
            f"Observed colleagues use {primary}, but did not perform the work directly.",
            _generic_evidence(unrelated, index + 1),
            "Prepared general project notes and status updates.",
        ]
    elif stratum == "mention_only":
        evidence = [
            f"Read documentation about {primary} without completing hands-on work.",
            f"Listed {primary} as a future learning goal.",
            _generic_evidence(adjacent, index + 1),
            _generic_evidence(unrelated, index + 2),
        ]
    elif stratum == "compound_partial":
        evidence = [
            _direct_evidence(primary, index),
            f"Read documentation about {adjacent} without applying it.",
            _generic_evidence(unrelated, index + 1),
            "Coordinated timelines and maintained project documentation.",
        ]
    else:
        evidence = [
            _generic_evidence(unrelated, index),
            "Coordinated routine project updates with stakeholders.",
            "Maintained documentation for an unrelated operational process.",
            f"Planned to study {primary} in the future but had not started.",
        ]
    return requirement, evidence


def fictional_challenge_records(*, random_state: int = 42) -> list[dict[str, Any]]:
    """Return balanced, varied records whose labels still require human judgment."""
    records: list[dict[str, Any]] = []
    for family, concepts in ROLE_CONCEPTS.items():
        for index, (primary, adjacent, unrelated) in enumerate(concepts):
            stratum = STRATA[index % len(STRATA)]
            requirement, evidence = _scenario_text(
                primary,
                adjacent,
                unrelated,
                stratum,
                index,
            )
            identity = f"v2:{family}:{index}:{requirement}"
            records.append(
                {
                    "resume_text": ". ".join(evidence),
                    "job_text": f"{ROLE_TITLES[family]}. {requirement}",
                    "resume_hash": hashlib.sha256(f"{identity}:resume".encode()).hexdigest(),
                    "job_hash": hashlib.sha256(f"{identity}:job".encode()).hexdigest(),
                    "label": "Unreviewed",
                    "source_dataset": "local_fictional_challenges_v2",
                    "requirement_sentences": [requirement],
                    "evidence_sentences": evidence,
                    "preferred_evidence": evidence[0],
                    "sampling_stratum": stratum,
                    "template_family": f"requirement_{index % len(REQUIREMENT_TEMPLATES)}",
                }
            )
    random.Random(random_state).shuffle(records)
    return records
