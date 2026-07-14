"""Explicit eligibility and hard-constraint checks for job-fit analysis."""

from __future__ import annotations

import re

from scoring_config import DEGREE_RANK
from scoring_extraction import (
    contains_alias,
    infer_candidate_experience_profile,
    is_preferred_line,
    normalize_text,
    split_job_description_lines,
)


def _eligibility_reason(code: str, message: str, evidence: str) -> dict[str, str]:
    return {"code": code, "message": message, "evidence": evidence}


def _required_experience_years(job_text: str) -> tuple[float | None, str]:
    """Extract an explicit required minimum, excluding preferred wording."""
    values: list[tuple[float, str]] = []
    for line in split_job_description_lines(job_text):
        if is_preferred_line(line):
            continue
        normalized = normalize_text(line)
        match = re.search(r"\b(\d+(?:\.\d+)?)\+?\s*(?:years|yrs)\b", normalized)
        if not match:
            continue
        is_required = bool(
            "+" in line
            or re.search(r"\b(?:required|requires?|must|minimum|at least)\b", normalized)
        )
        if is_required:
            values.append((float(match.group(1)), line[:180]))
    return max(values, default=(None, ""), key=lambda item: item[0])


def _required_degree(job_text: str) -> tuple[str | None, bool, str]:
    """Extract a required graduate degree and equivalent-experience qualifier."""
    for line in split_job_description_lines(job_text):
        if is_preferred_line(line):
            continue
        normalized = normalize_text(line)
        degree = None
        if re.search(r"\b(?:ph\s*d|doctorate|doctoral degree)\b", normalized):
            degree = "phd"
        elif re.search(r"\b(?:master\s+s|masters|master of|m\s+s)\b", normalized):
            degree = "master"
        if degree and re.search(r"\b(?:required|requires?|must|minimum)\b", normalized):
            equivalent = "equivalent" in normalized and "experience" in normalized
            return degree, equivalent, line[:180]
    return None, False, ""


def _hard_seniority_requirement(job_text: str) -> tuple[bool, str]:
    """Detect title/requirement seniority without matching incidental wording."""
    for raw_line in split_job_description_lines(job_text):
        line = raw_line.strip("#*: -")
        if re.search(r"^(?:role|title)\s*:\s*", line, re.I):
            line = line.split(":", 1)[1].strip()
        if re.search(r"^(?:senior|staff|principal)\s+[A-Za-z]", line, re.I):
            return True, raw_line[:180]
        if re.search(r"^(?:engineering|product|data|machine learning) manager\b", line, re.I):
            return True, raw_line[:180]
        if re.search(r"\b5\+?\s*years\s+in\s+a\s+senior\s+role\b", line, re.I):
            return True, raw_line[:180]
    return False, ""


def evaluate_eligibility(
    job_text: str,
    candidate_text: str,
    candidate_profile: dict[str, object] | None = None,
) -> dict[str, object]:
    """Evaluate explicit gating constraints separately from role-fit scoring."""
    profile = candidate_profile or infer_candidate_experience_profile(candidate_text)
    reasons: list[dict[str, str]] = []
    failed = False
    manual_review = False
    level = str(profile.get("career_level", "unknown"))
    years = profile.get("years_experience")

    required_years, years_evidence = _required_experience_years(job_text)
    if required_years is not None:
        if isinstance(years, (int, float)):
            if float(years) < required_years:
                failed = True
                reasons.append(
                    _eligibility_reason(
                        "minimum_experience",
                        f"The role requires {required_years:g}+ years, but the candidate source states {float(years):g}.",
                        years_evidence,
                    )
                )
        elif level in {"student", "new_grad"} and required_years >= 3:
            failed = True
            reasons.append(
                _eligibility_reason(
                    "minimum_experience",
                    f"The role requires {required_years:g}+ years and the candidate is explicitly identified as {level.replace('_', ' ')}.",
                    years_evidence,
                )
            )
        else:
            manual_review = True
            reasons.append(
                _eligibility_reason(
                    "minimum_experience",
                    f"Confirm the explicit {required_years:g}+ year requirement against candidate experience.",
                    years_evidence,
                )
            )

    required_degree, equivalent_allowed, degree_evidence = _required_degree(job_text)
    if required_degree:
        candidate_degree = str(profile.get("highest_degree", "unknown"))
        if equivalent_allowed and DEGREE_RANK.get(candidate_degree, 0) < DEGREE_RANK[required_degree]:
            manual_review = True
            reasons.append(
                _eligibility_reason(
                    "graduate_degree_equivalent",
                    "The graduate-degree requirement allows equivalent experience; confirm equivalence manually.",
                    degree_evidence,
                )
            )
        elif candidate_degree == "unknown":
            manual_review = True
            reasons.append(
                _eligibility_reason(
                    "graduate_degree",
                    f"Confirm the required {required_degree} degree against candidate education.",
                    degree_evidence,
                )
            )
        elif DEGREE_RANK[candidate_degree] < DEGREE_RANK[required_degree]:
            failed = True
            reasons.append(
                _eligibility_reason(
                    "graduate_degree",
                    f"The role requires a {required_degree} degree; the candidate source explicitly shows {candidate_degree} as highest degree.",
                    degree_evidence,
                )
            )

    seniority_required, seniority_evidence = _hard_seniority_requirement(job_text)
    if seniority_required:
        if level in {"student", "new_grad", "junior"}:
            failed = True
            reasons.append(
                _eligibility_reason(
                    "seniority_requirement",
                    f"The role is explicitly senior-level while candidate evidence indicates {level.replace('_', ' ')}.",
                    seniority_evidence,
                )
            )
        elif level == "unknown":
            manual_review = True
            reasons.append(
                _eligibility_reason(
                    "seniority_requirement",
                    "Confirm the explicit seniority requirement against candidate experience.",
                    seniority_evidence,
                )
            )

    normalized_job = normalize_text(job_text)
    normalized_candidate = normalize_text(candidate_text)
    authorization_mentioned = any(
        contains_alias(normalized_job, phrase)
        for phrase in ["work authorization", "right to work", "visa sponsorship", "sponsorship", "citizenship"]
    )
    existing_authorization_required = any(
        contains_alias(normalized_job, phrase)
        for phrase in [
            "must be authorized",
            "must have the right to work",
            "currently have the right to work",
            "current right to work",
            "sponsorship is not available",
            "no visa sponsorship",
        ]
    )
    candidate_incompatible = any(
        contains_alias(normalized_candidate, phrase)
        for phrase in ["requires visa sponsorship", "require visa sponsorship", "not authorized to work"]
    )
    candidate_confirmed = any(
        contains_alias(normalized_candidate, phrase)
        for phrase in ["authorized to work", "right to work", "citizen", "permanent resident"]
    )
    if authorization_mentioned and not candidate_confirmed:
        if existing_authorization_required and candidate_incompatible:
            failed = True
            reasons.append(
                _eligibility_reason(
                    "work_authorization",
                    "The role requires existing authorization and the candidate source explicitly states an incompatible status.",
                    "Explicit authorization requirement in job text.",
                )
            )
        else:
            manual_review = True
            reasons.append(
                _eligibility_reason(
                    "work_authorization",
                    "Work authorization or sponsorship must be confirmed; no compatible status is assumed.",
                    "Authorization or sponsorship language appears in the job text.",
                )
            )

    status = "failed" if failed else "manual_review" if manual_review else "passed"
    return {"status": status, "reasons": reasons}
