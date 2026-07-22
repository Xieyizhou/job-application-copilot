"""Typed payload contracts shared by scoring and dashboard presentation code."""

from __future__ import annotations

from typing import Any, Literal, NotRequired, TypedDict


class ScoreCategoryConfig(TypedDict):
    points: int
    keywords: list[str]


class PenaltyRule(TypedDict):
    name: str
    points: int
    patterns: list[str]


class RoleFocusRule(TypedDict):
    name: str
    title_patterns: list[str]
    candidate_aliases: list[str]


class CandidateProfile(TypedDict):
    career_level: str
    years_experience: float | None
    highest_degree: str
    evidence: list[str]


class ParsedJob(TypedDict):
    required_skills: list[str]
    preferred_skills: list[str]
    experience_level: list[str]
    degree_requirements: list[str]
    domain_keywords: list[str]
    red_flags: list[str]


class ScoreBreakdownItem(TypedDict):
    category: str
    earned: float | None
    possible: int
    active_terms: list[str]
    matched: list[str]
    partial: list[str]
    missing: list[str]
    note: str


class Penalty(TypedDict):
    name: str
    points: int


class EligibilityReason(TypedDict):
    code: str
    message: str
    evidence: str


class EligibilityResult(TypedDict):
    status: Literal["passed", "manual_review", "failed"]
    reasons: list[EligibilityReason]


class RoleAlignment(TypedDict):
    detected: bool
    focus: str
    title: str
    score: int | None
    matched_evidence: list[str]


class RoleFocusAdjustment(TypedDict):
    applied: bool
    coverage_score: int
    adjusted_score: int
    role_alignment: RoleAlignment


class ScoreCalibration(TypedDict):
    applied: bool
    observed_score: int
    calibrated_score: int
    active_requirement_count: int
    role_signal_count: int
    effective_evidence_count: int
    evidence_target: int
    evidence_factor: float
    reason: str


class ScoringConfidence(TypedDict):
    level: Literal["low", "medium", "high"]
    active_requirement_count: int
    candidate_evidence_count: int
    job_description_quality: dict[str, Any]
    reasons: list[str]
    role_alignment: NotRequired[RoleAlignment]
    coverage_score: NotRequired[int]
    observed_score: NotRequired[int]
    score_calibration: NotRequired[ScoreCalibration]


class ScoringResult(TypedDict):
    score: int
    coverage_score: int
    observed_score: int
    role_alignment: RoleAlignment
    role_focus_adjustment: RoleFocusAdjustment
    score_calibration: ScoreCalibration
    recommendation: str
    score_breakdown: list[ScoreBreakdownItem]
    eligibility: EligibilityResult
    confidence: ScoringConfidence
    candidate_profile: CandidateProfile
    parsed_job: ParsedJob
    job_keywords: list[str]
    resume_keywords: list[str]
    penalties: list[Penalty]


class StructuredAnalysis(ScoringResult):
    analysis_available: NotRequired[bool]
    matched_skills: list[str]
    partial_matches: list[str]
    missing_skills: list[str]
    main_reason: str
    main_risk: str
    matched_strengths: list[str]
    weak_areas: list[str]
    matched_keywords: list[str]
    missing_keywords: list[str]
    optional_keywords: list[str]
    resume_suggestions: list[str]
    jd_evidence: list[str]
    profile_evidence: list[str]
    semantic_evidence: dict[str, Any]
    jd_quality: dict[str, Any]
    raw_analysis: str


class DashboardJob(TypedDict, total=False):
    path: Any
    company: str
    role: str
    title: str
    location: str
    source: str
    url: str
    label: str
    analysis_result: dict[str, Any]
    analysis: dict[str, Any]
    analysis_available: bool
    jd_quality: dict[str, Any]
    confidence: dict[str, Any]
    eligibility: dict[str, Any]
    score_breakdown: list[dict[str, Any]]
    score: int | None
    legacy_score: int | None
    legacy_recommendation: str
    match_score: int | None
    recommendation: str
    status: str
    package_status: str
    tracker_status: str
    ml_relevance: dict[str, Any]
    last_seen_at: str
    first_seen_at: str
    high_level_region: str
    jd_fetch_status: str
    description_word_count: int
    company_needs_review: bool


class RequirementSummary(TypedDict):
    matched_required: list[str]
    matched_preferred: list[str]
    partial_required: list[str]
    partial_preferred: list[str]
    missing_required: list[str]
    missing_preferred: list[str]
    active_requirement_count: int
    matched_requirement_count: int


class FitPresentation(TypedDict):
    analysis_available: bool
    role_fit: str
    card_status: str
    score: int | None
    recommendation: str
    eligibility: dict[str, Any]
    confidence: dict[str, Any]
    terms: RequirementSummary
    coverage_score: int | None


class RegionOption(TypedDict):
    key: str
    label: str
    type: str
    value: str
    count: int


class TrackerRow(TypedDict, total=False):
    id: int
    company: str
    role: str
    location: str
    job_url: str
    match_score: int | None
    recommendation: str
    status: str
    resume_file: str
    cover_letter_file: str
    notes: str
    created_at: str
    applied_date: str | None
