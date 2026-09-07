"""Failure-closed Web shadow integration for the frozen MiniLM candidate."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import threading
from typing import Any

from ml.evidence import (
    build_semantic_evidence_index,
    extract_requirement_records,
    extract_resume_evidence_records,
    score_transparent_evidence_pair,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUNDLE_DIR = (
    PROJECT_ROOT
    / "data"
    / "ml"
    / "distillation"
    / "web_replacement_v1"
    / "web_candidate_minilm_v21_e15"
)
DEFAULT_REQUIREMENT_BUNDLE_DIR = (
    PROJECT_ROOT
    / "data"
    / "ml"
    / "distillation"
    / "minilm_requirement_extractor_v1"
    / "minilm_requirement_extractor_e4_bundle"
)
DEFAULT_SHADOW_ROOT = PROJECT_ROOT / "data" / "ml" / "shadow" / "web_candidate"
CANDIDATE_LIMIT = 4
MAX_REQUIREMENTS = 8
WEB_SHADOW_EXTRACTOR_VERSION = "minilm-two-stage-v3"
WEB_SHADOW_ENABLED_ENV = "JOB_COPILOT_WEB_SHADOW_ENABLED"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _manifest_hash(bundle_dir: Path) -> str:
    manifest = bundle_dir / "manifest.json"
    return _sha256_bytes(manifest.read_bytes()) if manifest.is_file() else "missing"


def web_shadow_enabled() -> bool:
    """Return whether optional background Shadow inference is explicitly enabled."""
    return os.environ.get(WEB_SHADOW_ENABLED_ENV, "").strip().casefold() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _submit_daemon(callback: Callable[[], None]) -> None:
    """Run optional Shadow work outside the user-visible analysis request."""
    threading.Thread(
        target=callback,
        name="job-copilot-web-shadow",
        daemon=True,
    ).start()


def privacy_safe_shadow_report(report: dict[str, Any]) -> dict[str, Any]:
    """Remove requirement, evidence, paths, and exception details before persistence."""
    safe = {
        key: report[key]
        for key in (
            "schema_version",
            "mode",
            "status",
            "reason_code",
            "requirement_count",
            "legacy_requirement_count",
            "evidence_count",
            "candidate_limit",
            "comparison_counts",
            "shadow_prediction_counts",
            "shadow_accepted_count",
            "product_state_modified",
            "user_visible_state_modified",
            "decision_policy",
            "observation_id",
            "created_at",
            "job_text_sha256",
            "resume_text_sha256",
            "extractor_version",
            "requirement_bundle_manifest_sha256",
            "evidence_bundle_manifest_sha256",
        )
        if key in report
    }
    if safe.get("status") == "unavailable":
        safe["reason_code"] = "shadow_runtime_unavailable"
    safe["input_paths_recorded"] = False
    safe["raw_text_persisted"] = False
    safe["content_hashes_only"] = True
    return safe


def _observation_id(
    job_text: str,
    resume_text: str,
    bundle_dir: Path,
    requirement_bundle_dir: Path,
) -> str:
    material = "\0".join(
        [
            _sha256_bytes(job_text.encode("utf-8")),
            _sha256_bytes(resume_text.encode("utf-8")),
            _manifest_hash(bundle_dir),
            _manifest_hash(requirement_bundle_dir),
            WEB_SHADOW_EXTRACTOR_VERSION,
        ]
    )
    return _sha256_bytes(material.encode("utf-8"))


@lru_cache(maxsize=2)
def _load_candidate(bundle_dir_value: str, device: str) -> Any:
    bundle_dir = Path(bundle_dir_value)
    runtime_path = bundle_dir / "runtime.py"
    spec = importlib.util.spec_from_file_location(
        f"job_copilot_web_candidate_{hashlib.sha256(bundle_dir_value.encode()).hexdigest()[:12]}",
        runtime_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError("Candidate runtime could not be loaded.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.WebEvidenceCandidate(bundle_dir, device=device)


@lru_cache(maxsize=2)
def _load_requirement_extractor(bundle_dir_value: str) -> Any:
    bundle_dir = Path(bundle_dir_value)
    runtime_path = bundle_dir / "runtime.py"
    spec = importlib.util.spec_from_file_location(
        f"job_copilot_requirement_extractor_"
        f"{hashlib.sha256(bundle_dir_value.encode()).hexdigest()[:12]}",
        runtime_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError("Requirement extractor runtime could not be loaded.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.RequirementExtractor(bundle_dir)


def _default_device() -> str:
    requested = os.environ.get("JOB_COPILOT_WEB_SHADOW_DEVICE", "cpu").strip().lower()
    return requested if requested in {"cpu", "mps"} else "cpu"


def build_web_candidate_shadow_report(
    job_text: str,
    resume_text: str,
    *,
    candidate: Any,
    requirement_extractor: Any | None = None,
    max_requirements: int = MAX_REQUIREMENTS,
) -> dict[str, Any]:
    """Compare the product baseline and candidate without changing either output."""
    # The comparison module imports optional ML dependencies. Load it only when
    # an opted-in shadow report is actually built, not during base Personal use.
    from ml.evidence_shadow import comparison_status

    legacy_requirements = extract_requirement_records(job_text)[:max_requirements]
    requirements: list[dict[str, Any]]
    if requirement_extractor is None:
        requirements = [dict(row) for row in legacy_requirements]
        extraction = {
            "available": True,
            "requirements": legacy_requirements,
            "model_version": "legacy-rule-extractor",
        }
    else:
        extraction = requirement_extractor.extract_safe(job_text)
        if not extraction.get("available"):
            raise RuntimeError(str(extraction.get("reason", "Requirement extraction failed.")))
        requirements = [
            {
                "text": str(row["text"]),
                "demand": (
                    "preferred"
                    if row.get("classification") == "Preferred"
                    else "required"
                ),
                "classification": str(row.get("classification", "Required")),
                "section": str(row.get("section", "")),
                "probability": float(row.get("probability", 0.0)),
            }
            for row in extraction.get("requirements", [])
        ][:max_requirements]
    evidence_records = extract_resume_evidence_records(resume_text)
    baseline = build_semantic_evidence_index(
        job_text,
        resume_text,
        max_requirements=max_requirements,
    )
    baseline_by_requirement = {
        str(match["requirement"]): match for match in baseline["matches"]
    }
    comparisons: list[dict[str, Any]] = []
    for requirement in requirements:
        requirement_text = str(requirement["text"])
        ordered = sorted(
            evidence_records,
            key=lambda row: (
                -float(
                    score_transparent_evidence_pair(
                        requirement_text, str(row["text"])
                    )["similarity"]
                ),
                int(row["line_index"]),
            ),
        )
        shortlist: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row in ordered:
            text = str(row["text"])
            if text not in seen:
                shortlist.append(row)
                seen.add(text)
            if len(shortlist) == CANDIDATE_LIMIT:
                break
        prediction = candidate.predict_task_safe(
            requirement_text,
            [str(row["text"]) for row in shortlist],
        )
        baseline_match = baseline_by_requirement.get(requirement_text)
        shadow_accepted = bool(prediction.get("available") and prediction.get("accepted"))
        shadow_evidence = str(prediction.get("strongest_evidence", "")) if shadow_accepted else ""
        baseline_evidence = str(baseline_match.get("evidence", "")) if baseline_match else ""
        baseline_accepted = bool(baseline_match and baseline_match["accepted"])
        comparisons.append(
            {
                "requirement": requirement_text,
                "demand": str(requirement["demand"]),
                "baseline": {
                    "requirement_found": baseline_match is not None,
                    "accepted": baseline_accepted,
                    "evidence": baseline_evidence,
                    "similarity": (
                        float(baseline_match["similarity"]) if baseline_match else 0.0
                    ),
                    "match_type": (
                        str(baseline_match["match_type"])
                        if baseline_match
                        else "requirement_not_extracted"
                    ),
                },
                "shadow": prediction,
                "comparison": (
                    comparison_status(
                        baseline_accepted=baseline_accepted,
                        shadow_accepted=shadow_accepted,
                        baseline_evidence=baseline_evidence,
                        shadow_evidence=shadow_evidence,
                    )
                    if baseline_match
                    else "baseline_missing_requirement"
                ),
                "shortlisted_candidate_count": len(shortlist),
            }
        )
    candidate_segments = extraction.get("candidate_segments", 0)
    return {
        "schema_version": 1,
        "mode": "web_shadow_only",
        "requirement_count": len(requirements),
        "legacy_requirement_count": len(legacy_requirements),
        "requirement_extractor": {
            "available": bool(extraction.get("available")),
            "model_version": str(extraction.get("model_version", "unknown")),
            "candidate_segments": (
                int(candidate_segments)
                if isinstance(candidate_segments, (int, float, str))
                else 0
            ),
        },
        "evidence_count": len(evidence_records),
        "candidate_limit": CANDIDATE_LIMIT,
        "comparisons": comparisons,
        "comparison_counts": dict(Counter(row["comparison"] for row in comparisons)),
        "shadow_prediction_counts": dict(
            Counter(str(row["shadow"].get("prediction", "unknown")) for row in comparisons)
        ),
        "shadow_accepted_count": sum(
            bool(row["shadow"].get("available") and row["shadow"].get("accepted"))
            for row in comparisons
        ),
        "product_state_modified": False,
        "user_visible_state_modified": False,
        "decision_policy": (
            "Diagnostic only: this report cannot affect Role Fit, evidence selection, "
            "Cover Letter content, or recommendations."
        ),
    }


def _record_web_candidate_shadow(
    job_text: str,
    resume_text: str,
    *,
    workspace_mode: str,
    bundle_dir: Path = DEFAULT_BUNDLE_DIR,
    requirement_bundle_dir: Path = DEFAULT_REQUIREMENT_BUNDLE_DIR,
    shadow_root: Path = DEFAULT_SHADOW_ROOT,
    candidate_loader: Callable[[str, str], Any] = _load_candidate,
    requirement_extractor_loader: Callable[[str], Any] = _load_requirement_extractor,
) -> Path | None:
    """Implement one content-addressed personal-Web observation."""
    if (
        workspace_mode != "personal"
        or not job_text.strip()
        or not resume_text.strip()
    ):
        return None
    observation_id = _observation_id(
        job_text, resume_text, bundle_dir, requirement_bundle_dir
    )
    output = shadow_root / f"{observation_id}.json"
    if output.is_file():
        return output
    try:
        candidate = candidate_loader(str(bundle_dir.resolve()), _default_device())
        requirement_extractor = requirement_extractor_loader(
            str(requirement_bundle_dir.resolve())
        )
        report = build_web_candidate_shadow_report(
            job_text,
            resume_text,
            candidate=candidate,
            requirement_extractor=requirement_extractor,
        )
        report["status"] = "recorded"
    except Exception as error:  # noqa: BLE001 - optional shadow must fail closed
        report = {
            "schema_version": 1,
            "mode": "web_shadow_only",
            "status": "unavailable",
            "reason_code": "shadow_runtime_unavailable",
            "comparisons": [],
            "comparison_counts": {},
            "product_state_modified": False,
            "user_visible_state_modified": False,
        }
        _ = error
    report = privacy_safe_shadow_report(report)
    report.update(
        {
            "observation_id": observation_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "job_text_sha256": _sha256_bytes(job_text.encode("utf-8")),
            "resume_text_sha256": _sha256_bytes(resume_text.encode("utf-8")),
            "extractor_version": WEB_SHADOW_EXTRACTOR_VERSION,
            "requirement_bundle_manifest_sha256": _manifest_hash(
                requirement_bundle_dir
            ),
            "evidence_bundle_manifest_sha256": _manifest_hash(bundle_dir),
        }
    )
    report = privacy_safe_shadow_report(report)
    shadow_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    payload = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
    try:
        descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return output
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
    return output


def record_web_candidate_shadow(
    job_text: str,
    resume_text: str,
    *,
    workspace_mode: str,
    bundle_dir: Path = DEFAULT_BUNDLE_DIR,
    requirement_bundle_dir: Path = DEFAULT_REQUIREMENT_BUNDLE_DIR,
    shadow_root: Path = DEFAULT_SHADOW_ROOT,
    candidate_loader: Callable[[str, str], Any] = _load_candidate,
    requirement_extractor_loader: Callable[[str], Any] = _load_requirement_extractor,
) -> Path | None:
    """Record a Web observation, failing closed before errors reach the product path."""
    try:
        return _record_web_candidate_shadow(
            job_text,
            resume_text,
            workspace_mode=workspace_mode,
            bundle_dir=bundle_dir,
            requirement_bundle_dir=requirement_bundle_dir,
            shadow_root=shadow_root,
            candidate_loader=candidate_loader,
            requirement_extractor_loader=requirement_extractor_loader,
        )
    except Exception:  # noqa: BLE001 - shadow telemetry can never break Web analysis
        return None


def enqueue_web_candidate_shadow(
    job_text: str,
    resume_text: str,
    *,
    workspace_mode: str,
    recorder: Callable[..., Path | None] = record_web_candidate_shadow,
    submitter: Callable[[Callable[[], None]], None] = _submit_daemon,
) -> bool:
    """Schedule opt-in Shadow work without blocking the product analysis response."""
    if workspace_mode != "personal" or not web_shadow_enabled():
        return False

    def run() -> None:
        recorder(job_text, resume_text, workspace_mode=workspace_mode)

    try:
        submitter(run)
    except Exception:  # noqa: BLE001 - scheduling can never break product analysis
        return False
    return True


def summarize_web_candidate_shadow_reports(
    report_paths: list[Path],
    *,
    discovered_jobs: int,
    parseable_jobs: int,
) -> dict[str, Any]:
    """Aggregate private shadow observations without assigning correctness labels."""
    comparison_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    requirement_count = 0
    legacy_requirement_count = 0
    semantic_only_observations = 0
    shadow_accepted_count = 0
    shadow_prediction_counts: Counter[str] = Counter()
    usable_observations = 0
    seen_observations: set[str] = set()
    for path in report_paths:
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            status_counts["unreadable"] += 1
            continue
        observation_id = str(report.get("observation_id", ""))
        if not observation_id or observation_id in seen_observations:
            continue
        seen_observations.add(observation_id)
        status = str(report.get("status", "unknown"))
        status_counts[status] += 1
        if status != "recorded":
            continue
        usable_observations += 1
        requirement_count += int(report.get("requirement_count", 0) or 0)
        legacy_count = int(report.get("legacy_requirement_count", 0) or 0)
        legacy_requirement_count += legacy_count
        semantic_only_observations += legacy_count == 0 and int(
            report.get("requirement_count", 0) or 0
        ) > 0
        shadow_accepted_count += int(report.get("shadow_accepted_count", 0) or 0)
        predictions = report.get("shadow_prediction_counts", {})
        if isinstance(predictions, dict):
            shadow_prediction_counts.update(
                {str(key): int(value) for key, value in predictions.items()}
            )
        counts = report.get("comparison_counts", {})
        if isinstance(counts, dict):
            comparison_counts.update(
                {str(key): int(value) for key, value in counts.items()}
            )
    disagreements = sum(
        count
        for status, count in comparison_counts.items()
        if status not in {"both_accept_same_evidence", "both_reject"}
    )
    return {
        "schema_version": 1,
        "mode": "web_shadow_summary_only",
        "discovered_jobs": discovered_jobs,
        "parseable_jobs": parseable_jobs,
        "unparseable_jobs": max(0, discovered_jobs - parseable_jobs),
        "usable_observations": usable_observations,
        "requirement_count": requirement_count,
        "legacy_requirement_count": legacy_requirement_count,
        "semantic_only_observations": semantic_only_observations,
        "shadow_accepted_count": shadow_accepted_count,
        "shadow_acceptance_rate": (
            shadow_accepted_count / requirement_count if requirement_count else 0.0
        ),
        "shadow_prediction_counts": dict(shadow_prediction_counts),
        "status_counts": dict(status_counts),
        "comparison_counts": dict(comparison_counts),
        "disagreement_count": disagreements,
        "accuracy_claimed": False,
        "teacher_or_human_labels_used": False,
        "product_state_modified": False,
        "user_visible_state_modified": False,
    }
