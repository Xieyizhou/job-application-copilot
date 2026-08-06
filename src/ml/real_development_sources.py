"""Load eligible requirements and resume statements from public real-text data."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
from typing import Any

import pandas as pd

from ml.annotation_generation import infer_role_family as infer_annotation_role_family
from ml.evidence import extract_requirement_records
from ml.real_development import (
    LEGACY_EVIDENCE_EXTRACTION,
    ROLE_FAMILIES,
    extract_action_evidence,
    infer_role_family,
    source_hash,
    validate_evidence_extraction_policy,
)
from ml.source_text_quality import (
    LEGACY_SOURCE_TEXT_QUALITY,
    SUCCESSOR_V5_SOURCE_TEXT_QUALITY,
    SUCCESSOR_V6_SOURCE_TEXT_QUALITY,
    SUCCESSOR_V7_SOURCE_TEXT_QUALITY,
    SUCCESSOR_V8_SOURCE_TEXT_QUALITY,
    canonical_public_text,
    diverse_quality_evidence,
    requirement_quality_reasons_for_policy,
    validate_source_text_quality_policy,
)

REQUIREMENT_SIGNAL = (
    "experience",
    "knowledge",
    "proficiency",
    "skill",
    "degree",
    "responsib",
    "ability",
    "able to",
    "must",
    "required",
    "familiar",
    "understanding",
    "work with",
    "develop",
    "build",
    "manage",
    "analy",
    "support",
    "design",
    "create",
    "lead",
    "collaborat",
)


def development_source_revision(
    djinni_manifest_path: Path,
    *,
    ats_dataset_dir: Path | None = None,
) -> str:
    """Commit to local source bytes without exposing any source content."""
    djinni_digest = hashlib.sha256(djinni_manifest_path.read_bytes()).hexdigest()
    if ats_dataset_dir is None:
        return f"djinni-manifest-sha256:{djinni_digest}"
    files = sorted(path for path in ats_dataset_dir.rglob("*") if path.is_file())
    if not files:
        raise ValueError("ATS dataset directory is empty.")
    ats_hasher = hashlib.sha256()
    for path in files:
        relative = path.relative_to(ats_dataset_dir).as_posix()
        ats_hasher.update(relative.encode())
        ats_hasher.update(b"\0")
        ats_hasher.update(hashlib.sha256(path.read_bytes()).digest())
    payload = json.dumps(
        {
            "ats_dataset_sha256": ats_hasher.hexdigest(),
            "djinni_manifest_sha256": djinni_digest,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return f"djinni+ats-sha256:{hashlib.sha256(payload).hexdigest()}"
NON_REQUIREMENT_SIGNAL = (
    "bonus",
    "incentive",
    "salary",
    "discount",
    "benefit",
    "industry:",
    "we offer",
    "opportunity to join",
    "looking to expand",
    "start-up environment",
    "pharmaceutical sciences",
    "biochemistry biotechnology",
)
SPECIALIZED_PROFILE_ROLE_SIGNALS = {
    "ML": (
        "machine learning",
        "data scientist",
        "artificial intelligence",
        "computer vision",
        "deep learning",
        "nlp",
        "pytorch",
        "tensorflow",
    ),
    "Data": (
        "data analyst",
        "analytics",
        "business intelligence",
        "power bi",
        "tableau",
        "reporting analyst",
    ),
}


def _extract_profile_evidence(
    text: str,
    *,
    policy: str,
) -> list[str]:
    """Keep the legacy call contract stable for frozen source builders."""
    if policy == LEGACY_EVIDENCE_EXTRACTION:
        return extract_action_evidence(text)
    return extract_action_evidence(text, policy=policy)
SKILLSPAN_REQUIREMENT_SIGNAL = (
    "experience",
    "knowledge",
    "proficiency",
    "skill",
    "degree",
    "responsible",
    "ability",
    "able to",
    "must",
    "required",
    "familiar",
    "understanding",
    "expected to",
    "work with",
)


def load_djinni_requirements(
    parquet_path: Path,
    excluded_job_hashes: set[str],
    *,
    random_state: int,
    family_limit: int = 180,
    text_quality_policy: str = LEGACY_SOURCE_TEXT_QUALITY,
    included_job_hashes: set[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Load privacy-screened JD requirements, balanced by broad role family."""
    validate_source_text_quality_policy(text_quality_policy)
    jobs = pd.read_parquet(
        parquet_path,
        columns=["id", "Position", "Primary Keyword", "Long Description"],
    )
    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in jobs.sample(frac=1, random_state=random_state).to_dict("records"):
        raw_id = str(row.get("id", ""))
        job_hash = source_hash("djinni_job", raw_id)
        if (
            not raw_id
            or job_hash in excluded_job_hashes
            or (
                included_job_hashes is not None
                and job_hash not in included_job_hashes
            )
        ):
            continue
        family = infer_role_family(
            str(row.get("Position", "")),
            str(row.get("Primary Keyword", "")),
        )
        if family not in ROLE_FAMILIES or len(pools[family]) >= family_limit:
            continue
        requirements = extract_requirement_records(
            str(row.get("Long Description", ""))
        )
        usable: list[str] = []
        for record in requirements:
            source_text = str(record["text"])
            candidate = (
                canonical_public_text(source_text)
                if text_quality_policy
                in {
                    SUCCESSOR_V6_SOURCE_TEXT_QUALITY,
                    SUCCESSOR_V7_SOURCE_TEXT_QUALITY,
                    SUCCESSOR_V8_SOURCE_TEXT_QUALITY,
                }
                else source_text
            )
            if (
                6 <= len(source_text.split()) <= 48
                and _is_requirement_text(candidate)
                and (
                    text_quality_policy == LEGACY_SOURCE_TEXT_QUALITY
                    or not requirement_quality_reasons_for_policy(
                        source_text,
                        text_quality_policy,
                    )
                )
            ):
                usable.append(candidate)
        if usable:
            pools[family].append(
                {
                    "requirement": usable[0],
                    "source_hash": job_hash,
                    "source_dataset": "djinni_job",
                    "family": family,
                }
            )
        if all(len(pools[item]) >= family_limit for item in ROLE_FAMILIES):
            break
    return _ordered_pools(pools, seed=f"jobs:{random_state}")


def load_skillspan_requirements(
    directory: Path,
    excluded_job_hashes: set[str],
    *,
    random_state: int,
) -> dict[str, list[dict[str, Any]]]:
    """Load tagged, de-identified SkillSpan sentences as requirement candidates."""
    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_sources: set[str] = set()
    for path in sorted(directory.glob("*.json")):
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            tokens = list(row.get("tokens", []))
            if not 6 <= len(tokens) <= 45 or not _has_skill_tag(row):
                continue
            requirement = " ".join(str(token) for token in tokens)
            if (
                not _is_requirement_text(requirement)
                or not any(
                    term in requirement.lower()
                    for term in SKILLSPAN_REQUIREMENT_SIGNAL
                )
            ):
                continue
            family = infer_role_family(requirement)
            source_id = (
                f"{path.stem}:{row.get('source', '')}:{row.get('idx', '')}"
            )
            job_hash = source_hash("skillspan_job", source_id)
            if (
                family not in ROLE_FAMILIES
                or source_id in seen_sources
                or job_hash in excluded_job_hashes
            ):
                continue
            pools[family].append(
                {
                    "requirement": requirement,
                    "source_hash": job_hash,
                    "source_dataset": "skillspan",
                    "family": family,
                }
            )
            seen_sources.add(source_id)
    return _ordered_pools(pools, seed=f"skillspan:{random_state}")


def load_djinni_profiles(
    parquet_path: Path,
    excluded_resume_hashes: set[str],
    *,
    random_state: int,
    family_limit: int = 240,
    include_supplemental_profile_text: bool = False,
    prefer_specialized_role: bool = False,
    minimum_evidence: int = 5,
    text_quality_policy: str = LEGACY_SOURCE_TEXT_QUALITY,
    included_resume_hashes: set[str] | None = None,
    evidence_extraction_policy: str = LEGACY_EVIDENCE_EXTRACTION,
) -> dict[str, list[dict[str, Any]]]:
    """Load profiles containing enough privacy-screened action evidence."""
    validate_source_text_quality_policy(text_quality_policy)
    validate_evidence_extraction_policy(evidence_extraction_policy)
    if minimum_evidence < 4:
        raise ValueError("Djinni development profiles need four evidence statements.")
    text_columns = (
        ["CV", "Highlights", "Moreinfo", "Looking For"]
        if include_supplemental_profile_text
        else ["CV"]
    )
    profiles = pd.read_parquet(
        parquet_path,
        columns=["id", "Position", "Primary Keyword", *text_columns],
    )
    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rows = profiles.sample(frac=1, random_state=random_state + 1)
    for row in rows.to_dict("records"):
        raw_id = str(row.get("id", ""))
        resume_hash = source_hash("djinni_profile", raw_id)
        if (
            not raw_id
            or resume_hash in excluded_resume_hashes
            or (
                included_resume_hashes is not None
                and resume_hash not in included_resume_hashes
            )
        ):
            continue
        role_text = (
            f"{row.get('Position', '')} {row.get('Primary Keyword', '')}"
        ).lower()
        specialized = next(
            (
                family_name
                for family_name, signals in (
                    SPECIALIZED_PROFILE_ROLE_SIGNALS.items()
                )
                if any(signal in role_text for signal in signals)
            ),
            None,
        )
        family = (
            specialized
            if prefer_specialized_role and specialized
            else infer_role_family(
                str(row.get("Position", "")),
                str(row.get("Primary Keyword", "")),
            )
        )
        if family not in ROLE_FAMILIES or len(pools[family]) >= family_limit:
            continue
        evidence = _extract_profile_evidence(
            "\n".join(str(row.get(column, "")) for column in text_columns),
            policy=evidence_extraction_policy,
        )
        if text_quality_policy in {
            SUCCESSOR_V5_SOURCE_TEXT_QUALITY,
            SUCCESSOR_V6_SOURCE_TEXT_QUALITY,
            SUCCESSOR_V7_SOURCE_TEXT_QUALITY,
            SUCCESSOR_V8_SOURCE_TEXT_QUALITY,
        }:
            evidence = diverse_quality_evidence(
                evidence,
                policy=text_quality_policy,
            )
        if len(evidence) >= minimum_evidence:
            pools[family].append(
                {
                    "source_hash": resume_hash,
                    "source_dataset": "djinni_profile",
                    "family": family,
                    "evidence": evidence,
                }
            )
        if all(len(pools[item]) >= family_limit for item in ROLE_FAMILIES):
            break
    return _ordered_pools(pools, seed=f"profiles:{random_state}")


def load_ats_profiles(
    dataset_dir: Path,
    excluded_resume_hashes: set[str],
    *,
    random_state: int,
    family_limit: int = 120,
    minimum_evidence: int = 4,
    text_quality_policy: str = LEGACY_SOURCE_TEXT_QUALITY,
    included_resume_hashes: set[str] | None = None,
    evidence_extraction_policy: str = LEGACY_EVIDENCE_EXTRACTION,
) -> dict[str, list[dict[str, Any]]]:
    """Load new privacy-screened evidence profiles from the local ATS corpus."""
    validate_source_text_quality_policy(text_quality_policy)
    validate_evidence_extraction_policy(evidence_extraction_policy)
    if minimum_evidence < 4:
        raise ValueError("ATS development profiles need four evidence statements.")
    try:
        from datasets import load_from_disk
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "datasets is required to load the local ATS research corpus."
        ) from error
    dataset = load_from_disk(str(dataset_dir))
    rows = [
        str(row["text"])
        for split in sorted(dataset)
        for row in dataset[split]
    ]
    random.Random(random_state).shuffle(rows)
    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for text in rows:
        raw_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        resume_hash = source_hash("resume_ats_profile", raw_hash)
        if (
            resume_hash in excluded_resume_hashes
            or (
                included_resume_hashes is not None
                and resume_hash not in included_resume_hashes
            )
        ):
            continue
        family = infer_annotation_role_family(text)
        if family not in ROLE_FAMILIES or len(pools[family]) >= family_limit:
            continue
        evidence = _extract_profile_evidence(
            text,
            policy=evidence_extraction_policy,
        )
        if text_quality_policy in {
            SUCCESSOR_V5_SOURCE_TEXT_QUALITY,
            SUCCESSOR_V6_SOURCE_TEXT_QUALITY,
            SUCCESSOR_V7_SOURCE_TEXT_QUALITY,
            SUCCESSOR_V8_SOURCE_TEXT_QUALITY,
        }:
            evidence = diverse_quality_evidence(
                evidence,
                policy=text_quality_policy,
            )
        if len(evidence) < minimum_evidence:
            continue
        pools[family].append(
            {
                "source_hash": resume_hash,
                "source_dataset": "resume_ats_profile",
                "family": family,
                "evidence": evidence,
            }
        )
        if all(len(pools[item]) >= family_limit for item in ROLE_FAMILIES):
            break
    return _ordered_pools(pools, seed=f"ats-profiles:{random_state}")


def load_talentclef_profiles(
    task_a_dir: Path,
    excluded_resume_hashes: set[str],
    *,
    random_state: int,
    family_limit: int = 120,
) -> dict[str, list[dict[str, Any]]]:
    """Load human-reviewed privacy-protecting synthetic TalentCLEF profiles."""
    if not task_a_dir.exists():
        raise FileNotFoundError(
            "TalentCLEF TaskA is missing. Download Zenodo record "
            "10.5281/zenodo.19652670 and extract TaskA.zip first."
        )
    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    paths = [
        path
        for split in ("development", "test")
        for path in sorted((task_a_dir / split / "en" / "corpus").glob("*"))
        if path.is_file()
    ]
    random.Random(random_state).shuffle(paths)
    for path in paths:
        split = path.parents[2].name
        resume_hash = source_hash(
            f"talentclef_2026_{split}_profile",
            path.name,
        )
        if resume_hash in excluded_resume_hashes:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        family = infer_annotation_role_family(text)
        if family not in ROLE_FAMILIES or len(pools[family]) >= family_limit:
            continue
        evidence = extract_action_evidence(text)
        if len(evidence) < 5:
            continue
        pools[family].append(
            {
                "source_hash": resume_hash,
                "source_dataset": (
                    f"talentclef_2026_{split}_synthetic_profile"
                ),
                "family": family,
                "evidence": evidence,
            }
        )
        if all(len(pools[item]) >= family_limit for item in ROLE_FAMILIES):
            break
    return _ordered_pools(pools, seed=f"talentclef-profiles:{random_state}")


def _has_skill_tag(row: dict[str, Any]) -> bool:
    return any(
        tag != "O"
        for tag in [
            *row.get("tags_skill", []),
            *row.get("tags_knowledge", []),
        ]
    )


def _is_requirement_text(text: str) -> bool:
    lowered = text.lower()
    return (
        "http" not in lowered
        and "@" not in text
        and "<" not in text
        and not any(term in lowered for term in NON_REQUIREMENT_SIGNAL)
        and any(term in lowered for term in REQUIREMENT_SIGNAL)
    )


def _ordered_pools(
    pools: dict[str, list[dict[str, Any]]],
    *,
    seed: str,
) -> dict[str, list[dict[str, Any]]]:
    return {
        family: sorted(
            pools[family],
            key=lambda row: hashlib.sha256(
                f"{seed}:{family}:{row['source_hash']}".encode()
            ).hexdigest(),
        )
        for family in ROLE_FAMILIES
    }
