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
    ROLE_FAMILIES,
    extract_action_evidence,
    infer_role_family,
    source_hash,
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
) -> dict[str, list[dict[str, Any]]]:
    """Load privacy-screened JD requirements, balanced by broad role family."""
    jobs = pd.read_parquet(
        parquet_path,
        columns=["id", "Position", "Primary Keyword", "Long Description"],
    )
    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in jobs.sample(frac=1, random_state=random_state).to_dict("records"):
        raw_id = str(row.get("id", ""))
        job_hash = source_hash("djinni_job", raw_id)
        if not raw_id or job_hash in excluded_job_hashes:
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
        usable = [
            str(record["text"])
            for record in requirements
            if 6 <= len(str(record["text"]).split()) <= 48
            and _is_requirement_text(str(record["text"]))
        ]
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
) -> dict[str, list[dict[str, Any]]]:
    """Load profiles containing enough privacy-screened action evidence."""
    profiles = pd.read_parquet(
        parquet_path,
        columns=["id", "Position", "Primary Keyword", "CV"],
    )
    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rows = profiles.sample(frac=1, random_state=random_state + 1)
    for row in rows.to_dict("records"):
        raw_id = str(row.get("id", ""))
        resume_hash = source_hash("djinni_profile", raw_id)
        if not raw_id or resume_hash in excluded_resume_hashes:
            continue
        family = infer_role_family(
            str(row.get("Position", "")),
            str(row.get("Primary Keyword", "")),
        )
        if family not in ROLE_FAMILIES or len(pools[family]) >= family_limit:
            continue
        evidence = extract_action_evidence(str(row.get("CV", "")))
        if len(evidence) >= 5:
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
) -> dict[str, list[dict[str, Any]]]:
    """Load new privacy-screened evidence profiles from the local ATS corpus."""
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
        if resume_hash in excluded_resume_hashes:
            continue
        family = infer_annotation_role_family(text)
        if family not in ROLE_FAMILIES or len(pools[family]) >= family_limit:
            continue
        evidence = extract_action_evidence(text)
        if len(evidence) < 5:
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
