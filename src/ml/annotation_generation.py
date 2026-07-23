"""Build privacy-conscious requirement/evidence queues from local pair data."""

from __future__ import annotations

import hashlib
import random
import re
from typing import Any, Iterable

from ml.annotation import SCHEMA_VERSION
from ml.evidence import score_evidence_pair


SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
CONTACT_PATTERN = re.compile(
    r"(?:https?://\S+|www\.\S+|[\w.+-]+@[\w.-]+\.\w+|\+?\d[\d\s().-]{7,}\d)",
    re.IGNORECASE,
)
ACTION_TERMS = {
    "built", "created", "developed", "designed", "implemented", "analyzed",
    "evaluated", "led", "improved", "automated", "researched", "supported",
    "deployed", "optimized", "managed", "conducted", "produced", "trained",
    "delivered", "collaborated", "maintained",
}
REQUIREMENT_TERMS = {
    "required", "must", "responsible", "experience", "proficient", "proficiency",
    "knowledge", "skills", "qualification", "ability", "develop", "build", "manage",
}
TECHNICAL_TERMS = {
    "python", "sql", "machine learning", "data", "software", "api", "cloud",
    "statistics", "analytics", "model", "pipeline", "database", "java", "excel",
    "dashboard", "research", "product", "business", "stakeholder",
}
EVIDENCE_TERMS = {
    "experience", "project", "degree", "certification", "coursework", "research",
    "developed", "implemented", "managed", "analyzed", "designed", "built",
}
ROLE_TERMS = {
    "Data": ("data analyst", "analytics", "sql", "dashboard", "business intelligence"),
    "ML": ("machine learning", "data scientist", "model training", "deep learning", "nlp"),
    "Software": ("software", "backend", "frontend", "full stack", "api", "developer"),
    "Business": ("business", "product", "stakeholder", "operations", "strategy", "marketing"),
}


def normalize_text(text: str) -> str:
    """Return a stable comparison key without retaining formatting noise."""
    return " ".join(re.sub(r"[^a-z0-9+#]+", " ", text.lower()).split())


def sanitize_snippet(text: str) -> str:
    """Remove contact strings and normalize one local annotation snippet."""
    cleaned = CONTACT_PATTERN.sub("[redacted]", " ".join(text.split()))
    return cleaned.strip(" -–—•")


def split_sentences(text: str) -> list[str]:
    """Split long single-line corpus records into reviewable sentences."""
    return [sanitize_snippet(part) for part in SENTENCE_BOUNDARY.split(" ".join(text.split()))]


def extract_requirement_sentences(job_text: str) -> list[str]:
    """Select compact requirement-like sentences from plain job text."""
    selected: list[str] = []
    for sentence in split_sentences(job_text):
        lowered = sentence.lower()
        word_count = len(sentence.split())
        if not 6 <= word_count <= 70 or "[redacted]" in sentence:
            continue
        has_requirement = any(term in lowered for term in REQUIREMENT_TERMS)
        has_domain_detail = sum(term in lowered for term in TECHNICAL_TERMS) >= 2
        if not (has_requirement or has_domain_detail):
            continue
        key = normalize_text(sentence)
        if key and key not in {normalize_text(item) for item in selected}:
            selected.append(sentence)
    return selected[:8]


def extract_evidence_sentences(resume_text: str) -> list[str]:
    """Select factual, action-led resume statements without contact headers."""
    selected: list[str] = []
    for sentence in split_sentences(resume_text):
        lowered = sentence.lower()
        words = sentence.split()
        if not 6 <= len(words) <= 100 or "[redacted]" in sentence:
            continue
        action_led = any(re.search(rf"\b{re.escape(term)}\b", lowered) for term in ACTION_TERMS)
        technical = any(term in lowered for term in TECHNICAL_TERMS)
        factual = any(term in lowered for term in EVIDENCE_TERMS)
        if not (action_led or technical or factual):
            continue
        key = normalize_text(sentence)
        if key and key not in {normalize_text(item) for item in selected}:
            selected.append(sentence)
    return selected[:30]


def prepared_snippets(value: object, *, minimum_words: int = 4) -> list[str]:
    """Validate structured fictional snippets without retaining source containers."""
    if not isinstance(value, (list, tuple)):
        return []
    snippets: list[str] = []
    for item in value:
        snippet = sanitize_snippet(str(item))
        if minimum_words <= len(snippet.split()) <= 70 and "[redacted]" not in snippet:
            snippets.append(snippet)
    return snippets


def infer_role_family(text: str) -> str:
    """Assign one broad role family for balanced annotation sampling."""
    lowered = text.lower()
    if any(
        term in lowered
        for term in (
            "machine learning", "ml engineer", "data scientist", "deep learning",
            "natural language processing", "computer vision", "nlp",
        )
    ):
        return "ML"
    scored = [
        (sum(lowered.count(term) for term in terms), -index, family)
        for index, (family, terms) in enumerate(ROLE_TERMS.items())
    ]
    score, _rank, family = max(scored)
    return family if score else "Other"


def candidate_records(
    requirement: str,
    evidence: list[str],
    *,
    randomizer: random.Random,
) -> list[dict[str, Any]]:
    """Select useful candidates, then hide retrieval order through shuffling."""
    scored = [
        (score_evidence_pair(requirement, sentence), sentence)
        for sentence in evidence
    ]
    scored.sort(key=lambda item: float(item[0]["similarity"]), reverse=True)
    chosen = scored[:3]
    if len(scored) > 3:
        hard_negative = scored[-1]
        if hard_negative[1] not in {sentence for _score, sentence in chosen}:
            chosen.append(hard_negative)
    records: list[dict[str, Any]] = []
    for _score, sentence in chosen:
        candidate_id = hashlib.sha256(normalize_text(sentence).encode()).hexdigest()[:16]
        records.append(
            {
                "candidate_id": candidate_id,
                "evidence": sentence,
            }
        )
    randomizer.shuffle(records)
    return records


def build_tasks_from_records(
    records: Iterable[dict[str, Any]],
    *,
    unique_count: int,
    blind_repeat_fraction: float = 0.1,
    random_state: int = 42,
) -> list[dict[str, Any]]:
    """Build a balanced deterministic queue without retaining full documents."""
    randomizer = random.Random(random_state)
    source_rows = list(records)
    randomizer.shuffle(source_rows)
    families = tuple(ROLE_TERMS)
    family_target = max(1, unique_count // len(families))
    family_pool_limit = max(12, family_target * 2)
    task_pools: dict[str, list[dict[str, Any]]] = {family: [] for family in families}
    seen_requirements: set[str] = set()
    preferred_position_cursor = 0

    for row in source_rows:
        job_text = str(row.get("job_text", ""))
        resume_text = str(row.get("resume_text", ""))
        declared_family = str(row.get("role_family", ""))
        family = declared_family if declared_family in task_pools else infer_role_family(job_text)
        if family not in task_pools or len(task_pools[family]) >= family_pool_limit:
            continue
        evidence = prepared_snippets(row.get("evidence_sentences"))
        if not evidence:
            evidence = extract_evidence_sentences(resume_text)
        if len(evidence) < 3:
            continue
        requirements = prepared_snippets(row.get("requirement_sentences"), minimum_words=3)
        if not requirements:
            requirements = extract_requirement_sentences(job_text)
        for requirement in requirements[:4]:
            requirement_key = normalize_text(requirement)
            if not requirement_key or requirement_key in seen_requirements:
                continue
            task_material = "\0".join(
                [str(row.get("resume_hash", "")), str(row.get("job_hash", "")), requirement_key]
            )
            task_id = "ann-" + hashlib.sha256(task_material.encode()).hexdigest()[:16]
            order_seed = int(
                hashlib.sha256(f"{random_state}:{task_material}".encode()).hexdigest()[:16],
                16,
            )
            candidates = candidate_records(
                requirement,
                evidence,
                randomizer=random.Random(order_seed),
            )
            if len(candidates) < 3:
                continue
            preferred_evidence = str(row.get("preferred_evidence", ""))
            preferred_index = next(
                (
                    index
                    for index, candidate in enumerate(candidates)
                    if candidate["evidence"] == preferred_evidence
                ),
                None,
            )
            if preferred_index is not None:
                target_index = preferred_position_cursor % len(candidates)
                candidates[target_index], candidates[preferred_index] = (
                    candidates[preferred_index],
                    candidates[target_index],
                )
                preferred_position_cursor += 1
            task_pools[family].append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "task_id": task_id,
                    "role_family": family,
                    "requirement": requirement,
                    "candidates": candidates,
                    "source_resume_hash": str(row.get("resume_hash", "")),
                    "source_job_hash": str(row.get("job_hash", "")),
                    "source_dataset": str(row.get("source_dataset", "local_pairs")),
                    "blind_duplicate_of": None,
                }
            )
            seen_requirements.add(requirement_key)
            break
        if all(len(pool) >= family_target for pool in task_pools.values()) and sum(
            len(pool) for pool in task_pools.values()
        ) >= unique_count:
            break

    tasks: list[dict[str, Any]] = []
    while len(tasks) < unique_count and any(task_pools.values()):
        for family in families:
            if task_pools[family] and len(tasks) < unique_count:
                tasks.append(task_pools[family].pop(0))
    if len(tasks) < unique_count:
        raise ValueError(
            f"Only {len(tasks)} unique annotation tasks could be built; requested {unique_count}."
        )
    repeat_count = round(unique_count * blind_repeat_fraction)
    for original in randomizer.sample(tasks, k=min(repeat_count, len(tasks))):
        duplicate = dict(original)
        duplicate["task_id"] = original["task_id"] + "-repeat"
        duplicate["blind_duplicate_of"] = original["task_id"]
        duplicate_candidates = list(original["candidates"])
        randomizer.shuffle(duplicate_candidates)
        if duplicate_candidates == original["candidates"] and len(duplicate_candidates) > 1:
            duplicate_candidates = duplicate_candidates[1:] + duplicate_candidates[:1]
        duplicate["candidates"] = duplicate_candidates
        tasks.append(duplicate)
    randomizer.shuffle(tasks)
    return tasks
