"""Generate expanded fictional annotation scenarios with style-label independence."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import random
from typing import Any

from ml.annotation_scenario_catalog import (
    COMPOUND_REQUIREMENT_TEMPLATES,
    GENERIC_STRONG_CORES,
    HANDS_ON_CORES,
    LIMITED_CORES,
    MENTION_CORES,
    REQUIREMENT_TEMPLATES_V3,
    ROLE_CONCEPTS_V3,
    ROLE_TITLES_V3,
    SURFACE_STYLES,
)


STRATUM_PLAN = (
    "direct",
    "adjacent",
    "mention_only",
    "direct",
    "scope_partial",
    "unsupported",
    "direct",
    "compound_partial",
    "adjacent",
    "mention_only",
    "direct",
    "unsupported",
    "scope_partial",
    "direct",
    "adjacent",
    "compound_partial",
    "mention_only",
    "direct",
    "unsupported",
    "direct",
)
EXPECTED_RELATION = {
    "direct": "Direct",
    "adjacent": "Partial",
    "scope_partial": "Partial",
    "compound_partial": "Partial",
    "mention_only": "No Support",
    "unsupported": "No Support",
}


def _lower_first(text: str) -> str:
    return text[:1].lower() + text[1:]


def _without_subject(text: str) -> str:
    return _lower_first(text)


def _surface(core: str, style_index: int) -> tuple[str, str]:
    style_name = f"surface_{style_index % len(SURFACE_STYLES):02d}"
    template = SURFACE_STYLES[style_index % len(SURFACE_STYLES)]
    sentence = template.format(
        clause=core,
        clause_lower=_lower_first(core),
        clause_no_subject=_without_subject(core),
    )
    return sentence, style_name


def _hands_on(skill: str, index: int) -> str:
    return HANDS_ON_CORES[index % len(HANDS_ON_CORES)].format(skill=skill)


def _limited(skill: str, index: int) -> str:
    return LIMITED_CORES[index % len(LIMITED_CORES)].format(skill=skill)


def _mention(skill: str, index: int) -> str:
    return MENTION_CORES[index % len(MENTION_CORES)].format(skill=skill)


def _candidate_cores(
    primary: str,
    adjacent: str,
    unrelated: str,
    stratum: str,
    index: int,
) -> tuple[list[str], list[str], int | None]:
    unrelated_core = _hands_on(unrelated, index + 5)
    generic_core = GENERIC_STRONG_CORES[index % len(GENERIC_STRONG_CORES)]
    if stratum == "direct":
        return [
            _hands_on(primary, index),
            _hands_on(adjacent, index + 3),
            _mention(primary, index + 2),
            unrelated_core,
        ], ["primary_hands_on", "adjacent_hands_on", "primary_mention", "unrelated_hands_on"], 0
    if stratum == "adjacent":
        return [
            _hands_on(adjacent, index),
            _mention(primary, index + 1),
            unrelated_core,
            generic_core,
        ], ["adjacent_hands_on", "primary_mention", "unrelated_hands_on", "generic_strong"], 0
    if stratum == "scope_partial":
        return [
            _limited(primary, index),
            _hands_on(adjacent, index + 2),
            unrelated_core,
            generic_core,
        ], ["primary_limited", "adjacent_hands_on", "unrelated_hands_on", "generic_strong"], 0
    if stratum == "compound_partial":
        return [
            _hands_on(primary, index),
            _limited(adjacent, index + 2),
            unrelated_core,
            generic_core,
        ], ["primary_hands_on", "adjacent_limited", "unrelated_hands_on", "generic_strong"], 0
    if stratum == "mention_only":
        return [
            _mention(primary, index),
            _mention(adjacent, index + 2),
            unrelated_core,
            generic_core,
        ], ["primary_mention", "adjacent_mention", "unrelated_hands_on", "generic_strong"], None
    return [
        unrelated_core,
        _hands_on(unrelated, index + 4),
        generic_core,
        GENERIC_STRONG_CORES[(index + 3) % len(GENERIC_STRONG_CORES)],
    ], ["unrelated_hands_on", "unrelated_hands_on", "generic_strong", "generic_strong"], None


def _requirement(
    primary: str,
    adjacent: str,
    stratum: str,
    template_index: int,
) -> tuple[str, str]:
    if stratum == "compound_partial":
        index = template_index % len(COMPOUND_REQUIREMENT_TEMPLATES)
        return (
            COMPOUND_REQUIREMENT_TEMPLATES[index].format(
                primary=primary,
                adjacent=adjacent,
            ),
            f"compound_{index:02d}",
        )
    index = template_index % len(REQUIREMENT_TEMPLATES_V3)
    return REQUIREMENT_TEMPLATES_V3[index].format(skill=primary), f"requirement_{index:02d}"


def expanded_fictional_records(*, random_state: int = 73) -> list[dict[str, Any]]:
    """Return 160 tasks whose evidence style is independent of semantic relation."""
    records: list[dict[str, Any]] = []
    for role_index, (family, concepts) in enumerate(ROLE_CONCEPTS_V3.items()):
        for local_index in range(40):
            concept_index = local_index % len(concepts)
            variant = local_index // len(concepts)
            primary, adjacent, unrelated = concepts[concept_index]
            stratum = STRATUM_PLAN[local_index % len(STRATUM_PLAN)]
            global_index = role_index * 40 + local_index
            template_index = global_index * 7 + role_index * 3 + variant * 11
            requirement, requirement_style = _requirement(
                primary,
                adjacent,
                stratum,
                template_index,
            )
            cores, semantic_roles, preferred_index = _candidate_cores(
                primary,
                adjacent,
                unrelated,
                stratum,
                global_index,
            )
            evidence: list[str] = []
            surface_styles: list[str] = []
            for candidate_index, core in enumerate(cores):
                style_index = global_index * 5 + candidate_index * 7 + role_index * 3
                sentence, style_name = _surface(core, style_index)
                evidence.append(sentence)
                surface_styles.append(style_name)
            identity = f"v3:{family}:{local_index}:{requirement}"
            records.append(
                {
                    "resume_text": " ".join(evidence),
                    "job_text": f"{ROLE_TITLES_V3[family]} role. {requirement}",
                    "resume_hash": hashlib.sha256(f"{identity}:resume".encode()).hexdigest(),
                    "job_hash": hashlib.sha256(f"{identity}:job".encode()).hexdigest(),
                    "label": "Unreviewed",
                    "role_family": family,
                    "source_dataset": "local_fictional_challenges_v3",
                    "requirement_sentences": [requirement],
                    "evidence_sentences": evidence,
                    "preferred_evidence": evidence[preferred_index] if preferred_index is not None else "",
                    "sampling_stratum": stratum,
                    "expected_relation": EXPECTED_RELATION[stratum],
                    "requirement_style": requirement_style,
                    "candidate_surface_styles": surface_styles,
                    "candidate_semantic_roles": semantic_roles,
                }
            )
    random.Random(random_state).shuffle(records)
    return records


def scenario_design_audit(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Return aggregate style/role/stratum diagnostics without task text."""
    style_by_relation: dict[str, Counter[str]] = defaultdict(Counter)
    style_by_semantic_role: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        relation = str(record["expected_relation"])
        for style, semantic_role in zip(
            record["candidate_surface_styles"],
            record["candidate_semantic_roles"],
        ):
            style_by_relation[relation][str(style)] += 1
            style_by_semantic_role[str(semantic_role)][str(style)] += 1
    style_relation_max_share = max(
        (
            count / sum(relation_counts[style] for relation_counts in style_by_relation.values())
            for relation_counts in style_by_relation.values()
            for style, count in relation_counts.items()
        ),
        default=0.0,
    )
    semantic_role_style_max_share = max(
        (
            count / sum(role_counts.values())
            for role_counts in style_by_semantic_role.values()
            for count in role_counts.values()
        ),
        default=0.0,
    )
    return {
        "records": len(records),
        "role_counts": dict(Counter(str(record["job_text"]).split(" role.", 1)[0] for record in records)),
        "stratum_counts": dict(Counter(str(record["sampling_stratum"]) for record in records)),
        "expected_relation_counts": dict(
            Counter(str(record["expected_relation"]) for record in records)
        ),
        "requirement_style_count": len({record["requirement_style"] for record in records}),
        "candidate_surface_style_count": len(
            {
                style
                for record in records
                for style in record["candidate_surface_styles"]
            }
        ),
        "largest_style_relation_share": style_relation_max_share,
        "largest_style_share_within_semantic_role": semantic_role_style_max_share,
    }
