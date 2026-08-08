"""User-facing requirement-to-evidence presentation for Review Jobs."""

from __future__ import annotations

import html
from typing import Any, Callable, Literal, TypedDict

import streamlit as st

from dashboard_evidence_styles import render_evidence_map_styles


EvidenceStatus = Literal["Direct", "Partial", "No Support"]
Sanitizer = Callable[[Any], str]


class EvidenceCard(TypedDict):
    """Stable UI contract shared by transparent and learned evidence matchers."""

    requirement: str
    context: str
    evidence: str
    status: EvidenceStatus
    explanation: str


def evidence_status(match: dict[str, Any]) -> EvidenceStatus:
    """Normalize current and future matcher labels into three user states."""
    prediction = str(match.get("prediction", "")).strip().lower().replace("_", " ")
    if prediction in {"direct", "direct support"}:
        return "Direct"
    if prediction in {"partial", "partial support", "semantic support"}:
        return "Partial"
    if prediction in {"no support", "unsupported", "insufficient evidence"}:
        return "No Support"
    if not match.get("accepted"):
        return "No Support"
    match_type = str(match.get("match_type", "")).lower()
    return "Direct" if "direct" in match_type else "Partial"


def missing_dimension(match: dict[str, Any], status: EvidenceStatus) -> str:
    """Explain the evidence boundary without inventing candidate experience."""
    if status == "Direct":
        return "The resume provides explicit evidence for this requirement."
    if status == "No Support":
        return "No resume statement passed the evidence threshold."
    if match.get("numeric_constraint_supported") is False:
        return "Related experience was found, but the stated duration or scale is not supported."
    if match.get("compound_requirement_supported") is False:
        return "The evidence covers part of this compound requirement, but not every dimension."
    return "Related evidence was found, but the scope, outcome, or ownership is not explicit."


def build_evidence_cards(analysis: dict[str, Any]) -> list[EvidenceCard]:
    """Build display cards from the canonical semantic-evidence contract."""
    semantic = dict(analysis.get("semantic_evidence", {}) or {})
    cards: list[EvidenceCard] = []
    for raw_match in semantic.get("matches", []) or []:
        if not isinstance(raw_match, dict):
            continue
        requirement = str(raw_match.get("requirement", "")).strip()
        if not requirement:
            continue
        status = evidence_status(raw_match)
        evidence = str(
            raw_match.get("strongest_evidence") or raw_match.get("evidence") or ""
        ).strip()
        cards.append(
            {
                "requirement": requirement,
                "context": str(raw_match.get("demand") or "Requirement").replace("_", " ").title(),
                "evidence": evidence,
                "status": status,
                "explanation": missing_dimension(raw_match, status),
            }
        )
    if cards:
        return cards

    # Snippet-only JDs may expose keyword signals before they expose complete
    # sentence-level requirements. Keep those signals visible, but never promote
    # keyword overlap to Direct evidence.
    fallback_groups = (
        (analysis.get("matched_keywords", []), "Partial"),
        (analysis.get("partial_matches", []), "Partial"),
        (analysis.get("missing_keywords", []), "No Support"),
    )
    seen: set[str] = set()
    for values, status_value in fallback_groups:
        status = status_value  # type: ignore[assignment]
        for raw_value in values or []:
            requirement = str(raw_value).split(" (", 1)[0].strip()
            normalized = requirement.casefold()
            if not requirement or normalized in seen:
                continue
            seen.add(normalized)
            has_overlap = status == "Partial"
            cards.append(
                {
                    "requirement": requirement,
                    "context": "Keyword signal",
                    "evidence": (
                        "Keyword overlap detected; sentence-level evidence is unavailable."
                        if has_overlap
                        else ""
                    ),
                    "status": status,
                    "explanation": (
                        "The saved JD is incomplete, so this signal cannot be treated as "
                        "direct resume evidence."
                        if has_overlap
                        else "No resume statement passed the available evidence threshold."
                    ),
                }
            )
    return cards


def evidence_counts(cards: list[EvidenceCard]) -> dict[EvidenceStatus, int]:
    """Count normalized states for the summary strip."""
    return {
        "Direct": sum(card["status"] == "Direct" for card in cards),
        "Partial": sum(card["status"] == "Partial" for card in cards),
        "No Support": sum(card["status"] == "No Support" for card in cards),
    }


def filter_evidence_cards(cards: list[EvidenceCard], selected: str) -> list[EvidenceCard]:
    """Filter cards without changing their original requirement order."""
    if selected == "All":
        return cards
    return [card for card in cards if card["status"] == selected]


def render_evidence_map(
    analysis: dict[str, Any],
    sanitize: Sanitizer,
    render_analysis_action: Callable[[], None] | None = None,
) -> bool:
    """Render the evidence summary and cards; return whether evidence data existed."""
    cards = build_evidence_cards(analysis)
    if not cards:
        return False
    render_evidence_map_styles()
    counts = evidence_counts(cards)
    title_column, filter_column, action_column = st.columns(
        [0.34, 0.52, 0.14], vertical_alignment="center", gap="small"
    )
    with title_column:
        st.markdown('<div class="evidence-map-title">Requirement evidence</div>', unsafe_allow_html=True)
    options = [
        f"All ({len(cards)})",
        f'Direct ({counts["Direct"]})',
        f'Partial ({counts["Partial"]})',
        f'Missing ({counts["No Support"]})',
    ]
    with filter_column:
        selected_label = st.segmented_control(
            "Evidence status",
            options,
            default=options[0],
            selection_mode="single",
            key="evidence_map_segment",
            label_visibility="collapsed",
            width="stretch",
        ) or options[0]
    if render_analysis_action is not None:
        with action_column:
            render_analysis_action()
    selected = selected_label.split(" (", 1)[0].replace("Missing", "No Support")
    visible_cards = filter_evidence_cards(cards, selected)
    st.markdown(
        '<div class="evidence-map-table-head">'
        "<span>Requirement</span><span>Strongest matching resume evidence</span><span>Status</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    if not visible_cards:
        st.markdown('<div class="evidence-map-filter-empty">No requirements in this state.</div>', unsafe_allow_html=True)
        return True
    for index, card in enumerate(visible_cards):
        status_slug = card["status"].lower().replace(" ", "-")
        expanded_key = f"evidence_map_expanded_{index}_{status_slug}"
        with st.container(key=f"evidence_map_row_{index}_{status_slug}"):
            requirement_column, evidence_column, status_column = st.columns(
                [0.26, 0.56, 0.18], vertical_alignment="center"
            )
            with requirement_column:
                st.markdown(
                    '<div class="evidence-map-requirement">'
                    f'<strong>{html.escape(sanitize(card["requirement"]))}</strong>'
                    f'<span>{html.escape(sanitize(card["context"]))}</span>'
                    "</div>",
                    unsafe_allow_html=True,
                )
            with evidence_column:
                evidence = sanitize(card["evidence"])
                content = html.escape(evidence) if evidence else "No reliable resume evidence found."
                st.markdown(f'<div class="evidence-map-quote">{content}</div>', unsafe_allow_html=True)
            with status_column:
                badge, action = st.columns([0.7, 0.3], vertical_alignment="center", gap="small")
                with badge:
                    st.markdown(
                        f'<span class="evidence-map-status evidence-map-status-{status_slug}">{card["status"]}</span>',
                        unsafe_allow_html=True,
                    )
                with action:
                    expanded = bool(st.session_state.get(expanded_key, False))
                    if st.button(
                        "Collapse" if expanded else "Expand",
                        key=f"evidence_map_toggle_{index}_{status_slug}",
                        icon=":material/keyboard_arrow_down:" if expanded else ":material/chevron_right:",
                        type="tertiary",
                        help="Hide evidence explanation" if expanded else "Show evidence explanation",
                    ):
                        st.session_state[expanded_key] = not expanded
                        st.rerun()
            if st.session_state.get(expanded_key, False):
                st.markdown(
                    '<div class="evidence-map-expanded"><strong>Why this label:</strong> '
                    f'{html.escape(sanitize(card["explanation"]))}</div>',
                    unsafe_allow_html=True,
                )
    return True
