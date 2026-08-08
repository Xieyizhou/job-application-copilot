"""Review Jobs filter state and controls."""

from __future__ import annotations

from typing import Any, Callable, Protocol

import streamlit as st

from dashboard_regions import (
    build_region_options,
    dynamic_source_options,
    filtered_region_option_keys,
)
from dashboard_review import REVIEW_FIT_BANDS, REVIEW_INBOX_OPTIONS, review_fit_band_counts
from scoring_types import DashboardJob, RegionOption, TrackerRow


SORT_OPTIONS = [
    "Role Fit high to low",
    "Newest first",
    "Recommendation",
    "Company A-Z",
    "Cover letter status",
    "Tracker status",
]
RECOMMENDATION_OPTIONS = [
    "all",
    "Apply",
    "Apply / Maybe Apply",
    "Maybe Apply",
    "Manual Review",
    "Skip or Low Priority",
    "Skip / Not Eligible",
]
LEGACY_INBOX_NAMES = {
    "Needs Review": "Needs attention",
    "Cover Letter Ready": "Ready",
    "Not Tracked": "All",
    "Ignored": "All",
    "All Jobs": "All",
}


class ReviewFilterServices(Protocol):
    """Operations used by the Review Jobs filter toolbar."""

    @property
    def default_review_inbox_view(self) -> Callable[..., str]: ...

    @property
    def demo_mode_enabled(self) -> Callable[[], bool]: ...

    @property
    def save_recent_region_key(self) -> Callable[[str], None]: ...


def initialize_review_state(
    all_jobs: list[DashboardJob],
    tracker_rows: list[TrackerRow],
    services: ReviewFilterServices,
) -> dict[str, Any]:
    """Initialize workspace-specific review defaults and return option models."""
    is_demo = services.demo_mode_enabled()
    default_inbox_view = services.default_review_inbox_view(
        all_jobs,
        tracker_rows,
        demo=is_demo,
    )
    workspace_state_key = "demo" if is_demo else "personal"
    if st.session_state.get("review_workspace_mode") != workspace_state_key:
        st.session_state["review_workspace_mode"] = workspace_state_key
        st.session_state["review_inbox_view"] = default_inbox_view
        if is_demo:
            st.session_state.update(
                {
                    "review_minimum_score": 0,
                    "review_hide_hard_red_flags": False,
                    "review_hide_degree_required": False,
                    "review_hide_current_student_only": False,
                }
            )
    source_options = dynamic_source_options(all_jobs)
    _apply_review_defaults(default_inbox_view)
    _normalize_review_options(default_inbox_view, source_options)
    return {
        "is_demo": is_demo,
        "default_inbox_view": default_inbox_view,
        "inbox_options": REVIEW_INBOX_OPTIONS,
        "sort_options": SORT_OPTIONS,
        "source_options": source_options,
        "recommendation_options": RECOMMENDATION_OPTIONS,
    }


def _apply_review_defaults(default_inbox_view: str) -> None:
    defaults = {
        "review_inbox_view": default_inbox_view,
        "review_sort_by": "Role Fit high to low",
        "review_source_filter": "all",
        "review_recommendation_filter": "all",
        "review_tracker_filter": "all",
        "review_search_text": "",
        "review_fit_band": "All",
        "region_search_query": "",
        "review_minimum_score": 0,
        "review_hide_hard_red_flags": False,
        "review_hide_degree_required": False,
        "review_hide_current_student_only": False,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _normalize_review_options(default_inbox_view: str, source_options: list[str]) -> None:
    inbox_view = LEGACY_INBOX_NAMES.get(
        st.session_state["review_inbox_view"],
        st.session_state["review_inbox_view"],
    )
    st.session_state["review_inbox_view"] = (
        inbox_view if inbox_view in REVIEW_INBOX_OPTIONS else default_inbox_view
    )
    if st.session_state["review_sort_by"] not in SORT_OPTIONS:
        st.session_state["review_sort_by"] = "Role Fit high to low"
    if st.session_state["review_source_filter"] not in source_options:
        st.session_state["review_source_filter"] = "all"
    if st.session_state["review_recommendation_filter"] not in RECOMMENDATION_OPTIONS:
        st.session_state["review_recommendation_filter"] = "all"


def reset_review_filters(
    default_inbox_view: str,
    is_demo: bool,
    *,
    show_all: bool = False,
) -> None:
    """Reset review widgets to workspace defaults."""
    st.session_state.pop("review_job_table", None)
    st.session_state.update(
        {
            "review_search_text": "",
            "review_fit_band": "All",
            "review_inbox_view": "All" if show_all else default_inbox_view,
            "review_sort_by": "Role Fit high to low",
            "selected_region_key": "all",
            "review_source_filter": "all",
            "review_recommendation_filter": "all",
            "review_tracker_filter": "all",
            "review_minimum_score": 0,
            "review_hide_hard_red_flags": False,
            "review_hide_degree_required": False,
            "review_hide_current_student_only": False,
        }
    )


def reset_review_table_selection() -> None:
    """Clear row-selection state before filters or ordering change."""
    st.session_state.pop("review_job_table", None)


def _demo_review_filters(
    options_by_key: dict[str, RegionOption],
    clear_filters: Callable[[], None],
) -> dict[str, Any]:
    return {
        "inbox_view": "All",
        "search_text": "",
        "sort_by": "Role Fit high to low",
        "selected_region_key": "all",
        "selected_region": options_by_key["all"],
        "source_filter": "all",
        "recommendation_filter": "all",
        "tracker_filter": "all",
        "minimum_score": 0,
        "hide_hard": False,
        "hide_degree": False,
        "hide_student": False,
        "clear_filters": clear_filters,
    }


def _render_region_filter(
    options_by_key: dict[str, RegionOption],
    services: ReviewFilterServices,
) -> tuple[str, RegionOption]:
    region_query = st.text_input(
        "Search region",
        placeholder="Beijing, China, Remote",
        key="region_search_query",
        on_change=reset_review_table_selection,
    )
    region_keys = filtered_region_option_keys(options_by_key, region_query)
    if region_query.strip() and region_keys == ["all"]:
        st.info("No matching regions found.")
    selected_key = st.session_state.get("selected_region_key", "all")
    if selected_key not in region_keys and selected_key in options_by_key:
        region_keys = [selected_key, *region_keys]
    selected_region_key = st.selectbox(
        "Region",
        region_keys,
        index=region_keys.index(selected_key) if selected_key in region_keys else 0,
        key="selected_region_key",
        format_func=lambda key: options_by_key.get(key, options_by_key["all"])["label"],
        on_change=reset_review_table_selection,
    )
    services.save_recent_region_key(selected_region_key)
    return selected_region_key, options_by_key.get(
        selected_region_key,
        options_by_key["all"],
    )


def _render_advanced_filters(
    options_by_key: dict[str, RegionOption],
    state: dict[str, Any],
    services: ReviewFilterServices,
    clear_filters: Callable[[], None],
) -> dict[str, Any]:
    with st.popover(
        "Filters",
        type="tertiary",
        icon=":material/tune:",
        width="stretch",
    ):
        sort_by = st.selectbox(
            "Sort by",
            state["sort_options"],
            key="review_sort_by",
            on_change=reset_review_table_selection,
        )
        selected_region_key, selected_region = _render_region_filter(
            options_by_key,
            services,
        )
        outcome_filters = _render_outcome_filters(state)
        exclusion_filters = _render_exclusion_filters()
        st.button(
            "Reset filters",
            key="review_clear_filters",
            type="tertiary",
            on_click=clear_filters,
        )
    return {
        "sort_by": sort_by,
        "selected_region_key": selected_region_key,
        "selected_region": selected_region,
        **outcome_filters,
        **exclusion_filters,
    }


def _render_outcome_filters(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_filter": st.selectbox(
            "Source",
            state["source_options"],
            key="review_source_filter",
            on_change=reset_review_table_selection,
        ),
        "recommendation_filter": st.selectbox(
            "Recommendation",
            state["recommendation_options"],
            key="review_recommendation_filter",
            on_change=reset_review_table_selection,
        ),
        "tracker_filter": st.selectbox(
            "Tracker",
            ["all", "Not tracked", "Tracked", "Ignored"],
            key="review_tracker_filter",
            on_change=reset_review_table_selection,
        ),
        "minimum_score": st.slider(
            "Minimum Role Fit score",
            min_value=0,
            max_value=100,
            key="review_minimum_score",
            on_change=reset_review_table_selection,
        ),
    }


def _render_exclusion_filters() -> dict[str, bool]:
    return {
        "hide_hard": st.checkbox(
            "Hide hard red flags",
            key="review_hide_hard_red_flags",
            on_change=reset_review_table_selection,
        ),
        "hide_degree": st.checkbox(
            "Hide PhD / Master's required roles",
            key="review_hide_degree_required",
            on_change=reset_review_table_selection,
        ),
        "hide_student": st.checkbox(
            "Hide current-student-only internships",
            key="review_hide_current_student_only",
            on_change=reset_review_table_selection,
        ),
    }


def render_review_filter_controls(
    all_jobs: list[DashboardJob],
    state: dict[str, Any],
    services: ReviewFilterServices,
) -> dict[str, Any]:
    """Render one compact toolbar and return the active filtering model."""

    def clear_filters() -> None:
        reset_review_filters(state["default_inbox_view"], state["is_demo"])

    options_by_key = build_region_options(all_jobs)
    base_filters = _demo_review_filters(options_by_key, clear_filters)
    search_col, filter_col = st.columns([0.78, 0.22] if not state["is_demo"] else [1, 0.0001])
    with search_col:
        search_text = st.text_input(
            "Search company or role",
            key="review_search_text",
            placeholder="Search jobs…",
            label_visibility="collapsed",
            icon=":material/search:",
            on_change=reset_review_table_selection,
        )
    if not state["is_demo"]:
        with filter_col:
            base_filters.update(_render_advanced_filters(options_by_key, state, services, clear_filters))
    counts = review_fit_band_counts(all_jobs)
    fit_band = st.segmented_control(
        "Fit group",
        REVIEW_FIT_BANDS,
        key="review_fit_band",
        format_func=lambda band: f"{band} ({counts[band]})",
        selection_mode="single",
        label_visibility="collapsed",
        width="stretch",
        on_change=reset_review_table_selection,
    ) or "All"
    return {
        **base_filters,
        "inbox_view": "All",
        "search_text": search_text,
        "fit_band": fit_band,
        "clear_filters": clear_filters,
    }
