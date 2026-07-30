"""Page-scoped desktop workspace styles for dense Streamlit workflows."""

from __future__ import annotations

import streamlit as st


def render_review_workspace_styles() -> None:
    """Keep the Review Jobs master-detail workspace inside the viewport."""
    st.markdown(
        """
        <style>
        .st-key-review_job_list_panel,.st-key-review_detail_shell {
            height:calc(100vh - 4.7rem);overflow:hidden;
            margin-top:0 !important;
        }
        .st-key-review_job_list_panel {padding-right:.15rem}
        .st-key-review_job_list_panel > div,
        .st-key-review_detail_shell > div {padding-top:0 !important}
        .st-key-review_detail_shell > [data-testid="stElementContainer"]:first-child {
            margin-top:-1rem !important;
        }
        .st-key-review_detail_panel {
            height:calc(100vh - 16.8rem) !important;min-height:300px;
            overflow-y:auto;padding-right:.45rem;
        }
        .st-key-review_job_list {
            height:calc(100vh - 9.8rem) !important;min-height:190px;overflow-y:auto;
        }
        [class*="st-key-review_job_row_"] {
            background:color-mix(in srgb,currentColor 5%,transparent);
            border-bottom:1px solid color-mix(in srgb,currentColor 18%,transparent);
            padding:.62rem .7rem .58rem;border-left:3px solid transparent;
            position:relative;min-height:4.65rem;
        }
        [class*="st-key-review_job_row_"]:hover {
            background:color-mix(in srgb,currentColor 9%,transparent);
        }
        [class*="st-key-review_job_row_"]:has([data-testid="stBaseButton-secondary"]) {
            border-left:3px solid var(--primary-color,#ff4b4b);
            background:color-mix(
                in srgb,var(--primary-color,#ff4b4b) 11%,transparent
            );
        }
        [class*="st-key-review_job_row_"] [class*="st-key-review_job_select_"] {
            position:absolute;top:0;right:-6px;bottom:-1px;left:-3px;z-index:2;
        }
        [class*="st-key-review_job_row_"] [data-testid="stButton"],
        [class*="st-key-review_job_row_"] [data-testid="stButton"] button {
            width:100%;height:100%;
        }
        [class*="st-key-review_job_row_"] [data-testid="stButton"] button {
            opacity:0;border:0 !important;padding:0 !important;
        }
        [class*="st-key-review_job_row_"]:has(button:focus-visible) {
            outline:2px solid var(--primary-color,#ff4b4b);outline-offset:-2px;
        }
        .review-job-row-copy {pointer-events:none}
        .review-job-row-title {
            color:inherit;font-size:.95rem;font-weight:600;line-height:1.25;
        }
        .review-job-row-signals {
            color:inherit;opacity:.76;font-size:.8rem;line-height:1.25;
            margin-top:.72rem;
        }
        .review-selected-job-header {margin:0;padding:0}
        .review-selected-job-title {
            color:inherit;font-size:.95rem;font-weight:700;line-height:1.25;
        }
        .review-selected-job-location {
            color:inherit;font-size:.8rem;line-height:1.25;
            opacity:.66;margin-top:.34rem;
        }
        @media (max-width:1099px) {
            .st-key-review_job_list_panel,.st-key-review_detail_shell,
            .st-key-review_detail_panel,.st-key-review_job_list {
                height:auto !important;min-height:0;overflow:visible;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_manual_workspace_styles() -> None:
    """Keep Add Target Job within the desktop viewport."""
    st.markdown(
        """
        <style>
        .st-key-manual_jd_panel,
        .st-key-manual_verify_panel {
            height:calc(100vh - 11.2rem) !important;
            max-height:calc(100vh - 11.2rem) !important;
            min-height:0 !important;
            flex:0 0 calc(100vh - 11.2rem) !important;
            overflow-y:auto !important;
            overscroll-behavior:contain;
            scrollbar-gutter:stable;
            padding-right:.45rem;
        }
        .st-key-manual_jd_panel [data-testid="stTextArea"] textarea {
            height:calc(100vh - 26rem) !important;
            min-height:220px;
        }
        .st-key-manual_verify_panel [data-testid="stForm"] {
            border:0 !important;padding:0 !important;margin-bottom:.35rem;
        }
        @media (max-width:1099px) {
            .st-key-manual_jd_panel,
            .st-key-manual_verify_panel {
                height:auto !important;
                max-height:none !important;
                min-height:0 !important;
                flex:1 1 auto !important;
                overflow:visible !important;
            }
            .st-key-manual_jd_panel [data-testid="stTextArea"] textarea {
                height:340px !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_cover_letter_workspace_styles() -> None:
    """Keep cover-letter context and draft aligned in a desktop workspace."""
    st.markdown(
        """
        <style>
        .st-key-cover_letter_context_panel,
        .st-key-cover_letter_document_panel {
            height:calc(100vh - 4.7rem) !important;
            max-height:calc(100vh - 4.7rem) !important;
            min-height:0 !important;
            flex:0 0 calc(100vh - 4.7rem) !important;
            overflow-y:auto !important;
            overscroll-behavior:contain;
            scrollbar-gutter:stable;
            margin-top:0 !important;padding-right:.4rem;
        }
        .st-key-cover_letter_context_panel > div,
        .st-key-cover_letter_document_panel > div {
            padding-top:0 !important;
        }
        .st-key-cover_letter_context_panel [data-testid="stColumn"] {
            min-width:100% !important;
        }
        @media (max-width:1099px) {
            .st-key-cover_letter_context_panel,
            .st-key-cover_letter_document_panel {
                height:auto !important;max-height:none !important;
                min-height:0 !important;flex:1 1 auto !important;
                overflow:visible !important;padding-right:0;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
