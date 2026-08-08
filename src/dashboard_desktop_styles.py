"""Page-scoped desktop workspace styles for dense Streamlit workflows."""

from __future__ import annotations

import streamlit as st


def render_review_workspace_styles() -> None:
    """Keep the Review Jobs master-detail workspace inside the viewport."""
    st.markdown(
        """
        <style>
        .st-key-review_job_list_panel {
            height:calc(100vh - 4.7rem);overflow:hidden;
            margin-top:0 !important;min-width:0 !important;
        }
        .st-key-review_detail_shell {
            height:calc(100vh - 4.7rem);max-height:calc(100vh - 4.7rem);
            overflow-y:auto;overflow-x:hidden;margin-top:0 !important;min-width:0 !important;
            overscroll-behavior:contain;scrollbar-gutter:stable;padding:0 .35rem 1.35rem 0;
            scrollbar-width:thin;scrollbar-color:var(--app-border-strong,#cdd3da) transparent;
        }
        .st-key-review_job_list_panel [data-testid="stColumn"],
        .st-key-review_detail_shell [data-testid="stColumn"] {min-width:0 !important}
        .st-key-review_job_list_panel {
            padding-right:.3rem;border-right:1px solid var(--app-border,#dfe3e8);
        }
        .st-key-review_job_list_panel > div,
        .st-key-review_detail_shell > div {padding-top:0 !important}
        .saved-jobs-title {font-size:1rem;font-weight:720;letter-spacing:-.018em}
        .saved-jobs-mobile-title {display:none}
        .st-key-review_add_target_job button {min-width:2.15rem !important;padding:.25rem !important;border:0 !important}
        .st-key-review_add_target_job button p,
        .st-key-review_job_list_panel [data-testid="stPopover"] button p,
        [class*="st-key-evidence_map_toggle_"] button p {
            position:absolute !important;width:1px !important;height:1px !important;
            padding:0 !important;margin:-1px !important;overflow:hidden !important;
            clip-path:inset(50%) !important;white-space:nowrap !important;border:0 !important;
        }
        .st-key-review_job_list_panel [data-testid="stPopover"] button {
            min-width:2.15rem !important;padding:.3rem !important;
        }
        .st-key-review_job_list_panel [data-testid="stTextInput"] {margin-top:.55rem}
        .st-key-review_job_list_panel [data-baseweb="button-group"] {
            border-bottom:1px solid var(--app-border,#dfe3e8);padding-bottom:.1rem;
            display:flex !important;flex-wrap:nowrap !important;
        }
        .st-key-review_job_list_panel [data-baseweb="button-group"] button {
            min-width:0 !important;flex:1 1 0 !important;padding:.38rem .08rem !important;border:0 !important;
            border-radius:0 !important;font-size:.68rem !important;white-space:nowrap;
        }
        .st-key-review_job_list_panel [data-baseweb="button-group"] button p {font-size:.68rem !important}
        .st-key-review_job_list_panel [data-baseweb="button-group"] button[aria-pressed="true"] {
            color:#e13f35 !important;box-shadow:inset 0 -2px 0 #f04438 !important;background:transparent !important;
        }
        .st-key-review_mobile_job_picker {display:none}
        .selected-job-context {
            min-height:2.8rem;display:flex;align-items:center;gap:.72rem;
            color:var(--app-text,inherit);white-space:nowrap;overflow:hidden;min-width:0;
        }
        .selected-job-context-icon {
            flex:0 0 auto;font-family:"Material Symbols Rounded";font-size:1.55rem;color:#303640;
        }
        .selected-job-context-role {font-size:1.2rem;font-weight:760;letter-spacing:-.025em;overflow:hidden;text-overflow:ellipsis;min-width:0}
        .selected-job-context-meta {overflow:hidden;text-overflow:ellipsis;min-width:0}
        .selected-job-context-meta,.selected-job-context-separator {color:var(--app-muted,inherit);font-size:.84rem}
        .review-decision-grid {background:#fff !important;margin:.55rem 0 .65rem !important}
        .review-decision-field {display:flex;align-items:center;gap:.7rem;padding:.72rem .85rem !important}
        .review-decision-icon {
            font-family:"Material Symbols Rounded";font-size:1.55rem;color:#39414d;font-weight:400;line-height:1;
        }
        .review-decision-value {font-size:1.05rem !important}
        .review-decision-value small {font-size:.72rem;color:var(--app-muted,inherit);font-weight:500}
        .review-decision-success .review-decision-value {color:#0f8a4d}
        .review-decision-warning .review-decision-value {color:#c77b00}
        .review-decision-danger .review-decision-value {color:#c33f45}
        .review-action-note {display:none}
        .st-key-review_detail_panel {
            min-height:300px;overflow:visible;padding-right:.25rem;margin-top:-.65rem;
        }
        .st-key-review_detail_tabs {margin:.35rem 0 .2rem}
        .st-key-review_detail_tabs [data-baseweb="button-group"] {
            display:flex !important;gap:0 !important;width:fit-content !important;
            border:1px solid var(--app-border-strong,#cdd3da) !important;
            border-radius:7px !important;background:#f8f9fb !important;overflow:hidden;
        }
        .st-key-review_detail_tabs [data-baseweb="button-group"] button {
            position:relative;flex:0 0 auto !important;min-width:7.2rem !important;height:2.35rem !important;
            padding:.3rem .72rem !important;border:0 !important;border-right:1px solid #dfe3e8 !important;
            border-radius:0 !important;background:transparent !important;box-shadow:none !important;color:#303640 !important;
        }
        .st-key-review_detail_tabs [data-baseweb="button-group"] button:last-child {border-right:0 !important}
        .st-key-review_detail_tabs [data-baseweb="button-group"] button p {font-size:.8rem !important;font-weight:620 !important}
        .st-key-review_detail_tabs [data-baseweb="button-group"] button[aria-pressed="true"] {
            color:var(--app-accent,#d94f55) !important;background:#fff !important;
            box-shadow:inset 0 -2px 0 var(--app-accent,#d94f55) !important;
        }
        .st-key-review_job_list {
            height:calc(100vh - 14.5rem) !important;min-height:24rem;overflow-y:auto;
            overscroll-behavior:contain;scrollbar-gutter:stable;
            scrollbar-width:thin;scrollbar-color:var(--app-border-strong,#cdd3da) transparent;
        }
        [class*="st-key-review_job_row_"] {
            background:transparent;border-bottom:1px solid var(--app-border,#dfe3e8);
            padding:.72rem .78rem .66rem;border-left:3px solid transparent;
            position:relative;min-height:5.35rem;
            transition:background-color 140ms ease,border-color 140ms ease;
        }
        [class*="st-key-review_job_row_"]:hover {
            background:#f7f8fa;
        }
        [class*="st-key-review_job_row_"]:has([data-testid="stBaseButton-secondary"]) {
            border-left:3px solid var(--app-accent,#d94f55);
            background:var(--app-accent-soft,#fff0f1);
        }
        [class*="st-key-review_job_row_"] [class*="st-key-review_job_select_"] {
            position:absolute;top:0;right:.55rem;bottom:-1px;left:-3px;z-index:2;
        }
        [class*="st-key-review_job_row_"] [data-testid="stButton"],
        [class*="st-key-review_job_row_"] [data-testid="stButton"] button {
            width:100%;height:100%;
        }
        [class*="st-key-review_job_row_"] [data-testid="stButton"] button {
            opacity:0;border:0 !important;padding:0 !important;
        }
        [class*="st-key-review_job_row_"]:has(button:focus-visible) {
            outline:2px solid var(--app-accent,var(--primary-color,currentColor));outline-offset:-2px;
        }
        .review-job-row-copy {pointer-events:none}
        .review-job-row-heading {display:flex;align-items:baseline;justify-content:space-between;gap:.65rem}
        .review-job-row-title {
            color:var(--app-text,inherit);font-size:.93rem;font-weight:680;line-height:1.3;
            letter-spacing:-.012em;min-width:0;overflow-wrap:anywhere;
            display:-webkit-box;-webkit-box-orient:vertical;-webkit-line-clamp:2;overflow:hidden;
        }
        .review-job-row-score {font-size:.82rem;font-weight:750;flex:0 0 auto}
        .review-job-row-score-strong {color:#0d8a4b}.review-job-row-score-review {color:#c77700}.review-job-row-score-weak {color:#6d7580}
        .review-job-row-company,.review-job-row-context {color:var(--app-muted,inherit);font-size:.76rem;line-height:1.35;margin-top:.28rem;max-width:100%;overflow-wrap:anywhere}
        .review-job-row-context {
            display:-webkit-box;-webkit-box-orient:vertical;-webkit-line-clamp:2;
            overflow:hidden;white-space:normal;
        }
        .review-selected-job-header {margin:0;padding:0}
        .review-selected-job-title {
            color:var(--app-text,inherit);font-size:1rem;font-weight:720;line-height:1.25;
            letter-spacing:-.016em;
        }
        .review-selected-job-location {
            color:var(--app-muted,inherit);font-size:.78rem;line-height:1.25;
            margin-top:.34rem;
        }
        @media (min-width:1300px) and (max-width:1499px) {
            .review-decision-field {padding:.62rem .58rem !important;gap:.48rem}
            .review-decision-icon {font-size:1.35rem}
            .review-decision-value {font-size:.92rem !important}
            .selected-job-context {gap:.45rem}
            .selected-job-context-role {font-size:1.05rem}
            .selected-job-context-meta {font-size:.76rem}
        }
        @media (max-width:1299px) {
            [data-testid="stHorizontalBlock"]:has(.st-key-review_job_list_panel):has(.st-key-review_detail_shell) {
                flex-direction:column !important;gap:.65rem !important;
            }
            [data-testid="stHorizontalBlock"]:has(.st-key-review_job_list_panel):has(.st-key-review_detail_shell)
            > [data-testid="stColumn"] {
                flex:1 1 100% !important;width:100% !important;max-width:100% !important;
            }
            .st-key-review_job_list_panel,.st-key-review_detail_shell,
            .st-key-review_detail_panel,.st-key-review_job_list {
                height:auto !important;min-height:0;overflow:visible;
            }
            .st-key-review_job_list_panel {
                border-right:0;border-bottom:1px solid var(--app-border,#dfe3e8);
                padding:0 0 .75rem;
            }
            .saved-jobs-desktop-title {display:none}.saved-jobs-mobile-title {display:inline}
            .st-key-review_mobile_job_picker {display:block;margin-top:.4rem}
            .st-key-review_job_list_panel [data-testid="stTextInput"],
            .st-key-review_job_list_panel [data-baseweb="button-group"],
            .st-key-review_job_list_panel [data-testid="stPopover"],
            .st-key-review_job_list {display:none !important}
            .st-key-review_job_list_panel [data-testid="stLayoutWrapper"]:has(.st-key-review_job_list),
            .st-key-review_job_list_panel [data-testid="stLayoutWrapper"]:has([data-testid="stTextInput"]) {
                display:none !important;height:0 !important;margin:0 !important;
            }
            .selected-job-context {white-space:normal;flex-wrap:wrap;gap:.35rem .55rem}
            .selected-job-context-role {font-size:1.05rem;max-width:calc(100% - 2.2rem)}
            .selected-job-context-meta {font-size:.76rem}
            .st-key-review_detail_shell [data-baseweb="button-group"] {
                display:flex;flex-wrap:nowrap;overflow-x:auto;
                scrollbar-width:none;padding-bottom:1px;
            }
            .st-key-review_detail_shell [data-baseweb="button-group"]::-webkit-scrollbar {
                display:none;
            }
            .st-key-review_detail_shell [data-baseweb="button-group"] button {
                flex:0 0 auto;padding-left:.55rem;padding-right:.55rem;
            }
            .st-key-review_detail_tabs [data-baseweb="button-group"] {gap:0 !important}
            .st-key-review_detail_tabs [data-baseweb="button-group"] button {
                min-width:6.5rem !important;padding-left:.5rem !important;padding-right:.5rem !important;
            }
        }
        @media (min-width:1100px) and (max-width:1299px) {
            .st-key-review_detail_shell {
                height:calc(100vh - 4.7rem) !important;
                max-height:calc(100vh - 4.7rem) !important;
                overflow-y:auto !important;overflow-x:hidden !important;
                padding-bottom:1.35rem !important;
            }
        }
        @media (max-width:760px) {
            .review-decision-grid {grid-template-columns:repeat(2,minmax(0,1fr)) !important}
            .review-decision-field {padding:.62rem .55rem !important;gap:.48rem}
            .review-decision-icon {font-size:1.3rem}.review-decision-value {font-size:.9rem !important}
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
        @media (min-width:1100px) {
            div[data-testid="stMainBlockContainer"]:has(.desktop-workspace-marker):has(.st-key-manual_back_bar) {
                margin-top:0 !important;
            }
        }
        .st-key-manual_back_bar {margin:.45rem 0 .9rem}
        .st-key-manual_back_to_review_jobs button {
            border-color:var(--app-border-strong,#cdd3da) !important;
            background:var(--app-surface,#fff) !important;
            padding-left:.8rem !important;padding-right:.9rem !important;
        }
        .st-key-manual_jd_panel,
        .st-key-manual_verify_panel {
            height:auto !important;max-height:none !important;
            min-height:34rem !important;overflow:visible !important;
            padding:.95rem 1rem .8rem !important;
            background:var(--app-surface,#fff);
            border:1px solid var(--app-border,#dfe3e8) !important;
            border-radius:var(--app-radius-md,12px) !important;
            box-shadow:var(--app-shadow,0 8px 24px rgba(20,24,32,.04));
        }
        .st-key-manual_jd_panel [data-testid="stTextArea"] textarea {
            height:clamp(260px,34vh,390px) !important;min-height:260px;
        }
        .st-key-manual_verify_panel [data-testid="stForm"] {
            border:0 !important;padding:0 !important;margin-bottom:.35rem;
        }
        @media (max-width:1099px) {
            .st-key-manual_jd_panel,
            .st-key-manual_verify_panel {
                height:auto !important;max-height:none !important;
                min-height:0 !important;flex:1 1 auto !important;
                overflow:visible !important;padding:.8rem !important;
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
        @media (min-width:1100px) {
            [data-testid="stHorizontalBlock"]:has(.cover-letter-editor-detail)
            > [data-testid="stColumn"]:has(.st-key-cover_letter_context_panel) {
                flex:0 0 calc(24% - 2.1rem) !important;
                width:calc(24% - 2.1rem) !important;max-width:calc(24% - 2.1rem) !important;
            }
            [data-testid="stHorizontalBlock"]:has(.cover-letter-editor-detail)
            > [data-testid="stColumn"]:has(.st-key-cover_letter_document_panel) {
                flex:0 0 calc(76% - 2.1rem) !important;
                width:calc(76% - 2.1rem) !important;max-width:calc(76% - 2.1rem) !important;
            }
        }
        .cover-letter-editor-state {display:none}
        .st-key-cover_letter_document_panel [data-testid="stElementContainer"]:has([data-testid="stTextArea"]) {
            height:auto !important;min-height:0 !important;
        }
        .st-key-cover_letter_document_panel [data-testid="stTextArea"] {
            height:548px !important;min-height:0 !important;
            transition:height 160ms ease;
        }
        .st-key-cover_letter_document_panel [data-testid="stTextAreaRootElement"] {
            height:calc(100% - 1.75rem) !important;min-height:0 !important;
        }
        .st-key-cover_letter_document_panel [data-testid="stTextAreaRootElement"] [data-baseweb="base-input"],
        .st-key-cover_letter_document_panel [data-testid="stTextArea"] textarea {
            height:100% !important;min-height:0 !important;
        }
        .st-key-cover_letter_demo_draft {
            max-height:390px;overflow-y:auto;padding:.72rem .88rem !important;
            transition:max-height 160ms ease;
            scrollbar-width:thin;scrollbar-color:var(--app-border-strong,#cdd3da) transparent;
        }
        .st-key-cover_letter_document_panel:has(.cover-letter-editor-supporting) [data-testid="stTextArea"] {
            height:258px !important;
        }
        .st-key-cover_letter_document_panel:has(.cover-letter-editor-detail) [data-testid="stTextArea"] {
            height:228px !important;
        }
        .st-key-cover_letter_document_panel:has(.cover-letter-editor-supporting) .st-key-cover_letter_demo_draft {
            max-height:190px !important;
        }
        .st-key-cover_letter_document_panel:has(.cover-letter-editor-detail) .st-key-cover_letter_demo_draft {
            max-height:150px !important;
        }
        .st-key-cover_letter_supporting_toggle {margin-top:.35rem}
        .st-key-cover_letter_supporting_toggle [data-testid="stButton"] button {
            min-height:2.7rem !important;justify-content:flex-start !important;
            padding:.48rem .72rem !important;background:var(--app-surface,#fff) !important;
            border-color:var(--app-border,#dfe3e8) !important;box-shadow:none !important;
        }
        .st-key-cover_letter_supporting_toggle [data-testid="stButton"] button p {
            flex:1;text-align:left;font-weight:620;font-size:.86rem !important;
        }
        .st-key-cover_letter_supporting_panel {
            margin-top:.45rem;padding:.72rem .78rem .6rem !important;
            box-shadow:none !important;border-radius:9px !important;
        }
        .st-key-cover_letter_supporting_panel [data-testid="stDataFrame"] {
            margin-bottom:.35rem;
        }
        .st-key-cover_letter_focused_detail {
            margin-top:.55rem;padding:.65rem .72rem .72rem !important;
            border-top:2px solid var(--app-accent,#d94f55) !important;
            box-shadow:0 8px 24px rgba(20,24,32,.06) !important;
        }
        .st-key-cover_letter_focused_detail_body {
            margin-top:.2rem;padding:.65rem .72rem !important;
            background:#f8f9fb;border-radius:8px;overflow-x:hidden !important;
        }
        .st-key-cover_letter_focused_detail_body [data-testid="stMarkdownContainer"] {
            font-size:.84rem;line-height:1.52;overflow-wrap:anywhere;
        }
        .st-key-cover_letter_focused_detail_body code {white-space:normal;overflow-wrap:anywhere}
        @media (max-width:1099px) {
            .st-key-cover_letter_context_panel,
            .st-key-cover_letter_document_panel {
                height:auto !important;max-height:none !important;
                min-height:0 !important;flex:1 1 auto !important;
                overflow:visible !important;padding-right:0;
            }
            .st-key-cover_letter_focused_detail {margin-left:0;margin-right:0}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
