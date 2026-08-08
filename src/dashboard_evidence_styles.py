"""Scoped styles for the requirement evidence map."""

from __future__ import annotations

import streamlit as st


def render_evidence_map_styles() -> None:
    """Install the evidence-map visual vocabulary."""
    st.markdown(
        """
        <style>
        .evidence-map-title {
            font-size:1rem;font-weight:750;letter-spacing:-.018em;margin:.45rem 0 .5rem;
            white-space:nowrap;
        }
        .st-key-evidence_map_segment {margin:.34rem 0 .48rem;width:100% !important}
        .st-key-evidence_map_segment [data-baseweb="button-group"] {
            display:flex !important;gap:0 !important;width:100% !important;
            border:1px solid var(--app-border,#dfe3e8) !important;
            border-radius:6px !important;background:#fafbfc !important;overflow:hidden;
        }
        .st-key-evidence_map_segment [data-baseweb="button-group"] button {
            position:relative;flex:1 1 0 !important;min-width:0 !important;height:2.05rem !important;
            border:0 !important;border-right:1px solid #e0e4e8 !important;border-radius:0 !important;
            background:transparent !important;padding:.2rem .35rem !important;font-size:.78rem !important;box-shadow:none !important;
        }
        .st-key-evidence_map_segment [data-baseweb="button-group"] button:last-child {border-right:0 !important}
        .st-key-evidence_map_segment [data-baseweb="button-group"] button p {font-size:.78rem !important;white-space:nowrap}
        .st-key-evidence_map_segment [data-baseweb="button-group"] button[aria-pressed="true"] {
            color:#d94147 !important;background:#fff !important;
            box-shadow:inset 0 -2px 0 var(--app-accent,#e05258) !important;
        }
        .st-key-evidence_map_segment [data-baseweb="button-group"] button:nth-child(n+2) p::before {
            content:"";display:inline-block;width:.48rem;height:.48rem;border-radius:50%;
            margin:0 .3rem .02rem 0;background:#1b8a55;
        }
        .st-key-evidence_map_segment [data-baseweb="button-group"] button:nth-child(3) p::before {background:#c98519}
        .st-key-evidence_map_segment [data-baseweb="button-group"] button:nth-child(4) p::before {background:#9299a3}
        .st-key-evidence_view_analysis {display:flex;justify-content:flex-end;margin:.52rem 0 .72rem}
        .st-key-evidence_view_analysis button {
            border:0 !important;background:transparent !important;box-shadow:none !important;
            color:#39414d !important;padding:.3rem .1rem !important;white-space:nowrap;
        }
        .st-key-evidence_view_analysis button:hover {color:var(--app-accent,#d94f55) !important}
        .evidence-map-table-head {
            display:grid;grid-template-columns:minmax(7.5rem,1fr) minmax(0,2.15fr) minmax(7rem,.65fr);
            gap:.9rem;align-items:start;
            border:0;border-bottom:1px solid var(--app-border,#dfe3e8);border-radius:0;
            background:#fff;color:#57606d;padding:.5rem .72rem;font-size:.68rem;font-weight:650;
        }
        [class*="st-key-evidence_map_row_"] {
            border:0;border-bottom:1px solid var(--app-border,#dfe3e8);
            padding:.55rem .72rem;font-size:.79rem;line-height:1.38;
            transition:background-color 140ms ease;min-height:3.75rem;
        }
        [class*="st-key-evidence_map_row_"]:hover {background:#fafbfc}
        .evidence-map-status {
            display:inline-block;border-radius:5px;padding:.18rem .48rem;
            background:#e8f4ec;color:#166534;font-size:.69rem;font-weight:720;
            white-space:nowrap;
        }
        .evidence-map-status-partial {background:#fff2d8;color:#8a4d00}
        .evidence-map-status-no-support {background:#edf0f3;color:#46515f}
        .evidence-map-requirement {display:flex;flex-direction:column;gap:.2rem;color:var(--app-text,inherit)}
        .evidence-map-requirement strong {font-size:.82rem;font-weight:700;line-height:1.35}
        .evidence-map-requirement span {color:var(--app-muted,inherit);font-size:.69rem}
        .evidence-map-quote {color:var(--app-text,inherit);line-height:1.45}
        .evidence-map-expanded {
            color:#56606c;margin:.4rem 0 0 26%;padding:.58rem .75rem;
            border-left:2px solid #d7dce2;background:#f8f9fa;font-size:.75rem;
        }
        .evidence-map-filter-empty {color:var(--app-muted,inherit);padding:1rem .15rem;font-size:.82rem}
        [class*="st-key-evidence_map_toggle_"] button {min-width:2rem !important;padding:.25rem !important;border:0 !important}
        .evidence-gap-row {
            display:grid;grid-template-columns:auto 8rem minmax(0,1fr);align-items:center;
            gap:.65rem;border:0;border-bottom:1px solid #efc87f;background:#fffaf0;border-radius:0;
            margin-top:.28rem;padding:.52rem .72rem;font-size:.74rem;line-height:1.35;
        }
        .evidence-gap-row i {font-family:"Material Symbols Rounded";font-style:normal;color:#a56500;font-size:1.05rem}
        .evidence-gap-row strong {color:#7a4a00}
        .evidence-gap-row span {color:#5e4b2d}
        [data-testid="stDialog"] {justify-content:flex-end !important;padding:0 !important}
        [data-testid="stDialog"] > div {
            width:100% !important;height:100vh !important;align-items:stretch !important;
            justify-content:flex-end !important;
        }
        [data-testid="stDialog"] [role="dialog"] {
            width:min(29rem,94vw) !important;max-width:min(29rem,94vw) !important;
            height:100vh !important;max-height:100vh !important;margin:0 !important;
            border-radius:0 !important;border:0 !important;border-left:1px solid #e1e5ea !important;
            box-shadow:-18px 0 48px rgba(25,31,40,.13) !important;
        }
        [data-testid="stDialog"] [role="dialog"] > div {padding:1.1rem 1.2rem 1.5rem !important}
        [data-testid="stDialog"] button[aria-label="Close"] {
            border:0 !important;outline:0 !important;box-shadow:none !important;background:transparent !important;
        }
        .analysis-drawer-section {
            font-size:.9rem;font-weight:730;margin:1.15rem 0 .5rem;padding-top:.85rem;
            border-top:1px solid #e6e9ed;
        }
        .analysis-drawer-section:first-child {border-top:0;margin-top:.1rem;padding-top:0}
        .st-key-evidence_analysis_inspector {
            border-left:1px solid var(--app-border,#dfe3e8);padding-left:.9rem;
            scrollbar-width:thin;scrollbar-color:#cdd3da transparent;
        }
        .st-key-review_detail_shell:has(.st-key-evidence_workspace) {
            overflow-y:clip !important;
        }
        .st-key-evidence_main_scroll,
        .st-key-evidence_analysis_inspector {
            height:max(15rem,calc(100vh - 27.5rem)) !important;
            min-height:15rem;max-height:none;
            flex:0 0 max(15rem,calc(100vh - 27.5rem)) !important;
            overflow-y:auto;overscroll-behavior:contain;scrollbar-gutter:stable;
            scrollbar-width:thin;scrollbar-color:#cdd3da transparent;
        }
        .st-key-evidence_main_scroll {padding-right:.42rem}
        .analysis-inspector-title {
            font-size:1rem;font-weight:750;letter-spacing:-.015em;margin:.4rem 0 .55rem;
        }
        .st-key-evidence_analysis_inspector .analysis-drawer-section {
            margin:.78rem 0 .35rem;padding-top:.62rem;font-size:.8rem;
        }
        .st-key-evidence_analysis_inspector .analysis-drawer-metric,
        .st-key-evidence_analysis_inspector .analysis-drawer-signal {font-size:.72rem;padding:.3rem 0}
        .st-key-evidence_analysis_inspector .analysis-detail-group {
            grid-template-columns:6.5rem minmax(0,1fr);gap:.5rem;padding:.3rem 0;font-size:.7rem;
        }
        @media (min-width:1500px) {
            .st-key-evidence_view_analysis {display:none !important}
            .st-key-evidence_main_scroll,
            .st-key-evidence_analysis_inspector {
                height:max(24rem,calc(100vh - 19rem)) !important;
                flex-basis:max(24rem,calc(100vh - 19rem)) !important;
            }
        }
        @media (max-width:1499px) {
            .st-key-evidence_workspace [data-testid="stColumn"]:has(.st-key-evidence_main_column) {
                flex:1 1 100% !important;width:100% !important;max-width:100% !important;
            }
            .st-key-evidence_workspace [data-testid="stColumn"]:has(.st-key-evidence_inspector_column) {
                display:none !important;
            }
            .st-key-evidence_workspace [data-testid="stHorizontalBlock"]:has(.st-key-evidence_main_column) {
                gap:0 !important;
            }
        }
        .analysis-drawer-metric {display:flex;justify-content:space-between;gap:1rem;padding:.38rem 0;font-size:.78rem}
        .analysis-drawer-metric strong {font-weight:650}.analysis-drawer-direct {color:#167347}
        .analysis-drawer-partial {color:#9a5b00}.analysis-drawer-missing {color:#68717e}.analysis-drawer-total {color:#303640}
        .analysis-drawer-signal {display:flex;justify-content:space-between;gap:1rem;padding:.52rem 0;border-bottom:1px solid #edf0f3;font-size:.78rem}
        .analysis-drawer-signal div {display:flex;flex-direction:column;gap:.14rem}.analysis-drawer-signal span {color:#737c88;font-size:.7rem}
        .analysis-drawer-signal em {font-style:normal;font-weight:650;white-space:nowrap}
        .analysis-drawer-no-support {color:#68717e}
        .analysis-detail-lead {color:var(--app-muted,inherit);font-size:.76rem;margin:.15rem 0 .75rem}
        .analysis-detail-group {display:grid;grid-template-columns:8rem minmax(0,1fr);gap:.8rem;padding:.42rem 0;border-bottom:1px solid #edf0f3;font-size:.77rem}
        .analysis-detail-group strong {color:#48515e}.analysis-detail-group span {color:#687280}
        .analysis-detail-section {font-size:.82rem;font-weight:700;margin:1rem 0 .45rem}
        .analysis-source-item {color:#596370;font-size:.76rem;line-height:1.45;padding:.32rem 0;overflow-wrap:anywhere}
        @media (max-width: 760px) {
            .st-key-review_detail_shell:has(.st-key-evidence_workspace) {overflow-y:visible !important}
            .st-key-evidence_main_scroll,.st-key-evidence_analysis_inspector {
                height:auto !important;min-height:0;max-height:none;overflow:visible;
                flex:1 1 auto !important;
            }
            .evidence-map-title {margin-bottom:.35rem}
            .st-key-evidence_map_segment [data-baseweb="button-group"] {gap:0 !important;overflow-x:auto}
            .st-key-evidence_view_analysis {justify-content:flex-start;margin-top:0}
            .evidence-map-table-head {display:none}
            [class*="st-key-evidence_map_row_"] [data-testid="stHorizontalBlock"] {gap:.35rem !important}
            .evidence-map-expanded {margin-left:0}
            .evidence-gap-row {grid-template-columns:auto 1fr;gap:.3rem .55rem}
            .evidence-gap-row span {grid-column:1 / -1;padding-top:.2rem}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
