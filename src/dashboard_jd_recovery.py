"""Incomplete-JD recovery controls shared by Review Jobs sections."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Callable, Protocol

import streamlit as st


class JDRecoveryServices(Protocol):
    """Operations needed by the recovery controls."""

    @property
    def demo_mode_enabled(self) -> Callable[[], bool]: ...

    @property
    def enrich_saved_job_description(self) -> Callable[..., dict[str, Any]]: ...

    @property
    def enrich_saved_job_description_from_url(self) -> Callable[..., dict[str, Any]]: ...

    @property
    def jsearch_configured(self) -> Callable[[], bool]: ...

    @property
    def replace_saved_job_description(self) -> Callable[..., dict[str, Any]]: ...


def _job_url_from_markdown(selected_path: Path) -> str:
    try:
        text = selected_path.read_text(encoding="utf-8")
    except OSError:
        return ""
    match = re.search(r"^Job URL:\s*(\S+)\s*$", text, flags=re.MULTILINE | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _notice_key(key_prefix: str) -> str:
    return f"{key_prefix}_recovery_notice"


def _set_notice(key_prefix: str, tone: str, message: str) -> None:
    st.session_state[_notice_key(key_prefix)] = (tone, message)


def _render_notice(key_prefix: str) -> None:
    notice = st.session_state.get(_notice_key(key_prefix))
    if not isinstance(notice, tuple) or len(notice) != 2:
        return
    tone, message = notice
    if tone == "success":
        st.success(str(message))
    elif tone == "warning":
        st.warning(str(message))
    else:
        st.info(str(message))


@st.dialog("Paste full job description", width="large")
def _paste_full_jd_dialog(
    selected_path: Path,
    *,
    key_prefix: str,
    services: JDRecoveryServices,
) -> None:
    st.caption(
        "Paste the employer's complete responsibilities and qualifications. "
        "The local record is replaced only after completeness checks pass."
    )
    pasted_jd = st.text_area(
        "Full job description",
        key=f"{key_prefix}_manual_full_jd",
        height=300,
        placeholder="Paste the complete responsibilities and qualifications here.",
    )
    if st.button(
        "Verify and save",
        key=f"{key_prefix}_save_manual_full_jd",
        type="primary",
        disabled=not pasted_jd.strip(),
    ):
        result = services.replace_saved_job_description(selected_path, pasted_jd)
        if result.get("updated"):
            _set_notice(key_prefix, "success", str(result["message"]))
        else:
            _set_notice(
                key_prefix,
                "warning",
                str(result.get("message", "The pasted JD was not complete enough.")),
            )
        st.rerun()


def render_jd_workspace_styles() -> None:
    st.markdown(
        """
        <style>
        .jd-recovery-head {display:flex;justify-content:space-between;gap:1rem;align-items:baseline;margin:.2rem 0 .25rem}
        .jd-recovery-head strong {font-size:.96rem;color:#242a33}
        .jd-recovery-head span {font-size:.72rem;color:#697381}
        .jd-recovery-note {font-size:.72rem;color:#697381;line-height:1.4;margin:0 0 .48rem}
        .st-key-jd_recovery_actions {margin:.15rem 0 .5rem}
        .st-key-jd_recovery_actions [data-testid="stHorizontalBlock"] {gap:.45rem !important}
        .st-key-jd_recovery_actions button,
        .st-key-jd_recovery_actions a {min-height:2.25rem !important;font-size:.75rem !important}
        .jd-document-head {display:flex;justify-content:space-between;align-items:center;gap:1rem;margin:.35rem 0 .28rem}
        .jd-document-head strong {font-size:1rem;letter-spacing:-.015em}
        .jd-document-head span {font-size:.72rem;color:#687280}
        .st-key-jd_document_body {
            border-top:1px solid var(--app-border,#dfe3e8);padding:1rem .35rem 1.6rem;
            font-size:.84rem;line-height:1.65;max-width:68rem;color:#303640;
        }
        .st-key-jd_document_body p {margin:.22rem 0 .78rem !important;max-width:76ch}
        .st-key-jd_document_body h1,.st-key-jd_document_body h2,.st-key-jd_document_body h3 {
            font-size:.92rem !important;margin:1.05rem 0 .45rem !important;
            padding-left:.62rem;border-left:3px solid var(--app-accent,#d94f55);
            letter-spacing:-.01em !important;
        }
        .st-key-jd_document_body ul {margin:.28rem 0 .9rem !important;padding-left:1.35rem !important;max-width:80ch}
        .st-key-jd_document_body li {margin:.28rem 0 !important;padding-left:.16rem;line-height:1.55}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_full_jd_recovery(
    selected_path: Path,
    quality: dict[str, Any],
    *,
    key_prefix: str,
    services: JDRecoveryServices,
    job_url: str = "",
) -> bool:
    """Render reliable and fallback JD recovery actions."""
    if quality.get("reliable_scoring_ready", False):
        return True
    if services.demo_mode_enabled():
        st.caption(
            "This fictional sample intentionally uses an incomplete JD to show why "
            "fit remains provisional."
        )
        return False
    render_jd_workspace_styles()
    resolved_url = job_url or _job_url_from_markdown(selected_path)
    st.markdown(
        '<div class="jd-recovery-head"><strong>Complete this job description</strong></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="jd-recovery-note">Employer sites often block server-side requests. '
        "Open the posting in your browser, try the safe direct fetch, paste the text, "
        "or search the configured provider index.</div>",
        unsafe_allow_html=True,
    )
    _render_notice(key_prefix)
    with st.container(key="jd_recovery_actions"):
        open_column, fetch_column, paste_column, provider_column = st.columns(4, gap="small")
        with open_column:
            if resolved_url:
                st.link_button(
                    "Open job page",
                    resolved_url,
                    icon=":material/open_in_new:",
                    type="secondary",
                    width="stretch",
                )
            else:
                st.button("Open job page", disabled=True, width="stretch")
        with fetch_column:
            fetch_clicked = st.button(
                "Fetch page",
                key=f"{key_prefix}_fetch_original_jd",
                icon=":material/download:",
                type="secondary",
                width="stretch",
                disabled=not resolved_url,
                help="Attempt a safe server-side fetch. Some employer sites block this request.",
            )
        with paste_column:
            paste_clicked = st.button(
                "Paste full JD",
                key=f"{key_prefix}_open_manual_full_jd",
                icon=":material/content_paste:",
                type="secondary",
                width="stretch",
            )
        with provider_column:
            provider_clicked = st.button(
                "Search provider",
                key=f"{key_prefix}_find_full_jd",
                icon=":material/search:",
                type="secondary",
                width="stretch",
                disabled=not services.jsearch_configured(),
                help=(
                    "Search the configured JSearch index for a matching full posting."
                    if services.jsearch_configured()
                    else "JSearch is unavailable because JSEARCH_API_KEY is not configured."
                ),
            )
    if paste_clicked:
        _paste_full_jd_dialog(selected_path, key_prefix=key_prefix, services=services)
    if fetch_clicked:
        try:
            with st.spinner("Reading and verifying the original job page…"):
                result = services.enrich_saved_job_description_from_url(selected_path)
        except Exception as error:  # noqa: BLE001
            _set_notice(key_prefix, "warning", f"Direct fetch could not read this page: {error}")
            st.rerun()
        if result.get("updated"):
            _set_notice(key_prefix, "success", str(result["message"]))
        else:
            _set_notice(
                key_prefix,
                "warning",
                str(result.get("message", "No complete JD was found on the original page.")),
            )
        st.rerun()
    if provider_clicked:
        try:
            with st.spinner("Checking this workspace, then the provider index…"):
                result = services.enrich_saved_job_description(selected_path)
        except Exception as error:  # noqa: BLE001
            _set_notice(key_prefix, "warning", f"Provider search failed: {error}")
            st.rerun()
        if result.get("updated"):
            _set_notice(key_prefix, "success", str(result["message"]))
        else:
            _set_notice(
                key_prefix,
                "warning",
                str(result.get("message", "No safe provider match was found.")),
            )
        st.rerun()
    return False
