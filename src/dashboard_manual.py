"""Add Target Job page and manual-job workflow for the dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import streamlit as st

from apply_package import create_application_package
from company_verification import verification_status_label
from dashboard_manual_entry import render_jd_capture, render_jd_quality, render_verification_form
from dashboard_regions import normalize_location
from dashboard_titles import display_title_from_value
from fetch_jobs import jsearch_configured
from jd_enrichment import enrich_saved_job_description
from manual_jobs import (
    SOURCE_OPTIONS,
    STATUS_OPTIONS,
    clean_extracted_job_text,
    duplicate_manual_job_exists,
    extract_text_from_upload,
    is_valid_url,
    load_manual_jobs,
    parse_job_description_suggestions,
    save_manual_job,
    sync_manual_job_from_markdown,
    update_manual_job,
)
from ml.jd_quality import classify_jd_quality
from output_cleanup import delete_directory_tree


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SHOW_DEBUG_UI = False


@dataclass(frozen=True)
class ManualPageServices:
    """Shared dashboard operations required by the manual-job page."""

    company_generation_allowed: Callable[[dict[str, Any]], bool]
    current_workspace: Callable[[], Any]
    demo_mode_enabled: Callable[[], bool]
    relative_path: Callable[[Path], str]
    render_manual_company_confirmation: Callable[[dict[str, Any], str], dict[str, Any]]
    render_page_header: Callable[[str, str | None], None]
    run_with_captured_output: Callable[..., tuple[Any, str]]


MANUAL_FORM_STATE_KEYS = {
    "manual_company",
    "manual_title",
    "manual_location",
    "manual_url",
    "manual_salary_range",
    "manual_visa_note",
    "manual_notes",
    "manual_job_description",
}
MANUAL_TRANSIENT_STATE_KEYS = {
    "manual_pending_suggestions",
    "manual_pending_clean_text",
    "manual_extracted_text",
    "manual_raw_extracted_text",
    "manual_cleaned_extracted_text",
    "manual_source_upload_filenames",
    "manual_parser_suggestions",
    "manual_extraction_reports",
    "manual_generated_summary",
    "manual_generated_backend_output",
    "manual_generated_error",
    "manual_last_extracted_upload_signature",
    "manual_parser_display_mode",
}
MANUAL_SELECTION_STATE_KEY_PREFIXES = (
    "manual_upload_",
    "manual_saved_selected",
    "manual_generate_selected",
    "manual_edit_status_",
    "manual_edit_notes_",
    "manual_generate_",
)


def clear_manual_job_session_state(clear_upload: bool = True) -> None:
    """Clear transient Add Target Job UI state without deleting saved jobs.

    Streamlit file upload widgets cannot be assigned directly after creation, so
    upload reset is handled by bumping a key suffix before the widget is rendered.
    """
    keys_to_clear = set(MANUAL_FORM_STATE_KEYS) | set(MANUAL_TRANSIENT_STATE_KEYS)
    keys_to_clear.update(
        key
        for key in st.session_state.keys()
        if isinstance(key, str)
        and key != "manual_upload_key_suffix"
        and key.startswith(MANUAL_SELECTION_STATE_KEY_PREFIXES)
    )
    if clear_upload:
        current_suffix = int(st.session_state.get("manual_upload_key_suffix", 0) or 0)
        st.session_state["manual_upload_key_suffix"] = current_suffix + 1
    for key in keys_to_clear:
        st.session_state.pop(key, None)
    st.session_state["manual_status"] = "Saved"
    st.session_state["manual_source"] = SOURCE_OPTIONS[0]
    st.session_state["manual_last_cleanup_timestamp"] = datetime.now().replace(microsecond=0).isoformat()


def clear_manual_state_for_new_extraction() -> None:
    """Reset stale form/parser state while preserving the current upload widget."""
    clear_manual_job_session_state(clear_upload=False)


def clean_generated_outputs(services: ManualPageServices) -> list[Path]:
    """Delete generated cover-letter bundles only; never delete saved manual jobs."""
    workspace = services.current_workspace()
    workspace.require_writable()
    output_root = workspace.generated_dir.resolve()
    if output_root != (workspace.root / "generated").resolve():
        raise ValueError("Generated output cleanup is restricted to the Personal workspace.")
    output_root.mkdir(parents=True, exist_ok=True)

    deleted_paths = []
    for child in output_root.iterdir():
        if child.name == ".gitkeep":
            continue
        if child.is_dir():
            delete_directory_tree(child)
        else:
            child.unlink()
        deleted_paths.append(child)
    return deleted_paths


def render_manual_cleanup_controls(services: ManualPageServices) -> None:
    """Render clearly separated cleanup actions for manual workflow state."""
    cleanup_left, cleanup_right = st.columns(2)
    with cleanup_left:
        if st.button("Clear Current Target Job Form", key="clear_manual_job_form"):
            clear_manual_job_session_state(clear_upload=True)
            st.session_state["manual_cleanup_message"] = "Current target job form and extraction state cleared."
            st.rerun()
    with cleanup_right:
        if st.button("Clean Generated Outputs", key="clean_generated_outputs"):
            try:
                deleted_paths = clean_generated_outputs(services)
                for key in ["manual_generated_summary", "manual_generated_backend_output", "manual_generated_error"]:
                    st.session_state.pop(key, None)
                st.session_state["manual_cleanup_message"] = (
                    f"Generated outputs cleaned. Removed {len(deleted_paths)} item(s). "
                    "Saved target jobs and tracker records were not deleted."
                )
            except Exception as error:  # noqa: BLE001
                st.session_state["manual_cleanup_message"] = f"Could not clean generated outputs: {error}"
            st.rerun()


MANUAL_SUGGESTION_FIELD_RULES = {
    "manual_company": ("company", {"high", "medium"}, "company_confidence"),
    "manual_title": ("title", {"high", "medium"}, "job_title_confidence"),
    "manual_location": ("location", {"high", "medium"}, "location_confidence"),
    "manual_source": ("source", {"high", "medium"}, "source_confidence"),
    "manual_url": ("url", {"high", "medium"}, "url_confidence"),
    "manual_salary_range": ("salary_range", {"high", "medium"}, "salary_confidence"),
    "manual_visa_note": ("visa_note", {"high", "medium"}, "visa_confidence"),
    "manual_status": ("status", {"high", "medium"}, "status_confidence"),
    "manual_notes": ("notes", {"high", "medium"}, "notes_confidence"),
    "manual_job_description": ("job_description", {"high", "medium"}, "job_description_confidence"),
}


def apply_suggestions_to_empty_fields(suggestions: dict[str, Any]) -> None:
    """Synchronize parser suggestions into the exact form widget keys.

    This runs before the form widgets are instantiated. Title suggestions with
    high or medium confidence are accepted into `manual_title` when it is empty,
    so the compact summary and the editable field cannot contradict each other.
    """
    confidence_rules = {
        key: rule
        for key, rule in MANUAL_SUGGESTION_FIELD_RULES.items()
        if key in {
            "manual_company",
            "manual_title",
            "manual_location",
            "manual_visa_note",
            "manual_source",
            "manual_url",
            "manual_salary_range",
            "manual_status",
            "manual_notes",
            "manual_job_description",
        }
    }
    for state_key, (suggestion_key, allowed_confidences, confidence_key) in confidence_rules.items():
        suggested_value = suggestions.get(suggestion_key)
        confidence = str(suggestions.get(confidence_key) or ("medium" if suggested_value else "")).lower()
        if st.session_state.get(state_key) or not suggested_value or confidence not in allowed_confidences:
            continue
        if state_key == "manual_source" and suggested_value not in SOURCE_OPTIONS:
            continue
        if state_key == "manual_status" and suggested_value not in STATUS_OPTIONS:
            continue
        st.session_state[state_key] = suggested_value


def current_manual_suggestions(current_text: str) -> dict[str, Any]:
    """Use one shared parser suggestion object for summary, actions, and save."""
    if not current_text.strip():
        st.session_state["manual_parser_suggestions"] = {}
        return {}
    suggestions = parse_job_description_suggestions(current_text, current_manual_source_metadata())
    st.session_state["manual_parser_suggestions"] = suggestions
    return suggestions


def manual_source_metadata_from_reports(reports: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Expose PDF/browser metadata to parser heuristics without changing uploads."""
    metadata_titles = []
    for report in reports:
        metadata_title = " ".join(str(report.get("metadata_title", "") or "").split())
        if metadata_title and metadata_title not in metadata_titles:
            metadata_titles.append(metadata_title)
    return {"metadata_titles": metadata_titles}


def current_manual_source_metadata() -> dict[str, list[str]]:
    """Return parser metadata for the current extracted upload."""
    return manual_source_metadata_from_reports(st.session_state.get("manual_extraction_reports", []) or [])


def combine_upload_extraction_results(uploaded_files: list[Any]) -> tuple[str, str, str, list[str], list[dict[str, Any]]]:
    """Extract each uploaded file in order and combine text with visible separators."""
    raw_parts = []
    cleaned_parts = []
    messages = []
    filenames = []
    reports = []

    # Multi-file uploads matter for long LinkedIn posts where one screenshot only
    # captures the visible viewport. Each file keeps a separator for traceability.
    for index, uploaded_file in enumerate(uploaded_files, start=1):
        file_bytes = uploaded_file.getvalue()
        filenames.append(uploaded_file.name)
        result = extract_text_from_upload(uploaded_file.name, file_bytes)
        if result.error:
            messages.append(f"{uploaded_file.name}: {result.error}")
        if result.warning:
            messages.append(f"{uploaded_file.name}: {result.warning}")
        if result.report:
            report = {"file_name": uploaded_file.name, **result.report}
            reports.append(report)
        if not result.text:
            continue

        header = f"--- Extracted text from file {index}: {uploaded_file.name} ---"
        raw_parts.append(f"{header}\n{result.text}")
        cleaned_parts.append(f"{header}\n{clean_extracted_job_text(result.text)}")

    return "\n\n".join(raw_parts).strip(), "\n\n".join(cleaned_parts).strip(), "\n".join(messages), filenames, reports


def render_extraction_reports(reports: list[dict[str, Any]]) -> None:
    """Render compact extraction status with detailed report collapsed."""
    if not reports:
        return
    total_pages = sum(int(report.get("pages_processed", 0) or 0) for report in reports)
    total_chars = sum(int(report.get("characters_extracted", 0) or 0) for report in reports)
    total_sections = sum(int(report.get("sections_detected", 0) or 0) for report in reports)
    methods = ", ".join(sorted({str(report.get("method", "unknown")) for report in reports}))
    st.success(
        f"Extraction complete: {total_pages or '-'} pages, {total_chars:,} characters extracted, "
        f"{total_sections} sections detected. Method: {methods}."
    )
    if not SHOW_DEBUG_UI:
        return
    with st.expander("Advanced: extraction report", expanded=False):
        for report in reports:
            st.write(f"File name: {report.get('file_name', '-')}")
            st.write(f"Pages processed: {report.get('pages_processed', '-')}")
            st.write(f"Characters extracted: {report.get('characters_extracted', '-')}")
            st.write(f"Extraction method used: {report.get('method', '-')}")
            if report.get("metadata_title"):
                st.write(f"PDF/browser title metadata: {report.get('metadata_title')}")
            headings = report.get("detected_section_headings") or []
            st.write("Detected section headings: " + (", ".join(headings) if headings else "-"))
            warnings = report.get("warnings") or []
            if warnings:
                st.write("Warnings:")
                for warning in warnings:
                    st.write(f"- {warning}")


def sorted_manual_records() -> list[dict[str, Any]]:
    """Return saved manual records newest-first for selectors and tables."""
    return sorted(load_manual_jobs(), key=lambda record: str(record.get("created_at", "")), reverse=True)


def manual_record_label(record: dict[str, Any]) -> str:
    """Build a compact stable label for saved manual job selectors."""
    return (
        f"{record.get('company', '')} | {record.get('title', '')} | "
        f"{record.get('created_at', '')} | {record.get('id', '')}"
    )


def select_manual_record(records: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    """Render a saved manual job selector and return the selected record."""
    if not records:
        st.info("No target jobs saved yet.")
        return None
    labels = [manual_record_label(record) for record in records]
    selected_label = st.selectbox("Select saved target job", labels, key=key)
    return records[labels.index(selected_label)]


def render_manual_record_long_details(record: dict[str, Any]) -> None:
    """Keep long saved-job details collapsed by default."""
    url = str(record.get("url", "") or "")
    if url:
        st.link_button("Open Job URL", url)
    else:
        st.write("Job URL: -")

    with st.expander("Full job description", expanded=False):
        st.text(record.get("job_description", ""))
    if SHOW_DEBUG_UI:
        with st.expander("Advanced: raw Markdown paths and uploads", expanded=False):
            st.write(f"Markdown file: `{record.get('markdown_path', '')}`")
            upload_filenames = record.get("source_upload_filenames") or []
            if upload_filenames:
                st.write("Uploads:")
                for filename in upload_filenames:
                    st.write(f"- `{filename}`")
            elif record.get("source_upload_filename"):
                st.write(f"Upload: `{record.get('source_upload_filename')}`")

        with st.expander("Advanced: OCR and parser details", expanded=False):
            st.write("Raw extracted OCR text")
            st.text(record.get("raw_extracted_text", "") or record.get("extracted_text", "") or "")
            st.write("Cleaned OCR text")
            st.text(record.get("cleaned_extracted_text", "") or "")
            st.write("Parser metadata")
            st.json(record.get("parser_suggestions", {}) or {})


def generate_package_for_manual_record(
    record: dict[str, Any],
    button_key: str,
    services: ManualPageServices,
) -> None:
    """Run the existing package workflow for a saved manual job."""
    markdown_path = PROJECT_ROOT / str(record.get("markdown_path", ""))
    if not markdown_path.exists():
        st.error("Saved Markdown file was not found for this target job.")
        return

    fields = services.render_manual_company_confirmation(record, f"{button_key}_manual_company")
    if not services.company_generation_allowed(fields):
        st.info(
            "Company name needs confirmation before looking up a full JD or generating a cover letter. "
            "This prevents using the wrong employer in your application."
        )
        return

    jd_quality = classify_jd_quality(markdown_path.read_text(encoding="utf-8"))
    if not jd_quality["reliable_scoring_ready"]:
        st.warning(
            f"A full JD is required before generating a cover letter. "
            f"Current quality: {jd_quality['display_label']}."
        )
        st.caption(str(jd_quality["next_action"]))
        if jsearch_configured():
            if st.button("Find and verify full JD", key=f"{button_key}_find_full_jd", type="primary"):
                result = enrich_saved_job_description(markdown_path)
                if result.get("updated"):
                    sync_manual_job_from_markdown(str(record["id"]), markdown_path)
                    st.success(str(result["message"]))
                    st.rerun()
                else:
                    st.warning(str(result.get("message", "No safe full-JD match was found.")))
        else:
            st.info("Configure JSEARCH_API_KEY to find the full posting automatically, or paste the complete JD.")
        return

    if st.button("Generate Cover Letter", key=button_key, type="primary"):
        st.session_state.pop("manual_generated_error", None)
        try:
            summary, output = services.run_with_captured_output(
                create_application_package,
                job_description_path=markdown_path,
                workspace=services.current_workspace(),
                company=str(record.get("company", "")).strip(),
                role=str(record.get("title", "")).strip(),
                location=str(record.get("location", "")).strip(),
                job_url=str(record.get("url", "")).strip(),
            )
            sync_manual_job_from_markdown(str(record["id"]), markdown_path)
            update_manual_job(str(record["id"]), status="Cover Letter Generated", notes=str(record.get("notes", "")))
            st.session_state["manual_generated_summary"] = {
                "match_score": summary["match_score"],
                "recommendation": summary["recommendation"],
                "analysis_path": services.relative_path(summary["analysis_path"]),
                "cover_letter_path": services.relative_path(summary["cover_letter_path"]),
                "cover_letter_docx_path": services.relative_path(summary["cover_letter_docx_path"]),
                "tracker_id": summary["tracker_id"],
            }
            st.session_state["manual_generated_backend_output"] = output
            st.success("Manual job analyzed and cover letter generated.")
            st.write(f"Overall score: {summary['match_score']}/100")
            st.write(f"Recommendation: {summary['recommendation']}")
            st.write(f"Analysis file: `{services.relative_path(summary['analysis_path'])}`")
            st.write(f"Cover letter Markdown: `{services.relative_path(summary['cover_letter_path'])}`")
            st.write(f"Cover letter DOCX: `{services.relative_path(summary['cover_letter_docx_path'])}`")
            st.write(f"Tracker id: {summary['tracker_id']}")
            if summary.get("uk_review_notes"):
                st.warning("UK work authorization review")
                for note in summary["uk_review_notes"]:
                    st.write(f"- {note}")
            if output:
                with st.expander("Technical output (advanced)", expanded=False):
                    st.text(output)
        except Exception as error:  # noqa: BLE001
            st.session_state["manual_generated_error"] = str(error)
            st.error(str(error))


def prepare_manual_job_session_state() -> None:
    """Apply queued state updates before widgets are created."""
    pending_suggestions = st.session_state.pop("manual_pending_suggestions", None)
    if pending_suggestions:
        apply_suggestions_to_empty_fields(pending_suggestions)

    pending_clean_text = st.session_state.pop("manual_pending_clean_text", None)
    if pending_clean_text is not None:
        st.session_state["manual_job_description"] = pending_clean_text


def render_manual_upload_controls() -> list[Any]:
    """Render upload and OCR cleanup controls, returning selected files."""
    uploaded_files = st.file_uploader(
        "Upload job file",
        type=["png", "jpg", "jpeg", "webp", "pdf", "txt", "md"],
        key=f"manual_upload_{st.session_state.get('manual_upload_key_suffix', 0)}",
        accept_multiple_files=True,
    ) or []
    if uploaded_files:
        st.caption("Selected uploads: " + ", ".join(f"`{uploaded_file.name}`" for uploaded_file in uploaded_files))
    if uploaded_files and st.button("Extract Text from Upload"):
        clear_manual_state_for_new_extraction()
        st.session_state["manual_source_upload_filenames"] = [item.name for item in uploaded_files]
        st.session_state["manual_last_extracted_upload_signature"] = " | ".join(
            f"{item.name}:{item.size}" for item in uploaded_files
        )
        raw_text, cleaned_text, messages, filenames, reports = combine_upload_extraction_results(uploaded_files)
        if messages:
            st.warning(messages)
        st.session_state["manual_extraction_reports"] = reports
        if cleaned_text:
            st.session_state.update(
                {
                    "manual_job_description": cleaned_text,
                    "manual_extracted_text": cleaned_text,
                    "manual_raw_extracted_text": raw_text,
                    "manual_cleaned_extracted_text": cleaned_text,
                    "manual_source_upload_filenames": filenames,
                }
            )
            suggestions = parse_job_description_suggestions(cleaned_text, manual_source_metadata_from_reports(reports))
            st.session_state["manual_parser_suggestions"] = suggestions
            apply_suggestions_to_empty_fields(suggestions)
            st.success("Extracted and cleaned text added to the editable job description.")
        elif not messages:
            st.warning("No text could be extracted. Please paste the job description manually.")
    if st.session_state.get("manual_job_description") and st.button("Clean OCR Text"):
        cleaned_text = clean_extracted_job_text(st.session_state.get("manual_job_description", ""))
        st.session_state["manual_pending_clean_text"] = cleaned_text
        st.session_state["manual_cleaned_extracted_text"] = cleaned_text
        st.rerun()
    render_extraction_reports(st.session_state.get("manual_extraction_reports", []) or [])
    return uploaded_files


def render_manual_job_form() -> dict[str, Any]:
    """Render inferred job fields and return one submission payload."""
    job_description = str(st.session_state.get("manual_job_description", "") or "")
    suggestions = current_manual_suggestions(job_description)
    apply_suggestions_to_empty_fields(suggestions)
    return render_verification_form(suggestions=suggestions, job_description=job_description)


def save_manual_form_submission(payload: dict[str, Any], uploaded_files: list[Any]) -> None:
    """Validate and persist one manual-job form submission."""
    if not payload["submitted"]:
        return
    if not payload["title"]:
        st.error("Job title is required.")
        return
    if not str(payload["job_description"]).strip():
        st.error("Job description is required. Paste text manually or extract it from an upload.")
        return
    if not is_valid_url(payload["url"]):
        st.error("Enter a valid http(s) Job URL, or leave it blank.")
        return
    if duplicate_manual_job_exists(payload["company"], payload["title"], payload["url"]):
        st.error("Duplicate target job found with the same company, title, and URL.")
        return
    try:
        record = save_manual_job(
            company=payload["company"], title=payload["title"], location=payload["location"],
            source=payload["source"], url=payload["url"], salary_range=payload["salary_range"],
            visa_note=payload["visa_note"], status=payload["status"], notes=payload["notes"],
            job_description=payload["job_description"], extracted_text=st.session_state.get("manual_extracted_text", ""),
            raw_extracted_text=st.session_state.get("manual_raw_extracted_text", ""),
            cleaned_extracted_text=st.session_state.get("manual_cleaned_extracted_text", ""),
            parser_suggestions=payload["suggestions"],
            upload_files=[(item.name, item.getvalue()) for item in uploaded_files],
        )
        st.session_state["manual_generate_selected"] = manual_record_label(record)
        st.success("Target job saved. Continue to Generate Cover Letter when you are ready.")
        st.info("Open the Generate Cover Letter tab next. This saved job will be preselected there.")
        if SHOW_DEBUG_UI:
            with st.expander("Advanced: raw Markdown path", expanded=False):
                st.write(f"Saved Markdown: `{record['markdown_path']}`")
    except Exception as error:  # noqa: BLE001
        st.error(f"Could not save target job: {error}")


def render_manual_add_extract_tab(services: ManualPageServices) -> None:
    """Render a JD-first, linear target-job workflow."""
    prepare_manual_job_session_state()
    cleanup_message = st.session_state.pop("manual_cleanup_message", "")
    if cleanup_message:
        st.success(cleanup_message)
    uploaded_files = render_jd_capture(render_manual_upload_controls)
    job_description = str(st.session_state.get("manual_job_description", "") or "")
    render_jd_quality(job_description)
    payload = render_manual_job_form()
    save_manual_form_submission(payload, uploaded_files)
    with st.expander("Maintenance", expanded=False):
        st.caption("Clear the current form or remove generated bundles. Saved target jobs are retained.")
        render_manual_cleanup_controls(services)


def render_saved_manual_jobs_tab(services: ManualPageServices) -> None:
    """Render compact saved manual job table and edit controls."""
    st.caption("Review saved targets, confirm company details, and update status.")
    records = sorted_manual_records()
    if not records:
        st.info("No target jobs saved yet.")
        return

    st.dataframe(
        [
            {
                "Company": record.get("company", ""),
                "Company status": verification_status_label(record),
                "Job title": display_title_from_value(record.get("title"), fallback="Sample Job"),
                "Location": normalize_location(str(record.get("location", ""))),
                "Source": record.get("source", ""),
                "Status": record.get("status", ""),
                "JD words": len(str(record.get("job_description", "") or "").split()),
                "Next": (
                    "Confirm company"
                    if bool(record.get("company_needs_review"))
                    else "Review fit / generate cover letter"
                ),
                "Created date": str(record.get("created_at", ""))[:10],
            }
            for record in records
        ],
        width="stretch",
        hide_index=True,
    )

    selected_record = select_manual_record(records, key="manual_saved_selected")
    if selected_record is None:
        return

    record_id = str(selected_record["id"])
    selected_url = str(selected_record.get("url", "") or "").strip()
    summary_left, summary_right = st.columns([0.68, 0.32], gap="large")
    with summary_left:
        st.markdown(f"**{selected_record.get('company', '') or '-'}**")
        st.write(display_title_from_value(selected_record.get("title"), fallback="Sample Job"))
        st.caption(
            f"{normalize_location(str(selected_record.get('location', ''))) or '-'} | "
            f"{selected_record.get('source', '') or '-'}"
        )
    with summary_right:
        st.write(f"Status: {selected_record.get('status', '-')}")
        st.write(f"Company: {verification_status_label(selected_record)}")
        if is_valid_url(selected_url):
            st.link_button("Open Job URL", selected_url, width="stretch")

    overview_tab, verification_tab, jd_tab, notes_tab = st.tabs(
        ["Overview", "Verification", "Full Job Description", "Notes / Status"]
    )
    with overview_tab:
        st.write(f"Company: {selected_record.get('company', '') or '-'}")
        st.write(f"Role: {display_title_from_value(selected_record.get('title'), fallback='Sample Job')}")
        st.write(f"Location: {normalize_location(str(selected_record.get('location', ''))) or '-'}")
        st.write(f"Source: {selected_record.get('source', '') or '-'}")
        st.write(f"Status: {selected_record.get('status', '-')}")
        notes = str(selected_record.get("notes", "") or "").strip()
        if notes:
            st.caption(notes)
        generate_package_for_manual_record(
            selected_record,
            button_key=f"manual_saved_generate_{record_id}",
            services=services,
        )
    with verification_tab:
        services.render_manual_company_confirmation(selected_record, f"manual_saved_{record_id}")
    with jd_tab:
        render_manual_record_long_details(selected_record)
    with notes_tab:
        edit_left, edit_right = st.columns([1, 2])
        with edit_left:
            current_status = str(selected_record.get("status", "Saved"))
            status_index = STATUS_OPTIONS.index(current_status) if current_status in STATUS_OPTIONS else 0
            edited_status = st.selectbox(
                "Edit status",
                STATUS_OPTIONS,
                index=status_index,
                key=f"manual_edit_status_{record_id}",
            )
        with edit_right:
            edited_notes = st.text_area(
                "Edit notes",
                value=str(selected_record.get("notes", "")),
                height=100,
                key=f"manual_edit_notes_{record_id}",
            )
        if st.button("Update Saved Target Job", key=f"manual_update_{record_id}"):
            updated = update_manual_job(record_id, status=edited_status, notes=edited_notes)
            if updated:
                st.success("Saved target job updated.")
                st.rerun()
            else:
                st.error("Could not find that target job record.")


def render_manual_generate_package_tab(services: ManualPageServices) -> None:
    """Render package generation for a selected saved manual job."""
    st.caption("Choose a saved job and generate a resume-grounded cover letter, match report, and evidence trace.")
    selected_record = select_manual_record(sorted_manual_records(), key="manual_generate_selected")
    if selected_record is None:
        return
    st.write(f"Company: {selected_record.get('company', '')}")
    st.write(f"Company status: {verification_status_label(selected_record)}")
    st.write(f"Job title: {display_title_from_value(selected_record.get('title'), fallback='Sample Job')}")
    st.write(f"Location: {normalize_location(str(selected_record.get('location', ''))) or '-'}")
    generate_package_for_manual_record(
        selected_record,
        button_key=f"manual_generate_{selected_record['id']}",
        services=services,
    )
    if st.session_state.get("manual_generated_summary"):
        with st.expander("Latest cover-letter generation output", expanded=False):
            st.json(st.session_state["manual_generated_summary"])
    if st.session_state.get("manual_generated_backend_output"):
        with st.expander("Latest technical output (advanced)", expanded=False):
            st.text(st.session_state["manual_generated_backend_output"])
    if st.session_state.get("manual_generated_error"):
        st.error(st.session_state["manual_generated_error"])


def render_manual_debug_tab() -> None:
    """Render collapsed OCR and parser debugging details."""
    st.info(
        "This section is for troubleshooting OCR, PDF extraction, and parsing issues. "
        "Most users do not need it during normal job entry."
    )
    with st.expander("Cleanup and current state", expanded=False):
        st.write("Current uploaded file names:")
        st.json(st.session_state.get("manual_source_upload_filenames", []) or [])
        st.write(f"Current upload signature: {st.session_state.get('manual_last_extracted_upload_signature', '-')}")
        st.write(
            "Parser suggestion source: "
            + ("current job description text" if st.session_state.get("manual_parser_suggestions") else "-")
        )
        st.write(f"Extracted text source: {st.session_state.get('manual_last_extracted_upload_signature', '-')}")
        st.write(f"Selected target job for generation: {st.session_state.get('manual_generate_selected', '-')}")
        st.write(f"Last cleanup timestamp: {st.session_state.get('manual_last_cleanup_timestamp', '-')}")
    with st.expander("Advanced: raw extracted text", expanded=False):
        st.text(st.session_state.get("manual_raw_extracted_text", ""))
    with st.expander("Advanced: cleaned extracted text", expanded=False):
        st.text(st.session_state.get("manual_cleaned_extracted_text", ""))
    with st.expander("Advanced: parser suggestions", expanded=False):
        st.json(st.session_state.get("manual_parser_suggestions", {}) or {})
    with st.expander("Advanced: parser evidence", expanded=False):
        suggestions = st.session_state.get("manual_parser_suggestions", {}) or {}
        for label, key in [
            ("Company", "company_evidence"),
            ("Job title", "job_title_evidence"),
            ("Location", "location_evidence"),
            ("Visa / work authorization", "visa_evidence"),
            ("Employment type", "employment_type_evidence"),
        ]:
            evidence = suggestions.get(key)
            if evidence:
                st.write(f"{label}: {evidence}")
    with st.expander("Advanced: extraction reports", expanded=False):
        st.json(st.session_state.get("manual_extraction_reports", []) or [])


def manual_job_target_tab(services: ManualPageServices) -> None:
    """Render the target-job workflow with JD capture as the primary task."""
    services.render_page_header(
        "Add Target Job",
        "Capture the complete posting once so fit, documents, and interview prep share the same source.",
    )
    if services.demo_mode_enabled():
        st.info("Demo workspace uses bundled sample jobs. Select Personal to save a new target job locally.")
        return

    tab_labels = ["Add Job", "Saved Jobs", "Cover Letter"]
    if SHOW_DEBUG_UI:
        tab_labels.append("Advanced / Debug")
    tabs = st.tabs(tab_labels)
    tab_add, tab_saved, tab_generate = tabs[:3]
    with tab_add:
        render_manual_add_extract_tab(services)
    with tab_generate:
        render_manual_generate_package_tab(services)
    with tab_saved:
        render_saved_manual_jobs_tab(services)
    if SHOW_DEBUG_UI:
        with tabs[3]:
            render_manual_debug_tab()
