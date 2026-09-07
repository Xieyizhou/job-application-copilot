"""Local job upload extraction with optional PDF and OCR backends."""

from __future__ import annotations

from manual_jd_parser import canonical_section_key
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SUPPORTED_UPLOAD_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".pdf", ".txt", ".md"}


@dataclass
class ExtractionResult:
    """Result from uploaded job file extraction."""

    text: str
    warning: str = ""
    error: str = ""
    report: dict[str, Any] | None = None


def extract_text_from_upload(filename: str, file_bytes: bytes) -> ExtractionResult:
    """Extract text from supported uploads without making OCR a hard dependency."""
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_UPLOAD_EXTENSIONS:
        return ExtractionResult("", error=f"Unsupported file type: {suffix or 'unknown'}")

    try:
        if suffix in {".txt", ".md"}:
            return ExtractionResult(file_bytes.decode("utf-8", errors="replace").strip())
        if suffix == ".pdf":
            return extract_text_from_pdf(file_bytes)
        if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
            return extract_text_from_image(file_bytes)
    except Exception as error:  # noqa: BLE001
        return ExtractionResult("", error=f"Could not read upload: {error}")

    return ExtractionResult("", error="Unsupported file type.")


def extract_text_from_pdf(file_bytes: bytes) -> ExtractionResult:
    """Extract PDF text using optional libraries without making startup brittle.

    The pipeline tries text-native extraction first, then falls back to PDFium.
    If a PDF appears scanned/image-only, PDFium page rendering plus Tesseract OCR
    is attempted only when both pytesseract and the tesseract binary are present.
    """
    reports = []
    warnings = []

    plumber_result = extract_pdf_with_pdfplumber(file_bytes)
    if plumber_result.text and not pdf_extraction_is_low_quality(plumber_result):
        return plumber_result
    if plumber_result.warning:
        warnings.append(plumber_result.warning)
    if plumber_result.report:
        reports.append(plumber_result.report)

    pdfium_result = extract_pdf_with_pdfium(file_bytes)
    if pdfium_result.text and (
        not plumber_result.text
        or len(pdfium_result.text) > len(plumber_result.text)
        or not pdf_extraction_is_low_quality(pdfium_result)
    ):
        if warnings:
            pdfium_result.warning = join_unique_warnings([*warnings, pdfium_result.warning])
        return pdfium_result
    if plumber_result.text:
        plumber_result.warning = join_unique_warnings([*warnings, plumber_result.warning])
        return plumber_result
    if pdfium_result.warning:
        warnings.append(pdfium_result.warning)
    if pdfium_result.report:
        reports.append(pdfium_result.report)

    ocr_result = extract_pdf_with_pdfium_ocr(file_bytes)
    if ocr_result.text:
        if warnings:
            ocr_result.warning = join_unique_warnings([*warnings, ocr_result.warning])
        return ocr_result
    if ocr_result.warning:
        warnings.append(ocr_result.warning)

    return ExtractionResult(
        "",
        warning=join_unique_warnings(warnings) or "PDF text extraction is not available locally. Paste the job description manually.",
        report={"method": "none", "fallback_reports": reports},
    )


def pdf_extraction_is_low_quality(result: ExtractionResult) -> bool:
    """Return True when extracted PDF text looks partial or sectionless."""
    report = result.report or {}
    return bool(report.get("warnings")) or len(result.text.strip()) < 500


def join_unique_warnings(warnings: list[str]) -> str:
    """Join warning strings once while preserving first-seen order."""
    seen = set()
    parts = []
    for warning in warnings:
        for line in str(warning or "").splitlines():
            line = line.strip()
            if line and line not in seen:
                seen.add(line)
                parts.append(line)
    return "\n".join(parts)


def extract_pdf_with_pdfplumber(file_bytes: bytes) -> ExtractionResult:
    """Extract all pages with pdfplumber when available."""
    try:
        import pdfplumber
    except ImportError:
        return ExtractionResult(
            "",
            warning="pdfplumber is not installed; trying PDF fallback extraction if available.",
            report={"method": "pdfplumber", "available": False},
        )

    import io

    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        page_text = [page.extract_text(x_tolerance=1, y_tolerance=3) or "" for page in pdf.pages]
        metadata = dict(pdf.metadata or {})
    text = format_pdf_pages(page_text)
    report = build_pdf_extraction_report("pdfplumber", page_text, text, metadata)
    warning = "\n".join(report["warnings"])
    return ExtractionResult(text, warning=warning, report=report)


def extract_pdf_with_pdfium(file_bytes: bytes) -> ExtractionResult:
    """Extract all pages with PDFium when available."""
    try:
        import pypdfium2 as pdfium
    except ImportError:
        return ExtractionResult(
            "",
            warning="pypdfium2 is not installed; PDF fallback extraction is unavailable.",
            report={"method": "pdfium", "available": False},
        )

    page_text = []
    document = pdfium.PdfDocument(file_bytes)
    try:
        metadata = dict(document.get_metadata_dict() or {})
        for page in document:
            text_page = page.get_textpage()
            try:
                page_text.append(text_page.get_text_range() or "")
            finally:
                text_page.close()
                page.close()
    finally:
        document.close()
    text = format_pdf_pages(page_text)
    report = build_pdf_extraction_report("pdfium", page_text, text, metadata)
    warning = "\n".join(report["warnings"])
    return ExtractionResult(text, warning=warning, report=report)


def extract_pdf_with_pdfium_ocr(file_bytes: bytes) -> ExtractionResult:
    """OCR scanned PDFs only when PDFium, Pillow, pytesseract, and Tesseract exist."""
    if shutil.which("tesseract") is None:
        return ExtractionResult(
            "",
            warning="PDF text extraction may be incomplete. OCR is not available locally. Mac: brew install tesseract",
            report={"method": "pdfium_ocr", "available": False},
        )

    try:
        import pypdfium2 as pdfium
        import pytesseract
    except ImportError:
        return ExtractionResult(
            "",
            warning="PDF OCR dependencies are not installed. Install pypdfium2, Pillow, and pytesseract or paste manually.",
            report={"method": "pdfium_ocr", "available": False},
        )

    page_text = []
    document = pdfium.PdfDocument(file_bytes)
    try:
        metadata = dict(document.get_metadata_dict() or {})
        for page in document:
            bitmap = page.render(scale=2)
            try:
                page_text.append(pytesseract.image_to_string(bitmap.to_pil()) or "")
            finally:
                bitmap.close()
                page.close()
    finally:
        document.close()

    text = format_pdf_pages(page_text)
    report = build_pdf_extraction_report("pdfium_ocr", page_text, text, metadata)
    warning = "\n".join(report["warnings"])
    return ExtractionResult(text, warning=warning, report=report)


def format_pdf_pages(page_text: list[str]) -> str:
    """Preserve page order with internal page separators."""
    parts = []
    for index, text in enumerate(page_text, start=1):
        cleaned = clean_pdf_page_text(text)
        if cleaned:
            parts.append(f"--- Page {index} ---\n{cleaned}")
        else:
            parts.append(f"--- Page {index} ---")
    return "\n\n".join(parts).strip()


def clean_pdf_page_text(text: str) -> str:
    """Preserve bullets/headings while fixing common PDF extraction artifacts."""
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if re.fullmatch(r"(?:page\s*)?\d+\s*(?:of\s*\d+)?", line, flags=re.IGNORECASE):
            continue
        lines.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def detect_section_headings(text: str) -> list[str]:
    """Find recognized job-section headings in extracted text."""
    headings = []
    for line in text.splitlines():
        heading_key = canonical_section_key(line)
        if heading_key and line.strip() not in headings:
            headings.append(line.strip())
    return headings


def build_pdf_extraction_report(
    method: str,
    page_text: list[str],
    text: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize PDF extraction quality for the UI report expander."""
    pages_processed = len(page_text)
    empty_pages = sum(1 for page in page_text if not page.strip())
    section_headings = detect_section_headings(text)
    metadata = metadata or {}
    metadata_title = " ".join(str(metadata.get("title", "") or "").split())
    warnings = []
    if len(text.strip()) < 500 or empty_pages == pages_processed:
        warnings.append(
            "PDF text extraction may be incomplete. This PDF may be scanned, image-based, or formatted in a way that requires OCR."
        )
    if not section_headings:
        warnings.append("No common job-description sections were detected in the extracted PDF text.")
    return {
        "method": method,
        "pages_processed": pages_processed,
        "pages_with_no_text": empty_pages,
        "characters_extracted": len(text),
        "detected_section_headings": section_headings,
        "sections_detected": len(section_headings),
        "metadata_title": metadata_title,
        "warnings": warnings,
    }


def extract_text_from_image(file_bytes: bytes) -> ExtractionResult:
    """Run OCR only when Pillow, pytesseract, and the tesseract binary exist."""
    if shutil.which("tesseract") is None:
        return ExtractionResult(
            "",
            warning="OCR is not available locally. Please paste the job description manually, or install Tesseract OCR. Mac: brew install tesseract",
        )

    try:
        from PIL import Image
        import pytesseract
    except ImportError:
        return ExtractionResult(
            "",
            warning="OCR is not available locally. Please paste the job description manually, or install Tesseract OCR. Mac: brew install tesseract",
        )

    import io

    try:
        image = Image.open(io.BytesIO(file_bytes))
        return ExtractionResult(pytesseract.image_to_string(image).strip())
    except Exception as error:  # noqa: BLE001
        return ExtractionResult("", error=f"OCR failed: {error}")
