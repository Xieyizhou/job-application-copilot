# Changelog

All notable public changes are documented here.

## [1.0.0] - Unreleased

### Added

- Local Personal workspaces with candidate document import, job review, cover-letter and
  evidence-bundle export, and an application tracker.
- Read-only fictional Demo workspace.
- Public Greenhouse and Lever job-description recovery with safe ordinary-page fallback.
- Local Chrome page-import companion with experimental, unverified Edge compatibility.
- Public MiniLM v21 validation contract and reproducible semantic checks.
- PDF text extraction and OCR rendering through the permissively licensed PDFium backend.

### Changed

- Simplified internal module boundaries while preserving public commands, extension
  messages, persisted workspace records, and model artifact formats.
- Made JD completeness, eligibility, confidence, and role fit independent review signals.
- Replaced the AGPL/commercial PyMuPDF dependency with pypdfium2 while preserving PDF
  upload and OCR fallback behavior.

### Security and privacy

- Personal files, credentials, generated documents, caches, private ML data, and model
  artifacts remain local and are excluded from the public repository.
- Employer-facing output remains gated by explicit human review.

### Fixed

- Preserved the active companion connection when a second launcher encounters an occupied port.
- Kept base Personal startup independent of optional ML packages.
- Allowed manually entered titles to reach form validation when title inference is empty.
- Distinguished JD headings from prose and ignored empty visa metadata in risk warnings.
- Added advertised Markdown recovery for job pages with incomplete HTML extraction.
- Aligned newly generated report and Tracker scores with the dashboard analysis; historical exports remain snapshots.
- Refreshed Demo evidence and made narrow review controls readable.

### Known limitations

- The stable desktop acceptance target is macOS with Python 3.11 or 3.12; Windows is not
  supported in this release.
- Browser-import acceptance covers Chrome on macOS. Edge is unverified and experimental;
  Safari extension import is outside the release scope.
- Credential-backed job-search services remain experimental; minimal live searches passed,
  but broader provider behavior and quotas were not exhaustively validated.
- Private training data and unpublished model weights are not distributed, so the public
  repository validates the v21 contract but cannot reproduce private training.

[1.0.0]: https://github.com/Xieyizhou/job-application-copilot/releases/tag/v1.0.0
