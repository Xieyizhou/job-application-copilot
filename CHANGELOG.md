# Changelog

All notable public changes are documented here.

## [1.0.0] - Unreleased

### Added

- Local Personal workspaces with candidate document import, job review, cover-letter and
  evidence-bundle export, and an application tracker.
- Read-only fictional Demo workspace.
- Public Greenhouse and Lever job-description recovery with safe ordinary-page fallback.
- Local Chrome and Edge page-import companion.
- Public MiniLM v21 validation contract and reproducible semantic checks.

### Changed

- Simplified internal module boundaries while preserving public commands, extension
  messages, persisted workspace records, and model artifact formats.
- Made JD completeness, eligibility, confidence, and role fit independent review signals.

### Security and privacy

- Personal files, credentials, generated documents, caches, private ML data, and model
  artifacts remain local and are excluded from the public repository.
- Employer-facing output remains gated by explicit human review.

### Known limitations

- The stable desktop acceptance target is macOS with Python 3.11 or 3.12; Windows is not
  supported in this release.
- Credential-backed job-search services are experimental unless listed as verified in the
  final release acceptance report.
- Private training data and unpublished model weights are not distributed, so the public
  repository validates the v21 contract but cannot reproduce private training.

[1.0.0]: https://github.com/Xieyizhou/job-application-copilot/releases/tag/v1.0.0
