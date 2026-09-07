"""Stable keys used to collapse exact syndicated job previews."""

from __future__ import annotations

import hashlib
import re

from analyze_job import extract_job_description_body


def description_fingerprint(job_text: str) -> str:
    """Return a stable fingerprint for exact-equivalent JD body text."""
    body = extract_job_description_body(job_text)
    normalized = re.sub(r"[^a-z0-9]+", " ", body.lower()).strip()
    if len(normalized.split()) < 12:
        return ""
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
