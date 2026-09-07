"""Shared lossless boundary recovery for flattened employer descriptions."""

import re


def normalize_jd_boundaries(text: str) -> str:
    """Recover list and heading boundaries without changing statement wording."""
    text = re.sub(r"(?<!\S)\*(?!\*)\s+(?=[A-Z])", "\n- ", text)
    text = re.sub(r"\s*[•▪◦]\s*", "\n- ", text)
    text = re.sub(
        r"(?<![A-Za-z])((?:Preferred |Required )?Qualifications|Requirements|Responsibilities)\s*:?(?=\s+(?:[A-Z]|[-*]))",
        lambda m: f"\n## {m.group(1)}\n", text,
    )
    text = re.sub(r"(?i)Key Deliverables and Success Criteria\s*:", "\n## Responsibilities\n", text)
    text = re.sub(
        r"(?i)Individuals who do well in this role[^:\n]{0,80}usually possess\s*:",
        "\n## Qualifications\n", text,
    )
    # Separate company prose from the preceding qualification list.
    text = re.sub(r"(?<!\w)(Sustainable impact is)", r"\n## About Company\n\1", text)
    return text
