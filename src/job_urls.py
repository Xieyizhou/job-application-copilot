"""Public job URLs for fetched postings, bundles, and tracker records."""

import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


TRACKING_QUERY_PARAMETERS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "app_id",
    "app_key",
    "aztt",
}


def sanitize_job_url(job_url: str) -> str:
    """Remove tracking parameters and canonicalize common Adzuna redirect URLs."""
    if not job_url:
        return ""

    split_url = urlsplit(job_url)

    # Adzuna API results often contain /land/ad/<id> links with tracking
    # parameters. The cleaner public details URL is easier to save and compare.
    land_ad_match = re.match(r"^/land/ad/(\d+)", split_url.path)
    if "adzuna." in split_url.netloc.lower() and land_ad_match:
        return urlunsplit(
            (
                split_url.scheme,
                split_url.netloc,
                f"/details/{land_ad_match.group(1)}",
                "",
                "",
            )
        )

    safe_query_pairs = [
        (key, value)
        for key, value in parse_qsl(split_url.query, keep_blank_values=True)
        if key.lower() not in TRACKING_QUERY_PARAMETERS
    ]
    safe_query = urlencode(safe_query_pairs)
    return urlunsplit(
        (
            split_url.scheme,
            split_url.netloc,
            split_url.path,
            safe_query,
            split_url.fragment,
        )
    )
