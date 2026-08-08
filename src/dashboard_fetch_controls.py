"""Search-market controls and provider-specific location mapping."""

from __future__ import annotations

from typing import Any


REGION_OPTIONS = [
    "Remote", "Singapore", "United States", "United Kingdom", "Canada", "China",
    "Australia", "India", "Germany", "France", "Netherlands", "Japan",
    "South Korea", "New Zealand", "Brazil",
]
ADZUNA_SUPPORTED_COUNTRIES = {
    "sg", "gb", "us", "ca", "au", "nz", "de", "fr", "it", "nl", "pl", "br", "za", "in"
}
REGION_COUNTRY_CODES = {
    "Remote": "us", "Singapore": "sg", "United States": "us",
    "United Kingdom": "gb", "Canada": "ca", "China": "cn", "Australia": "au",
    "India": "in", "Germany": "de", "France": "fr", "Netherlands": "nl",
    "Japan": "jp", "South Korea": "kr", "New Zealand": "nz", "Brazil": "br",
}
REGION_CONFIG = {
    region: {
        "adzuna_country": country,
        "adzuna_location": region,
        "jooble_location": region,
    }
    for region, country in REGION_COUNTRY_CODES.items()
}
REGION_CONFIG["Remote"] = {
    "adzuna_country": "us",
    "adzuna_location": "Remote",
    "jooble_location": "Remote",
}
REGION_COUNTRY_MARKERS = {
    "ca": ("canada", "toronto", "vancouver", "montreal"),
    "cn": ("china", "beijing", "shanghai", "shenzhen", "hangzhou"),
    "de": ("germany", "berlin", "munich", "frankfurt"),
    "fr": ("france", "paris", "lyon"),
    "gb": ("united kingdom", "uk", "london", "england", "scotland"),
    "in": ("india", "bangalore", "bengaluru", "hyderabad", "mumbai", "delhi"),
    "jp": ("japan", "tokyo", "osaka"),
    "kr": ("south korea", "korea", "seoul"),
    "sg": ("singapore",),
    "au": ("australia", "sydney", "melbourne", "brisbane"),
    "nz": ("new zealand", "auckland", "wellington"),
    "br": ("brazil", "sao paulo", "rio de janeiro"),
    "nl": ("netherlands", "amsterdam", "rotterdam"),
}


def render_region_fields(ui: Any, *, show_debug_ui: bool) -> tuple[str, str, str, bool]:
    """Render a searchable suggestion field that also accepts arbitrary locations."""
    region = ui.selectbox(
        "Region",
        REGION_OPTIONS,
        index=None,
        key="fetch_region",
        placeholder="Type a country, city, or Remote",
        accept_new_options=True,
        help="Choose a suggested market or enter any city, country, or region.",
    )
    location_text = str(region or "").strip()
    region_config = REGION_CONFIG.get(location_text)
    if region_config is None:
        lowered = location_text.lower()
        country = next(
            (code for code, markers in REGION_COUNTRY_MARKERS.items() if any(marker in lowered for marker in markers)),
            "us",
        )
        region_config = {
            "adzuna_country": country,
            "adzuna_location": location_text,
            "jooble_location": location_text,
        }
    adzuna_country = region_config["adzuna_country"]
    if show_debug_ui:
        adzuna_country = ui.text_input("Developer: Adzuna country", value=adzuna_country)
    return (
        adzuna_country,
        region_config["adzuna_location"],
        region_config["jooble_location"],
        adzuna_country.lower() in ADZUNA_SUPPORTED_COUNTRIES,
    )


def render_advanced_fetch_options(
    ui: Any,
    *,
    minimum_recommendations: int,
    maximum_recommendations: int,
    default_recommendations: int,
    default_per_source: int,
    maximum_per_source: int,
) -> tuple[int, int]:
    """Keep provider-volume controls available without dominating the search flow."""
    with ui.expander(
        "Advanced options",
        expanded=False,
        icon=":material/tune:",
    ):
        recommendations, per_source = ui.columns(2)
        with recommendations:
            recommendation_limit = ui.slider(
                "Number of recommendations",
                min_value=minimum_recommendations,
                max_value=maximum_recommendations,
                value=default_recommendations,
                help="How many ranked jobs to display after duplicate removal.",
            )
        with per_source:
            fetch_limit_per_source = ui.slider(
                "Jobs per source",
                min_value=5,
                max_value=maximum_per_source,
                value=default_per_source,
                help="How many jobs to request from each source before filtering.",
            )
    return recommendation_limit, fetch_limit_per_source


def render_fetch_search_styles(ui: Any) -> None:
    """Apply the accepted compact search-workspace visual system."""
    ui.markdown(
        """
        <style>
        .st-key-fetch_search_panel {
            width:min(100%,74rem);margin-top:1.15rem;
        }
        .st-key-fetch_search_panel > div[data-testid="stVerticalBlockBorderWrapper"] {
            padding:.35rem .5rem .45rem;border-color:#dfe3e8;
            border-radius:10px;box-shadow:none;background:#fff;
        }
        .st-key-fetch_search_panel [data-testid="stForm"] {border:0;padding:0}
        .st-key-fetch_search_panel [data-testid="stExpander"] {
            margin:.3rem 0 .7rem;border-color:#dfe3e8;border-radius:8px;background:#fff;
        }
        .st-key-fetch_search_panel [data-testid="stFormSubmitButton"] button {
            min-width:9.25rem;font-weight:650;
        }
        .st-key-fetch_search_panel [data-baseweb="input"],
        .st-key-fetch_search_panel [data-baseweb="select"] > div {
            min-height:3rem;
        }
        @media (max-width:900px) {
            .st-key-fetch_search_panel {width:100%}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
