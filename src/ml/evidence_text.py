"""Pure lexical signals shared by evidence retrieval and constraint checks."""

import re


USEFUL_TOKEN_PATTERN = re.compile(r"[a-z0-9+#.-]{2,}")


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "have",
    "in", "into", "is", "it", "of", "on", "or", "our", "that", "the", "their",
    "this", "to", "using", "we", "who", "will", "with", "work", "role", "job",
    "candidate", "responsible", "required", "requirement", "requirements", "must",
    "preferred", "experience", "data",
}


ACTION_PATTERN = re.compile(
    r"^(?:built|created|developed|designed|implemented|analyzed|evaluated|led|improved|"
    r"reduced|increased|delivered|automated|researched|supported|produced|managed|"
    r"coordinated|conducted|presented|wrote|deployed|optimized)\b",
    re.IGNORECASE,
)


CONCEPT_ALIASES = {
    "data_pipeline": (
        "data pipeline", "data pipelines", "etl", "data ingestion", "data integration",
        "data workflow", "data workflows", "processing pipeline", "processing pipelines",
    ),
    "analytics": (
        "data analysis", "data analytics", "analyze data", "analysed data", "analytics",
        "statistical analysis", "insights",
    ),
    "dashboard_reporting": (
        "dashboard", "dashboards", "reporting", "business intelligence", "bi report",
        "visualization", "visualisation", "tableau",
    ),
    "automation": ("automate", "automated", "automation", "workflow automation"),
    "machine_learning": (
        "machine learning", "classification", "predictive model", "predictive models",
        "model training", "training workflow", "training workflows", "pytorch", "jax",
        "onnx", "scikit-learn", "sklearn", "ml pipeline", "ml systems",
    ),
    "model_evaluation": (
        "model evaluation", "cross-validation", "cross validation", "f1", "confusion matrix",
        "precision", "recall", "roc auc", "evaluation pipeline", "evaluation pipelines", "evals",
    ),
    "communication": (
        "communicate", "communicated", "communicating", "communication", "presented", "presentation", "stakeholder",
        "stakeholders", "documentation", "documented", "technical writing",
    ),
    "software_delivery": (
        "production system", "production systems", "deployed", "deployment", "api",
        "service", "services", "software development", "inference", "onnx packaging",
    ),
    "delivery_automation": (
        "continuous integration", "continuous delivery", "ci cd", "cicd", "ci/cd",
        "github actions", "build pipeline", "build pipelines",
    ),
    "database": ("sql", "database", "databases", "postgres", "mysql", "warehouse"),
    "python": ("python", "pandas", "numpy"),
    "cloud": ("aws", "azure", "gcp", "cloud platform", "cloud services"),
    "java_enterprise": ("java", "j2ee", "jee", "spring mvc", "spring framework"),
    "spreadsheet_analysis": (
        "excel", "pivot table", "pivot tables", "pivottable", "pivottables", "vlookup",
    ),
    "collaboration": (
        "collaborated", "collaboration", "cross-functional", "cross functional", "teamwork",
    ),
}


YEAR_WORDS = {
    "one": 1.0,
    "two": 2.0,
    "three": 3.0,
    "four": 4.0,
    "five": 5.0,
    "six": 6.0,
    "seven": 7.0,
    "eight": 8.0,
    "nine": 9.0,
    "ten": 10.0,
}


def useful_tokens(text: str) -> set[str]:
    """Return specific lexical terms used by the transparent similarity layer."""
    return {
        token.strip(".-")
        for token in USEFUL_TOKEN_PATTERN.findall(text.lower())
        if token.strip(".-") and token.strip(".-") not in STOPWORDS
    }


def concept_tags(text: str) -> set[str]:
    """Map common job/resume paraphrases to reviewable canonical concepts."""
    normalized = " " + re.sub(r"[^a-z0-9+#]+", " ", text.lower()).strip() + " "
    return {
        concept
        for concept, aliases in CONCEPT_ALIASES.items()
        if any(f" {alias} " in normalized for alias in aliases)
    }


def stated_years(text: str) -> list[float]:
    """Extract numeric or short word-form years-of-experience statements."""
    lowered = text.lower()
    values = [
        float(value)
        for value in re.findall(r"\b(\d+(?:\.\d+)?)\+?\s*(?:years|yrs)\b", lowered)
    ]
    values.extend(
        number
        for word, number in YEAR_WORDS.items()
        if re.search(rf"\b{word}\s+(?:years|yrs)\b", lowered)
    )
    return values


def clean_source_line(raw_line: str) -> str:
    """Clean Markdown decoration while preserving factual source wording."""
    line = raw_line.strip()
    line = re.sub(r"^[-*•]+\s*", "", line)
    line = re.sub(r"^\d+[.)]\s*", "", line)
    line = re.sub(r"\*\*([^*]+)\*\*", r"\1", line)
    line = re.sub(r"`([^`]+)`", r"\1", line)
    line = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", line)
    return re.sub(r"\s+", " ", line).strip()
