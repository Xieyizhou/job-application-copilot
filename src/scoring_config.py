"""Shared constants for deterministic job-fit analysis."""

from __future__ import annotations

from scoring_types import PenaltyRule, RoleFocusRule, ScoreCategoryConfig

REQUIRED_REQUIREMENT_WEIGHT = 1.0
PREFERRED_REQUIREMENT_WEIGHT = 0.5
DIRECT_MATCH_STRENGTH = 1.0
PARTIAL_MATCH_STRENGTH = 0.6
NO_MATCH_STRENGTH = 0.0

CAREER_LEVELS = {"student", "new_grad", "junior", "mid", "senior", "unknown"}
DEGREE_RANK = {"unknown": 0, "bachelor": 1, "master": 2, "phd": 3}

KEYWORD_CATALOG = {
    "Python": ["python"],
    "pandas": ["pandas"],
    "scikit-learn": ["scikit-learn", "sklearn"],
    "SQL": ["sql"],
    "machine learning": ["machine learning", "ml"],
    "model evaluation": ["model evaluation", "evaluate models", "model performance"],
    "data visualization": ["data visualization", "visualization", "charts"],
    "data analysis": ["data analysis", "analytics"],
    "business analysis": ["business analysis", "business analyst"],
    "requirements gathering": [
        "requirements gathering",
        "gather requirements",
        "business requirements",
    ],
    "process improvement": [
        "process improvement",
        "business process improvement",
        "workflow improvement",
    ],
    "UAV": ["uav", "drone", "aerial"],
    "robotics": ["robotics", "robot"],
    "sensor data": ["sensor data", "inspection data"],
    "thermal data": ["thermal data", "thermal-analysis", "temperature-difference"],
    "route planning": ["route planning", "route-planning", "path planning"],
    "game AI": ["game ai", "npc", "game artificial intelligence"],
    "econometrics": ["econometrics"],
    "statistical analysis": ["statistical analysis", "statistics"],
    "entry-level": ["entry-level", "entry level"],
    "new grad": ["new grad", "new graduate"],
    "intern": ["intern", "internship"],
    "junior": ["junior"],
    "recent graduate": ["recent graduate", "recent grad", "undergraduate graduate"],
    "classification": ["classification", "supervised classification"],
    "PCA": ["pca", "principal component analysis"],
    "feature engineering": ["feature engineering", "engineered timing-summary features"],
    "CNN": ["cnn", "cnns", "convolutional neural network"],
    "reinforcement learning": ["reinforcement learning"],
    "causal inference": ["causal inference", "difference-in-differences", "did"],
    "communication": ["communication", "technical explanation", "instructional sessions"],
    "teaching": ["teaching", "peer instruction", "instructional assistant"],
    "teamwork": ["teamwork", "collaboration", "collaborate", "collaborative"],
    "documentation": ["documentation", "document", "write reports"],
    "presentation": ["presentation", "present", "presenting"],
    "stakeholder management": ["stakeholder management"],
    "R": [" r ", "r programming"],
    "MATLAB": ["matlab"],
    "C#": ["c#", "c sharp"],
    "Excel": ["excel", "spreadsheets"],
    "NumPy": ["numpy"],
}

SCORE_CATEGORIES: dict[str, ScoreCategoryConfig] = {
    "Core technical skills": {
        "points": 40,
        "keywords": [
            "Python", "pandas", "scikit-learn", "SQL", "machine learning",
            "model evaluation", "data visualization", "data analysis", "Excel",
            "business analysis", "requirements gathering", "process improvement",
        ],
    },
    "Domain fit": {
        "points": 25,
        "keywords": [
            "UAV", "robotics", "sensor data", "thermal data", "route planning",
            "game AI", "econometrics", "statistical analysis",
        ],
    },
    "Experience level fit": {
        "points": 15,
        "keywords": [
            "entry-level", "new grad", "intern", "junior", "recent graduate",
            "mid-level", "senior-level",
        ],
    },
    "Project relevance": {
        "points": 10,
        "keywords": [
            "classification", "PCA", "feature engineering", "CNN",
            "reinforcement learning", "causal inference",
        ],
    },
    "Communication / collaboration fit": {
        "points": 10,
        "keywords": [
            "communication", "teaching", "teamwork", "documentation", "presentation",
            "stakeholder management",
        ],
    },
}

PENALTY_RULES: list[PenaltyRule] = []

RED_FLAG_RULES = [
    {
        "name": "Citizenship, permanent residency, or work authorization requirement",
        "patterns": [
            r"\bcitizenship\b", r"\bcitizen\b", r"\bpermanent\s+residen(?:t|cy)\b",
            r"\bgreen\s+card\b", r"\bwork\s+authorization\b",
            r"\bauthorized\s+to\s+work\b", r"\bvisa\b", r"\bsponsorship\b",
        ],
    }
]

UK_HPI_NOTE = (
    "UK work authorization note: verify the candidate's current status and any available visa route "
    "against official guidance. Do not claim work authorization unless the candidate has confirmed it."
)
UK_HPI_MANUAL_REVIEW_WARNING = (
    "Manual review required: confirm whether the employer's sponsorship and work-authorization "
    "requirements match the candidate's actual status."
)
UK_ALREADY_AUTHORIZED_WARNING = (
    "Manual review required: this JD appears to require candidates to already or currently "
    "have the right to work in the UK. Current authorization is not assumed; eligibility is evaluated separately."
)

PARTIAL_RESUME_MATCHES = {
    "robotics": (
        {"UAV", "route planning"},
        "Partial match: the candidate source contains adjacent UAV or route-planning evidence.",
    ),
    "sensor data": (
        {"UAV", "thermal data"},
        "Partial match: the candidate source contains adjacent UAV or thermal-data evidence.",
    ),
}

PREFERRED_LANGUAGE = ["plus", "preferred", "nice to have", "bonus", "desired", "would be helpful"]

EXPERIENCE_LEVEL_KEYWORDS = [
    "entry-level", "new grad", "intern", "junior", "recent graduate", "mid-level", "senior-level",
]

EXPERIENCE_THEMES = {
    "Python data analysis": ["Python", "pandas", "NumPy", "data analysis", "statistics", "data visualization"],
    "Machine learning model evaluation": ["machine learning", "model evaluation", "scikit-learn", "classification", "PCA"],
    "UAV inspection algorithms": ["UAV", "sensor data", "route planning", "obstacle avoidance"],
    "Game AI and reinforcement learning": ["game AI", "reinforcement learning", "CNN", "C#"],
    "Econometrics and statistical reasoning": ["econometrics", "regression", "causal inference", "statistical analysis"],
    "Teaching and communication": ["communication", "teaching", "Python", "data analysis"],
}

# Ordered from narrow/specialized titles to broader role families. Title focus is
# a separate signal from requirement coverage: it can reduce a superficially high
# keyword match, but it never fills an explicitly missing requirement.
ROLE_FOCUS_RULES: list[RoleFocusRule] = [
    {
        "name": "Physics",
        "title_patterns": [r"\bphysics\b"],
        "candidate_aliases": ["physics", "physical science", "mechanics", "electromagnetism", "quantum"],
    },
    {
        "name": "Audio engineering",
        "title_patterns": [r"\baudio\b", r"\bacoustic"],
        "candidate_aliases": ["audio engineering", "audio", "acoustic", "signal processing", "speech processing"],
    },
    {
        "name": "Data entry / annotation",
        "title_patterns": [r"\bdata entry\b", r"\bdata annotation\b", r"\bdata label"],
        "candidate_aliases": ["data entry", "data annotation", "data labeling", "data labelling", "video annotation"],
    },
    {
        "name": "Data migration",
        "title_patterns": [r"\bdata migration\b"],
        "candidate_aliases": ["data migration", "etl", "data integration", "data warehouse"],
    },
    {
        "name": "Data engineering",
        "title_patterns": [r"\bdata engineer"],
        "candidate_aliases": ["data engineering", "data pipeline", "etl", "data warehouse", "spark"],
    },
    {
        "name": "Business intelligence / analysis",
        "title_patterns": [r"\bbusiness intelligence\b", r"\bbusiness (?:and )?data analyst\b", r"\bbusiness analyst\b"],
        "candidate_aliases": ["business intelligence", "power bi", "tableau", "business analysis", "requirements gathering"],
    },
    {
        "name": "Data analytics",
        "title_patterns": [r"\bdata analyst\b", r"\banalytics analyst\b"],
        "candidate_aliases": ["data analysis", "analytics", "sql", "statistical analysis", "data visualization"],
    },
    {
        "name": "Data science",
        "title_patterns": [r"\bdata scientist\b", r"\bdata science\b"],
        "candidate_aliases": ["data science", "machine learning", "statistical analysis", "model evaluation"],
    },
    {
        "name": "Machine learning / AI engineering",
        "title_patterns": [
            r"\bmachine learning\b",
            r"\bai engineer\b",
            r"\bai developer\b",
            r"\breinforcement learning\b",
        ],
        "candidate_aliases": [
            "machine learning", "model evaluation", "scikit-learn", "sklearn",
            "deep learning", "reinforcement learning",
        ],
    },
]
