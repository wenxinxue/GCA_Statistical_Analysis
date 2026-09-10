from __future__ import annotations

RCT_PDQI_DOMAINS = [
    "Up-to-date",
    "Accurate",
    "Thorough",
    "Useful",
    "Organized",
    "Comprehensible",
    "Succinct",
    "Synthesized",
    "Internally consistent",
]

PATIENT_REPORTED_ITEMS = [
    "Communication convenience",
    "Perceived physician attention",
    "Perceived empathy",
    "Overall satisfaction",
]

CORE_CONTINUOUS_OUTCOMES = [
    "Total admission workflow time",
    "Interview time",
    "Physician follow-up work time",
    "Text editing workload",
    "PDQI-9 total score",
]

IMPLEMENTATION_OUTCOMES = [
    "Useful top-5 diagnostic recommendations",
    "Useful test recommendations",
    "Useful graph-based prompts",
    "Perceived workload reduction",
    "Reported workflow interference",
    "Willingness to use AI assistance again",
]

CLINICAL_ERROR_SEVERITY_RCT_HEADING = (
    "Clinical error severity of AI-generated drafts, n (%)"
)
CLINICAL_ERROR_SEVERITY_MULTICENTRE_HEADING = (
    "Clinical error severity of GCA-generated drafts, n (%)"
)
CLINICAL_ERROR_SEVERITY_LEVELS = [
    "Level 0: no medically substantive error",
    "Level 1: non-serious medically substantive error",
    "Level 2: potentially serious clinical error",
]
CHARACTER_LEVEL_FACTUAL_CONFLICT_HEADING = "Character-level factual conflicts"

CENTRE_FILES = [
    ("Centre 1", "HM.xlsx"),
    ("Centre 2", "JJ.xlsx"),
    ("Centre 3", "JX.xlsx"),
    ("Centre 4", "WHRM.xlsx"),
    ("Centre 5", "WHZX.xlsx"),
]
