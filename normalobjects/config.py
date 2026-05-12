import os
import re

from dotenv import load_dotenv


load_dotenv()

SUPPORTED_CATEGORIES = {"portal", "monster", "psychic", "environmental"}
ALL_CATEGORIES = SUPPORTED_CATEGORIES | {"other"}
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
STAGE_ICONS = {
    "INTAKE": "📥",
    "VALIDATE": "✅",
    "INVESTIGATE": "🔎",
    "RESOLVE": "🛠️",
    "CLOSE": "🔒",
}
RELATIVE_DATE_PATTERN = re.compile(
    r"\b(today|tonight|yesterday|tomorrow|last night|this morning|this afternoon|this evening)\b",
    re.IGNORECASE,
)
TIME_PATTERN = re.compile(
    r"\b(?:at\s*)?(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)?\b",
    re.IGNORECASE,
)
RESOLUTION_RESULTS = {"applied", "blocked", "escalated", "linked_case"}
EFFECTIVENESS_RATINGS = {"high", "medium", "low"}
SATISFACTION_STATUSES = {"satisfied", "unsatisfied", "unreachable", "pending"}
