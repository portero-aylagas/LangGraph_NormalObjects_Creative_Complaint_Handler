import json
import os
import re
from datetime import date, datetime, time, timedelta
from typing import Any, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph


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


class ComplaintState(TypedDict, total=False):
    complaint: str
    complaint_id: str
    complainant: str
    submitted_at: str
    occurred_at: str
    location: str
    category: str
    category_reasoning: str
    extracted_metadata: dict[str, str]
    issue_signature: str
    needs_clarification: bool
    missing_details: list[str]
    detail_notes: str
    duplicate_detected: bool
    duplicate_of: str | None
    historical_similar_cases: list[str]
    category_validation_passed: bool
    validation_notes: list[str]
    is_valid: bool
    investigation: dict[str, Any]
    resolution: dict[str, Any]
    closure: dict[str, Any]
    status: str
    workflow_path: list[str]


COMPLAINT_DATABASE: list[dict[str, str]] = [
    {
        "complaint_id": "NO-2026-0410-001",
        "complainant": "Joyce Byers",
        "submitted_at": "2026-04-10",
        "category": "portal",
        "issue_signature": "portal opens near byers house after midnight",
    },
    {
        "complaint_id": "NO-2026-0424-002",
        "complainant": "Dustin Henderson",
        "submitted_at": "2026-04-24",
        "category": "monster",
        "issue_signature": "demogorgon packs near hawkins lab loading dock",
    },
    {
        "complaint_id": "NO-2026-0331-003",
        "complainant": "Eleven Hopper",
        "submitted_at": "2026-03-31",
        "category": "psychic",
        "issue_signature": "psychic lift limit for heavy quarry rocks",
    },
    {
        "complaint_id": "NO-2026-0320-004",
        "complainant": "Hawkins Power",
        "submitted_at": "2026-03-20",
        "category": "environmental",
        "issue_signature": "power lines flicker during storm at melvald alley",
    },
]


DOWNSIDE_UP_PROTOCOLS: dict[str, list[dict[str, str]]] = {
    "portal": [
        {
            "protocol_id": "DU-PORTAL-01",
            "protocol_name": "Portal Timing Stabilization",
            "procedure": "Stabilize reported portal timing and recheck the portal location against the complaint details.",
            "specialized_team": "",
        }
    ],
    "monster": [
        {
            "protocol_id": "DU-MONSTER-01",
            "protocol_name": "Creature Behavior Containment",
            "procedure": "Contain the reported creature behavior pattern and escalate if the investigation risk is high or unknown.",
            "specialized_team": "Monster Response Unit",
        }
    ],
    "psychic": [
        {
            "protocol_id": "DU-PSYCHIC-01",
            "protocol_name": "Psychic Ability Baseline Review",
            "procedure": "Compare the reported psychic limitation with the expected ability baseline and recommend rest or retesting.",
            "specialized_team": "",
        }
    ],
    "environmental": [
        {
            "protocol_id": "DU-ENV-01",
            "protocol_name": "Environmental Anomaly Isolation",
            "procedure": "Isolate the reported electrical, weather, or physical anomaly and escalate if risk is high or unknown.",
            "specialized_team": "Environmental Anomaly Team",
        }
    ],
}
RESOLUTION_RESULTS = {"applied", "blocked", "escalated", "linked_case"}
EFFECTIVENESS_RATINGS = {"high", "medium", "low"}
SATISFACTION_STATUSES = {"satisfied", "unsatisfied", "unreachable", "pending"}


def _llm() -> ChatOpenAI:
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError(
            "OPENAI_API_KEY not found. Add it to your environment or .env file."
        )
    return ChatOpenAI(model=MODEL_NAME, temperature=0)


def _stage_label(stage: str) -> str:
    return f"{STAGE_ICONS[stage]} [{stage}]"


def _strip_code_fence(content: str) -> str:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _llm_json(system_prompt: str, user_prompt: str) -> dict[str, Any]:
    response = _llm().invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )
    content = _strip_code_fence(response.content)
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError(f"LLM returned invalid JSON: {content}") from error
    if not isinstance(parsed, dict):
        raise ValueError(f"LLM returned JSON that is not an object: {content}")
    return parsed


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).date()
    except ValueError:
        return None


def _today() -> date:
    return date.today()


def _state_date(state: ComplaintState) -> date:
    return (
        _parse_date(state.get("submitted_at"))
        or _parse_date(state.get("occurred_at"))
        or _today()
    )


def _parse_time_from_phrase(value: str) -> time | None:
    match = TIME_PATTERN.search(value.replace(".", ""))
    if not match:
        return None

    hour = int(match.group(1))
    minute = int(match.group(2) or 0)
    meridiem = (match.group(3) or "").lower().replace(".", "")

    if minute > 59:
        return None
    if meridiem:
        if not 1 <= hour <= 12:
            return None
        if meridiem.startswith("p") and hour != 12:
            hour += 12
        elif meridiem.startswith("a") and hour == 12:
            hour = 0
    elif not 0 <= hour <= 23:
        return None

    return time(hour=hour, minute=minute)


def _normalise_relative_datetime(
    value: str | None, reference_date: date | None = None
) -> str:
    if not value:
        return ""

    text = value.strip()
    if not text:
        return ""

    try:
        return date.fromisoformat(text).isoformat()
    except ValueError:
        pass

    try:
        return datetime.fromisoformat(text).isoformat(timespec="minutes")
    except ValueError:
        pass

    lowered = text.lower()
    base_date = reference_date or _today()
    if "day before yesterday" in lowered:
        occurred_date = base_date - timedelta(days=2)
    elif "yesterday" in lowered or "last night" in lowered:
        occurred_date = base_date - timedelta(days=1)
    elif "tomorrow" in lowered:
        occurred_date = base_date + timedelta(days=1)
    elif RELATIVE_DATE_PATTERN.search(lowered):
        occurred_date = base_date
    else:
        return text

    parsed_time = _parse_time_from_phrase(text)
    if parsed_time:
        return datetime.combine(occurred_date, parsed_time).isoformat(timespec="minutes")
    return occurred_date.isoformat()


def _same_issue_with_llm(
    state: ComplaintState, issue_signature: str, record: dict[str, str]
) -> bool:
    result = _llm_json(
        "You compare complaint records for duplicate detection. Return strict JSON only.",
        f"""
Decide whether these two complaint records describe the same underlying issue.

Rules:
- Match based on the recurring issue, not exact wording.
- Treat small wording differences as the same issue if the complainant would reasonably be following up about the same problem.
- Do not match merely because both records are in the same broad category.

Current complaint:
{json.dumps({
    "complaint": state.get("complaint", ""),
    "complainant": state.get("complainant", ""),
    "category": state.get("category", ""),
    "location": state.get("location", ""),
    "occurred_at": state.get("occurred_at", ""),
    "issue_signature": issue_signature,
}, indent=2)}

Existing record:
{json.dumps(record, indent=2)}

Return JSON with this shape:
{{
  "same_issue": true,
  "reasoning": "short reason"
}}
""",
    )
    return bool(result.get("same_issue"))


def _categorize_complaint(state: ComplaintState) -> tuple[str, str]:
    result = _llm_json(
        "You classify Downside Up complaint intake items. Return strict JSON only.",
        f"""
Choose exactly one category for this complaint:
- portal: portal timing, portal location, portal stability, rifts, gates, openings
- monster: creature behavior, sightings, attacks, demogorgons, demodogs, mind flayer
- psychic: psychic abilities, visions, mind powers, limits, side effects
- environmental: electricity, lights, weather, air, temperature, physical environment
- other: anything not covered above

Complaint text:
{state.get("complaint", "")}

Known metadata:
complainant: {state.get("complainant", "")}
submitted_at: {state.get("submitted_at", "")}
occurred_at: {state.get("occurred_at", "")}
location: {state.get("location", "")}

Return JSON with this shape:
{{
  "category": "portal|monster|psychic|environmental|other",
  "reasoning": "short reason"
}}
""",
    )
    category = str(result.get("category", "other")).strip().lower()
    if category not in ALL_CATEGORIES:
        category = "other"
    return category, str(result.get("reasoning", "")).strip()


def _extract_metadata(state: ComplaintState) -> dict[str, str]:
    result = _llm_json(
        "You extract explicit complaint metadata. Return strict JSON only.",
        f"""
Extract metadata that is explicitly present in this complaint text.

Rules:
- Do not invent missing values.
- If a value is not stated, return an empty string for that field.
- complainant is the person or organization making the complaint, if named.
- occurred_at should be an ISO date if the complaint gives a specific date.
- If the complaint gives a relative date or time such as today, today at 9, yesterday, or last night, keep the full relative phrase.
- location is where the issue happened, if stated.

Complaint text:
{state.get("complaint", "")}

Return JSON with this shape:
{{
  "complainant": "name or empty string",
  "occurred_at": "ISO date, relative date phrase, or empty string",
  "location": "place or empty string"
}}
""",
    )
    metadata = {
        "complainant": str(result.get("complainant", "")).strip(),
        "occurred_at": str(result.get("occurred_at", "")).strip(),
        "location": str(result.get("location", "")).strip(),
    }
    return metadata


def _check_essential_details(state: ComplaintState) -> tuple[list[str], str]:
    result = _llm_json(
        "You check complaint intake completeness. Return strict JSON only.",
        f"""
Determine whether the complaint has the four essential details.

Rules:
- who: who is complaining or who experienced the issue. Metadata can satisfy this.
- what: what happened or what the issue is.
- when: when it happened. Relative dates such as today, yesterday, last night, or metadata can satisfy this.
- where: where it happened. Metadata can satisfy this.

Complaint text:
{state.get("complaint", "")}

Known metadata:
complainant: {state.get("complainant", "")}
submitted_at: {state.get("submitted_at", "")}
occurred_at: {state.get("occurred_at", "")}
location: {state.get("location", "")}

Return JSON with this shape:
{{
  "present": {{"who": true, "what": true, "when": true, "where": true}},
  "missing_details": ["who", "what", "when", "where"],
  "notes": "short explanation"
}}
""",
    )
    missing = result.get("missing_details", [])
    if not isinstance(missing, list):
        missing = []
    present = result.get("present", {})
    if isinstance(present, dict):
        for detail in ["who", "what", "when", "where"]:
            if present.get(detail) is False:
                missing.append(detail)
    allowed = {"who", "what", "when", "where"}
    clean_missing = []
    for item in missing:
        detail = str(item)
        if detail in allowed and detail not in clean_missing:
            clean_missing.append(detail)
    return clean_missing, str(result.get("notes", "")).strip()


def _create_issue_signature(state: ComplaintState) -> str:
    result = _llm_json(
        "You create stable duplicate-detection signatures. Return strict JSON only.",
        f"""
Create a short canonical issue signature for this complaint.

The signature should:
- describe the core recurring issue, not the desired resolution
- be lowercase
- be 5 to 12 words
- omit names unless the name is essential to the issue
- use stable wording likely to match a future duplicate of the same issue

Category: {state.get("category", "other")}
Complaint text:
{state.get("complaint", "")}

Known metadata:
complainant: {state.get("complainant", "")}
submitted_at: {state.get("submitted_at", "")}
occurred_at: {state.get("occurred_at", "")}
location: {state.get("location", "")}

Return JSON with this shape:
{{"issue_signature": "short stable signature"}}
""",
    )
    signature = str(result.get("issue_signature", "")).strip().lower()
    if not signature:
        raise ValueError("LLM did not return an issue_signature.")
    return signature


def _validate_category_rule(state: ComplaintState) -> tuple[bool, str]:
    category = state.get("category", "other")
    if category == "other":
        return False, "No automated category-specific validation exists for other."

    result = _llm_json(
        "You validate complaint categories for an intake workflow. Return strict JSON only.",
        f"""
Validate whether this complete complaint is a real supported-category complaint.

Category being validated: {category}

Category definitions:
- portal: portal timing, portal location, portal stability, rifts, gates, openings
- monster: creature behavior, sightings, attacks, demogorgons, demodogs, mind flayer
- psychic: psychic abilities, visions, mind powers, limits, side effects
- environmental: electricity, lights, weather, air, temperature, physical environment

Pass only if the complaint genuinely belongs in the selected category and includes an actionable issue for that category.

Complaint text:
{state.get("complaint", "")}

Known metadata:
complainant: {state.get("complainant", "")}
submitted_at: {state.get("submitted_at", "")}
occurred_at: {state.get("occurred_at", "")}
location: {state.get("location", "")}

Return JSON with this shape:
{{
  "passed": true,
  "notes": "short explanation"
}}
""",
    )
    return bool(result.get("passed")), str(result.get("notes", "")).strip()


def _find_related_complaints(
    state: ComplaintState, issue_signature: str
) -> tuple[dict[str, str] | None, list[str]]:
    complainant = state.get("complainant", "").strip().lower()
    submitted_at = _state_date(state)
    historical: list[str] = []

    for record in COMPLAINT_DATABASE:
        same_complainant = complainant and complainant == record["complainant"].lower()
        same_category = state.get("category") == record.get("category")
        record_date = _parse_date(record.get("submitted_at"))
        if not record_date or not same_category:
            continue

        age_days = abs((submitted_at - record_date).days)
        should_compare = (same_complainant and age_days <= 30) or age_days > 30
        if not should_compare:
            continue

        same_issue = _same_issue_with_llm(state, issue_signature, record)
        if same_issue and same_complainant and age_days <= 30:
            return record, historical
        if same_issue and age_days > 30:
            historical.append(record["complaint_id"])

    return None, historical


def _append_demo_record(state: ComplaintState) -> None:
    COMPLAINT_DATABASE.append(
        {
            "complaint_id": state["complaint_id"],
            "complainant": state.get("complainant", "Unknown"),
            "submitted_at": state.get("submitted_at", _today().isoformat()),
            "category": state["category"],
            "issue_signature": state["issue_signature"],
        }
    )


def intake_node(state: ComplaintState) -> ComplaintState:
    """Step 1: LLM intake, category assignment, completeness, and related cases."""
    print(f"\n{_stage_label('INTAKE')} Processing complaint...")

    complaint_id = state.get("complaint_id") or f"NO-DEMO-{len(COMPLAINT_DATABASE) + 1:03d}"
    submitted_at = _normalise_relative_datetime(state.get("submitted_at"))
    reference_date = _parse_date(submitted_at) or _today()

    category, category_reasoning = _categorize_complaint(state)
    extracted_metadata = _extract_metadata(state)
    occurred_at = state.get("occurred_at") or extracted_metadata["occurred_at"]
    occurred_at = _normalise_relative_datetime(occurred_at, reference_date)
    intake_state: ComplaintState = {
        **state,
        "complaint_id": complaint_id,
        "complainant": state.get("complainant") or extracted_metadata["complainant"],
        "occurred_at": occurred_at,
        "location": state.get("location") or extracted_metadata["location"],
        "category": category,
        "category_reasoning": category_reasoning,
        "extracted_metadata": extracted_metadata,
    }
    if submitted_at:
        intake_state["submitted_at"] = submitted_at

    missing_details, detail_notes = _check_essential_details(intake_state)
    issue_signature = _create_issue_signature(intake_state)
    duplicate, historical = _find_related_complaints(intake_state, issue_signature)

    new_state: ComplaintState = {
        **intake_state,
        "submitted_at": submitted_at or reference_date.isoformat(),
        "issue_signature": issue_signature,
        "needs_clarification": bool(missing_details),
        "missing_details": missing_details,
        "detail_notes": detail_notes,
        "duplicate_detected": duplicate is not None,
        "duplicate_of": duplicate["complaint_id"] if duplicate else None,
        "historical_similar_cases": historical,
        "workflow_path": state.get("workflow_path", []) + ["intake"],
        "status": "intake_complete",
    }
    _append_demo_record(new_state)

    print(f"{_stage_label('INTAKE')} Categorized as: {category}")
    if missing_details:
        print(f"{_stage_label('INTAKE')} Missing details: {', '.join(missing_details)}")
    if duplicate:
        print(f"{_stage_label('INTAKE')} Duplicate of: {duplicate['complaint_id']}")
    if historical:
        print(f"{_stage_label('INTAKE')} Historical similar cases: {', '.join(historical)}")
    return new_state


def process_complaint(complaint: str) -> ComplaintState:
    """Run the graph for the lab's plain-string complaint input format."""
    return app.invoke({"complaint": complaint})


def validate_node(state: ComplaintState) -> ComplaintState:
    """Step 2: Apply validation priority order with LLM category validation."""
    print(f"\n{_stage_label('VALIDATE')} Validating complaint...")

    category = state.get("category", "other")
    missing_details = state.get("missing_details", [])
    notes: list[str] = []

    if missing_details:
        rule_passed = False
        is_valid = False
        status = "pending_clarification"
        notes.append("Missing essential details take priority.")
    elif category == "other":
        rule_passed = False
        is_valid = False
        status = "manual_review_required"
        notes.append("Complete other-category complaints require manual review.")
    else:
        rule_passed, rule_note = _validate_category_rule(state)
        notes.append(rule_note)
        if not rule_passed:
            is_valid = False
            status = "validation_failed"
        elif state.get("duplicate_detected"):
            is_valid = True
            status = "validated_duplicate"
            notes.append(f"Linked to duplicate case {state.get('duplicate_of')}.")
        else:
            is_valid = True
            status = "validated"

    new_state: ComplaintState = {
        **state,
        "category_validation_passed": rule_passed,
        "validation_notes": notes,
        "is_valid": is_valid,
        "status": status,
        "workflow_path": state.get("workflow_path", []) + ["validate"],
    }
    print(f"{_stage_label('VALIDATE')} Status: {status}")
    return new_state


def _investigate_with_llm(state: ComplaintState) -> dict[str, Any]:
    status = state.get("status", "")
    result = _llm_json(
        "You are the investigation step in a strict complaint workflow. Return strict JSON only.",
        f"""
Create a structured investigation report for this complaint.

Workflow status before investigation: {status}

Rules:
- This is a demo workflow. You may create plausible investigation findings from the complaint state, category, location, date, and issue description.
- Do not claim that real external tools, logs, databases, sensors, or teams were actually checked.
- Phrase invented findings as internal demo investigation conclusions, not verified outside evidence.
- If status is pending_clarification, explain that investigation is blocked and list the missing details needed next.
- If status is manual_review_required, explain that automated investigation is limited because the category is other.
- If status is validation_failed, explain that investigation is blocked by validation failure.
- If status is validated_duplicate, focus on linked-case guidance using duplicate_of.
- If status is validated, create category-specific findings and list evidence that should be checked next.
- Keep findings concrete and useful for the next resolution step.

Complaint state:
{json.dumps({
    "complaint": state.get("complaint", ""),
    "complaint_id": state.get("complaint_id", ""),
    "complainant": state.get("complainant", ""),
    "submitted_at": state.get("submitted_at", ""),
    "occurred_at": state.get("occurred_at", ""),
    "location": state.get("location", ""),
    "category": state.get("category", ""),
    "category_reasoning": state.get("category_reasoning", ""),
    "issue_signature": state.get("issue_signature", ""),
    "missing_details": state.get("missing_details", []),
    "duplicate_detected": state.get("duplicate_detected", False),
    "duplicate_of": state.get("duplicate_of"),
    "historical_similar_cases": state.get("historical_similar_cases", []),
    "validation_notes": state.get("validation_notes", []),
    "is_valid": state.get("is_valid", False),
    "status": status,
}, indent=2)}

Return JSON with exactly these top-level keys:
{{
  "result": "blocked|manual_review|linked_case|evidence_collected",
  "summary": "one sentence investigation summary",
  "findings": ["finding 1", "finding 2"],
  "evidence_to_check": ["evidence item 1", "evidence item 2"],
  "risk_level": "low|medium|high|unknown",
  "recommended_next_action": "short action for the next workflow step",
  "notes": "short note"
}}
""",
    )

    required_fields = {
        "result": str,
        "summary": str,
        "findings": list,
        "evidence_to_check": list,
        "risk_level": str,
        "recommended_next_action": str,
        "notes": str,
    }
    missing_fields = [field for field in required_fields if field not in result]
    if missing_fields:
        raise ValueError(
            f"LLM investigation response missing fields: {', '.join(missing_fields)}"
        )

    if not isinstance(result["findings"], list):
        raise ValueError("LLM investigation response field 'findings' must be a list.")
    if not isinstance(result["evidence_to_check"], list):
        raise ValueError(
            "LLM investigation response field 'evidence_to_check' must be a list."
        )

    return {
        "result": str(result["result"]).strip(),
        "summary": str(result["summary"]).strip(),
        "findings": [str(item).strip() for item in result["findings"]],
        "evidence_to_check": [
            str(item).strip() for item in result["evidence_to_check"]
        ],
        "risk_level": str(result["risk_level"]).strip(),
        "recommended_next_action": str(result["recommended_next_action"]).strip(),
        "notes": str(result["notes"]).strip(),
    }


def investigate_node(state: ComplaintState) -> ComplaintState:
    """Step 3: Record an investigation result for every validation outcome."""
    print(f"\n{_stage_label('INVESTIGATE')} Investigating complaint...")

    investigation = _investigate_with_llm(state)

    new_state: ComplaintState = {
        **state,
        "investigation": investigation,
        "workflow_path": state.get("workflow_path", []) + ["investigate"],
    }
    print(f"{_stage_label('INVESTIGATE')} Result: {investigation['result']}")
    return new_state


def _blocked_resolution(reason: str) -> dict[str, Any]:
    return {
        "result": "blocked",
        "protocol_id": "",
        "protocol_name": "",
        "action": "No resolution applied.",
        "effectiveness": "low",
        "escalation_required": False,
        "specialized_team": "",
        "rationale": reason,
        "notes": reason,
    }


def _linked_case_resolution(state: ComplaintState) -> dict[str, Any]:
    duplicate_of = state.get("duplicate_of") or "the linked complaint"
    return {
        "result": "linked_case",
        "protocol_id": "",
        "protocol_name": "",
        "action": f"Use the documented resolution path from linked case {duplicate_of}.",
        "effectiveness": "medium",
        "escalation_required": False,
        "specialized_team": "",
        "rationale": "Duplicate complaints should be consolidated instead of applying a new independent fix.",
        "notes": f"Linked to existing case {duplicate_of}.",
    }


def _has_documented_investigation(state: ComplaintState) -> bool:
    investigation = state.get("investigation")
    if not isinstance(investigation, dict):
        return False

    result = str(investigation.get("result", "")).strip()
    if result in {"blocked", "manual_review"}:
        return False

    findings = investigation.get("findings", [])
    has_findings = isinstance(findings, list) and bool(findings)
    return bool(
        result
        and (
            str(investigation.get("summary", "")).strip()
            or has_findings
            or str(investigation.get("notes", "")).strip()
        )
    )


def _bool_from_llm(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1"}
    return bool(value)


def _validate_resolution_response(
    result: dict[str, Any], state: ComplaintState, protocols: list[dict[str, str]]
) -> dict[str, Any]:
    required_fields = {
        "result": str,
        "protocol_id": str,
        "protocol_name": str,
        "action": str,
        "effectiveness": str,
        "escalation_required": bool,
        "specialized_team": str,
        "rationale": str,
        "notes": str,
    }
    missing_fields = [field for field in required_fields if field not in result]
    if missing_fields:
        raise ValueError(
            f"LLM resolution response missing fields: {', '.join(missing_fields)}"
        )

    resolution_result = str(result["result"]).strip().lower()
    if resolution_result not in RESOLUTION_RESULTS:
        raise ValueError(f"Unknown resolution result: {result['result']}")

    effectiveness = str(result["effectiveness"]).strip().lower()
    if effectiveness not in EFFECTIVENESS_RATINGS:
        raise ValueError(f"Unknown effectiveness rating: {result['effectiveness']}")

    protocols_by_id = {protocol["protocol_id"]: protocol for protocol in protocols}
    protocol_id = str(result["protocol_id"]).strip()
    protocol = protocols_by_id.get(protocol_id)
    if resolution_result in {"applied", "escalated"} and not protocol:
        raise ValueError(f"Resolution used an unknown protocol_id: {protocol_id}")

    category = state.get("category", "")
    risk_level = str(state.get("investigation", {}).get("risk_level", "")).lower()
    should_escalate = category in {"monster", "environmental"} and risk_level in {
        "high",
        "unknown",
    }
    if resolution_result in {"applied", "escalated"}:
        resolution_result = "escalated" if should_escalate else "applied"
        escalation_required = should_escalate
        specialized_team = protocol["specialized_team"] if should_escalate else ""
        protocol_name = protocol["protocol_name"]
    else:
        escalation_required = _bool_from_llm(result["escalation_required"])
        specialized_team = str(result["specialized_team"]).strip()
        protocol_name = str(result["protocol_name"]).strip()

    return {
        "result": resolution_result,
        "protocol_id": protocol_id,
        "protocol_name": protocol_name,
        "action": str(result["action"]).strip(),
        "effectiveness": effectiveness,
        "escalation_required": escalation_required,
        "specialized_team": specialized_team,
        "rationale": str(result["rationale"]).strip(),
        "notes": str(result["notes"]).strip(),
    }


def _resolve_with_llm(
    state: ComplaintState, protocols: list[dict[str, str]]
) -> dict[str, Any]:
    result = _llm_json(
        "You are the resolution step in a strict complaint workflow. Return strict JSON only.",
        f"""
Create a resolution for this investigated complaint.

Rules:
- Use the documented investigation result below. Do not apply a resolution without it.
- Choose exactly one protocol_id from allowed_protocols.
- The action must be specific to the complaint category.
- The action must explicitly reference the selected Downside Up protocol.
- Monster and environmental complaints with high or unknown risk must be escalated to the protocol's specialized_team.
- Portal and psychic complaints should not escalate unless the protocol includes a specialized_team.
- effectiveness must be exactly high, medium, or low.
- Do not invent protocols that are not listed in allowed_protocols.

Complaint state:
{json.dumps({
    "complaint": state.get("complaint", ""),
    "complaint_id": state.get("complaint_id", ""),
    "complainant": state.get("complainant", ""),
    "submitted_at": state.get("submitted_at", ""),
    "occurred_at": state.get("occurred_at", ""),
    "location": state.get("location", ""),
    "category": state.get("category", ""),
    "category_reasoning": state.get("category_reasoning", ""),
    "status": state.get("status", ""),
    "issue_signature": state.get("issue_signature", ""),
    "duplicate_detected": state.get("duplicate_detected", False),
    "duplicate_of": state.get("duplicate_of"),
    "investigation": state.get("investigation", {}),
}, indent=2)}

allowed_protocols:
{json.dumps(protocols, indent=2)}

Return JSON with exactly these top-level keys:
{{
  "result": "applied|blocked|escalated|linked_case",
  "protocol_id": "one allowed protocol_id, or empty string if blocked/linked_case",
  "protocol_name": "selected protocol name, or empty string if blocked/linked_case",
  "action": "specific resolution action",
  "effectiveness": "high|medium|low",
  "escalation_required": true,
  "specialized_team": "team name or empty string",
  "rationale": "why this protocol fits the investigation",
  "notes": "short note"
}}
""",
    )
    return _validate_resolution_response(result, state, protocols)


def resolve_node(state: ComplaintState) -> ComplaintState:
    """Step 4: Apply category-specific resolution rules."""
    print(f"\n{_stage_label('RESOLVE')} Resolving complaint...")

    investigation = state.get("investigation", {})
    if not _has_documented_investigation(state):
        resolution = _blocked_resolution(
            "No resolution can be applied without documented investigation results."
        )
    elif state.get("status") == "validated_duplicate" or investigation.get(
        "result"
    ) == "linked_case":
        resolution = _linked_case_resolution(state)
    else:
        category = state.get("category", "other")
        protocols = DOWNSIDE_UP_PROTOCOLS.get(category, [])
        if not state.get("is_valid") or not protocols:
            resolution = _blocked_resolution(
                "No category-specific Downside Up protocol is available for this complaint."
            )
        else:
            resolution = _resolve_with_llm(state, protocols)

    print(f"{_stage_label('RESOLVE')} Result: {resolution['result']}")
    return {
        **state,
        "resolution": resolution,
        "workflow_path": state.get("workflow_path", []) + ["resolve"],
    }


def _customer_satisfaction_with_llm(state: ComplaintState) -> dict[str, str]:
    result = _llm_json(
        "You create demo-only customer satisfaction verification records. Return strict JSON only.",
        f"""
Create a customer satisfaction verification record for this complaint closure step.

Rules:
- This is a demo workflow record, not real customer outreach.
- Do not claim that a real customer was contacted, called, emailed, surveyed, or interviewed.
- Describe the verification as a demo satisfaction attempt/status based on the complaint and resolution state.
- Use pending when the resolution is not applied or when satisfaction cannot be reasonably inferred.
- Keep all fields short and operational.

Complaint closure context:
{json.dumps({
    "complaint": state.get("complaint", ""),
    "complaint_id": state.get("complaint_id", ""),
    "complainant": state.get("complainant", ""),
    "category": state.get("category", ""),
    "status": state.get("status", ""),
    "resolution": state.get("resolution", {}),
}, indent=2)}

Return JSON with exactly these top-level keys:
{{
  "status": "satisfied|unsatisfied|unreachable|pending",
  "summary": "short summary",
  "next_contact_action": "short action"
}}
""",
    )

    required_fields = {"status": str, "summary": str, "next_contact_action": str}
    missing_fields = [field for field in required_fields if field not in result]
    if missing_fields:
        raise ValueError(
            "LLM satisfaction response missing fields: "
            + ", ".join(missing_fields)
        )

    satisfaction_status = str(result["status"]).strip().lower()
    if satisfaction_status not in SATISFACTION_STATUSES:
        raise ValueError(f"Unknown satisfaction status: {result['status']}")

    return {
        "status": satisfaction_status,
        "summary": str(result["summary"]).strip(),
        "next_contact_action": str(result["next_contact_action"]).strip(),
    }


def close_node(state: ComplaintState) -> ComplaintState:
    """Step 5: Close only complaints with confirmed applied resolutions."""
    print(f"\n{_stage_label('CLOSE')} Closing complaint...")

    resolution = state.get("resolution", {})
    resolution_result = str(resolution.get("result", "")).strip().lower()
    resolution_confirmed = resolution_result == "applied"
    closure_result = "closed" if resolution_confirmed else "pending_closure"
    closure_timestamp = datetime.now().isoformat(timespec="seconds")

    follow_up_required = str(resolution.get("effectiveness", "")).strip().lower() == "low"
    follow_up_at = ""
    if follow_up_required:
        follow_up_date = datetime.fromisoformat(closure_timestamp).date() + timedelta(
            days=30
        )
        follow_up_at = follow_up_date.isoformat()

    customer_satisfaction = _customer_satisfaction_with_llm(state)
    resolution_action = str(
        resolution.get("action") or resolution.get("result") or "No resolution recorded."
    ).strip()
    notes = (
        "Resolution application confirmed; complaint is fully closed."
        if resolution_confirmed
        else f"Closure remains pending because resolution result is {resolution_result or 'unknown'}."
    )

    closure = {
        "result": closure_result,
        "resolution_confirmed": resolution_confirmed,
        "customer_satisfaction_attempted": True,
        "customer_satisfaction": customer_satisfaction,
        "closure_log": {
            "category": state.get("category", "other"),
            "resolution": resolution_action,
            "outcome": closure_result,
            "timestamp": closure_timestamp,
        },
        "follow_up_required": follow_up_required,
        "follow_up_at": follow_up_at,
        "notes": notes,
    }

    print(f"{_stage_label('CLOSE')} Result: {closure_result}")
    return {
        **state,
        "closure": closure,
        "workflow_path": state.get("workflow_path", []) + ["close"],
    }


workflow = StateGraph(ComplaintState)
workflow.add_node("intake", intake_node)
workflow.add_node("validate", validate_node)
workflow.add_node("investigate", investigate_node)
workflow.add_node("resolve", resolve_node)
workflow.add_node("close", close_node)

workflow.set_entry_point("intake")
workflow.add_edge("intake", "validate")
workflow.add_edge("validate", "investigate")
workflow.add_edge("investigate", "resolve")
workflow.add_edge("resolve", "close")
workflow.add_edge("close", END)

app = workflow.compile()


def _format_demo_summary(state: ComplaintState) -> str:
    resolution = state.get("resolution", {})
    closure = state.get("closure", {})
    workflow_path = " -> ".join(state.get("workflow_path", []))
    summary_parts = [
        f"id={state.get('complaint_id', 'unknown')}",
        f"status={state.get('status', 'unknown')}",
        f"category={state.get('category', 'unknown')}",
        f"resolution={resolution.get('result', 'unknown')}",
        f"closure={closure.get('result', 'unknown')}",
        f"path={workflow_path}",
    ]
    duplicate_of = state.get("duplicate_of")
    historical_cases = state.get("historical_similar_cases", [])
    if duplicate_of:
        summary_parts.append(f"duplicate_of={duplicate_of}")
    if historical_cases:
        summary_parts.append(f"historical={', '.join(historical_cases)}")
    return "Final summary: " + " | ".join(summary_parts)


def run_demo() -> None:
    test_complaints = [
        # Expected: portal, complete details, validated.
        "I am Nancy Wheeler. Today at 9pm, the Downside Up portal at Hawkins Lab opened three hours late.",
        # Expected: monster, complete details, validated.
        "I am Steve Harrington. Last night at Hawkins High, two demogorgons fought each other instead of chasing students.",
        # Expected: psychic, complete details, validated.
        "I am Mike Wheeler. This morning at Hawkins Quarry, El could move small stones with her mind but could not lift the heavy rocks.",
        # Expected: environmental, complete details, validated.
        "I am Robin Buckley. Today at Melvald Alley, the power lines flickered whenever the red storm clouds moved overhead.",
        # Expected: other, complete details, manual_review_required.
        "I am Lucas Sinclair. Yesterday at Palace Arcade, the vending machine took my quarters and gave me warm grape soda.",
        # Expected: portal category, missing who/when/where, pending_clarification.
        "The Downside Up portal opens at different times each day. How do I predict when?",
        # Expected: monster category, missing who/when/where, pending_clarification.
        "Demogorgons sometimes work together and sometimes fight. What's their deal?",
        # Expected: psychic category, missing reporter/when/where, pending_clarification.
        "El can move things with her mind but can't lift heavy rocks. Why?",
        # Expected: environmental or monster/environmental mix, missing who/when/where, pending_clarification.
        "Why do creatures and power lines react so strangely together?",
        # Expected: other, lacks a real actionable complaint, manual_review_required or validation-style rejection.
        "This is not a valid complaint about something random",
        # Expected: duplicate within 30 days of seeded Dustin Henderson case, validated_duplicate if validation passes.
        "I am Dustin Henderson. Today at the Hawkins Lab loading dock, demogorgon packs gathered again and blocked the door.",
        # Expected: similar to seeded Eleven Hopper case but older than 30 days, historical_similar_cases noted, treated as new.
        "I am Eleven Hopper. Today at Hawkins Quarry, my psychic lift limit for heavy quarry rocks failed again.",
        # Expected: duplicate-like portal issue but missing who/when, pending_clarification takes priority.
        "The portal opens near the Byers house after midnight.",
        # Expected: other with too little detail, pending_clarification.
        "Something weird happened.",
    ]

    print("\nTesting workflow with sample complaints...\n")
    for complaint in test_complaints:
        print(f"\nProcessing complaint: {complaint}")
        result = process_complaint(complaint)
        print(_format_demo_summary(result))


if __name__ == "__main__":
    run_demo()
