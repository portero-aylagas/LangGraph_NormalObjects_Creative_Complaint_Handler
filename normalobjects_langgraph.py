import json
import os
from datetime import date, datetime
from typing import Any, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph


load_dotenv()

SUPPORTED_CATEGORIES = {"portal", "monster", "psychic", "environmental"}
ALL_CATEGORIES = SUPPORTED_CATEGORIES | {"other"}
TODAY = date(2026, 5, 12)
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


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


def _llm() -> ChatOpenAI:
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError(
            "OPENAI_API_KEY not found. Add it to your environment or .env file."
        )
    return ChatOpenAI(model=MODEL_NAME, temperature=0)


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


def _state_date(state: ComplaintState) -> date:
    return (
        _parse_date(state.get("submitted_at"))
        or _parse_date(state.get("occurred_at"))
        or TODAY
    )


def _normalise_for_matching(value: str) -> str:
    characters = [char.lower() if char.isalnum() else " " for char in value]
    return " ".join("".join(characters).split())


def _similar_signature(left: str, right: str) -> bool:
    left_words = set(_normalise_for_matching(left).split())
    right_words = set(_normalise_for_matching(right).split())
    if not left_words or not right_words:
        return False
    overlap = len(left_words & right_words)
    smaller_size = min(len(left_words), len(right_words))
    return overlap / smaller_size >= 0.7


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
- If the complaint only gives a relative date such as today, yesterday, or last night, keep that phrase.
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
        record_date = _parse_date(record.get("submitted_at"))
        if not record_date:
            continue

        age_days = abs((submitted_at - record_date).days)
        exact_signature = _normalise_for_matching(issue_signature) == _normalise_for_matching(
            record["issue_signature"]
        )
        similar_signature = exact_signature or _similar_signature(
            issue_signature, record["issue_signature"]
        )

        if exact_signature and same_complainant and age_days <= 30:
            return record, historical
        if similar_signature and age_days > 30:
            historical.append(record["complaint_id"])

    return None, historical


def _append_demo_record(state: ComplaintState) -> None:
    COMPLAINT_DATABASE.append(
        {
            "complaint_id": state["complaint_id"],
            "complainant": state.get("complainant", "Unknown"),
            "submitted_at": state.get("submitted_at", TODAY.isoformat()),
            "category": state["category"],
            "issue_signature": state["issue_signature"],
        }
    )


def intake_node(state: ComplaintState) -> ComplaintState:
    """Step 1: LLM intake, category assignment, completeness, and related cases."""
    print("\n[INTAKE] Processing complaint...")

    complaint_id = state.get("complaint_id") or f"NO-DEMO-{len(COMPLAINT_DATABASE) + 1:03d}"
    submitted_at = state.get("submitted_at")

    category, category_reasoning = _categorize_complaint(state)
    extracted_metadata = _extract_metadata(state)
    intake_state: ComplaintState = {
        **state,
        "complaint_id": complaint_id,
        "complainant": state.get("complainant") or extracted_metadata["complainant"],
        "occurred_at": state.get("occurred_at") or extracted_metadata["occurred_at"],
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
        "submitted_at": submitted_at or TODAY.isoformat(),
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

    print(f"[INTAKE] Categorized as: {category}")
    if missing_details:
        print(f"[INTAKE] Missing details: {', '.join(missing_details)}")
    if duplicate:
        print(f"[INTAKE] Duplicate of: {duplicate['complaint_id']}")
    if historical:
        print(f"[INTAKE] Historical similar cases: {', '.join(historical)}")
    return new_state


def process_complaint(complaint: str) -> ComplaintState:
    """Run the graph for the lab's plain-string complaint input format."""
    return app.invoke({"complaint": complaint})


def validate_node(state: ComplaintState) -> ComplaintState:
    """Step 2: Apply validation priority order with LLM category validation."""
    print("\n[VALIDATE] Validating complaint...")

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
    print(f"[VALIDATE] Status: {status}")
    return new_state


def investigate_node(state: ComplaintState) -> ComplaintState:
    """Step 3: Record an investigation result for every validation outcome."""
    print("\n[INVESTIGATE] Investigating complaint...")

    status = state.get("status", "")
    category = state.get("category", "other")

    if status == "pending_clarification":
        investigation = {
            "result": "blocked",
            "notes": f"Investigation blocked until these details are supplied: {', '.join(state.get('missing_details', []))}.",
        }
    elif status == "manual_review_required":
        investigation = {
            "result": "manual_review",
            "notes": "Automated investigation is unavailable for other complaints.",
        }
    elif status == "validation_failed":
        investigation = {
            "result": "blocked",
            "notes": "Investigation blocked because the complaint failed category validation.",
        }
    elif status == "validated_duplicate":
        investigation = {
            "result": "linked_case",
            "notes": f"Use evidence and resolution history from linked case {state.get('duplicate_of')}.",
        }
    else:
        evidence_notes = {
            "portal": "Check portal timing logs, location drift records, and recent gate activity.",
            "monster": "Review creature sighting reports, pack behavior notes, and containment logs.",
            "psychic": "Compare claimed ability limits with prior psychic exertion and recovery records.",
            "environmental": "Inspect power anomalies, weather readings, and affected physical infrastructure.",
        }
        investigation = {
            "result": "evidence_collected",
            "notes": evidence_notes.get(category, "No category-specific evidence path is available."),
        }

    new_state: ComplaintState = {
        **state,
        "investigation": investigation,
        "workflow_path": state.get("workflow_path", []) + ["investigate"],
    }
    print(f"[INVESTIGATE] Result: {investigation['result']}")
    return new_state


def resolve_node(state: ComplaintState) -> ComplaintState:
    """Step 4: Placeholder resolution node."""
    print("\n[RESOLVE] Resolution placeholder...")
    return {
        **state,
        "resolution": {
            "result": "not_implemented",
            "notes": "Resolution logic will be added in a later pass.",
        },
        "workflow_path": state.get("workflow_path", []) + ["resolve"],
    }


def close_node(state: ComplaintState) -> ComplaintState:
    """Step 5: Placeholder closure node."""
    print("\n[CLOSE] Closure placeholder...")
    return {
        **state,
        "closure": {
            "result": "closed_for_demo",
            "notes": "Complaint reached the end of the demo workflow.",
        },
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
        print(f"Final state: {result}")


if __name__ == "__main__":
    run_demo()
