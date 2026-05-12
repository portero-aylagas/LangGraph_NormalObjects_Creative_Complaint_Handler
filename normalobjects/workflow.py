from datetime import date, datetime, time, timedelta
from typing import Any

from langgraph.graph import END, StateGraph

from .config import (
    ALL_CATEGORIES,
    EFFECTIVENESS_RATINGS,
    RELATIVE_DATE_PATTERN,
    RESOLUTION_RESULTS,
    SATISFACTION_STATUSES,
    STAGE_ICONS,
    TIME_PATTERN,
)
from .data import COMPLAINT_DATABASE, DOWNSIDE_UP_PROTOCOLS
from .llm_client import _llm_json
from .prompts import (
    category_prompt,
    category_validation_prompt,
    customer_satisfaction_prompt,
    essential_details_prompt,
    investigation_prompt,
    issue_signature_prompt,
    metadata_prompt,
    resolution_prompt,
    same_issue_prompt,
)
from .state import ComplaintState


def _stage_label(stage: str) -> str:
    return f"{STAGE_ICONS[stage]} [{stage}]"


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
    result = _llm_json(*same_issue_prompt(state, issue_signature, record))
    return bool(result.get("same_issue"))


def _categorize_complaint(state: ComplaintState) -> tuple[str, str]:
    result = _llm_json(*category_prompt(state))
    category = str(result.get("category", "other")).strip().lower()
    if category not in ALL_CATEGORIES:
        category = "other"
    return category, str(result.get("reasoning", "")).strip()


def _extract_metadata(state: ComplaintState) -> dict[str, str]:
    result = _llm_json(*metadata_prompt(state))
    return {
        "complainant": str(result.get("complainant", "")).strip(),
        "occurred_at": str(result.get("occurred_at", "")).strip(),
        "location": str(result.get("location", "")).strip(),
    }


def _check_essential_details(state: ComplaintState) -> tuple[list[str], str]:
    result = _llm_json(*essential_details_prompt(state))
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
    result = _llm_json(*issue_signature_prompt(state))
    signature = str(result.get("issue_signature", "")).strip().lower()
    if not signature:
        raise ValueError("LLM did not return an issue_signature.")
    return signature


def _validate_category_rule(state: ComplaintState) -> tuple[bool, str]:
    category = state.get("category", "other")
    if category == "other":
        return False, "No automated category-specific validation exists for other."

    result = _llm_json(*category_validation_prompt(state))
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

    complaint_id = (
        state.get("complaint_id") or f"NO-DEMO-{len(COMPLAINT_DATABASE) + 1:03d}"
    )
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
    result = _llm_json(*investigation_prompt(state))
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
    result = _llm_json(*resolution_prompt(state, protocols))
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
    result = _llm_json(*customer_satisfaction_prompt(state))
    required_fields = {"status": str, "summary": str, "next_contact_action": str}
    missing_fields = [field for field in required_fields if field not in result]
    if missing_fields:
        raise ValueError(
            "LLM satisfaction response missing fields: " + ", ".join(missing_fields)
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


def process_complaint(complaint: str) -> ComplaintState:
    """Run the graph for the lab's plain-string complaint input format."""
    return app.invoke({"complaint": complaint})

