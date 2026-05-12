import json

from .state import ComplaintState


def same_issue_prompt(
    state: ComplaintState, issue_signature: str, record: dict[str, str]
) -> tuple[str, str]:
    return (
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


def category_prompt(state: ComplaintState) -> tuple[str, str]:
    return (
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


def metadata_prompt(state: ComplaintState) -> tuple[str, str]:
    return (
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


def essential_details_prompt(state: ComplaintState) -> tuple[str, str]:
    return (
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


def issue_signature_prompt(state: ComplaintState) -> tuple[str, str]:
    return (
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


def category_validation_prompt(state: ComplaintState) -> tuple[str, str]:
    return (
        "You validate complaint categories for an intake workflow. Return strict JSON only.",
        f"""
Validate whether this complete complaint is a real supported-category complaint.

Category being validated: {state.get("category", "other")}

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


def investigation_prompt(state: ComplaintState) -> tuple[str, str]:
    status = state.get("status", "")
    return (
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


def resolution_prompt(
    state: ComplaintState, protocols: list[dict[str, str]]
) -> tuple[str, str]:
    return (
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


def customer_satisfaction_prompt(state: ComplaintState) -> tuple[str, str]:
    return (
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

