from datetime import date

import pytest

from normalobjects import demo, workflow
from normalobjects.data import DOWNSIDE_UP_PROTOCOLS


def test_validate_resolution_response_escalates_high_risk_monster_case():
    protocols = DOWNSIDE_UP_PROTOCOLS["monster"]
    state = {
        "category": "monster",
        "investigation": {"risk_level": "high"},
    }
    llm_result = {
        "result": "applied",
        "protocol_id": "DU-MONSTER-01",
        "protocol_name": "Creature Behavior Containment",
        "action": "Apply the containment protocol.",
        "effectiveness": "high",
        "escalation_required": False,
        "specialized_team": "",
        "rationale": "High-risk monster behavior needs containment.",
        "notes": "Escalate because risk is high.",
    }

    result = workflow._validate_resolution_response(llm_result, state, protocols)

    assert result["result"] == "escalated"
    assert result["escalation_required"] is True
    assert result["specialized_team"] == "Monster Response Unit"


def test_validate_resolution_response_rejects_unknown_protocol():
    protocols = DOWNSIDE_UP_PROTOCOLS["portal"]
    state = {
        "category": "portal",
        "investigation": {"risk_level": "low"},
    }
    llm_result = {
        "result": "applied",
        "protocol_id": "DU-UNKNOWN",
        "protocol_name": "Unknown",
        "action": "Apply unknown protocol.",
        "effectiveness": "high",
        "escalation_required": False,
        "specialized_team": "",
        "rationale": "Invalid protocol.",
        "notes": "Invalid protocol.",
    }

    with pytest.raises(ValueError, match="unknown protocol_id"):
        workflow._validate_resolution_response(llm_result, state, protocols)


def test_format_demo_summary_includes_key_workflow_fields():
    state = {
        "complaint_id": "NO-DEMO-005",
        "status": "validated_duplicate",
        "category": "monster",
        "duplicate_of": "NO-2026-0424-002",
        "historical_similar_cases": ["NO-2026-0331-003"],
        "resolution": {"result": "linked_case"},
        "closure": {"result": "pending_closure"},
        "workflow_path": ["intake", "validate", "investigate", "resolve", "close"],
    }

    summary = demo._format_demo_summary(state)

    assert "id=NO-DEMO-005" in summary
    assert "status=validated_duplicate" in summary
    assert "resolution=linked_case" in summary
    assert "closure=pending_closure" in summary
    assert "duplicate_of=NO-2026-0424-002" in summary
    assert "historical=NO-2026-0331-003" in summary
    assert "path=intake -> validate -> investigate -> resolve -> close" in summary


def test_process_complaint_runs_full_graph_with_mocked_llm(monkeypatch):
    original_database = list(workflow.COMPLAINT_DATABASE)
    monkeypatch.setattr(workflow, "COMPLAINT_DATABASE", list(original_database))
    monkeypatch.setattr(workflow, "_today", lambda: date(2026, 5, 12))

    def fake_llm_json(system_prompt, user_prompt):
        if "classify Downside Up complaint intake items" in system_prompt:
            return {
                "category": "portal",
                "reasoning": "The complaint is about portal timing.",
            }
        if "extract explicit complaint metadata" in system_prompt:
            return {
                "complainant": "Nancy Wheeler",
                "occurred_at": "Today at 9pm",
                "location": "Hawkins Lab",
            }
        if "check complaint intake completeness" in system_prompt:
            return {
                "present": {"who": True, "what": True, "when": True, "where": True},
                "missing_details": [],
                "notes": "All essential details are present.",
            }
        if "create stable duplicate-detection signatures" in system_prompt:
            return {"issue_signature": "downside up portal opened late at hawkins lab"}
        if "compare complaint records for duplicate detection" in system_prompt:
            return {"same_issue": False, "reasoning": "Different issue."}
        if "validate complaint categories" in system_prompt:
            return {"passed": True, "notes": "Portal timing is supported."}
        if "investigation step" in system_prompt:
            return {
                "result": "evidence_collected",
                "summary": "Portal timing delay was documented.",
                "findings": ["The portal opened later than expected."],
                "evidence_to_check": ["Portal timing logs."],
                "risk_level": "medium",
                "recommended_next_action": "Apply portal timing stabilization.",
                "notes": "Demo investigation complete.",
            }
        if "resolution step" in system_prompt:
            return {
                "result": "applied",
                "protocol_id": "DU-PORTAL-01",
                "protocol_name": "Portal Timing Stabilization",
                "action": "Apply DU-PORTAL-01 to stabilize portal timing.",
                "effectiveness": "high",
                "escalation_required": False,
                "specialized_team": "",
                "rationale": "The complaint is a portal timing issue.",
                "notes": "Protocol applied.",
            }
        if "customer satisfaction verification" in system_prompt:
            return {
                "status": "pending",
                "summary": "Resolution was applied; satisfaction is pending.",
                "next_contact_action": "Monitor for updates.",
            }
        raise AssertionError(f"Unexpected prompt: {system_prompt}")

    monkeypatch.setattr(workflow, "_llm_json", fake_llm_json)

    result = workflow.process_complaint(
        "I am Nancy Wheeler. Today at 9pm, the Downside Up portal at Hawkins Lab opened three hours late."
    )

    assert result["workflow_path"] == [
        "intake",
        "validate",
        "investigate",
        "resolve",
        "close",
    ]
    assert result["category"] == "portal"
    assert result["status"] == "validated"
    assert result["is_valid"] is True
    assert result["investigation"]["result"] == "evidence_collected"
    assert result["resolution"]["result"] == "applied"
    assert result["closure"]["result"] == "closed"
    assert result["occurred_at"] == "2026-05-12T21:00"

