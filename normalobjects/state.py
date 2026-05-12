from typing import Any, TypedDict


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

