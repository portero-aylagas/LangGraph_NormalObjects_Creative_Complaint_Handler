import pytest

from normalobjects import workflow


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (True, True),
        (False, False),
        ("true", True),
        ("yes", True),
        ("1", True),
        ("false", False),
        ("no", False),
        ("0", False),
        ("", False),
    ],
)
def test_bool_from_llm(value, expected):
    assert workflow._bool_from_llm(value) is expected


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ({}, False),
        ({"investigation": {"result": "blocked", "summary": "Blocked."}}, False),
        (
            {
                "investigation": {
                    "result": "evidence_collected",
                    "summary": "Documented issue.",
                }
            },
            True,
        ),
        (
            {
                "investigation": {
                    "result": "evidence_collected",
                    "findings": ["Observed pattern."],
                }
            },
            True,
        ),
    ],
)
def test_has_documented_investigation(state, expected):
    assert workflow._has_documented_investigation(state) is expected


def test_validate_category_rule_rejects_other_without_llm():
    passed, notes = workflow._validate_category_rule({"category": "other"})

    assert passed is False
    assert "No automated category-specific validation" in notes

