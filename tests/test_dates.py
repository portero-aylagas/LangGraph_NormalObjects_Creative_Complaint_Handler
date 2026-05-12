from datetime import date

import pytest

from normalobjects.llm_client import _strip_code_fence
from normalobjects.workflow import _normalise_relative_datetime, _parse_time_from_phrase


def test_strip_code_fence_removes_json_fence():
    content = '```json\n{"result": "ok"}\n```'

    assert _strip_code_fence(content) == '{"result": "ok"}'


@pytest.mark.parametrize(
    ("phrase", "expected"),
    [
        ("Today at 9pm", "2026-05-12T21:00"),
        ("yesterday at 6:30 am", "2026-05-11T06:30"),
        ("last night", "2026-05-11"),
        ("tomorrow", "2026-05-13"),
        ("2026-05-01T14:45:30", "2026-05-01T14:45"),
    ],
)
def test_normalise_relative_datetime(phrase, expected):
    assert _normalise_relative_datetime(phrase, date(2026, 5, 12)) == expected


@pytest.mark.parametrize(
    ("phrase", "expected"),
    [
        ("at 9pm", "21:00:00"),
        ("6:30 am", "06:30:00"),
        ("12am", "00:00:00"),
        ("12pm", "12:00:00"),
    ],
)
def test_parse_time_from_phrase(phrase, expected):
    assert _parse_time_from_phrase(phrase).isoformat() == expected

