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

