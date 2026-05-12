from .state import ComplaintState
from .workflow import process_complaint


TEST_COMPLAINTS = [
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
    print("\nTesting workflow with sample complaints...\n")
    for complaint in TEST_COMPLAINTS:
        print(f"\nProcessing complaint: {complaint}")
        result = process_complaint(complaint)
        print(_format_demo_summary(result))


if __name__ == "__main__":
    run_demo()

