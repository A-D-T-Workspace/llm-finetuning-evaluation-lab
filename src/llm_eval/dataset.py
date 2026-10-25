import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def load_cases(dataset_path: str) -> list[dict]:
    path = PROJECT_ROOT / dataset_path

    with path.open("r", encoding="utf-8") as file:
        cases = json.load(file)

    if not isinstance(cases, list) or not cases:
        raise ValueError("Dataset must be a non-empty JSON list.")

    required_fields = {
        "id",
        "input",
        "expected_route",
        "expected_escalation",
    }

    seen_ids = set()

    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("Every dataset record must be a JSON object.")

        missing = required_fields - case.keys()
        if missing:
            raise ValueError(
                f"Record is missing required fields: {sorted(missing)}"
            )

        if not isinstance(case["id"], str) or not case["id"].strip():
            raise ValueError("Every record must have a non-empty string ID.")

        if case["id"] in seen_ids:
            raise ValueError(f"Duplicate record ID: {case['id']}")

        if not isinstance(case["input"], str) or not case["input"].strip():
            raise ValueError(f"Record {case['id']} has empty input.")

        if (
            not isinstance(case["expected_route"], str)
            or not case["expected_route"].strip()
        ):
            raise ValueError(f"Record {case['id']} has an invalid route.")

        if not isinstance(case["expected_escalation"], bool):
            raise ValueError(
                f"Record {case['id']} has an invalid escalation label."
            )

        seen_ids.add(case["id"])

    return cases

