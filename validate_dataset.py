import json
import sys
from collections import Counter
from pathlib import Path

VALID_ROUTES = {
    "account_support",
    "billing_support",
    "account_security",
    "order_support",
    "general_support",
}

REQUIRED_FIELDS = {
    "id",
    "input",
    "expected_route",
    "expected_escalation",
}

def validate_dataset(file_path):
    with Path(file_path).open("r", encoding="utf-8") as file:
        cases = json.load(file)

    if not isinstance(cases, list) or not cases:
        raise ValueError("Dataset must be a non-empty JSON list.")

    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("Every case must be a JSON object.")

        if not REQUIRED_FIELDS.issubset(case):
            raise ValueError(
                f"Missing fields in {case.get('id', 'unknown case')}"
            )

        if not isinstance(case["id"], str) or not case["id"].strip():
            raise ValueError("Case IDs must be non-empty strings.")

        if not isinstance(case["input"], str) or not case["input"].strip():
            raise ValueError(f"Empty input in {case['id']}")

        if case["expected_route"] not in VALID_ROUTES:
            raise ValueError(f"Invalid route in {case['id']}")

        if type(case["expected_escalation"]) is not bool:
            raise ValueError(f"Invalid escalation label in {case['id']}")

    ids = [case["id"] for case in cases]

    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate case IDs found!")

    routes = Counter(case["expected_route"] for case in cases)
    escalation = Counter(
        case["expected_escalation"] for case in cases
    )

    print(f"Validation passed: {file_path}")
    print("Total cases:", len(cases))
    print("Unique case IDs:", len(set(ids)))
    print("Route distribution:", dict(routes))
    print("Escalation distribution:", dict(escalation))

if __name__ == "__main__":
    dataset_path = (
        sys.argv[1] if len(sys.argv) > 1 else "evaluation_cases.json"
    )
    validate_dataset(dataset_path)
