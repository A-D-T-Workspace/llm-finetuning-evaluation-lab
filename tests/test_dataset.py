import json

import pytest

from src.llm_eval.dataset import load_cases, PROJECT_ROOT


def test_dataset_loads_valid_records():
    cases = load_cases("evaluation_cases.json")

    assert len(cases) == 20
    assert cases[0]["id"] == "case_001"


def test_dataset_labels_have_correct_types():
    cases = load_cases("evaluation_cases.json")

    for case in cases:
        assert isinstance(case["expected_route"], str)
        assert isinstance(case["expected_escalation"], bool)


def test_duplicate_ids_are_rejected(tmp_path):
    cases = [
        {
            "id": "duplicate",
            "input": "First message",
            "expected_route": "account_support",
            "expected_escalation": False,
        },
        {
            "id": "duplicate",
            "input": "Second message",
            "expected_route": "billing_support",
            "expected_escalation": False,
        },
    ]

    dataset_file = PROJECT_ROOT / "data" / "processed" / "invalid_test_data.json"
    dataset_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        dataset_file.write_text(json.dumps(cases), encoding="utf-8")

        with pytest.raises(ValueError, match="Duplicate record ID"):
            load_cases("data/processed/invalid_test_data.json")
    finally:
        dataset_file.unlink(missing_ok=True)

def test_empty_route_is_rejected():
    from src.llm_eval.dataset import load_cases, PROJECT_ROOT
    import json
    import pytest

    dataset_file = PROJECT_ROOT / "data" / "processed" / "invalid_route_test.json"
    dataset_file.parent.mkdir(parents=True, exist_ok=True)

    invalid_case = [
        {
            "id": "case_invalid",
            "input": "I need help with my account.",
            "expected_route": "   ",
            "expected_escalation": False,
        }
    ]

    try:
        dataset_file.write_text(json.dumps(invalid_case), encoding="utf-8")

        with pytest.raises(ValueError, match="invalid route"):
            load_cases("data/processed/invalid_route_test.json")
    finally:
        dataset_file.unlink(missing_ok=True)

