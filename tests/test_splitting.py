import pytest

from src.llm_eval.splitting import split_dataset
from src.llm_eval.dataset import load_cases


def test_split_fractions_must_sum_to_one():
    cases = load_cases("evaluation_cases.json")

    with pytest.raises(ValueError, match="sum to 1"):
        split_dataset(
            cases,
            train_fraction=0.80,
            validation_fraction=0.15,
            test_fraction=0.10,
        )


def test_tiny_dataset_reports_stratification_problem():
    cases = load_cases("evaluation_cases.json")

    with pytest.raises(ValueError, match="Cannot create stratified"):
        split_dataset(cases)

def test_split_ids_do_not_overlap():
    cases = [
        {
            "id": f"case_{i}",
            "input": f"Example message {i}",
            "expected_route": route,
            "expected_escalation": False,
        }
        for route in ["account_support", "billing_support"]
        for i in range(20 if route == "account_support" else 20, 40)
    ]

    splits = split_dataset(
        cases,
        train_fraction=0.60,
        validation_fraction=0.20,
        test_fraction=0.20,
        seed=42,
    )

    train_ids = {case["id"] for case in splits["train"]}
    validation_ids = {case["id"] for case in splits["validation"]}
    test_ids = {case["id"] for case in splits["test"]}

    assert train_ids.isdisjoint(validation_ids)
    assert train_ids.isdisjoint(test_ids)
    assert validation_ids.isdisjoint(test_ids)

def test_split_ids_do_not_overlap():
    cases = [
        {
            "id": f"{route}_{i}",
            "input": f"Example message {i}",
            "expected_route": route,
            "expected_escalation": False,
        }
        for route in ["account_support", "billing_support"]
        for i in range(20)
    ]

    splits = split_dataset(
        cases,
        train_fraction=0.60,
        validation_fraction=0.20,
        test_fraction=0.20,
        seed=42,
    )

    train_ids = {case["id"] for case in splits["train"]}
    validation_ids = {case["id"] for case in splits["validation"]}
    test_ids = {case["id"] for case in splits["test"]}

    assert train_ids.isdisjoint(validation_ids)
    assert train_ids.isdisjoint(test_ids)
    assert validation_ids.isdisjoint(test_ids)

def test_split_preserves_all_records():
    cases = [
        {
            "id": f"{route}_{i}",
            "input": f"Example message {i}",
            "expected_route": route,
            "expected_escalation": False,
        }
        for route in ["account_support", "billing_support"]
        for i in range(20)
    ]

    splits = split_dataset(
        cases,
        train_fraction=0.60,
        validation_fraction=0.20,
        test_fraction=0.20,
        seed=42,
    )

    original_ids = {case["id"] for case in cases}
    split_ids = {
        case["id"]
        for split in splits.values()
        for case in split
    }

    assert split_ids == original_ids
    assert sum(len(split) for split in splits.values()) == len(cases)
