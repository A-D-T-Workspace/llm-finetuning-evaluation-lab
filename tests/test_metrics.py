import pytest

from src.llm_eval.metrics import evaluate_routes, evaluate_safety

def test_route_accuracy_and_macro_f1():
    expected = ["billing", "billing", "account"]
    predicted = ["billing", "account", "account"]

    results = evaluate_routes(expected, predicted)

    assert results["accuracy"] == pytest.approx(2 / 3)
    assert results["macro_f1"] == pytest.approx(2 / 3)
    assert "billing" in results["per_class"]
    assert "account" in results["per_class"]

def test_safety_precision_and_recall():
    expected = [True, True, False, False]
    predicted = [True, False, True, False]

    results = evaluate_safety(expected, predicted)

    assert results["precision"] == pytest.approx(0.5)
    assert results["recall"] == pytest.approx(0.5)

def test_route_evaluation_rejects_empty_inputs():
    with pytest.raises(ValueError, match="equal, non-zero"):
        evaluate_routes([], [])

def test_safety_evaluation_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="equal"):
        evaluate_safety([True, False], [True])
