import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.llm_eval.llm_baseline import predict
from src.llm_eval.metrics import evaluate_routes, evaluate_safety

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = PROJECT_ROOT / "qwen-routing-lora"
TEST_PATH = PROJECT_ROOT / "data" / "splits" / "test.json"
REPORT_PATH = PROJECT_ROOT / "reports" / "frozen_test_comparison.json"

def load_test_cases():
    with TEST_PATH.open("r", encoding="utf-8") as file:
        cases = json.load(file)

    if not isinstance(cases, list) or not cases:
        raise ValueError("The test split must contain a non-empty list.")

    return cases

def evaluate_model(cases, tokenizer, model):
    predictions = []

    for case in cases:
        route, escalation, response, valid = predict(
            case["input"], tokenizer, model
        )

        predictions.append({
            "id": case["id"],
            "expected_route": case["expected_route"],
            "predicted_route": route,
            "expected_escalation": case["expected_escalation"],
            "predicted_escalation": escalation,
            "response_valid": valid,
            "raw_response": response,
            "exact_match": (
                route == case["expected_route"]
                and escalation == case["expected_escalation"]
            ),
        })

    expected_routes = [case["expected_route"] for case in cases]
    predicted_routes = [item["predicted_route"] for item in predictions]
    expected_safety = [case["expected_escalation"] for case in cases]
    predicted_safety = [item["predicted_escalation"] for item in predictions]

    return {
        "test_size": len(cases),
        "route_metrics": evaluate_routes(expected_routes, predicted_routes),
        "safety_metrics": evaluate_safety(expected_safety, predicted_safety),
        "exact_match_accuracy": sum(
            item["exact_match"] for item in predictions
        ) / len(predictions),
        "invalid_response_count": sum(
            not item["response_valid"] for item in predictions
        ),
        "predictions": predictions,
    }

def main():
    cases = load_test_cases()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    results = {
        "dataset": "data/splits/test.json",
        "test_ids": [case["id"] for case in cases],
        "test_size": len(cases),
        "base_model": MODEL_NAME,
        "adapter_path": str(ADAPTER_PATH),
        "note": (
            "Exploratory evaluation on a small frozen test split. "
            "Expanded dataset labels have not been independently adjudicated. "
            "Do not use these test results for model selection."
        ),
        "models": {},
    }

    print("Evaluating base model...")
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    base_model.eval()
    results["models"]["base"] = evaluate_model(
        cases, tokenizer, base_model
    )
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Evaluating LoRA adapter...")
    base_for_adapter = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    adapter_model = PeftModel.from_pretrained(
        base_for_adapter, str(ADAPTER_PATH)
    )
    adapter_model.eval()
    results["models"]["lora_adapter"] = evaluate_model(
        cases, tokenizer, adapter_model
    )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print("\nEvaluation complete.")
    print(f"Report saved to: {REPORT_PATH}")
    for name, metrics in results["models"].items():
        print(f"\n{name}")
        print(f"  Route accuracy: {metrics['route_metrics']['accuracy']:.3f}")
        print(f"  Macro F1: {metrics['route_metrics']['macro_f1']:.3f}")
        print(
            "  Escalation precision: "
            f"{metrics['safety_metrics']['precision']:.3f}"
        )
        print(
            "  Escalation recall: "
            f"{metrics['safety_metrics']['recall']:.3f}"
        )
        print(f"  Exact-match accuracy: {metrics['exact_match_accuracy']:.3f}")
        print(f"  Invalid responses: {metrics['invalid_response_count']}")

if __name__ == "__main__":
    main()
