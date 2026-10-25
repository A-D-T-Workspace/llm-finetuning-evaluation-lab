import json
from pathlib import Path

from src.llm_eval.config import load_config
from src.llm_eval.dataset import load_cases
from src.llm_eval.metrics import evaluate_routes, evaluate_safety

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def main():
    config = load_config()
    dataset_path = config["data"]["dataset_path"]
    cases = load_cases(dataset_path)

    predicted_routes = [case["expected_route"] for case in cases]
    predicted_escalations = [
        case["expected_escalation"] for case in cases
    ]

    route_results = evaluate_routes(
        [case["expected_route"] for case in cases],
        predicted_routes,
    )
    safety_results = evaluate_safety(
        [case["expected_escalation"] for case in cases],
        predicted_escalations,
    )

    results = {
        "num_cases": len(cases),
        "route_metrics": route_results,
        "safety_metrics": safety_results,
        "note": "Temporary perfect predictions; not a real model evaluation.",
    }

    output_dir = PROJECT_ROOT / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "evaluation_results.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print(f"Evaluated {len(cases)} cases.")
    print(f"Accuracy: {route_results['accuracy']:.2f}")
    print(f"Macro-F1: {route_results['macro_f1']:.2f}")
    print(f"Safety precision: {safety_results['precision']:.2f}")
    print(f"Safety recall: {safety_results['recall']:.2f}")
    print(f"Results saved to: {output_path}")

if __name__ == "__main__":
    main()
