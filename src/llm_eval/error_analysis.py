import json
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_FILE = PROJECT_ROOT / "reports" / "qwen_baseline_results.json"
OUTPUT_FILE = PROJECT_ROOT / "reports" / "qwen_error_analysis.json"

ROUTES = [
    "account_support",
    "billing_support",
    "account_security",
    "order_support",
    "general_support",
]

def main():
    with RESULTS_FILE.open("r", encoding="utf-8") as file:
        results = json.load(file)

    predictions = results["predictions"]

    confusion_matrix = {
        actual: {
            predicted: 0 for predicted in ROUTES
        }
        for actual in ROUTES
    }

    route_errors = []
    false_positives = []
    false_negatives = []
    route_prediction_counts = Counter()
    escalation_counts = Counter()

    true_positive = 0
    true_negative = 0
    false_positive_count = 0
    false_negative_count = 0

    for item in predictions:
        expected_route = item["expected_route"]
        predicted_route = item["predicted_route"]

        expected_escalation = item["expected_escalation"]
        predicted_escalation = item["predicted_escalation"]

        confusion_matrix[expected_route][predicted_route] += 1
        route_prediction_counts[predicted_route] += 1
        escalation_counts[str(predicted_escalation).lower()] += 1

        if expected_route != predicted_route:
            route_errors.append({
                "id": item["id"],
                "input": item["input"],
                "expected_route": expected_route,
                "predicted_route": predicted_route,
                "raw_response": item.get("raw_response"),
            })

        if expected_escalation and predicted_escalation:
            true_positive += 1
        elif not expected_escalation and not predicted_escalation:
            true_negative += 1
        elif predicted_escalation and not expected_escalation:
            false_positive_count += 1
            false_positives.append({
                "id": item["id"],
                "input": item["input"],
            })
        else:
            false_negative_count += 1
            false_negatives.append({
                "id": item["id"],
                "input": item["input"],
            })

    report = {
        "model": results.get("model", "unknown"),
        "num_cases": len(predictions),
        "route_prediction_counts": dict(route_prediction_counts),
        "confusion_matrix": confusion_matrix,
        "route_error_count": len(route_errors),
        "route_errors": route_errors,
        "escalation_prediction_counts": dict(escalation_counts),
        "escalation_confusion": {
            "true_positive": true_positive,
            "true_negative": true_negative,
            "false_positive": false_positive_count,
            "false_negative": false_negative_count,
        },
        "false_positive_cases": false_positives,
        "false_negative_cases": false_negatives,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("Error analysis completed.")
    print(f"Cases analysed: {len(predictions)}")
    print(f"Routing errors: {len(route_errors)}")
    print(f"Escalation false positives: {false_positive_count}")
    print(f"Escalation false negatives: {false_negative_count}")
    print(f"Report saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
