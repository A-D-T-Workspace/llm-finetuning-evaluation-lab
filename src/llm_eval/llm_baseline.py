import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.llm_eval.config import load_config
from src.llm_eval.dataset import load_cases
from src.llm_eval.metrics import evaluate_routes, evaluate_safety

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def predict(text, tokenizer, model):
    messages = [
        {
            "role": "system",
            "content": (
                "You classify customer support messages. Return only a valid JSON "
                "object with keys 'route' and 'escalation'. "
                "Choose exactly one route from these five options. "
                "ROUTING RULES: "
                "account_support: routine password resets, login problems, locked "
                "accounts, and account setting changes, unless there is evidence "
                "of unauthorized access. "
                "billing_support: duplicate charges, declined payments, refunds, "
                "receipts, and transactions the user disputes. "
                "account_security: unauthorized account access, compromised "
                "credentials, phishing, impersonation, and suspected theft of "
                "sensitive information. "
                "order_support: order status, tracking, delivery, and arrival issues. "
                "general_support: product questions and general enquiries that do "
                "not fit another category. "
                "ESCALATION RULES: "
                "ESCALATION RULES: "
                "Set escalation to true for clear security threats, including stolen "
                "payment details, unauthorized transactions, unexpected password-reset "
                "emails, account credentials changed by someone else, phishing, "
                "impersonation, or suspected fraud. "
                "An unexpected password-reset email must be escalated even if the user "
                "can still access their account. A person impersonating the company "
                "and requesting bank details must always be escalated. "
                "Do not escalate routine forgotten passwords, ordinary login problems, "
                "or an account locked after repeated login attempts unless there is "
                "additional evidence of unauthorized access or another security threat. "
                "Normal billing questions, delivery questions, and general product "
                "questions should not be escalated unless they indicate a security threat. "
                "The escalation value must be the JSON boolean true or false, not a string."
            ),
        },
        {"role": "user", "content": text},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)

    valid_routes = {
        "account_support",
        "billing_support",
        "account_security",
        "order_support",
        "general_support",
    }

    response_valid = True

    try:
        result = json.loads(response)

        if not isinstance(result, dict):
            raise ValueError("Response must be a JSON object.")

        route = result.get("route")
        escalation = result.get("escalation")

        if route not in valid_routes:
            response_valid = False
            route = "general_support"

        if not isinstance(escalation, bool):
            response_valid = False
            escalation = False

    except (json.JSONDecodeError, AttributeError, ValueError):
        response_valid = False
        route = "general_support"
        escalation = False

    return route, escalation, response, response_valid

def main():
    config = load_config()
    cases = load_cases(config["data"]["dataset_path"])
    model_name = config["training"]["base_model"]

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()

    predictions = []

    for index, case in enumerate(cases, start=1):
        route, escalation, raw_response, response_valid = predict(
            case["input"], tokenizer, model
        )

        predictions.append({
            "id": case["id"],
            "input": case["input"],
            "expected_route": case["expected_route"],
            "predicted_route": route,
            "expected_escalation": case["expected_escalation"],
            "predicted_escalation": escalation,
            "raw_response": raw_response,
            "response_valid": response_valid,
        })

        print(f"Processed {index}/{len(cases)}: {case['id']}")

    invalid_response_count = sum(
        not item["response_valid"] for item in predictions
    )

    route_metrics = evaluate_routes(
        [item["expected_route"] for item in predictions],
        [item["predicted_route"] for item in predictions],
    )
    safety_metrics = evaluate_safety(
        [item["expected_escalation"] for item in predictions],
        [item["predicted_escalation"] for item in predictions],
    )

    results = {
        "model": model_name,
        "num_cases": len(predictions),
        "invalid_response_count": invalid_response_count,
        "route_metrics": route_metrics,
        "safety_metrics": safety_metrics,
        "predictions": predictions,
        "note": "Exploratory evaluation on 20 handcrafted examples. Results are preliminary and should not be treated as production performance.",
    }

    output_dir = PROJECT_ROOT / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "qwen_baseline_results.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print("\nEvaluation results")
    print(f"Invalid responses: {invalid_response_count}")
    print(f"Route accuracy: {route_metrics['accuracy']:.2f}")
    print(f"Route macro-F1: {route_metrics['macro_f1']:.2f}")
    print(f"Safety precision: {safety_metrics['precision']:.2f}")
    print(f"Safety recall: {safety_metrics['recall']:.2f}")
    print(f"Saved results to: {output_path}")

if __name__ == "__main__":
    main()
