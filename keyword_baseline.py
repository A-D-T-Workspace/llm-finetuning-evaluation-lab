import json

def predict_route(text):
    text = text.lower()

    if any(word in text for word in ["unauthorized", "stolen", "accessed"]):
        return "account_security"
    if any(word in text for word in ["charged", "payment", "billing"]):
        return "billing_support"
    if any(word in text for word in ["order", "arrive", "delivery"]):
        return "order_support"
    if any(word in text for word in ["password", "login", "account"]):
        return "account_support"

    return "unknown"

with open("evaluation_cases.json", "r", encoding="utf-8") as file:
    cases = json.load(file)

correct = 0

for case in cases:
    prediction = predict_route(case["input"])
    if prediction == case["expected_route"]:
        correct += 1

accuracy = correct / len(cases) * 100

print("Keyword baseline")
print("Correct predictions:", correct)
print("Total cases:", len(cases))
print(f"Routing accuracy: {accuracy:.1f}%")
