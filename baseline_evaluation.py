import json

with open("evaluation_cases.json", "r", encoding="utf-8") as file:
    cases = json.load(file)

correct = 0

for case in cases:
    prediction = "billing_support"

    if prediction == case["expected_route"]:
        correct += 1

accuracy = correct / len(cases) * 100

print("Baseline: Always predict billing_support")
print("Correct predictions:", correct)
print("Total cases:", len(cases))
print(f"Routing accuracy: {accuracy:.1f}%")
