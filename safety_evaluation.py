import json

def predict_escalation(text):
    text = text.lower()
    risk_words = ["unauthorized", "stolen", "without permission"]
    return any(word in text for word in risk_words)

with open("evaluation_cases.json", "r", encoding="utf-8") as file:
    cases = json.load(file)

tp = fp = fn = tn = 0

for case in cases:
    actual = case["expected_escalation"]
    predicted = predict_escalation(case["input"])

    if actual and predicted:
        tp += 1
    elif not actual and predicted:
        fp += 1
    elif actual and not predicted:
        fn += 1
    else:
        tn += 1

precision = tp / (tp + fp) if tp + fp else 0
recall = tp / (tp + fn) if tp + fn else 0

print("Safety evaluation")
print("True positives:", tp)
print("False positives:", fp)
print("False negatives:", fn)
print("True negatives:", tn)
print(f"Precision: {precision:.1%}")
print(f"Recall: {recall:.1%}")
