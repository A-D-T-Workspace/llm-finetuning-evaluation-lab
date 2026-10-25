import json
from collections import Counter

with open("evaluation_cases.json", "r", encoding="utf-8") as file:
    cases = json.load(file)

required_fields = {
    "id",
    "input",
    "expected_route",
    "expected_escalation"
}

for case in cases:
    if not required_fields.issubset(case):
        raise ValueError(f"Missing fields in {case.get('id', 'unknown case')}")

ids = [case["id"] for case in cases]

if len(ids) != len(set(ids)):
    raise ValueError("Duplicate case IDs found!")

routes = Counter(case["expected_route"] for case in cases)

print("Dataset validation passed!")
print("Total cases:", len(cases))
print("Unique case IDs:", len(set(ids)))
print("Route distribution:")

for route, count in routes.items():
    print(f"  {route}: {count}")
