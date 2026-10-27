import json
from collections import Counter
from pathlib import Path

split_dir = Path("data/splits")

for filename in ["train.json", "validation.json", "test.json"]:
    with (split_dir / filename).open("r", encoding="utf-8") as file:
        cases = json.load(file)

    routes = Counter(case["expected_route"] for case in cases)
    escalation = Counter(case["expected_escalation"] for case in cases)

    print(f"\n{filename} ({len(cases)} cases)")
    print("Routes:", dict(routes))
    print("Escalation:", dict(escalation))
