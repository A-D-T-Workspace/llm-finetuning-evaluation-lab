import json
from pathlib import Path

from sklearn.model_selection import train_test_split

SEED = 42
DATA_PATH = Path("evaluation_cases_v3_frozen.json")
OUTPUT_DIR = Path("data/splits")

def main():
    with DATA_PATH.open("r", encoding="utf-8") as file:
        cases = json.load(file)

    train_val, test = train_test_split(
        cases,
        test_size=0.15,
        random_state=SEED,
        stratify=[case["expected_route"] for case in cases],
    )

    train, validation = train_test_split(
        train_val,
        test_size=6 / 34,
        random_state=SEED,
        stratify=[case["expected_route"] for case in train_val],
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    splits = {
        "train.json": train,
        "validation.json": validation,
        "test.json": test,
    }

    for filename, records in splits.items():
        output_path = OUTPUT_DIR / filename
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(records, file, indent=2, ensure_ascii=False)

        print(f"{filename}: {len(records)} cases")

    print(f"Random seed: {SEED}")
    print(f"Total: {len(train) + len(validation) + len(test)} cases")

if __name__ == "__main__":
    main()
