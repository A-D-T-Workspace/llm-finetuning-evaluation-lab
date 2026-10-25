from sklearn.model_selection import train_test_split

def split_dataset(
    cases: list[dict],
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> dict[str, list[dict]]:
    fractions = [train_fraction, validation_fraction, test_fraction]

    if any(fraction <= 0 or fraction >= 1 for fraction in fractions):
        raise ValueError("All split fractions must be between 0 and 1.")

    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("Split fractions must sum to 1.")

    if not cases:
        raise ValueError("Dataset must not be empty.")

    if len(cases) < 3:
        raise ValueError("At least 3 records are required to split the dataset.")

    labels = [case["expected_route"] for case in cases]

    try:
        train_cases, remaining_cases = train_test_split(
            cases,
            train_size=train_fraction,
            random_state=seed,
            stratify=labels,
        )
    except ValueError as error:
        raise ValueError(
            "Cannot stratify this dataset. Add more examples per route."
        ) from error

    remaining_fraction = validation_fraction + test_fraction
    test_share_of_remaining = test_fraction / remaining_fraction

    remaining_labels = [
        case["expected_route"] for case in remaining_cases
    ]

    try:
        validation_cases, test_cases = train_test_split(
            remaining_cases,
            test_size=test_share_of_remaining,
            random_state=seed,
            stratify=remaining_labels,
        )
    except ValueError as error:
        raise ValueError(
            "Cannot create stratified validation/test sets. "
            "Add more examples per route."
        ) from error

    return {
        "train": train_cases,
        "validation": validation_cases,
        "test": test_cases,
    }
