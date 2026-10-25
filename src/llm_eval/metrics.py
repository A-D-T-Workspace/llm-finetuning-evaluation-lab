from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

def evaluate_routes(expected: list[str], predicted: list[str]) -> dict:
    if not expected or len(expected) != len(predicted):
        raise ValueError(
            "Expected and predicted labels must have equal, non-zero lengths."
        )

    labels = sorted(set(expected) | set(predicted))

    return {
        "accuracy": accuracy_score(expected, predicted),
        "macro_f1": f1_score(
            expected,
            predicted,
            labels=labels,
            average="macro",
            zero_division=0,
        ),
        "per_class": classification_report(
            expected,
            predicted,
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
    }

def evaluate_safety(expected: list[bool], predicted: list[bool]) -> dict:
    if not expected or len(expected) != len(predicted):
        raise ValueError(
            "Expected and predicted safety labels must have equal, "
            "non-zero lengths."
        )

    return {
        "precision": precision_score(expected, predicted, zero_division=0),
        "recall": recall_score(expected, predicted, zero_division=0),
    }
