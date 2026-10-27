# Original LoRA Adapter: Held-Out Test Analysis

## Experiment context

- Base model: `Qwen/Qwen2.5-0.5B-Instruct`
- Fine-tuning method: LoRA
- Random seed: 42
- Training examples: 12
- Validation examples: 4
- Held-out test examples: 4

## Recorded results

| Metric | Base model | Fine-tuned model |
|---|---:|---:|
| Route accuracy | 25% | 0% |
| Escalation accuracy | 75% | 50% |
| Exact-match accuracy | 0% | 0% |

## Observed failure cases

- `case_001`: Password/login request classified as `account_security` instead of `account_support`; escalation was incorrectly set to true.
- `case_018`: Impersonation and bank-detail phishing classified as `billing_support` instead of `account_security`; escalation was correct.
- `case_016`: Premium-plan feature question classified as `billing_support` instead of `general_support`; escalation was correct.
- `case_002`: Duplicate-charge complaint classified as `order_support` instead of `billing_support`; escalation was incorrectly set to true.

## Interpretation

The recorded fine-tuned model achieved 0/4 route accuracy, 2/4 escalation accuracy, and 0/4 exact-match accuracy. Its validation metrics were 100% on only four examples, but this did not transfer to the held-out test set.

These results are preliminary because the test set contains only four cases. Possible explanations include limited training data, route-label ambiguity, overfitting, and a mismatch between training and inference formats. The evidence does not establish which explanation is responsible.

## Evaluation integrity

The original test cases and their labels must remain unchanged. Diagnose and compare future experiments using training and validation data first. Evaluate the frozen test set only for the final comparison, and report all failures rather than selecting only favorable examples.
