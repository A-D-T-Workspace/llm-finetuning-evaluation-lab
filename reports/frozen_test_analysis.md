# Frozen Test Set: Base Model vs. LoRA Adapter

## Experimental setup

The base model, `Qwen/Qwen2.5-0.5B-Instruct`, and its LoRA adapter were evaluated on the same six frozen test cases using the same prompt, output parser, and evaluation metrics.

The expanded dataset contains proposed labels that have not been independently adjudicated. Results are therefore exploratory, not definitive evidence of real-world performance.

## Results

| Metric | Base model | LoRA adapter |
|---|---:|---:|
| Route accuracy | 33.3% | 66.7% |
| Route macro F1 | 26.7% | 63.3% |
| Escalation precision | 20.0% | 20.0% |
| Escalation recall | 50.0% | 50.0% |
| Exact-match accuracy | 16.7% | 16.7% |
| Invalid responses | 1 | 1 |

## Error analysis

- **Routing:** The adapter corrected two routing errors, in `case_036` and `case_019`, without introducing new routing errors on the other cases.
- **Escalation:** Both models produced the same escalation decisions. They incorrectly escalated four cases and missed one case requiring escalation.
- **Output validity:** Both models produced one invalid response, which was converted to fallback values by the existing parser.
- **Exact match:** Both models correctly predicted both fields on only `case_015`.

## Interpretation

The adapter showed better routing performance on this small test set, while escalation behavior and exact-match accuracy did not improve. This suggests that improvements in one output dimension do not necessarily translate into improvements in overall task performance.

The results do not establish that the adapter generalizes better. The test set contains only six cases, and the labels have not been independently verified. Each individual test case has substantial influence on the reported metrics.

## Limitations and next steps

1. Independently review and adjudicate the expanded dataset labels.
2. Investigate invalid JSON and the fallback behavior.
3. Analyze escalation false positives and false negatives using a separate development set.
4. Keep the frozen test set unchanged during prompt or model selection.
5. Repeat evaluation on a larger, representative, independently labeled test set before making deployment claims.

**Conclusion:** In this exploratory run, LoRA improved route classification but did not improve escalation performance or exact-match accuracy. Further validation is required before drawing generalization or deployment conclusions.