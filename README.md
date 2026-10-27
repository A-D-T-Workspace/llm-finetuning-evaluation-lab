# LLM Fine-Tuning & Evaluation Lab

## Overview

An experimental ML engineering project investigating whether LoRA fine-tuning improves a small language model's performance on customer-support routing and security-escalation tasks.

The project emphasizes reproducible experiments, dataset validation, frozen evaluation data, multi-dimensional metrics, and honest reporting of limitations.

## Research Questions

- Does LoRA fine-tuning improve support-request routing over the base model?
- Do routing improvements translate into better security-escalation decisions?
- How do invalid structured outputs affect end-to-end task performance?
- Why should model selection rely on validation data rather than repeated tuning against a test set?

## Models and Tools

- **Base model:** `Qwen/Qwen2.5-0.5B-Instruct`
- **Fine-tuning method:** Low-Rank Adaptation (LoRA)
- **Python:** 3.12
- **PyTorch, Transformers, PEFT, scikit-learn**
- **Google Colab:** Cloud-based LoRA training workflow
- **Cursor:** Local development and evaluation
- **Hardware:** CPU-only local environment


The adapter weights are stored locally and excluded from Git. Reproduce the adapter evaluation only after obtaining the corresponding adapter artifact.

## Dataset and Experimental Design

The expanded dataset contains 40 support-request examples, divided into:

| Split | Cases |
|---|---:|
| Training | 28 |
| Validation | 6 |
| Frozen test | 6 |
| **Total** | **40** |

The test split is kept separate from model selection. The expanded labels are proposed labels and have not yet been independently adjudicated. Consequently, results should be treated as exploratory.

Tasks:
- Classify each request into one of five support routes.
- Predict whether the request requires security escalation.

## Evaluation

The base model and LoRA adapter were evaluated on the same six frozen test cases using the same system prompt, output parser, and metrics.

| Metric | Base model | LoRA adapter |
|---|---:|---:|
| Route accuracy | 33.3% | 66.7% |
| Route macro-F1 | 26.7% | 63.3% |
| Escalation precision | 20.0% | 20.0% |
| Escalation recall | 50.0% | 50.0% |
| Exact-match accuracy | 16.7% | 16.7% |
| Invalid responses | 1 | 1 |

### Interpretation

The adapter improved route classification on this test sample, but did not improve escalation precision, escalation recall, or exact-match accuracy. Both models produced one invalid response.

These results do not establish generalization or production readiness. With only six test cases, individual predictions have a large influence on the reported metrics.

Detailed results are saved in:
- `reports/frozen_test_comparison.json`
- `reports/frozen_test_analysis.md`

An earlier prompt-based baseline experiment on 20 handcrafted examples is documented in `reports/qwen_baseline_results.json`, with its analysis in `reports/original_adapter_test_analysis.md`. These are separate experiments and should not be compared directly with the six-case expanded test results.

## Reproducing the Evaluation

Create and activate a Python 3.12 virtual environment, then install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

Run the automated tests:

```powershell
python -m pytest -q
```

Run the frozen test comparison:

```powershell
python -m src.llm_eval.compare_models
```

The comparison command requires access to the Hugging Face base model and a compatible