# LLM Fine-Tuning & Evaluation Lab

## Project Overview

This project explores how to evaluate and improve fine-tuned language models using sound machine learning practices.

The focus is on LoRA fine-tuning, validation, loss analysis, dataset quality, checkpoint selection, and fair model comparisons.

## Objectives

- Understand training, validation, and test datasets.
- Identify overfitting and underfitting.
- Learn how LoRA fine-tuning works.
- Compare base and fine-tuned model performance.
- Design reproducible evaluation experiments.

## Tools

- Python 3.12
- PyTorch
- Hugging Face Transformers
- Cursor

## Current Status

Development environment configured and verified.

## Hardware Note

The current PyTorch installation is CPU-only. Initial exercises will use lightweight examples before attempting larger language models.

## Results and Evaluation

### Experimental Setup
- **Base model:** Qwen2.5-0.5B-Instruct
- **Evaluation dataset:** 20 handcrafted support-request examples
- **Tasks:** Support-request routing and security escalation
- **Evaluation metrics:** Accuracy, macro-F1, precision, and recall
- **Hardware:** CPU-only local environment

### Baseline Results

| Metric | Earlier prompt | Latest prompt |
|---|---:|---:|
| Routing accuracy | 0.40 | 0.30 |
| Routing macro-F1 | 0.26 | 0.21 |
| Escalation precision | 0.67 | 0.29 |
| Escalation recall | 0.67 | 0.83 |
| Invalid responses | 0 | 3 |

### Findings
- Prompt changes affected routing and escalation performance differently.
- The latest prompt improved escalation recall but produced substantially more false positives and invalid responses.
- The earlier prompt offered a better balance between escalation precision and recall.
- These results demonstrate why model evaluation must consider multiple metrics rather than accuracy alone.

### Limitations
- The dataset contains only 20 handcrafted examples and is too small to establish reliable generalization.
- Results are preliminary and should not be interpreted as production-level performance.
- No fine-tuning experiment has been completed yet; these results are from prompt-based baseline evaluation.
- Further work should use a larger, more diverse dataset and a controlled comparison between the baseline and a fine-tuned model.
