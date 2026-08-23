# Week 10 - LLMOps: Fine-Tuning Gemini on IRIS Pipeline

## Overview
This assignment demonstrates LLMOps principles by fine-tuning a Gemini model on the IRIS dataset using two different data representations and comparing their performance.

## Project Structure
- `task1_data_prep.ipynb` - Data preparation for V1 and V2 formats
- `task4_evaluation.ipynb` - Model evaluation and comparison
- `evaluate_llm.py` - CI/CD evaluation script
- `iris_test.csv` - Test dataset
- `iris_v1_train.jsonl` - V1 Raw format training data
- `iris_v2_train.jsonl` - V2 Natural language training data
- `.github/workflows/llmops_eval.yml` - CI/CD pipeline

## LLMOps Lifecycle

### Task 1 & 2 - Data Preparation
- Converted IRIS dataset into two JSONL formats
- V1 (Raw): `sepal_length: 5.1, sepal_width: 3.5...`
- V2 (Natural Language): `A flower specimen has a sepal length of 5.1 cm...`
- Uploaded to GCS: `gs://llmops-iris-week10/data/`

### Task 3 - Fine-Tuning on Vertex AI
- Base model: `gemini-2.5-flash-lite`
- Two fine-tuning jobs submitted on Vertex AI
- Same hyperparameters for both (controlled experiment)

### Task 4 - Evaluation Results

| Metric | V1 (Raw Format) | V2 (Natural Language) |
|--------|----------------|----------------------|
| Accuracy | 71.4% | 47.6% |
| Format Compliance | 76.2% | 85.7% |

### Why V1 Performed Better?
- IRIS is structured numerical data
- Raw format is more precise for numbers
- Natural language adds unnecessary complexity
- LLMs work better with natural language for text tasks, not numerical classification

### Task 5 - CI/CD Pipeline
- GitHub Actions workflow triggers on push to `week_10`
- Validates data format and checks accuracy threshold (60%)
- Fails pipeline if accuracy drops below threshold

## GCP Resources
- Project: `project-b38b370e-25fb-420a-a54`
- Region: `us-central1`
- GCS Bucket: `llmops-iris-week10`
- V1 Model: `gemini-2.5-flash-lite` fine-tuned on raw format
- V2 Model: `gemini-2.5-flash-lite` fine-tuned on natural language
