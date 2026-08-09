# Week 8: MLSecOps — Data Poisoning Simulation on IRIS Pipeline

## Overview
This assignment simulates a data poisoning attack on the IRIS classification
pipeline, measures its impact using MLflow, and reasons about mitigation
strategies in a production ML system.

## Files
- `poison_data.py` — generates poisoned variants of `data/v1data.csv`
  at 5%, 10%, and 50% corruption levels.
- `train_mlflow.py` — trains a DecisionTreeClassifier on the clean
  dataset and each poisoned variant, logging parameters and metrics
  (accuracy, precision, recall, F1) to MLflow.
- `data/iris_clean.csv`, `data/iris_poisoned_5.csv`,
  `data/iris_poisoned_10.csv`, `data/iris_poisoned_50.csv` — generated
  datasets used for training.

## Threat Vectors Covered (Task 1)
- **Data Poisoning** — attacker corrupts training data (this assignment's focus).
  Targets: data ingestion / training stage.
- **Adversarial Examples** — small crafted perturbations at inference time
  cause misclassification. Targets: inference stage.
- **Model Extraction** — attacker repeatedly queries a deployed model to
  reconstruct its behavior/parameters. Targets: inference API.
- **Prompt Injection** — malicious instructions embedded in user input to an
  LLM-based system, overriding intended behavior. Targets: natural language
  interfaces.

## Poisoning Method (Task 2)
For each corruption level, a random subset of rows equal to that percentage
of the dataset had all four features replaced with random values (drawn
from the observed min/max range of each feature) and the label reassigned
to a random class.

## Results (Task 3 & 4)

| Poison Level | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| 0% (clean)   | 90.48% | 0.9238 | 0.9048 | 0.8942 |
| 5%           | 85.71% | 0.8519 | 0.8571 | 0.8511 |
| 10%          | 85.71% | 0.8571 | 0.8571 | 0.8540 |
| 50%          | 61.90% | 0.6071 | 0.6190 | 0.6107 |

**Analysis:**
- The model degrades even at 5% corruption.
- All four metrics move together proportionally.
- At 50%, accuracy drops sharply but stays above pure-random (~33%).

## Mitigation Strategies (Task 5)
- Statistical validation / anomaly detection on incoming data.
- Schema enforcement at ingestion.
- Data provenance tracking.
- Data quantity vs. quality — more data does NOT compensate for poisoned
  data; clean/filter first, then assess if remaining clean data suffices.

## Reproduce
\`\`\`bash
python poison_data.py
python train_mlflow.py
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000
\`\`\`
