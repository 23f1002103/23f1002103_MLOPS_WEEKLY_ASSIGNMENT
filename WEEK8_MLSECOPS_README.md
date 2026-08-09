# Week 8: MLSecOps — Data Poisoning Simulation on IRIS Pipeline

## Overview
This assignment simulates a data poisoning attack on the IRIS classification
pipeline, measures its impact using MLflow, and reasons about mitigation
strategies in a production ML system.

## Files
- `scripts/poison_data.py` — generates poisoned variants of `data/v1data.csv`
  at 5%, 10%, and 50% corruption levels.
- `scripts/train_mlflow.py` — trains a DecisionTreeClassifier on the clean
  dataset and each poisoned variant, logging parameters and metrics
  (accuracy, precision, recall, F1) to MLflow.
- `data/iris_clean.csv`, `data/iris_poisoned_5.csv`,
  `data/iris_poisoned_10.csv`, `data/iris_poisoned_50.csv` — generated
  datasets used for training.

## Threat Vectors Covered (Task 1)
- **Data Poisoning** — attacker corrupts training data (this assignment's focus).
  Targets: data ingestion / training stage.
- **Adversarial Examples** — small crafted perturbations at inference time
  cause misclassification. Targets: inference stage. Model stays correct on
  clean data but fails on manipulated input.
- **Model Extraction** — attacker repeatedly queries a deployed model to
  reconstruct its behavior/parameters. Targets: inference API.
  Mitigation: rate limiting, returning only class labels (not full
  probability distributions).
- **Prompt Injection** — malicious instructions embedded in user input to an
  LLM-based system, overriding intended behavior. Targets: natural language
  interfaces. Analogous to SQL injection.

## Poisoning Method (Task 2)
For each corruption level, a random subset of rows equal to that percentage
of the dataset had all four features replaced with random values (drawn
from the observed min/max range of each feature) and the label reassigned
to a random class. Only training-relevant data was touched — this
simulates label-flipping / noise injection by a malicious data source.

## Results (Task 3 & 4)

| Poison Level | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| 0% (clean)   | 90.48% | 0.9238 | 0.9048 | 0.8942 |
| 5%           | 85.71% | 0.8519 | 0.8571 | 0.8511 |
| 10%          | 85.71% | 0.8571 | 0.8571 | 0.8540 |
| 50%          | 61.90% | 0.6071 | 0.6190 | 0.6107 |

**Analysis:**
- The model begins to noticeably degrade even at **5% corruption** — a ~5
  point drop in accuracy from the clean baseline.
- All four metrics (accuracy, precision, recall, F1) move together and
  roughly proportionally — no single metric is uniquely more sensitive at
  these poisoning levels in this small dataset.
- At **50% corruption**, performance drops sharply (61.9% accuracy) but
  does **not** collapse to pure random guessing (~33% for 3 balanced
  classes) — the model still extracts some signal from the remaining 50%
  clean data, but the decision boundary is significantly degraded.

## Mitigation Strategies (Task 5)
- **Statistical validation / anomaly detection** — profile incoming data
  (feature ranges, distributions) and flag samples that fall outside
  expected bounds before they enter training.
- **Schema enforcement** — reject rows with unexpected types, missing
  fields, or out-of-range values at ingestion.
- **Data provenance tracking** — know where every training sample came
  from; treat unverified/external sources with more scrutiny.
- **Data quantity vs. quality** — more data does NOT compensate for
  poisoned data; a larger poisoned pool amplifies the problem rather than
  diluting it. The priority is cleaning/filtering contaminated samples
  first, then evaluating if the remaining clean subset is sufficient in
  size. In production this means setting a minimum clean-data threshold
  and tracking the clean-to-poisoned ratio, not just raw dataset size.

## Reproduce
\`\`\`bash
python scripts/poison_data.py
python scripts/train_mlflow.py
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000
\`\`\`
