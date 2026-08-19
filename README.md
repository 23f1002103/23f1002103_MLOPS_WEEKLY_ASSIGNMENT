# Week 9 — ML Governance: Explainability, Fairness & Drift

This week's assignment demonstrates fairness auditing, model explainability, and drift detection on the IRIS classification pipeline.

## Files in this submission

| File | Purpose |
|---|---|
| `task1_location.py` | Adds a random binary `location` column to the IRIS dataset (sensitive attribute for fairness auditing, excluded from training) |
| `task2_fairness.py` | Trains a RandomForest classifier on the 4 original IRIS features, then audits fairness across `location` groups using Fairlearn's `MetricFrame` |
| `task3_shap.py` | Generates SHAP summary plots for all 3 IRIS classes to explain feature contributions to predictions |
| `task4_drift.py` | Simulates data drift by shifting `petal_length` in a copy of the dataset, then detects it using the Kolmogorov-Smirnov test |
| `MODEL_CARD.md` | Model card documenting intended use, training data, performance, fairness, and limitations (Task 5, bonus) |

## Outputs generated
- `data/iris_with_location.csv` — IRIS data with the added `location` column
- `iris_rf_model.bin`, `X_test.csv`, `y_test.csv` — trained model and test split, reused across tasks
- `shap_summary_setosa.png`, `shap_summary_versicolor.png`, `shap_summary_virginica.png` — SHAP summary plots per class
- `drift_comparison.png` — histogram comparison of original vs. simulated "production" feature distributions

## How to run
```bash
pip install fairlearn shap scikit-learn pandas matplotlib scipy joblib

python task1_location.py
python task2_fairness.py
python task3_shap.py
python task4_drift.py
```

## Key Results Summary

**Fairness (Task 2)**: Overall accuracy 0.952. Accuracy by `location` group: 0.875 (group 0) vs 1.000 (group 1). Since `location` is randomly assigned and unused in training, this gap reflects sampling noise on a small (21-sample) test set rather than real bias.

**Explainability (Task 3)**: For the `virginica` class, `petal_length` and `petal_width` are the dominant features — high values push predictions toward virginica, low values push away. Sepal measurements contribute minimally.

**Drift (Task 4)**: Simulating a +1.5 shift in `petal_length` produces a KS statistic of 0.416 (p≈0.000), correctly flagged as drift. Unshifted features correctly show no drift (p=1.000). Since `petal_length` is a top predictor for virginica, this kind of drift would likely degrade classification performance and should trigger model retraining.

## Author
Pruthvi Prasad S — 23f1002103
