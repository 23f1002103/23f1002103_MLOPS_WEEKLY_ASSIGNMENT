# Model Card: IRIS Species Classifier

## Model Overview
- **Model type**: Random Forest Classifier (scikit-learn)
- **Task**: Multi-class classification — predicts iris flower species (setosa, versicolor, virginica) from 4 physical measurements.
- **Intended use**: Educational/demo pipeline for practicing fairness auditing, explainability, and drift monitoring in an MLOps context. Not intended for production or any real-world biological/agricultural decision-making.

## Training Data
- **Source**: IRIS dataset (`data/v1data.csv`), 150 samples total, 4 features + species label.
- **Features used for training**: `sepal_length`, `sepal_width`, `petal_length`, `petal_width`.
- **Excluded feature**: `location` — a randomly generated binary attribute (0/1) added purely for fairness auditing. Not used in training since it carries no real signal and would only add noise.
- **Train/test split**: 80/20, stratified by species, `random_state=42`.

## Performance

### Overall (on 21-sample test set)
| Metric | Value |
|---|---|
| Accuracy | 0.952 |
| Precision (macro) | 0.958 |
| Recall (macro) | 0.944 |

### By sensitive attribute (`location`) — Fairlearn MetricFrame
| Location | Accuracy | Precision | Recall |
|---|---|---|---|
| 0 | 0.875 | 0.889 | 0.917 |
| 1 | 1.000 | 1.000 | 1.000 |

**Interpretation**: Since `location` was randomly assigned and never used in training, the small gap between groups reflects sampling variance from a small test set (only ~8-13 samples per group), not systematic bias. No fairness concerns are expected or found here, by design.

## Explainability (SHAP)
SHAP summary plots were generated for all three classes. For **virginica**, `petal_length` and `petal_width` are the dominant features — high values of both strongly push predictions toward virginica, while low values push away from it. `sepal_length` and `sepal_width` contribute minimally to this class. This aligns with domain knowledge: virginica is the largest of the three iris species.

## Drift Monitoring
A simulated production shift (petal_length +1.5) was tested using the Kolmogorov-Smirnov test:
- `petal_length`: KS=0.416, p≈0.000 → drift detected
- `sepal_length`, `sepal_width`, `petal_width`: KS=0.000, p=1.000 → no drift

Since `petal_length` is a top contributor to virginica predictions (per SHAP analysis), a real-world shift of this kind would likely degrade classification accuracy for virginica vs. versicolor, and should trigger retraining.

## Known Limitations
- Very small dataset (150 samples) — test set fairness metrics are highly sensitive to sampling noise.
- `location` is a synthetic, randomly assigned attribute, not a real demographic or geographic variable — this exercise demonstrates the *auditing methodology*, not a real-world fairness finding.
- No hyperparameter tuning was performed; default RandomForestClassifier settings were used.
- Drift detection was tested on a single simulated shift; real-world drift patterns may be more complex (e.g., affecting multiple features simultaneously, or concept drift where the same measurements map to different labels).

## Fairness Considerations
This model was audited for fairness across a synthetic `location` attribute as a demonstration of using Fairlearn's `MetricFrame`. In a real deployment involving actual sensitive attributes (e.g., geographic region, demographic group), this same methodology should be applied, alongside monitoring for proxy discrimination — where features correlated with a sensitive attribute could still cause biased outcomes even if the sensitive attribute itself is excluded from training.