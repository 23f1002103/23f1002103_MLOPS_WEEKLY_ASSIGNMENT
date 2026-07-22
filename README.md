# IRIS MLOps Pipeline - Week 5: MLflow Integration

## Files

### Code/Scripts
- `train.py` - Training pipeline with hyperparameter tuning and MLflow experiment tracking. Logs parameters, metrics, and registers best model in MLflow Model Registry.
- `evaluate.py` - Evaluation pipeline that fetches the latest model directly from MLflow Model Registry and computes accuracy, precision, and recall.

### Output Files
- `metrics.csv` - Contains accuracy, precision, and recall of the best trained model.
- `eval_data.csv` - Evaluation dataset generated during training.

## MLflow Setup
- Tracking URI: http://34.135.49.194:5000
- Experiment: iris-experiment
- Registered Model: iris-model
