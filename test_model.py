# Test suite for IRIS - Week 5
# Fetches model from MLflow Model Registry
import pandas as pd
import csv
import os
import mlflow
import mlflow.sklearn

DATA_PATH = "data/v1data.csv"
EVAL_DATA_PATH = "eval_data.csv"
METRICS_PATH = "metrics.csv"

MLFLOW_TRACKING_URI = "http://34.135.49.194:5000"
MODEL_ALIAS = "models:/iris-model/latest"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# ---------- Data validation tests ----------

def test_data_file_exists():
    assert os.path.exists(DATA_PATH), "Data file is missing"

def test_data_schema():
    data = pd.read_csv(DATA_PATH)
    expected_columns = {"sepal_length", "sepal_width", "petal_length", "petal_width", "species"}
    assert expected_columns.issubset(set(data.columns))

def test_no_missing_values():
    data = pd.read_csv(DATA_PATH)
    assert data.isnull().sum().sum() == 0

def test_feature_types():
    data = pd.read_csv(DATA_PATH)
    numeric_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    for col in numeric_cols:
        assert pd.api.types.is_numeric_dtype(data[col])

def test_value_ranges():
    data = pd.read_csv(DATA_PATH)
    numeric_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    for col in numeric_cols:
        assert (data[col] > 0).all()

# ---------- Model evaluation tests (from MLflow) ----------

def test_model_loads_from_registry():
    model = mlflow.sklearn.load_model(MODEL_ALIAS)
    assert model is not None, "Could not load model from MLflow registry"

def test_eval_data_exists():
    assert os.path.exists(EVAL_DATA_PATH), "eval_data.csv is missing"

def test_model_predicts():
    model = mlflow.sklearn.load_model(MODEL_ALIAS)
    data = pd.read_csv(EVAL_DATA_PATH)
    X = data.drop("species", axis=1)
    predictions = model.predict(X)
    assert len(predictions) == len(X)

def test_accuracy_threshold():
    assert os.path.exists(METRICS_PATH), "metrics.csv is missing"
    with open(METRICS_PATH) as f:
        reader = csv.DictReader(f)
        row = next(reader)
    accuracy = float(row["accuracy"])
    assert accuracy >= 0.90, f"Accuracy too low: {accuracy}"
# Test suite for IRIS - Week 5
# Fetches model from MLflow Model Registry
import pandas as pd
import csv
import os
import mlflow
import mlflow.sklearn

DATA_PATH = "data/v1data.csv"
EVAL_DATA_PATH = "eval_data.csv"
METRICS_PATH = "metrics.csv"

MLFLOW_TRACKING_URI = "http://34.135.49.194:5000"
MODEL_ALIAS = "models:/iris-model/latest"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# ---------- Data validation tests ----------

def test_data_file_exists():
    assert os.path.exists(DATA_PATH), "Data file is missing"

def test_data_schema():
    data = pd.read_csv(DATA_PATH)
    expected_columns = {"sepal_length", "sepal_width", "petal_length", "petal_width", "species"}
    assert expected_columns.issubset(set(data.columns))

def test_no_missing_values():
    data = pd.read_csv(DATA_PATH)
    assert data.isnull().sum().sum() == 0

def test_feature_types():
    data = pd.read_csv(DATA_PATH)
    numeric_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    for col in numeric_cols:
        assert pd.api.types.is_numeric_dtype(data[col])

def test_value_ranges():
    data = pd.read_csv(DATA_PATH)
    numeric_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    for col in numeric_cols:
        assert (data[col] > 0).all()

# ---------- Model evaluation tests (from MLflow) ----------

def test_model_loads_from_registry():
    model = mlflow.sklearn.load_model(MODEL_ALIAS)
    assert model is not None, "Could not load model from MLflow registry"

def test_eval_data_exists():
    assert os.path.exists(EVAL_DATA_PATH), "eval_data.csv is missing"

def test_model_predicts():
    model = mlflow.sklearn.load_model(MODEL_ALIAS)
    data = pd.read_csv(EVAL_DATA_PATH)
    X = data.drop("species", axis=1)
    predictions = model.predict(X)
    assert len(predictions) == len(X)

def test_accuracy_threshold():
    assert os.path.exists(METRICS_PATH), "metrics.csv is missing"
    with open(METRICS_PATH) as f:
        reader = csv.DictReader(f)
        row = next(reader)
    accuracy = float(row["accuracy"])
    assert accuracy >= 0.90, f"Accuracy too low: {accuracy}"